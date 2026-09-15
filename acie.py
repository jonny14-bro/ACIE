"""
acie.py — Adaptive Coding Intelligence Engine  (PRIMA + DELTA Edition)
=======================================================================
Full pipeline:
    router → PRIMA → memory → prompt → task planner →
    DELTA stream → auto-continue → goal-aware → confidence →
    self-correction → (optional refiner)
"""

import re
import os
from rich.console import Console
from rich.panel   import Panel

from router      import make_route, check_confidence, RouteDecision, Confidence
from memory      import MemorySystem
from engine      import build_prompt, stream_response, get_model_name
from engine      import _stream_with_repeat_guard, _has_repetition
from prima       import prima_score, ResourceVector
from delta       import delta_stream, DELTAConfig
from executor import execute_plan_v3

console = Console()


# ── Config ────────────────────────────────────────────────────────────────────

class ACIEConfig:
    confidence_retry_on_low:  bool  = True
    confidence_retry_on_med:  bool  = False
    show_route_info:          bool  = True
    show_prima_info:          bool  = True
    min_text_to_save:         int   = 30
    use_delta:                bool  = True
    use_prima:                bool  = True
    use_planner:              bool  = True
    use_goal_aware:           bool  = True
    use_self_correct:         bool  = True


config = ACIEConfig()


# ── Display helpers ───────────────────────────────────────────────────────────

_MODE_COLORS = {
    "debug":    "red",
    "generate": "cyan",
    "refactor": "yellow",
    "explain":  "blue",
    "general":  "dim white",
}

def _print_route(decision: RouteDecision):
    if not config.show_route_info:
        return
    color    = _MODE_COLORS.get(decision.mode, "white")
    lang     = decision.signals.dominant_language
    lang_str = f"  [{lang}]" if lang != "unknown" else ""
    console.print(
        f"[{color}]⟶  Mode: {decision.mode.upper()}{lang_str}[/{color}]  "
        f"[dim]domain:{decision.memory_domain}  "
        f"complexity:{decision.signals.complexity}[/dim]"
    )


def _print_prima(rv: ResourceVector):
    if not config.show_prima_info:
        return
    bar_len = int(rv.score * 20)
    bar     = "█" * bar_len + "░" * (20 - bar_len)
    console.print(
        f"[magenta]⬡ PRIMA[/magenta]  score:[bold]{rv.score:.2f}[/bold]  "
        f"[dim]{bar}[/dim]  "
        f"tokens:[yellow]{rv.tokens}[/yellow]  "
        f"temp:[yellow]{rv.temperature:.2f}[/yellow]  "
        f"k:[yellow]{rv.memory_k}[/yellow]"
    )


def _print_confidence(conf: Confidence):
    if conf == "high":
        console.print("[dim green]✓ confidence: high[/dim green]")
    elif conf == "medium":
        console.print("[dim yellow]~ confidence: medium[/dim yellow]")
    else:
        console.print("[dim red]⚠ confidence: low[/dim red]")


# ── Incomplete detection ──────────────────────────────────────────────────────

def is_incomplete(response: str) -> bool:
    stripped = response.strip()
    if stripped.count("```") % 2 != 0:
        return True
    if stripped.endswith(("...", "pass", "TODO")):
        return True
    if len(stripped.split()) < 15:
        return True
    if response.count("(") != response.count(")"):
        return True
    return False


# ── Goal extraction & checking ────────────────────────────────────────────────

def extract_goals(user_input: str) -> dict:
    goals = {}
    lower = user_input.lower()
    match = re.search(r"(\d+)\s*lines?", lower)
    if match:
        goals["min_lines"] = int(match.group(1))
    if "complete" in lower or "full" in lower:
        goals["complete"] = True
    if any(w in lower for w in ["code", "program", "script"]):
        goals["code_required"] = True
    return goals


def check_goals(response: str, goals: dict) -> bool:
    if "min_lines" in goals:
        if response.count("\n") < goals["min_lines"]:
            return False
    if goals.get("code_required") and "```" not in response:
        return False
    if goals.get("complete"):
        has_class = "class " in response
        has_func  = "def "   in response
        has_code  = "```"    in response
        if not (has_code and (has_class or has_func)):
            return False
    return True


# ── Repetition cleanup ────────────────────────────────────────────────────────

def _truncate_at_repetition(text: str, window: int = 120) -> str:
    if len(text) < window * 2:
        return text
    phrase = text[-window:]
    idx = text.find(phrase)
    if idx > 0 and idx < len(text) - window:
        return text[:idx].rstrip()
    return text


# ── Task planning ─────────────────────────────────────────────────────────────

def generate_plan(model, user_input, model_type, model_path, max_tokens, temperature):
    """
    Generate a 3-6 step module-level plan.
    Each step = one Python class or module, NOT a method inside a class.
    Explicitly forbids method-level steps with a wrong-example counter-example.
    """
    history = [
        {
            "role": "system",
            "content": (
                "You are a software architect planning a Python project.\n\n"
                "STRICT OUTPUT FORMAT:\n"
                "- Output ONLY a numbered list of 3 to 6 steps\n"
                "- Each step = ONE complete Python class or one main() function\n"
                "- FORBIDDEN: steps that say 'implement method X within class Y'\n"
                "- FORBIDDEN: steps about testing, documentation, or refactoring\n"
                "- FORBIDDEN: splitting one class into multiple steps\n"
                "- Every step must name a CLASS or 'main()' — nothing else\n"
                "- Do NOT write any code\n\n"
                "CORRECT EXAMPLE (follow this exactly):\n"
                "1. Tokenizer class\n"
                "2. SynonymDictionary class\n"
                "3. RuleBasedIntentInferrer class\n"
                "4. MemoryStore class\n"
                "5. Chatbot class integrating all components\n"
                "6. main() entry point with CLI loop\n\n"
                "WRONG EXAMPLE (never do this):\n"
                "2. Implement method tokenize_input within Chatbot  <- FORBIDDEN\n"
                "4. Create method get_synonyms within SynonymDictionary  <- FORBIDDEN\n"
            ),
        },
        {
            "role": "user",
            "content": f"List the classes needed to build this project:\n{user_input}",
        },
    ]
    prompt = build_prompt(history, model_type=model_type)
    return _stream_with_repeat_guard(
        model, prompt, min(max_tokens // 4, 350), temperature, model_type, model_path
    )


def parse_steps(plan: str) -> list:
    """
    Extract numbered steps from a plan, filter method-level steps,
    and deduplicate by class/module name (not verb).

    Two-stage filter:
    1. Drop method-level steps ("implement method X within class Y") —
       these cause the executor to re-generate the parent class, triggering
       false duplicate detection. We only want class/module-level steps.
    2. Deduplicate by class name extracted from backticks or CamelCase —
       NOT by the leading verb (create/implement/develop are all different
       verbs but may refer to the same class).
    """
    raw_steps = []
    for line in plan.split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            clean = re.sub(r"^\d+[\.\)\-\s]+", "", line).strip()
            if clean and len(clean) > 5:
                raw_steps.append(clean)

    # Filter out method-level steps — they always cause the model to
    # re-generate the entire parent class, which the dedup then catches
    # as a false positive and blocks.
    _method_step = re.compile(
        r"\b(method|function)\b.+\bwithin\b|\bwithin\b.+\b(class|module)\b",
        re.IGNORECASE,
    )
    class_steps = [s for s in raw_steps if not _method_step.search(s)]

    # Pre-filter using executor's _extract_definitions logic:
    # if two steps would resolve to the same meaningful def names, drop the
    # second immediately rather than burning retries in the executor.
    try:
        from executor import _extract_definitions as _exec_defs
    except ImportError:
        _exec_defs = None

    if _exec_defs is not None:
        seen_defs: set[str] = set()
        filtered_steps = []
        for step in class_steps:
            # Fabricate a minimal stub so _extract_definitions can parse the name
            # from the step description by checking CamelCase / backtick names.
            stub_names: set[str] = set()
            m = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)`", step)
            if m:
                stub_names.add(m.group(1))
            for camel in re.findall(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", step):
                stub_names.add(camel)

            # Drop if every extracted name was already seen
            if stub_names and stub_names.issubset(seen_defs):
                print(f"   [parse_steps] early-dedup drop: {step!r} overlaps {stub_names & seen_defs}")
                continue

            seen_defs.update(stub_names)
            filtered_steps.append(step)
        class_steps = filtered_steps

    # Extract the class/module name as the dedup key.
    # Priority: backtick name > CamelCase word > first long word.
    def _key(step: str) -> str:
        m = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)`", step)
        if m:
            return m.group(1).lower()
        m = re.search(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", step)
        if m:
            return m.group(1).lower()
        words = step.split()
        return next((w.lower() for w in words if len(w) > 4), words[0].lower() if words else step)

    seen_keys = set()
    deduped   = []
    for step in class_steps:
        key = _key(step)
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(step)

    return deduped[:6]


# ── Self-correction ───────────────────────────────────────────────────────────

def self_correct_response(model, response, user_input, model_type, model_path, max_tokens, temperature):
    history = [
        {"role": "system",    "content": "You are a software engineer fixing an incomplete response."},
        {"role": "user",      "content": user_input},
        {"role": "assistant", "content": response},
        {"role": "user",      "content": (
            "The response above is incomplete or incorrect. Fix it:\n"
            "- Do NOT remove correct parts\n"
            "- Complete any missing code\n"
            "- Output only the final corrected version"
        )},
    ]
    prompt = build_prompt(history, model_type=model_type)
    return stream_response(model, prompt, max_tokens, temperature, model_type, model_path)


# ── Core pipeline ─────────────────────────────────────────────────────────────

def _run_pipeline(
    user_input:    str,
    model,
    model_type:    str,
    model_path:    str,
    max_tokens:    int,
    temperature:   float,
    memory:        MemorySystem | None,
    base_history:  list[dict],
    refiner_model  = None,
) -> tuple[str, RouteDecision, Confidence]:

    # ── 1. Route ─────────────────────────────────────────────────────────────
    decision = make_route(user_input)
    _print_route(decision)

    # ── 2. PRIMA ─────────────────────────────────────────────────────────────
    if config.use_prima:
        resource = prima_score(
            user_input  = user_input,
            mode        = decision.mode,
            memory      = memory,
            domain      = decision.memory_domain,
            base_tokens = max_tokens,
            base_temp   = temperature,
            base_k      = 3,
        )
        _print_prima(resource)
        effective_tokens = resource.tokens
        effective_temp   = resource.temperature
        effective_k      = resource.memory_k
    else:
        effective_tokens = max_tokens
        effective_temp   = temperature
        effective_k      = 3
        if decision.mode in ("generate", "refactor") and decision.signals.complexity in ("medium", "high"):
            effective_tokens = min(max_tokens * 2, 3000)

    # ── 3. Memory retrieval ───────────────────────────────────────────────────
    temp_history = list(base_history)

    if temp_history and temp_history[0]["role"] == "system":
        temp_history[0] = {"role": "system", "content": decision.system_prompt}
    else:
        temp_history.insert(0, {"role": "system", "content": decision.system_prompt})

    if memory:
        ctx = memory.build_context(user_input, domain=decision.memory_domain)
        if ctx.strip():
            temp_history.append({"role": "system", "content": ctx})

    temp_history.append({"role": "user", "content": user_input})

    # ── 4. Prompt building ────────────────────────────────────────────────────
    prompt = build_prompt(temp_history, model_type=model_type)

    # ── 5. Task planning ──────────────────────────────────────────────────────
    ran_planner = False
    _PROJECT_SCALE = re.compile(
        r"\b(project|system|application|platform|app|cli|tool|engine|"
        r"framework|service|server|client|bot|dashboard|api)\b",
        re.IGNORECASE,
    )
    is_project_scale = bool(_PROJECT_SCALE.search(user_input))
    use_planner = (
        config.use_planner and
        decision.mode in ("generate", "refactor") and
        (decision.signals.complexity in ("medium", "high") or is_project_scale)
    )

    if use_planner:
        console.print("[magenta]🧠 Generating plan...[/magenta]")
        plan  = generate_plan(model, user_input, model_type, model_path, effective_tokens, effective_temp)
        steps = parse_steps(plan)

        if steps:
            console.print(f"[dim]Steps: {len(steps)}[/dim]")
            response    = execute_plan_v3(
                model, steps, user_input, memory,
                model_type, model_path, effective_tokens, effective_temp,
            )
            ran_planner = True
        else:
            console.print("[yellow]⚠ No steps parsed — falling back to direct generation[/yellow]")

    if not ran_planner:
        # ── 6. DELTA monitored streaming ──────────────────────────────────────
        if config.use_delta and memory is not None:
            response = delta_stream(
                model       = model,
                prompt      = prompt,
                user_input  = user_input,
                max_tokens  = effective_tokens,
                temperature = effective_temp,
                model_type  = model_type,
                model_path  = model_path,
                embedder    = memory.embedder,
                domain      = decision.memory_domain,
            )
        else:
            response = stream_response(
                model, prompt, effective_tokens, effective_temp, model_type, model_path
            )

    # ── 7. Auto-continue ──────────────────────────────────────────────────────
    max_loops  = 2
    loop_count = 0

    while is_incomplete(response) and loop_count < max_loops:
        if _has_repetition(response):
            console.print("[yellow]⚠ Repetition in response — skipping auto-continue[/yellow]")
            response = _truncate_at_repetition(response)
            break

        console.print("[yellow]↻ Continuing incomplete response...[/yellow]")
        tail = response.rstrip()[-500:]
        cont_history = [
            {"role": "system",    "content": "You are a software engineer. Continue the code from where it was cut off."},
            {"role": "user",      "content": f"Here is what you wrote so far:\n{tail}\n\nContinue from exactly where the code above stopped. Do not repeat any of it."},
        ]
        cont_prompt  = build_prompt(cont_history, model_type=model_type)
        continuation = _stream_with_repeat_guard(
            model, cont_prompt, effective_tokens, effective_temp, model_type, model_path,
        )
        if not _has_repetition(continuation):
            response += "\n" + continuation
        loop_count += 1

    # ── 8. Goal-aware enforcement ─────────────────────────────────────────────
    if config.use_goal_aware:
        goals = extract_goals(user_input)

        if goals:
            console.print("[blue]🎯 Applying goal-aware generation...[/blue]")
            count = 0

            while not check_goals(response, goals) and count < 2:
                if _has_repetition(response):
                    response = _truncate_at_repetition(response)
                    break

                console.print("[yellow]↻ Expanding to meet goals...[/yellow]")
                tail         = response.rstrip()[-500:]
                what_missing = []
                if "min_lines" in goals:
                    what_missing.append(f"at least {goals['min_lines']} lines of code")
                if goals.get("code_required"):
                    what_missing.append("a working code block")
                if goals.get("complete"):
                    what_missing.append("a full working implementation")

                goal_history = [
                    {"role": "system",    "content": "You are a software engineer completing a coding task."},
                    {"role": "user",      "content": user_input},
                    {"role": "assistant", "content": tail},
                    {"role": "user",      "content": (
                        f"The response above is not yet complete. "
                        f"Please add the missing parts: {', '.join(what_missing)}. "
                        f"Do not repeat any code already shown."
                    )},
                ]
                goal_prompt = build_prompt(goal_history, model_type=model_type)
                extra       = _stream_with_repeat_guard(
                    model, goal_prompt, effective_tokens, effective_temp, model_type, model_path,
                )
                if not _has_repetition(extra):
                    response += "\n" + extra
                count += 1

            if count > 0:
                console.print(f"[dim]✔ Goals satisfied after {count+1} passes[/dim]")

    # ── 9. Confidence evaluation ──────────────────────────────────────────────
    # Planner output: assembled multi-module code — don't penalise hedging words
    # that naturally appear in code comments or docstrings.
    if ran_planner:
        has_code   = "def " in response or "class " in response
        is_ok      = not is_incomplete(response)
        confidence = "high" if (has_code and is_ok) else "medium"
    else:
        confidence = check_confidence(response, decision.mode)
    _print_confidence(confidence)

    # ── 10. Self-correction (non-planner only) ────────────────────────────────
    if config.use_self_correct and confidence == "low" and not ran_planner:
        console.print("[red]↻ Attempting self-correction...[/red]")
        fixed = self_correct_response(
            model, response, user_input, model_type, model_path,
            effective_tokens, effective_temp,
        )
        if len(fixed.strip()) > len(response.strip()) * 0.6:
            response = fixed
        confidence = check_confidence(response, decision.mode)
        console.print(f"[dim]After correction: {confidence}[/dim]")

    # ── 11. Optional refiner model ────────────────────────────────────────────
    should_refine = (
        refiner_model is not None and (
            (confidence == "low"    and config.confidence_retry_on_low) or
            (confidence == "medium" and config.confidence_retry_on_med)
        )
    )

    if should_refine:
        console.print("[bold yellow]↻  Escalating to refiner model…[/bold yellow]")
        refined = stream_response(
            refiner_model, prompt, effective_tokens, effective_temp, model_type, model_path
        )
        if len(refined.strip()) > len(response.strip()) * 0.6:
            response = refined
        confidence = check_confidence(response, decision.mode)
        console.print(f"[dim]Refiner confidence: {confidence}[/dim]")

    return response, decision, confidence


# ── Main chat loop ────────────────────────────────────────────────────────────

def run_acie_chat(
    model,
    model_path:    str,
    model_type:    str   = "qwen",
    max_tokens:    int   = 2048,
    temperature:   float = 0.7,
    system_prompt: str   = None,
    memory:        MemorySystem | None = None,
    refiner_model        = None,
    refiner_path:  str   = None,
):
    model_name = get_model_name(model_path)

    console.print(Panel.fit(
        f"[bold cyan]{model_name}[/bold cyan]  via  [bold magenta]ACIE[/bold magenta]\n"
        "[dim]Adaptive Coding Intelligence Engine — PRIMA + DELTA Edition[/dim]\n"
        f"[dim]refiner:{' ✓' if refiner_model else ' ✗'}  "
        f"memory:{' ✓' if memory else ' ✗'}  "
        f"PRIMA:{' ✓' if config.use_prima else ' ✗'}  "
        f"DELTA:{' ✓' if config.use_delta else ' ✗'}[/dim]",
        border_style="cyan",
    ))
    console.print(
        "[dim]Type 'exit' to quit  ·  '/mode' to see last route  ·  "
        "'/prima' to toggle PRIMA  ·  '/delta' to toggle DELTA  ·  "
        "'/stats' for memory  ·  '/help' for all commands[/dim]\n"
    )

    base_history: list[dict] = [{
        "role":    "system",
        "content": system_prompt or "You are a helpful AI assistant running locally.",
    }]

    last_decision: RouteDecision = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        if user_input.lower() == "/mode" and last_decision:
            console.print(
                f"[dim]Last route: mode={last_decision.mode}  "
                f"domain={last_decision.memory_domain}  "
                f"lang={last_decision.signals.dominant_language}[/dim]"
            )
            continue

        if user_input.lower() == "/prima":
            config.use_prima = not config.use_prima
            console.print(f"[magenta]PRIMA {'enabled' if config.use_prima else 'disabled'}[/magenta]")
            continue

        if user_input.lower() == "/delta":
            config.use_delta = not config.use_delta
            console.print(f"[red]DELTA {'enabled' if config.use_delta else 'disabled'}[/red]")
            continue

        if user_input.lower() == "/help":
            console.print(
                "[dim]Commands:\n"
                "  /mode   — show last routing decision\n"
                "  /prima  — toggle PRIMA pre-generation tuning\n"
                "  /delta  — toggle DELTA drift control\n"
                "  /stats  — show memory domain counts\n"
                "  exit    — quit[/dim]"
            )
            continue

        if user_input.lower() == "/stats" and memory:
            for d, entries in memory.texts.items():
                if entries:
                    console.print(f"[dim]  {d}: {len(entries)} entries[/dim]")
            continue

        response, decision, confidence = _run_pipeline(
            user_input    = user_input,
            model         = model,
            model_type    = model_type,
            model_path    = model_path,
            max_tokens    = max_tokens,
            temperature   = temperature,
            memory        = memory,
            base_history  = base_history,
            refiner_model = refiner_model,
        )
        last_decision = decision

        base_history.append({"role": "user",      "content": user_input})
        base_history.append({"role": "assistant", "content": response})

        if memory:
            memory.add_message("user",      user_input)
            memory.add_message("assistant", response)

            combined = f"User: {user_input}\nAssistant: {response}"
            if len(combined) > config.min_text_to_save:
                memory.add_to_domain(combined, decision.memory_domain, quality=confidence)
                if confidence != "low":
                    memory.add_global_memory(combined)

            memory.save_session()
            memory.save_all_domains()
            memory.save_global_memory()
