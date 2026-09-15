"""
validator.py — ACIE Post-Generation Validator
===============================================
Three-layer validation pipeline that runs after execute_plan_v3() assembles
a project. Each layer catches a different class of failure and attempts to
fix it before the final output is returned.

    Layer 1 — Integrity checker
        Detects: missing imports, undefined names used across modules,
        skipped plan steps, broken __init__ signatures.
        Fix: static auto-patcher injects missing glue code.

    Layer 2 — Integration validator
        Detects: class wiring mismatches, method name drift between modules
        (Module A calls B.foo() but B only has B.bar()), broken data flow.
        Fix: LLM targeted re-generation of the single broken module.

    Layer 3 — Completeness audit
        Detects: goal vs output diff (plan steps never implemented),
        placeholder stubs (pass / raise NotImplementedError / TODO),
        missing entry point (no main() or if __name__ == '__main__').
        Fix: LLM gap-fill generation appended as a final module.

Usage (in executor.py after assemble_final()):
    from validator import validate_and_fix
    final_code = validate_and_fix(
        state       = state,
        user_input  = user_input,
        model       = model,
        model_type  = model_type,
        model_path  = model_path,
        max_tokens  = max_tokens,
        temperature = temperature,
    )
"""

import re
import ast
from dataclasses import dataclass, field
from rich.console import Console

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    layer:       int
    kind:        str    # "missing_import" | "undefined_ref" | "method_mismatch" | etc.
    module:      str    # which module the issue lives in
    detail:      str    # human-readable description
    fixable:     bool = True


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    patches_applied: list[str]    = field(default_factory=list)
    llm_repairs:     list[str]    = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0

    def add(self, issue: ValidationIssue):
        self.issues.append(issue)

    def summary(self) -> str:
        if self.ok:
            return "✓ All validation layers passed"
        lines = [f"  {i.layer}. [{i.kind}] in {i.module}: {i.detail}"
                 for i in self.issues]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_safe(code: str) -> ast.Module | None:
    """Try to parse Python code; return None on SyntaxError."""
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _defined_names(code: str) -> set[str]:
    """Return all class and function names defined at module level."""
    names = set()
    tree = _parse_safe(code)
    if not tree:
        return names
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def _called_names(code: str) -> set[str]:
    """Return all bare names that appear in Call nodes (best-effort)."""
    names = set()
    tree = _parse_safe(code)
    if not tree:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # foo() → Name id
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            # obj.method() → Attribute attr
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def _imported_names(code: str) -> set[str]:
    """Return all top-level imported names."""
    names = set()
    tree = _parse_safe(code)
    if not tree:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _stdlib_modules() -> set[str]:
    """A minimal set of standard library module names to avoid false positives."""
    return {
        "os", "sys", "re", "json", "math", "time", "datetime", "random",
        "socket", "threading", "asyncio", "pathlib", "typing", "dataclasses",
        "collections", "itertools", "functools", "io", "abc", "enum",
        "logging", "argparse", "subprocess", "shutil", "copy", "hashlib",
        "uuid", "struct", "contextlib", "traceback", "inspect", "textwrap",
    }


def _has_entry_point(code: str) -> bool:
    """Check for main() def or if __name__ == '__main__' block."""
    return (
        "def main(" in code
        or 'if __name__ == "__main__"' in code
        or "if __name__ == '__main__'" in code
    )


def _has_stubs(code: str) -> list[str]:
    """Return list of stub patterns found: pass-only bodies, TODO, NotImplementedError."""
    stubs = []
    if re.search(r"def \w+\([^)]*\):\s*\n\s+pass\s*$", code, re.MULTILINE):
        stubs.append("pass-only function body")
    if re.search(r"raise NotImplementedError", code):
        stubs.append("NotImplementedError stub")
    if re.search(r"#\s*(TODO|FIXME|PLACEHOLDER)", code, re.IGNORECASE):
        stubs.append("TODO/FIXME comment")
    return stubs


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — Integrity checker + static auto-patcher
# ─────────────────────────────────────────────────────────────────────────────

def _layer1_check(state) -> tuple[ValidationReport, dict[str, str]]:
    """
    Static analysis across all modules.
    Returns (report, patches) where patches is {module_name: extra_code_to_prepend}.
    """
    report  = ValidationReport()
    patches: dict[str, str] = {}

    # Collect all defined names across the whole project
    all_defined: set[str] = set()
    module_defs: dict[str, set[str]] = {}

    for name, code in state.modules.items():
        defs = _defined_names(code)
        module_defs[name] = defs
        all_defined.update(defs)

    stdlib = _stdlib_modules()

    for mod_name, code in state.modules.items():
        tree = _parse_safe(code)

        # ── 1a. SyntaxError ───────────────────────────────────────────────
        if tree is None:
            report.add(ValidationIssue(
                layer=1, kind="syntax_error", module=mod_name,
                detail="Module has a SyntaxError — cannot parse",
                fixable=False,
            ))
            continue

        imported = _imported_names(code)

        # ── 1b. Missing standard imports ──────────────────────────────────
        # Heuristic: if code uses "threading." or "socket." but doesn't import it
        for lib in ("threading", "socket", "asyncio", "json", "os", "sys", "re",
                    "logging", "time", "datetime", "pathlib", "queue", "uuid"):
            usage_pattern = re.compile(rf"\b{lib}\.", re.MULTILINE)
            if usage_pattern.search(code) and lib not in imported:
                report.add(ValidationIssue(
                    layer=1, kind="missing_import", module=mod_name,
                    detail=f"Uses '{lib}.' but does not import {lib}",
                ))
                # Auto-patch: inject import at top
                patches[mod_name] = patches.get(mod_name, "") + f"import {lib}\n"

        # ── 1c. Undefined cross-module references ─────────────────────────
        # Names called in this module that are neither imported nor defined here
        # AND don't exist in any other module = probable skip
        called = _called_names(code)
        local_defs = module_defs[mod_name]

        for name in called:
            if (
                name not in local_defs
                and name not in imported
                and name not in all_defined
                and name not in stdlib
                and not name.startswith("_")
                and len(name) > 2           # skip single-char names
                and name[0].isupper()       # only flag CapCase (likely class refs)
            ):
                report.add(ValidationIssue(
                    layer=1, kind="undefined_ref", module=mod_name,
                    detail=f"References '{name}' which is not defined in any module",
                ))

    # ── 1d. Skipped plan steps ────────────────────────────────────────────
    # Cross-check completed modules vs planned steps
    if hasattr(state, "todo"):
        for task_entry in state.todo:
            task = task_entry["task"] if isinstance(task_entry, dict) else task_entry
            if task not in state.completed:
                report.add(ValidationIssue(
                    layer=1, kind="skipped_step", module="(plan)",
                    detail=f"Step was planned but never completed: '{task}'",
                    fixable=True,
                ))

    return report, patches


def _apply_static_patches(state, patches: dict[str, str], report: ValidationReport):
    """Prepend import patches to affected modules in-place."""
    for mod_name, patch_code in patches.items():
        if mod_name in state.modules:
            state.modules[mod_name] = patch_code + state.modules[mod_name]
            report.patches_applied.append(
                f"Prepended imports to {mod_name}: {patch_code.strip()}"
            )
            console.print(f"[dim green]  ✓ patched {mod_name}: {patch_code.strip()}[/dim green]")


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — Integration validator + LLM repair
# ─────────────────────────────────────────────────────────────────────────────

def _layer2_check(state) -> ValidationReport:
    """
    Check that methods called across module boundaries actually exist.
    e.g. if module A does self.router.route(msg) and module B (Router) has no
    method called 'route', flag it.
    """
    report = ValidationReport()

    # Build a map: class_name → set of method names defined
    class_methods: dict[str, set[str]] = {}

    for mod_name, code in state.modules.items():
        tree = _parse_safe(code)
        if not tree:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = set()
                for item in ast.walk(node):
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.add(item.name)
                class_methods[node.name] = methods

    # Now scan each module for Attribute calls: obj.method_name()
    # and check if method_name exists in any known class
    for mod_name, code in state.modules.items():
        tree = _parse_safe(code)
        if not tree:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue

            method = node.func.attr

            # Skip dunders and very common generic names
            if method.startswith("__") or method in {
                "append", "extend", "update", "get", "set", "pop",
                "strip", "split", "join", "format", "encode", "decode",
                "read", "write", "close", "open", "flush", "send", "recv",
                "start", "run", "stop", "print", "items", "keys", "values",
                "lower", "upper", "replace", "search", "match", "findall",
            }:
                continue

            # Check if this method exists in ANY known class
            all_methods = set().union(*class_methods.values()) if class_methods else set()
            if method not in all_methods and len(method) > 3:
                report.add(ValidationIssue(
                    layer=2, kind="method_mismatch", module=mod_name,
                    detail=(
                        f"Calls .{method}() but no class in the project defines it. "
                        f"Available methods: {sorted(all_methods)[:8]}"
                    ),
                ))

    return report


def _layer2_repair(
    state,
    issues: list[ValidationIssue],
    model,
    model_type:  str,
    model_path:  str,
    max_tokens:  int,
    temperature: float,
    report:      ValidationReport,
):
    """
    For each broken module, ask the LLM to produce a corrected version
    with the correct method signatures, given the full project context.
    """
    from engine import build_prompt, _stream_with_repeat_guard
    from executor import extract_code

    # De-duplicate: repair each broken module once
    broken_modules = list(dict.fromkeys(i.module for i in issues if i.module in state.modules))

    for mod_name in broken_modules:
        console.print(f"[yellow]  ↻ Layer 2 repair: {mod_name}[/yellow]")

        context = state.get_context(max_chars=2500)
        issue_detail = "\n".join(
            f"  - {i.detail}" for i in issues if i.module == mod_name
        )

        history = [
            {
                "role": "system",
                "content": (
                    "You are fixing integration bugs in a generated Python project.\n"
                    "Your task: rewrite ONLY the specified module to fix the listed issues.\n"
                    "Keep ALL existing logic. Only fix method names or signatures.\n"
                    "Output only the corrected Python code, no prose."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Fix this module: {mod_name}\n\n"
                    f"Issues to fix:\n{issue_detail}\n\n"
                    f"Current code of {mod_name}:\n```python\n{state.modules[mod_name]}\n```\n\n"
                    f"Other modules in the project (for context — do NOT rewrite these):\n"
                    f"```python\n{context}\n```\n\n"
                    f"Output the corrected {mod_name} code only."
                ),
            },
        ]

        prompt = build_prompt(history, model_type=model_type)
        output = _stream_with_repeat_guard(
            model, prompt,
            max_tokens // 2,
            max(temperature * 0.85, 0.1),
            model_type, model_path,
        )

        fixed_code = extract_code(output)
        if len(fixed_code) > 30:
            state.modules[mod_name] = fixed_code
            report.llm_repairs.append(f"Layer 2 repaired: {mod_name}")
            console.print(f"[dim green]  ✓ {mod_name} repaired[/dim green]")
        else:
            console.print(f"[dim yellow]  ⚠ {mod_name} repair too short — keeping original[/dim yellow]")


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — Completeness audit + gap-fill generation
# ─────────────────────────────────────────────────────────────────────────────

def _layer3_check(state, user_input: str) -> ValidationReport:
    """
    Audit the assembled project for:
      - Missing entry point (main / __main__ block)
      - Placeholder stubs in any module
      - Plan steps that were skipped (if executor skipped them)
    """
    report = ValidationReport()

    all_code = "\n\n".join(state.modules.values())

    # ── 3a. Missing entry point ───────────────────────────────────────────
    if not _has_entry_point(all_code):
        report.add(ValidationIssue(
            layer=3, kind="missing_entry_point", module="(project)",
            detail="No main() function or if __name__ == '__main__' block found",
        ))

    # ── 3b. Stubs in each module ──────────────────────────────────────────
    for mod_name, code in state.modules.items():
        stubs = _has_stubs(code)
        for stub in stubs:
            report.add(ValidationIssue(
                layer=3, kind="stub", module=mod_name,
                detail=f"Contains unimplemented stub: {stub}",
            ))

    # ── 3c. Cross-check: skipped steps not yet caught by Layer 1 ─────────
    if hasattr(state, "todo"):
        skipped = [
            t["task"] if isinstance(t, dict) else t
            for t in state.todo
            if (isinstance(t, dict) and not t.get("done")) or
               (not isinstance(t, dict) and t not in state.completed)
        ]
        for task in skipped:
            report.add(ValidationIssue(
                layer=3, kind="skipped_step", module="(plan)",
                detail=f"Never implemented: '{task}'",
            ))

    return report


def _layer3_gapfill(
    state,
    issues: list[ValidationIssue],
    user_input: str,
    model,
    model_type:  str,
    model_path:  str,
    max_tokens:  int,
    temperature: float,
    report:      ValidationReport,
):
    """
    Ask the LLM to generate any missing pieces as a single 'glue' module:
    entry point, completed stubs, and any skipped steps.
    """
    from engine import build_prompt, _stream_with_repeat_guard
    from executor import extract_code

    console.print(f"[yellow]  ↻ Layer 3 gap-fill generation...[/yellow]")

    issue_list = "\n".join(f"  - [{i.kind}] {i.detail}" for i in issues)
    all_code   = "\n\n".join(
        f"# FILE: {n}.py\n{c}" for n, c in state.modules.items()
    )

    history = [
        {
            "role": "system",
            "content": (
                "You are completing a partially-generated Python project.\n"
                "You will be given the existing code and a list of what is missing.\n"
                "Your job: write ONLY the missing pieces as one Python code block.\n"
                "Do NOT rewrite existing classes. Add only what is listed as missing.\n"
                "Output only valid Python code."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original project goal: {user_input}\n\n"
                f"Missing pieces:\n{issue_list}\n\n"
                f"Existing code (do NOT repeat):\n```python\n{all_code[-3000:]}\n```\n\n"
                f"Write only the missing code (entry point, completed stubs, "
                f"or unimplemented steps). Keep it concise and correct."
            ),
        },
    ]

    prompt = build_prompt(history, model_type=model_type)
    output = _stream_with_repeat_guard(
        model, prompt,
        max_tokens // 2,
        max(temperature * 0.8, 0.1),
        model_type, model_path,
    )

    gap_code = extract_code(output)
    if len(gap_code) > 30:
        # Guard: skip if the gap-fill redefines classes already in a module
        from executor import _is_duplicate
        if _is_duplicate(gap_code, state):
            console.print(f"[dim yellow]  ⚠ gap-fill duplicates existing definitions — skipping[/dim yellow]")
            return
        state.add_module("_gapfill", gap_code)
        report.llm_repairs.append("Layer 3 gap-fill module added")
        console.print(f"[dim green]  ✓ gap-fill module added[/dim green]")
    else:
        console.print(f"[dim yellow]  ⚠ gap-fill too short — skipping[/dim yellow]")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_fix(
    state,
    user_input:  str,
    model,
    model_type:  str,
    model_path:  str,
    max_tokens:  int   = 2048,
    temperature: float = 0.7,
    verbose:     bool  = True,
) -> str:
    """
    Run all three validation layers on the project state, apply fixes,
    and return the final assembled code string.

    Parameters
    ----------
    state       : ProjectState from builder.py (mutated in-place with fixes).
    user_input  : Original user request (for Layer 3 completeness context).
    model       : Loaded Llama model.
    model_type  : "deepseek" | "qwen" | "mistral" | "llama" | "generic"
    model_path  : For display label.
    max_tokens  : Token budget for LLM repair passes.
    temperature : Sampling temperature (repair passes use a lower value).
    verbose     : Print layer-by-layer results.

    Returns
    -------
    Final assembled code string (same format as assemble_final()).
    """
    from executor import assemble_final

    report = ValidationReport()

    # ── Layer 1: Static integrity ─────────────────────────────────────────
    if verbose:
        console.print("\n[bold cyan]⬡ Validator — Layer 1: Integrity[/bold cyan]")

    l1_report, patches = _layer1_check(state)

    if patches:
        _apply_static_patches(state, patches, report)

    for issue in l1_report.issues:
        report.add(issue)

    if verbose:
        if l1_report.ok and not patches:
            console.print("[dim green]  ✓ Layer 1 passed[/dim green]")
        else:
            for i in l1_report.issues:
                console.print(f"  [dim yellow]⚠ [{i.kind}] {i.module}: {i.detail}[/dim yellow]")

    # ── Layer 2: Integration ──────────────────────────────────────────────
    if verbose:
        console.print("\n[bold cyan]⬡ Validator — Layer 2: Integration[/bold cyan]")

    l2_report = _layer2_check(state)
    l2_issues = [i for i in l2_report.issues if i.fixable]

    if l2_issues:
        _layer2_repair(
            state, l2_issues, model, model_type, model_path,
            max_tokens, temperature, report,
        )
    else:
        if verbose:
            console.print("[dim green]  ✓ Layer 2 passed[/dim green]")

    for issue in l2_report.issues:
        report.add(issue)

    # ── Layer 3: Completeness ─────────────────────────────────────────────
    if verbose:
        console.print("\n[bold cyan]⬡ Validator — Layer 3: Completeness[/bold cyan]")

    l3_report = _layer3_check(state, user_input)
    l3_issues = [i for i in l3_report.issues if i.fixable]

    if l3_issues:
        if verbose:
            for i in l3_issues:
                console.print(f"  [dim yellow]⚠ [{i.kind}] {i.module}: {i.detail}[/dim yellow]")
        _layer3_gapfill(
            state, l3_issues, user_input, model, model_type, model_path,
            max_tokens, temperature, report,
        )
    else:
        if verbose:
            console.print("[dim green]  ✓ Layer 3 passed[/dim green]")

    for issue in l3_report.issues:
        report.add(issue)

    # ── Summary ───────────────────────────────────────────────────────────
    if verbose:
        console.print(f"\n[bold]Validation complete[/bold]")
        if report.patches_applied:
            console.print(f"  Static patches : {len(report.patches_applied)}")
        if report.llm_repairs:
            console.print(f"  LLM repairs    : {len(report.llm_repairs)}")
        total = len(report.issues)
        if total:
            console.print(f"  Issues logged  : {total}")
        else:
            console.print("  [green]✓ Project is clean[/green]")

    return assemble_final(state)
