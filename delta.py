"""
delta.py — Divergence-Estimated Layer Token Allocation
=======================================================
Mid-generation semantic drift detection and correction.

Monitors token chunks as they stream out. If the output drifts
semantically from the original intent (measured via EMA cosine distance),
DELTA stops generation and re-prompts with a targeted correction directive.

Usage in acie.py _run_pipeline() — replace stream_response() call:
    from delta import delta_stream
    response = delta_stream(
        model        = model,
        prompt       = prompt,
        user_input   = user_input,
        max_tokens   = resource.tokens,
        temperature  = resource.temperature,
        model_type   = model_type,
        model_path   = model_path,
        embedder     = memory.embedder,  # reuse existing embedder
    )
"""

import re
import numpy as np
from rich.console import Console
from typing import Optional

console = Console()


# ── Config ────────────────────────────────────────────────────────────────────

class DELTAConfig:
    chunk_size:      int   = 40     # tokens per drift check
    drift_threshold: float = 0.35   # EMA distance to trigger intervention (general)
    alpha:           float = 0.4    # EMA smoothing (recent chunks weighted more)
    max_retries:     int   = 1      # max re-prompt attempts (keep low to avoid loops)
    min_tokens_before_check: int = 60  # don't check until this many tokens generated

    # Domain-aware thresholds: code has structural repetition that embeds
    # differently from plain prose, so it needs a higher drift tolerance.
    domain_thresholds: dict = {
        "coding":             0.55,
        "coding_python":      0.55,
        "coding_javascript":  0.55,
        "coding_cpp":         0.55,
        "debug":              0.50,
        "general":            0.35,
    }

    def threshold_for(self, domain: str) -> float:
        return self.domain_thresholds.get(domain, self.drift_threshold)


config = DELTAConfig()


# ── Cosine similarity helper ──────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D float32 vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0   # treat zero vectors as identical (no drift)
    return float(np.dot(a, b) / (norm_a * norm_b))


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - _cosine_sim(a, b)


# ── Re-prompt template ────────────────────────────────────────────────────────

def _build_correction_prompt(
    original_input:   str,
    partial_response: str,
    model_type:       str,
) -> str:
    """
    Build a correction prompt that includes what was generated so far
    and asks the model to continue correctly on-topic.
    """
    # Trim partial to last 600 chars to stay within context budget
    partial_trimmed = partial_response[-600:].strip()

    correction = (
        f"You were answering the following question:\n"
        f"{original_input}\n\n"
        f"You began responding but drifted off-topic. "
        f"Here is what you produced before the drift:\n\n"
        f"{partial_trimmed}\n\n"
        f"Continue the answer correctly from where it makes sense. "
        f"Stay strictly on the original question. "
        f"Do NOT repeat what was already answered correctly. "
        f"Only provide what is missing or needs fixing."
    )

    # Wrap in the model's chat format
    if model_type == "qwen":
        return (
            f"<|im_start|>user\n{correction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    elif model_type == "deepseek":
        return f"<|user|>\n{correction}\n<|assistant|>\n"
    elif model_type in ("mistral", "llama"):
        return f"### Instruction:\n{correction}\n\n### Response:\n"
    else:
        return f"USER: {correction}\nASSISTANT:\n"


# ── Chunk accumulator ────────────────────────────────────────────────────────

class _ChunkAccumulator:
    """
    Accumulates streaming tokens into fixed-size text chunks.
    Yields a chunk string every `size` tokens.
    """
    def __init__(self, size: int = 40):
        self.size    = size
        self.buffer  = []
        self.total   = 0

    def push(self, token: str):
        self.buffer.append(token)
        self.total += 1

    def ready(self) -> bool:
        return len(self.buffer) >= self.size

    def flush(self) -> str:
        chunk = "".join(self.buffer)
        self.buffer = []
        return chunk

    def flush_remaining(self) -> str:
        chunk = "".join(self.buffer)
        self.buffer = []
        return chunk


# ── DELTA streaming core ──────────────────────────────────────────────────────

def delta_stream(
    model,
    prompt:       str,
    user_input:   str,
    max_tokens:   int   = 2048,
    temperature:  float = 0.7,
    model_type:   str   = "qwen",
    model_path:   str   = None,
    embedder            = None,
    chunk_size:   int   = None,
    drift_thresh: float = None,
    alpha:        float = None,
    domain:       str   = "general",
) -> str:
    """
    DELTA-monitored stream generation.

    Streams tokens, checks semantic drift every `chunk_size` tokens via EMA
    cosine distance from the original intent anchor. If sustained drift is
    detected, stops early and re-prompts with correction directive.

    Falls back to standard streaming if embedder is None (DELTA disabled).

    Parameters
    ----------
    model        : Loaded llama_cpp Llama instance.
    prompt       : Formatted prompt string.
    user_input   : Original user query (for intent anchor + re-prompt).
    max_tokens   : Token budget.
    temperature  : Sampling temperature.
    model_type   : For stop token selection and re-prompt formatting.
    model_path   : For display label.
    embedder     : sentence_transformers model (from memory.embedder).
    chunk_size   : Override config.chunk_size.
    drift_thresh : Override config.drift_threshold.
    alpha        : Override config.alpha.

    Returns
    -------
    Final response string (possibly corrected).
    """
    from engine import get_model_name, stream_response

    # ── Resolve config overrides ──────────────────────────────────────────────
    c_size   = chunk_size   if chunk_size   is not None else config.chunk_size
    d_thresh = drift_thresh if drift_thresh is not None else config.threshold_for(domain)
    ema_a    = alpha        if alpha        is not None else config.alpha

    # ── Fallback: no embedder → plain stream ─────────────────────────────────
    if embedder is None:
        return stream_response(model, prompt, max_tokens, temperature, model_type, model_path)

    # ── Phase 1: Intent anchor ────────────────────────────────────────────────
    anchor: np.ndarray = embedder.encode(user_input).astype("float32")

    # ── Setup ─────────────────────────────────────────────────────────────────
    label       = get_model_name(model_path) if model_path else "Assistant"
    stop_tokens = ["<|im_end|>"] if model_type == "qwen" else []

    full_response = ""
    accumulator   = _ChunkAccumulator(size=c_size)
    ema           = 0.0
    prev_ema      = 0.0
    drift_triggered = False
    check_count   = 0

    console.print(f"\n[bold green]{label}:[/bold green] ", end="")

    # ── Phase 2 & 3: Stream + chunk-level drift monitoring ────────────────────
    for chunk in model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
        echo=False,
        stop=stop_tokens,
    ):
        token = chunk["choices"][0]["text"]
        token = token.replace("<think>", "").replace("</think>", "")

        full_response += token
        accumulator.push(token)
        print(token, end="", flush=True)

        # Only check drift after minimum token warmup
        if accumulator.total < config.min_tokens_before_check:
            continue

        # ── Phase 3: Drift measurement on full chunk ──────────────────────────
        if accumulator.ready():
            chunk_text = accumulator.flush()
            check_count += 1

            # Embed the chunk and compute distance from anchor
            try:
                chunk_emb = embedder.encode(chunk_text).astype("float32")
                dist      = _cosine_dist(anchor, chunk_emb)
            except Exception:
                continue   # embedding failed — skip check this round

            # EMA update: α × dist + (1-α) × prev_ema
            prev_ema = ema
            ema      = ema_a * dist + (1.0 - ema_a) * ema

            # ── Phase 4: Intervention trigger ────────────────────────────────
            # Trigger only if EMA exceeds threshold for TWO consecutive checks
            if ema > d_thresh and prev_ema > d_thresh and check_count > 2:
                drift_triggered = True
                console.print(
                    f"\n[bold red]⚠ DELTA: drift detected "
                    f"(EMA={ema:.3f} > {d_thresh}) — stopping generation[/bold red]"
                )
                break

    print("\n")

    # ── Phase 4: Re-prompt with correction directive ──────────────────────────
    if drift_triggered:
        for attempt in range(config.max_retries):
            console.print(
                f"[yellow]↻ DELTA re-prompt (attempt {attempt+1}/{config.max_retries})...[/yellow]"
            )

            correction_prompt = _build_correction_prompt(
                original_input   = user_input,
                partial_response = full_response,
                model_type       = model_type,
            )

            # Stream the corrected response
            corrected = ""
            console.print(f"\n[bold green]{label} (corrected):[/bold green] ", end="")

            for chunk in model(
                correction_prompt,
                max_tokens=max_tokens,
                temperature=max(temperature * 0.8, 0.1),  # slightly lower temp for correction
                stream=True,
                echo=False,
                stop=stop_tokens,
            ):
                token = chunk["choices"][0]["text"]
                token = token.replace("<think>", "").replace("</think>", "")
                corrected += token
                print(token, end="", flush=True)

            print("\n")

            # Accept correction only if it's substantial
            if len(corrected.strip()) > 50:
                # Merge: keep the correct part of original + correction
                # Find the last "good" sentence boundary in partial
                good_partial = _find_good_cutoff(full_response)
                full_response = good_partial + "\n\n" + corrected
                console.print("[dim green]✓ DELTA correction applied[/dim green]")
                break
            else:
                console.print("[dim yellow]DELTA correction too short — keeping original[/dim yellow]")

    return full_response


# ── Good cutoff helper ────────────────────────────────────────────────────────

def _find_good_cutoff(text: str, max_chars: int = 1200) -> str:
    """
    Find the last clean sentence/code-block boundary before drift,
    to use as the kept portion when merging original + correction.
    """
    # Prefer keeping up to last complete code block
    if "```" in text:
        blocks = text.split("```")
        # Keep everything up to the last complete block (even number of ```)
        if len(blocks) % 2 == 1:   # odd number of ``` means last block is closed
            return text[:max_chars]
        else:                       # even = last block unclosed, cut before it
            safe = "```".join(blocks[:-1])
            return safe[:max_chars]

    # Otherwise cut at last sentence boundary
    truncated = text[:max_chars]
    last_period = max(
        truncated.rfind(". "),
        truncated.rfind(".\n"),
        truncated.rfind(":\n"),
    )
    if last_period > max_chars // 2:
        return truncated[:last_period + 1]

    return truncated


# ── Drift stats helper (for /stats command) ───────────────────────────────────

def delta_status_line(last_ema: float, triggered: bool) -> str:
    """Return a formatted status string for the /stats command."""
    indicator = "🔴" if triggered else "🟢"
    return f"{indicator} DELTA last EMA: {last_ema:.3f}  (threshold: {config.drift_threshold})"


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick test of the EMA drift logic without a real model.
    Simulates chunk embeddings to verify the trigger mechanism.
    """
    import numpy as np

    class FakeEmbedder:
        def encode(self, text):
            # Simulate drift: later chunks diverge from the anchor
            if "drift" in text.lower():
                return np.random.randn(384).astype("float32") * 2  # far from anchor
            return np.array([1.0] * 384, dtype="float32")           # close to anchor

    embedder = FakeEmbedder()
    anchor   = embedder.encode("what is recursion in python")

    # Simulate EMA across 6 chunks
    chunks = [
        "Recursion is when a function calls itself.",
        "It needs a base case to stop.",
        "def factorial(n): return 1 if n==0 else n*factorial(n-1)",
        "DRIFT: unrelated tangent about pandas dataframes",
        "DRIFT: more unrelated content about matplotlib",
        "DRIFT: even more off-topic material",
    ]

    ema   = 0.0
    alpha = config.alpha
    print("\nDELTA EMA simulation:")
    print(f"{'Chunk':<50} {'dist':>6}  {'EMA':>6}  {'trigger':>7}")
    print("-" * 72)

    prev_ema = 0.0
    for i, c in enumerate(chunks):
        emb  = embedder.encode(c).astype("float32")
        dist = _cosine_dist(anchor, emb)
        prev_ema = ema
        ema  = alpha * dist + (1 - alpha) * ema
        trigger = "🔴 YES" if (ema > config.drift_threshold and prev_ema > config.drift_threshold and i > 1) else ""
        print(f"{c[:48]:<50} {dist:>6.3f}  {ema:>6.3f}  {trigger}")
