"""
prima.py — Predictive Resource & Intent Modulation Algorithm
============================================================
Pre-generation difficulty fingerprinting.

Computes a unified difficulty score from 5 weighted signals,
then derives a coupled resource vector (tokens, temperature, memory_k)
so all three parameters move together as a single unit.

Usage in acie.py _run_pipeline():
    from prima import prima_score
    resource = prima_score(
        user_input   = user_input,
        mode         = decision.mode,
        memory       = memory,
        domain       = decision.memory_domain,
        base_tokens  = max_tokens,
        base_temp    = temperature,
        base_k       = 3,
    )
    # Then use resource.tokens, resource.temperature, resource.memory_k
"""

import re
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Resource vector output ────────────────────────────────────────────────────

@dataclass
class ResourceVector:
    """Coupled output from PRIMA. All three values derived from one score."""
    tokens:      int
    temperature: float
    memory_k:    int
    score:       float   # raw difficulty score [0, 1] — for logging
    breakdown:   dict    # per-signal scores — for debugging


# ── Signal weights ────────────────────────────────────────────────────────────

_WEIGHTS = {
    "syntactic_density": 0.25,
    "ambiguity_index":   0.20,
    "domain_novelty":    0.25,
    "length_hint":       0.15,
    "error_complexity":  0.15,
}

# ── Signal 1: Syntactic density ────────────────────────────────────────────────
# Counts code keywords per word — high density = complex technical request

_SYNTAX_KW = re.compile(
    r"\bdef \b|\bclass \b|\bimport \b|\breturn \b|\bfor \b|\bwhile \b|"
    r"\btry:\b|\bexcept\b|\bwith \b|\basync \b|\bawait \b|"
    r"\bfunction\b|\bconst \b|\blet \b|\bvar \b|=>|"
    r"#include|\bstd::\b|\bpub fn\b|\bimpl \b|"
    r"\bSELECT\b|\bJOIN\b|\bWHERE\b",
    re.IGNORECASE,
)

def _syntactic_density(text: str) -> float:
    words = len(text.split()) or 1
    kw_count = len(_SYNTAX_KW.findall(text))
    # Normalise: 10+ keywords per 100 words = max density
    raw = kw_count / words * 100
    return min(raw / 10.0, 1.0)


# ── Signal 2: Ambiguity index ─────────────────────────────────────────────────
# Vague language and question marks suggest harder-to-answer queries

_VAGUE = re.compile(
    r"\bsomething\b|\bkind of\b|\bsort of\b|\bmaybe\b|\bperhaps\b|"
    r"\bi think\b|\bi guess\b|\bnot sure\b|\bsomehow\b|\bsomewhere\b|"
    r"\banything\b|\bwhatever\b|\bsome kind\b",
    re.IGNORECASE,
)

def _ambiguity_index(text: str) -> float:
    words = len(text.split()) or 1
    vague_hits  = len(_VAGUE.findall(text))
    q_marks     = text.count("?")
    raw = (vague_hits * 2 + q_marks) / words * 10
    return min(raw, 1.0)


# ── Signal 3: Domain novelty ──────────────────────────────────────────────────
# Cosine distance from nearest FAISS entry — high distance = no prior context

def _domain_novelty(
    user_input: str,
    memory,
    domain: str,
) -> float:
    """
    Returns how novel (distant) this query is from stored memory.
    Range [0, 1] — 1.0 means completely unseen domain.
    Falls back to 0.5 if memory is unavailable.
    """
    if memory is None:
        return 0.5

    try:
        results = memory.search_domain(user_input, domain, k=1)
        if not results:
            return 1.0   # No entries at all → fully novel

        best_dist = results[0]["score"]   # L2 distance, lower = more similar
        # all-MiniLM-L6-v2 typical L2 range: 0 (identical) → ~2.0 (very different)
        # Map [0, 2.0] → [0, 1] and invert: low dist = low novelty
        novelty = min(best_dist / 2.0, 1.0)
        return float(novelty)

    except Exception:
        return 0.5


# ── Signal 4: Output length hint ─────────────────────────────────────────────
# Explicit length or completeness words predict large output need

_LENGTH_PATTERNS = [
    (re.compile(r"(\d+)\s*lines?", re.IGNORECASE), "lines"),
    (re.compile(r"\b(full|complete|entire|whole|comprehensive)\b", re.IGNORECASE), "full"),
    (re.compile(r"\bstep[- ]by[- ]step\b", re.IGNORECASE), "step_by_step"),
    (re.compile(r"\bdetailed?\b|\bin detail\b", re.IGNORECASE), "detailed"),
    (re.compile(r"\bminimum\s+\d+\b", re.IGNORECASE), "minimum_count"),
    (re.compile(r"\b(project|system|application|platform)\b", re.IGNORECASE), "project_scale"),
]

def _length_hint(text: str) -> float:
    score = 0.0

    # Explicit line count
    m = re.search(r"(\d+)\s*lines?", text, re.IGNORECASE)
    if m:
        lines = int(m.group(1))
        if lines >= 300:   score += 1.0
        elif lines >= 100: score += 0.7
        elif lines >= 50:  score += 0.4
        else:              score += 0.2

    # Keyword signals
    for pattern, label in _LENGTH_PATTERNS[1:]:
        if pattern.search(text):
            score += 0.25

    return min(score, 1.0)


# ── Signal 5: Error complexity ────────────────────────────────────────────────
# Estimates debug difficulty from traceback depth and error type keywords

_ERROR_TYPES = re.compile(
    r"traceback|error:|exception:|syntaxerror|typeerror|valueerror|"
    r"runtimeerror|attributeerror|nameerror|indexerror|keyerror|"
    r"segmentation fault|null pointer|undefined|cuda error|oom|"
    r"out of memory|killed|failed|crashed|abort",
    re.IGNORECASE,
)

_STACK_DEPTH = re.compile(r"line \d+", re.IGNORECASE)

def _error_complexity(text: str) -> float:
    error_hits  = len(_ERROR_TYPES.findall(text))
    stack_lines = len(_STACK_DEPTH.findall(text))

    if error_hits == 0:
        return 0.0

    # Multiple error types or deep stack = complex bug
    raw = (error_hits * 0.2) + (stack_lines * 0.15)
    return min(raw, 1.0)


# ── Weighted combination ──────────────────────────────────────────────────────

def _compute_score(signals: dict) -> float:
    total = sum(_WEIGHTS[k] * v for k, v in signals.items())
    return float(np.clip(total, 0.0, 1.0))


# ── Resource vector derivation ────────────────────────────────────────────────

def _derive_resources(
    score:       float,
    base_tokens: int,
    base_temp:   float,
    base_k:      int,
) -> tuple[int, float, int]:
    """
    Derive coupled (tokens, temperature, memory_k) from one score.

    Formulas:
        tokens      = base × (1 + 0.8 × score)   →  1.0× to 1.8× base
        temperature = base × (1 − 0.4 × score)   →  1.0× to 0.6× base
        memory_k    = base_k + int(score × 4)     →  base_k to base_k+4
    """
    tokens      = int(base_tokens * (1.0 + 0.8 * score))
    temperature = round(base_temp  * (1.0 - 0.4 * score), 3)
    memory_k    = base_k + int(score * 4)

    # Hard clamps
    tokens      = min(tokens, 3500)
    temperature = max(temperature, 0.1)
    memory_k    = min(memory_k, 10)

    return tokens, temperature, memory_k


# ── Public API ────────────────────────────────────────────────────────────────

def prima_score(
    user_input:  str,
    mode:        str,
    memory       = None,
    domain:      str   = "general",
    base_tokens: int   = 2048,
    base_temp:   float = 0.7,
    base_k:      int   = 3,
) -> ResourceVector:
    """
    Compute PRIMA difficulty fingerprint and return a coupled ResourceVector.

    Parameters
    ----------
    user_input  : Raw user query string.
    mode        : Task mode from router ('generate', 'debug', etc.).
    memory      : MemorySystem instance for domain novelty signal (can be None).
    domain      : FAISS domain to search for novelty.
    base_tokens : Baseline max_tokens before PRIMA scaling.
    base_temp   : Baseline temperature before PRIMA scaling.
    base_k      : Baseline memory retrieval count.

    Returns
    -------
    ResourceVector with .tokens, .temperature, .memory_k, .score, .breakdown
    """
    # ── Compute all 5 signals ─────────────────────────────────────────────────
    signals = {
        "syntactic_density": _syntactic_density(user_input),
        "ambiguity_index":   _ambiguity_index(user_input),
        "domain_novelty":    _domain_novelty(user_input, memory, domain),
        "length_hint":       _length_hint(user_input),
        "error_complexity":  _error_complexity(user_input),
    }

    # Mode-specific boosts — debug tasks are inherently harder
    if mode == "debug":
        signals["error_complexity"] = min(signals["error_complexity"] + 0.3, 1.0)
        signals["syntactic_density"] = min(signals["syntactic_density"] + 0.2, 1.0)
    elif mode in ("generate", "refactor"):
        signals["length_hint"] = min(signals["length_hint"] + 0.1, 1.0)

    # ── Weighted score ────────────────────────────────────────────────────────
    score = _compute_score(signals)

    # ── Derive coupled resource vector ───────────────────────────────────────
    tokens, temperature, memory_k = _derive_resources(
        score, base_tokens, base_temp, base_k
    )

    return ResourceVector(
        tokens=tokens,
        temperature=temperature,
        memory_k=memory_k,
        score=score,
        breakdown=signals,
    )


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        ("what is a list in python?", "general"),
        ("explain recursion with examples", "explain"),
        ("write a complete python project with 200 lines for a CLI chatbot", "generate"),
        ("debug this cuda out of memory error: RuntimeError line 47", "debug"),
        ("refactor this full Django REST API and optimize all endpoints", "refactor"),
    ]

    print(f"\n{'─'*70}")
    print(f"{'Input':<45} {'Score':>6}  {'Tokens':>7}  {'Temp':>6}  {'k':>3}")
    print(f"{'─'*70}")

    for text, mode in test_cases:
        rv = prima_score(text, mode=mode, base_tokens=2048, base_temp=0.7, base_k=3)
        short = text[:42] + "..." if len(text) > 42 else text
        print(f"{short:<45} {rv.score:>6.3f}  {rv.tokens:>7}  {rv.temperature:>6.3f}  {rv.memory_k:>3}")

    print(f"{'─'*70}\n")
