"""
router.py — ACIE: Adaptive Coding Intelligence Engine
Handles:
  1. Token Analyzer   — structured input parsing
  2. Task Router      — assigns a mode (debug / generate / explain / general)
  3. Prompt Constructor — mode-specific system prompts
  4. Confidence Checker — post-response quality signal
"""

import re
from dataclasses import dataclass, field
from typing import Literal

# ── Types ─────────────────────────────────────────────────────────────────────

TaskMode = Literal["debug", "generate", "explain", "refactor", "general"]
Confidence = Literal["high", "medium", "low"]


@dataclass
class InputSignals:
    """Structured signals extracted from user input."""
    raw: str
    word_count: int
    has_code_block: bool       # fenced ``` block
    has_inline_code: bool      # backtick or known syntax tokens
    has_error_trace: bool      # traceback / error keywords
    has_refactor_hint: bool    # "refactor", "clean", "improve", "optimise"
    has_explain_hint: bool     # "explain", "what does", "how does", "why"
    has_generate_hint: bool    # "write", "create", "build", "implement"
    complexity: Literal["low", "medium", "high"]
    dominant_language: str     # "python", "js", "cpp", "unknown"


@dataclass
class RouteDecision:
    mode: TaskMode
    signals: InputSignals
    memory_domain: str         # which FAISS index to query
    system_prompt: str
    use_large_model: bool = False   # set True by confidence check later


# ── 1. TOKEN ANALYZER ────────────────────────────────────────────────────────

_ERROR_PATTERNS = re.compile(
    r"traceback|error:|exception:|syntaxerror|typeerror|valueerror|"
    r"nameerror|indexerror|keyerror|attributeerror|runtimeerror|"
    r"segmentation fault|undefined|null pointer|failed|crashed",
    re.IGNORECASE,
)

_LANGUAGE_HINTS = {
    "python": re.compile(r"\bdef \w+|import \w+|:\s*$|print\(|self\.", re.MULTILINE),
    "javascript": re.compile(r"\bfunction \w+|const |let |var |=>\s*{|console\.log"),
    "cpp": re.compile(r"#include|std::|cout|cin|int main"),
    "rust": re.compile(r"\bfn \w+|let mut |println!|->"),
    "sql": re.compile(r"\bSELECT\b|\bFROM\b|\bWHERE\b|\bINSERT\b", re.IGNORECASE),
}

_CODE_KEYWORDS = re.compile(
    r"\bdef |class |import |return |if |else:|for |while |try:|except:|"
    r"function\b|const |let |var |=>|#include|pub fn|SELECT\b",
    re.IGNORECASE,
)


def analyze_input(text: str) -> InputSignals:
    words = text.split()
    word_count = len(words)
    lower = text.lower()

    has_code_block = "```" in text
    has_inline_code = bool(_CODE_KEYWORDS.search(text)) or "`" in text

    has_error_trace = bool(_ERROR_PATTERNS.search(text))

    has_refactor_hint = any(w in lower for w in (
        "refactor", "clean up", "improve", "optimise", "optimize",
        "simplify", "rewrite", "restructure"
    ))
    has_explain_hint = any(w in lower for w in (
        "explain", "what does", "what is", "how does", "why does",
        "what's happening", "understand", "clarify", "describe"
    ))
    has_generate_hint = any(w in lower for w in (
        "write", "create", "build", "implement", "make", "generate",
        "add", "code", "function", "class", "script"
    ))

    # Complexity: rough heuristic
    if word_count < 15:
        complexity = "low"
    elif word_count < 60:
        complexity = "medium"
    else:
        complexity = "high"

    # Language detection
    dominant_language = "unknown"
    for lang, pattern in _LANGUAGE_HINTS.items():
        if pattern.search(text):
            dominant_language = lang
            break

    return InputSignals(
        raw=text,
        word_count=word_count,
        has_code_block=has_code_block,
        has_inline_code=has_inline_code,
        has_error_trace=has_error_trace,
        has_refactor_hint=has_refactor_hint,
        has_explain_hint=has_explain_hint,
        has_generate_hint=has_generate_hint,
        complexity=complexity,
        dominant_language=dominant_language,
    )


# ── 2. TASK ROUTER ────────────────────────────────────────────────────────────

def route(signals: InputSignals) -> TaskMode:
    """
    Priority order (highest → lowest):
      debug > refactor > generate > explain > general
    """
    if signals.has_error_trace:
        return "debug"
    if signals.has_refactor_hint and (signals.has_code_block or signals.has_inline_code):
        return "refactor"
    if signals.has_generate_hint or (signals.has_inline_code and not signals.has_explain_hint):
        return "generate"
    if signals.has_explain_hint:
        return "explain"
    return "general"


def memory_domain_for(mode: TaskMode, lang: str) -> str:
    """Map task mode to FAISS domain index key."""
    if mode in ("debug",):
        return "debug"
    if mode in ("generate", "refactor"):
        return f"coding_{lang}" if lang != "unknown" else "coding"
    return "general"


# ── 3. PROMPT CONSTRUCTOR ─────────────────────────────────────────────────────

_SYSTEM_PROMPTS: dict[TaskMode, str] = {
    "debug": (
        "You are an expert debugging assistant.\n"
        "Workflow: (1) Identify the exact error and line. "
        "(2) Explain WHY it happens in one sentence. "
        "(3) Provide the corrected code with inline comments on the fix. "
        "Be direct. No unnecessary preamble."
    ),
    "generate": (
        "You are a senior software engineer.\n"
        "When asked to write code: output clean, idiomatic, well-commented code. "
        "Include a brief docstring. No unnecessary prose before the code block. "
        "If the request is ambiguous, state your assumption then write the code."
    ),
    "refactor": (
        "You are a code quality specialist.\n"
        "Refactor the provided code for clarity, performance, and idiom. "
        "Output ONLY the refactored code followed by a short bullet list of changes. "
        "Do not restate the original unless asked."
    ),
    "explain": (
        "You are a patient technical educator.\n"
        "Explain concepts clearly with concrete examples. "
        "Use analogies where helpful. Keep explanations concise but complete. "
        "If code is present, walk through it line-by-line only if the user asks."
    ),
    "general": (
        "You are a knowledgeable and concise AI assistant. "
        "Answer accurately and directly. "
        "If a coding question arises, shift to precise technical language."
    ),
}


def build_system_prompt(mode: TaskMode, lang: str = "unknown") -> str:
    base = _SYSTEM_PROMPTS.get(mode, _SYSTEM_PROMPTS["general"])
    if lang != "unknown" and mode in ("debug", "generate", "refactor"):
        base = f"[Language context: {lang.upper()}]\n" + base
    return base


# ── 4. CONFIDENCE CHECKER ────────────────────────────────────────────────────

# Patterns that suggest the model is uncertain
_UNCERTAINTY_PATTERNS = re.compile(
    r"\bi (think|believe|guess|assume|suppose)\b|"
    r"\bnot sure\b|might be|could be|possibly|probably|"
    r"\bi('m| am) not (certain|sure|confident)\b|"
    r"\bit depends\b",
    re.IGNORECASE,
)

# Patterns that suggest incomplete/broken output
_INCOMPLETE_PATTERNS = re.compile(
    r"\.\.\.$|<\.\.\.>|\[TODO\]|\[FIXME\]|# \.\.\.|pass\s*$|NotImplementedError",
    re.IGNORECASE | re.MULTILINE,
)

# Error keywords in the response itself (model reproduced an error without fixing)
_UNRESOLVED_ERROR = re.compile(
    r"SyntaxError|NameError|AttributeError|TypeError|still throws|still fails",
    re.IGNORECASE,
)


def check_confidence(response: str, mode: TaskMode) -> Confidence:
    """
    Inspect the model's response and return a confidence signal.

    high   → response looks complete and certain
    medium → some hedging or minor issues
    low    → significant uncertainty or incomplete output detected
    """
    score = 0  # penalty points

    if _UNCERTAINTY_PATTERNS.search(response):
        score += 2
    if _INCOMPLETE_PATTERNS.search(response):
        score += 2
    if _UNRESOLVED_ERROR.search(response) and mode == "debug":
        score += 3

    # Mode-specific checks
    if mode in ("generate", "refactor") and "```" not in response and len(response) > 100:
        # Expected a code block but didn't get one
        score += 1

    if mode == "debug" and "fix" not in response.lower() and "solution" not in response.lower():
        score += 1

    # Very short responses to complex tasks
    if len(response.split()) < 20 and mode in ("generate", "debug", "refactor"):
        score += 2

    if score == 0:
        return "high"
    elif score <= 2:
        return "medium"
    else:
        return "low"


# ── 5. FULL ROUTE DECISION ────────────────────────────────────────────────────

def make_route(user_input: str) -> RouteDecision:
    """Top-level call: analyze input → produce full routing decision."""
    signals = analyze_input(user_input)
    mode = route(signals)
    domain = memory_domain_for(mode, signals.dominant_language)
    system_prompt = build_system_prompt(mode, signals.dominant_language)

    return RouteDecision(
        mode=mode,
        signals=signals,
        memory_domain=domain,
        system_prompt=system_prompt,
        use_large_model=False,
    )
