"""
triagerl.reward.llm_judge
=========================
Async LLM-as-judge grader for offline evaluation and reward-model training.

Responsibility
--------------
Call the Anthropic API to score an agent's clinical reasoning with a senior
emergency medicine attending physician persona.  Return a structured result.

What this module does NOT do
-----------------------------
*  Does NOT silently fall back to keyword grading on API failure.
   Failures surface as explicit exceptions (``LLMJudgeError`` subclasses)
   so callers decide what to do.  A convenience wrapper
   ``grade_with_llm_safe()`` provides optional controlled fallback for
   callers that explicitly want it.
*  Does NOT produce RL training gradients — online training uses
   ``grader.compute_final_score()``.  This module is offline-only.
*  Does NOT log.  All structured error information is in the exception chain.

Error hierarchy
---------------
    LLMJudgeError          — base for all judge failures
    ├── LLMUnavailableError  — httpx not installed
    ├── LLMAPIError          — non-2xx response or network failure
    └── LLMParseError        — response received but JSON malformed

Caller contract
---------------
    result = await grade_with_llm(action, task_dict)
    # or, with fallback:
    result = await grade_with_llm_safe(action, task_dict, grader_fn=my_fallback)

LLMJudgeResult fields
---------------------
    llm_reasoning_score : float ∈ [0.0, 1.0]
    llm_feedback        : str
    llm_undertriage_flag: bool
    llm_safety_concern  : bool
    model               : str
    raw_response        : str   (the raw text returned by the API)

Prompt design
-------------
The four scoring dimensions were chosen to penalise keyword stuffing
and reward genuine clinical reasoning:

  1. Clinical accuracy    (0–3): diagnosis and key risk factors
  2. Safety awareness     (0–2): undertriage risk, time-critical interventions
  3. Completeness         (0–2): key differentials and interventions addressed
  4. Coherence            (0–3): logical, medically sound vs. keyword salad

Total max = 10 → normalised to [0.0, 1.0] by the API.
The API is instructed to return normalised 0.0-1.0 directly.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, Optional

from triagerl.core.models import TriageAction


# ---------------------------------------------------------------------------
# Public result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMJudgeResult:
    """
    Structured result from the LLM-as-judge grader.

    All fields are read-only (frozen dataclass).

    Attributes
    ----------
    llm_reasoning_score : float
        Normalised score ∈ [0.0, 1.0].
    llm_feedback : str
        Brief critique from the attending physician persona.
    llm_undertriage_flag : bool
        True when correct ESI ≤ 2 and agent ESI > correct (computed
        from inputs, not from LLM response — LLM is not trusted for
        this safety-critical binary decision).
    llm_safety_concern : bool
        True when the LLM specifically flagged a safety concern.
    model : str
        The model string used for this evaluation.
    raw_response : str
        The raw text returned by the API before parsing.
    """
    llm_reasoning_score:  float
    llm_feedback:         str
    llm_undertriage_flag: bool
    llm_safety_concern:   bool
    model:                str
    raw_response:         str  = field(default="")


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class LLMJudgeError(Exception):
    """Base class for all LLM judge failures."""


class LLMUnavailableError(LLMJudgeError):
    """
    Raised when the ``httpx`` dependency is not installed.

    Install with: ``pip install httpx``
    """


class LLMAPIError(LLMJudgeError):
    """
    Raised when the API call fails (network error, non-2xx status, timeout).

    Attributes
    ----------
    status_code : int | None
        HTTP status code if available; None for network/timeout errors.
    """

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMParseError(LLMJudgeError):
    """
    Raised when the API returned a response but it could not be parsed.

    Attributes
    ----------
    raw_text : str
        The raw response text that failed to parse.
    """

    def __init__(self, message: str, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL:     str   = "claude-haiku-4-5-20251001"
_API_URL:           str   = "https://api.anthropic.com/v1/messages"
_API_VERSION:       str   = "2023-06-01"
_MAX_TOKENS:        int   = 256
_TIMEOUT_SECONDS:   float = 15.0
_CRITICAL_ESI_THRESHOLD: int = 2

_SYSTEM_PROMPT: str = """\
You are a senior emergency medicine attending physician evaluating an AI triage agent's clinical reasoning.

Score the agent's reasoning on four dimensions:
1. Clinical accuracy  (0–3): Correctly identifies the diagnosis and key risk factors.
2. Safety awareness   (0–2): Identifies undertriage risks and critical time-sensitive interventions.
3. Completeness       (0–2): Addresses key differentials and recommended interventions.
4. Coherence          (0–3): Reasoning is logical and medically sound (vs keyword stuffing).

Return ONLY a JSON object with no markdown or preamble:
{"score": <float 0.0–1.0>, "feedback": "<brief critique>", "safety_concern": <true|false>}

The "score" must be already normalised to [0.0, 1.0] (do not return raw point totals).
"safety_concern" must be true if you identify a clinically dangerous undertriage or \
missed critical diagnosis."""

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|```$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Prompt builder (pure function — testable without API)
# ---------------------------------------------------------------------------

def build_judge_prompt(
    action: TriageAction,
    task: Dict[str, Any],
    correct_esi: Optional[int],
    episode_summary: str = "",
) -> str:
    """
    Build the user-turn prompt for the LLM judge.

    Separated from the API call so the prompt can be unit-tested and
    logged independently.
    """
    lines = [
        f"Task scenario: {task.get('scenario', '(not provided)')}",
        f"Correct ESI: {correct_esi}",
        f"Agent-assigned ESI: {action.esi_level}",
        f"Agent reasoning: {action.reasoning}",
        f"Recommended actions: {', '.join((action.recommended_actions or [])[:10])}",
    ]
    if episode_summary:
        lines.append(f"Episode context: {episode_summary}")
    lines.append("\nEvaluate the clinical reasoning quality.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core async grader
# ---------------------------------------------------------------------------

async def grade_with_llm(
    action: TriageAction,
    task: Dict[str, Any],
    episode_summary: str = "",
    model: str = _DEFAULT_MODEL,
) -> LLMJudgeResult:
    """
    Score clinical reasoning quality using an LLM judge (async).

    Parameters
    ----------
    action : TriageAction
        The classify action to evaluate.
    task : dict
        Task configuration dict.  Must contain ``scenario`` and optionally
        ``esi_correct``/``correct_esi``.
    episode_summary : str
        Optional free-text episode context to include in the prompt.
    model : str
        Anthropic model string.  Defaults to claude-haiku-4-5 for speed.

    Returns
    -------
    LLMJudgeResult

    Raises
    ------
    LLMUnavailableError
        If ``httpx`` is not installed.
    LLMAPIError
        If the API returns a non-2xx response or times out.
    LLMParseError
        If the response body cannot be parsed as the expected JSON schema.

    Notes
    -----
    The ``llm_undertriage_flag`` field is computed from the inputs, not
    from the LLM response.  The model cannot reliably determine ESI values
    from clinical text alone, and this safety-critical decision must not
    depend on an LLM that may hallucinate.
    """
    try:
        import httpx  # type: ignore[import]
    except ImportError as exc:
        raise LLMUnavailableError(
            "httpx is required for grade_with_llm. "
            "Install it with: pip install httpx"
        ) from exc

    correct_esi = _resolve_esi_from_dict(task)
    undertriage_flag = _compute_undertriage(action.esi_level, correct_esi)

    prompt = build_judge_prompt(action, task, correct_esi, episode_summary)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.post(
                _API_URL,
                headers={
                    "x-api-key":          api_key,
                    "anthropic-version":  _API_VERSION,
                    "content-type":       "application/json",
                },
                json={
                    "model":      model,
                    "max_tokens": _MAX_TOKENS,
                    "system":     _SYSTEM_PROMPT,
                    "messages":   [{"role": "user", "content": prompt}],
                },
            )
    except httpx.TimeoutException as exc:
        raise LLMAPIError(
            f"Anthropic API timed out after {_TIMEOUT_SECONDS}s: {exc}"
        ) from exc
    except httpx.RequestError as exc:
        raise LLMAPIError(f"Network error calling Anthropic API: {exc}") from exc

    if response.status_code != 200:
        raise LLMAPIError(
            f"Anthropic API returned HTTP {response.status_code}: "
            f"{response.text[:300]}",
            status_code=response.status_code,
        )

    try:
        data = response.json()
        raw_text = data["content"][0]["text"].strip()
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise LLMParseError(
            f"Could not extract text from API response: {exc}",
            raw_text=response.text[:500],
        ) from exc

    # Strip code fences if the model wrapped the JSON.
    clean_text = _CODE_FENCE_RE.sub("", raw_text).strip()

    try:
        parsed = json.loads(clean_text)
    except json.JSONDecodeError as exc:
        raise LLMParseError(
            f"API response is not valid JSON: {exc}",
            raw_text=clean_text[:300],
        ) from exc

    # Validate expected keys — raise explicitly rather than silently defaulting.
    if "score" not in parsed:
        raise LLMParseError(
            "API response JSON missing required key 'score'.",
            raw_text=clean_text[:300],
        )

    llm_score = float(parsed["score"])
    if not (0.0 <= llm_score <= 1.0):
        raise LLMParseError(
            f"API returned score {llm_score!r} outside [0.0, 1.0].",
            raw_text=clean_text[:300],
        )

    return LLMJudgeResult(
        llm_reasoning_score=llm_score,
        llm_feedback=str(parsed.get("feedback", "")),
        llm_undertriage_flag=undertriage_flag,
        llm_safety_concern=bool(parsed.get("safety_concern", False)),
        model=model,
        raw_response=raw_text,
    )


# ---------------------------------------------------------------------------
# Safe wrapper with explicit fallback
# ---------------------------------------------------------------------------

FallbackFn = Callable[[TriageAction, Dict[str, Any]], Coroutine[Any, Any, LLMJudgeResult]]


async def grade_with_llm_safe(
    action: TriageAction,
    task: Dict[str, Any],
    episode_summary: str = "",
    model: str = _DEFAULT_MODEL,
    fallback: Optional[FallbackFn] = None,
) -> LLMJudgeResult:
    """
    ``grade_with_llm`` with an optional explicit fallback on failure.

    Unlike the original ``graders.py`` implementation, fallback is NOT
    automatic — callers must explicitly opt in by passing ``fallback``.
    This prevents silent degradation to keyword scoring without surfacing
    the failure.

    Parameters
    ----------
    action, task, episode_summary, model
        Same as ``grade_with_llm``.
    fallback : async callable | None
        If provided, called with ``(action, task)`` when the primary
        grader raises any ``LLMJudgeError``.  Must return ``LLMJudgeResult``.
        When None and the primary grader fails, the exception propagates.

    Returns
    -------
    LLMJudgeResult
        From primary grader, or from fallback if primary failed and
        fallback was provided.

    Raises
    ------
    LLMJudgeError subclass
        If the primary grader fails and no fallback is provided.
    """
    try:
        return await grade_with_llm(action, task, episode_summary, model)
    except LLMJudgeError:
        if fallback is None:
            raise
        return await fallback(action, task)


# ---------------------------------------------------------------------------
# Keyword-score fallback factory (explicit opt-in)
# ---------------------------------------------------------------------------

def make_keyword_fallback(
    grader_fn: Any,  # Callable[[TriageAction, dict], Any]
) -> FallbackFn:
    """
    Build a fallback ``FallbackFn`` that wraps the synchronous keyword grader.

    This is the explicit replacement for the silent fallback in the original
    ``grade_with_llm``.  Callers must deliberately construct and pass this
    to ``grade_with_llm_safe``.

    Parameters
    ----------
    grader_fn : callable
        Synchronous grader — typically ``triagerl.reward.grader.grade``.
        Must accept ``(TriageAction, dict)`` and return an object with
        ``reasoning_quality`` and ``feedback`` attributes.

    Returns
    -------
    FallbackFn (async callable)

    Example
    -------
    ::

        from triagerl.reward.grader import grade
        from triagerl.reward.llm_judge import grade_with_llm_safe, make_keyword_fallback

        result = await grade_with_llm_safe(
            action, task,
            fallback=make_keyword_fallback(grade),
        )
    """
    async def _fallback(action: TriageAction, task: Dict[str, Any]) -> LLMJudgeResult:
        fallback_result = grader_fn(action, task)
        correct_esi     = _resolve_esi_from_dict(task)
        undertriage     = _compute_undertriage(action.esi_level, correct_esi)
        return LLMJudgeResult(
            llm_reasoning_score=float(
                getattr(fallback_result, "reasoning_quality", 0.0)
            ),
            llm_feedback=(
                "[keyword fallback] "
                + str(getattr(fallback_result, "feedback", ""))
            ),
            llm_undertriage_flag=undertriage,
            llm_safety_concern=False,
            model="keyword-fallback",
            raw_response="",
        )

    return _fallback


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_esi_from_dict(task: Dict[str, Any]) -> Optional[int]:
    """Extract correct ESI from task dict supporting both field names."""
    v = task.get("esi_correct", task.get("correct_esi"))
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _compute_undertriage(
    predicted_esi: Optional[int],
    correct_esi: Optional[int],
) -> bool:
    """
    Compute undertriage flag from inputs (not from LLM response).

    The LLM is not trusted for this safety-critical binary decision.
    """
    if correct_esi is None or predicted_esi is None:
        return False
    return correct_esi <= _CRITICAL_ESI_THRESHOLD and predicted_esi > correct_esi