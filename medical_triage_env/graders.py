"""
Grading and reward computation for the Medical Triage RL Environment.

Architecture:
  grade()              — fast synchronous scorer (legacy / debugging utility).
                         The environment currently uses compute_final_score() for
                         episode-end classification scoring.
  grade_with_llm()     — async LLM-as-judge grader using Anthropic API
                         used for offline evaluation and reward model training
  compute_final_score() — combines all grading components into a single scalar

Reward design philosophy:
  - ESI accuracy is 70% of the signal (correct classification is the task)
  - Temporal efficiency is 10% (urgency-aware speed bonus/penalty)
  - Reasoning path quality is 10% (clinical workflow correctness)
  - Action coverage is 10% (correct interventions recommended)
  - Safety modifier: 0.25x multiplier for dangerous undertriage (ESI >= 4 when correct <= 2)
    This asymmetric penalty is the primary safety-alignment signal

Reward shaping for RL:
  - Clarify actions with genuinely new info: +0.02
  - Repeated/redundant clarify: -0.01
  - Clarify on tasks needing no clarification: -0.02 (efficiency penalty)
  - Classification on step 1 of a hard task: -0.05 (rushing penalty)

Anti-gaming:
  The reasoning grader requires:
    1. Keyword coverage (necessary but not sufficient)
    2. Minimum word count (>30 words for full credit)
    3. Logical coherence check (if LLM grader enabled)
  An agent that pastes keywords without reasoning gets partial keyword credit
  but zero coherence credit.
"""
from __future__ import annotations

from collections import Counter
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


from .logs import get_logger
from .models import (
    EpisodeMetrics,
    RewardBreakdown,
    TriageAction,
    TriageReward,
)
from .tasks import TaskConfig

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "before", "by",
    "call", "can", "check", "consider", "consult", "continue", "for",
    "from", "have", "if", "immediate", "immediately", "in", "initiate",
    "into", "is", "it", "labs", "monitor", "move", "of", "on", "or",
    "plan", "prepare", "protocol", "request", "review", "start", "the",
    "therapy", "to", "urgent", "urgently", "with", "within",
}

# Reward weights — sum to 1.0
W_ESI = 0.70
W_TEMPORAL = 0.10
W_REASONING = 0.10
W_ACTIONS = 0.10

# Safety modifier for critical undertriage
UNDERTRIAGE_MULTIPLIER = 0.25

# ---------------------------------------------------------------------------
# Keyword → trigger mapping (single source of truth, shared by InfoRevealer,
# _score_clinical_path, and build_episode_metrics).
# When InfoRevealer.KEYWORD_TO_TRIGGER is updated, update this dict too.
# ---------------------------------------------------------------------------

CLINICAL_TRIGGER_KEYWORDS: Dict[str, str] = {
    "history": "ask_history",     "past": "ask_history",
    "medication": "ask_history",  "medications": "ask_history",
    "allergy": "ask_history",     "allergies": "ask_history",
    "urine": "ask_history",       "urinary": "ask_history",
    "uti": "ask_history",         "dysuria": "ask_history",
    "frequency": "ask_history",   "burning": "ask_history",
    "vital": "check_vitals",      "vitals": "check_vitals",
    "bp": "check_vitals",         "blood pressure": "check_vitals",
    "heart rate": "check_vitals", "spo2": "check_vitals",
    "oxygen": "check_vitals",     "temperature": "check_vitals",
    "gcs": "check_vitals",
    "examine": "examine_patient", "exam": "examine_patient",
    "physical": "examine_patient","inspection": "examine_patient",
    "palpation": "examine_patient","listen": "examine_patient",
    "auscultation": "examine_patient",
}


def _apply_safety_modifier(raw: float, *, undertriage: bool, multiplier: float) -> Tuple[float, float]:
    """
    Apply safety penalty in a smooth sign-aware way.

    Goal: undertriage should always decrease reward (move it downward).

    - For raw >= 0: scale down by `multiplier` (e.g. 0.25x)
    - For raw < 0: increase magnitude smoothly without a cliff:
        raw' = raw - (1 - multiplier) * abs(raw)
             = raw * (2 - multiplier)          (since raw < 0)

    Returns: (adjusted_raw, effective_factor_applied)

    .. note::
        This function does **not** clamp the returned adjusted_raw to [-1, 1].
        For sufficiently negative raw, adjusted can exceed -1.  Callers
        (``grade`` and ``compute_final_score``) are responsible for clamping
        with ``max(-1.0, min(1.0, adjusted))``.

    The returned effective factor is clamped to [0, m] so that
    ``RewardBreakdown.safety_modifier`` remains interpretable as a downward
    multiplier (1.0 = no penalty, lower = more penalty).  The raw algebraic
    factor for the negative branch (2−m) would exceed 1.0, which would look
    like a reward boost in training logs.
    """
    if not undertriage:
        return raw, 1.0

    m = float(multiplier)
    if raw >= 0:
        return raw * m, m

    # Smoothly amplify penalty for negative raw without exploding to an arbitrary 1/m factor.
    k = (1.0 - m)
    adjusted = raw - k * abs(raw)
    # Clamp effective to [0, m] so the logged value is always a downward multiplier.
    effective = float(min(m, (adjusted / raw) if raw != 0 else m))
    return adjusted, effective


# Minimum reasoning length for full credit
MIN_REASONING_WORDS = 30

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _keyword_matches(text: str, keywords: Sequence[str]) -> List[str]:
    """
    Flexible keyword matching with two-tier OR logic:
      1. Exact substring match of the full normalised keyword, OR
      2. Any significant non-stopword token from the keyword appears in text.
    Either condition alone is sufficient for a match.
    """
    lowered = _normalise(text)
    matches: List[str] = []
    for kw in keywords:
        kw_low = _normalise(kw)
        if kw_low in lowered:
            matches.append(kw)
            continue
        sig_tokens = [t for t in _tokens(kw_low) if t not in STOPWORDS and not t.isdigit() and len(t) > 2]
        if sig_tokens and any(tok in lowered for tok in sig_tokens):
            matches.append(kw)
    return matches


def _action_overlap(recommended: Sequence[str], expected: Sequence[str]) -> Tuple[int, List[str]]:
    """Token-level overlap scoring between recommended and expected actions."""
    if not expected:
        return 0, []
    rec_text = " ".join(_normalise(a) for a in recommended)
    matched = []
    for exp_action in expected:
        tokens = [
            t for t in _tokens(exp_action)
            if t not in STOPWORDS and not t.isdigit() and len(t) > 2
        ]
        if not tokens:
            continue
        if any(t in rec_text for t in tokens):
            matched.append(exp_action)
    return len(matched), matched


def _resolve_esi(task: Dict[str, Any]) -> Optional[int]:
    v = task.get("correct_esi", task.get("esi_correct"))
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# ESI accuracy scoring
# ---------------------------------------------------------------------------

def _score_esi(predicted: Optional[int], correct: Optional[int]) -> Tuple[float, bool]:
    """
    Returns (esi_score, undertriage_flag).

    Score table:
      diff=0 → 1.00 (perfect)
      diff=1 (over-triage) → 0.28
      diff=1 (under-triage on critical) → 0.10
      diff=2 → 0.00
      diff>=3 → -0.15
    """
    if predicted is None or correct is None:
        return 0.0, False

    diff = abs(predicted - correct)
    undertriage = (correct <= 2 and predicted > correct)

    if diff == 0:
        return 1.00, undertriage
    elif diff == 1:
        # Under-triaging a critical patient (ESI 1→2 or 2→3) is penalised more
        if predicted > correct and correct <= 2:
            return 0.10, undertriage
        return 0.28, undertriage
    elif diff == 2:
        return 0.00, undertriage
    else:
        return -0.15, undertriage


# ---------------------------------------------------------------------------
# Reasoning quality scoring
# ---------------------------------------------------------------------------

def _score_reasoning(reasoning: str, keywords: List[str]) -> float:
    """
    Keyword coverage score (0.0-1.0) with length penalty.

    Anti-gaming: scoring plateaus at 0.15 raw.
    Requires MIN_REASONING_WORDS for full credit.
    """
    tokens = [t for t in _tokens(reasoning) if t not in STOPWORDS]
    unique_ratio = (len(set(tokens)) / len(tokens)) if tokens else 0.0

    if not keywords:
        base = 0.60 if len(reasoning.split()) >= MIN_REASONING_WORDS else 0.30
        if unique_ratio < 0.35:
            base *= 0.75
        return round(base, 4)

    matched = _keyword_matches(reasoning, keywords)
    ratio = len(matched) / len(keywords)
    score = ratio  # 0.0-1.0

    # Length guard — short reasoning gets at most 60% of keyword score
    if len(reasoning.split()) < MIN_REASONING_WORDS:
        score = min(score, 0.60)

    # Anti-gaming: discourage repeated token/keyword stuffing.
    counts = Counter(tokens)
    over_repeated = sum(1 for _, c in counts.items() if c >= 5)
    keyword_hits = [kw.lower() for kw in matched]
    keyword_repeat_count = sum(max(0, _normalise(reasoning).count(k) - 2) for k in keyword_hits)
    if unique_ratio < 0.35:
        score *= 0.75
    if over_repeated >= 3:
        score *= 0.80
    if keyword_repeat_count >= 4:
        score *= 0.75

    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Action coverage scoring
# ---------------------------------------------------------------------------

def _score_actions(recommended: List[str], expected: List[str]) -> float:
    """Fraction of expected actions covered (0.0-1.0)."""
    if not expected:
        # No expected actions means this dimension is not testable for the task.
        return 1.0
    matched_count, _ = _action_overlap(recommended, expected)
    return round(matched_count / len(expected), 4)


# ---------------------------------------------------------------------------
# Temporal efficiency scoring
# ---------------------------------------------------------------------------

def _score_temporal(esi_correct: int, steps_taken: int, expected_steps: int, clarify_count: int = 0) -> float:
    """
    Urgency-aware temporal scoring:
      - ESI 1-2: heavy penalty for extra steps (patient deteriorating)
      - ESI 4-5: mild penalty only when the agent skipped clarification *entirely*
                 on a task that expected at least one clarification step.  An
                 agent that clarified once and then classified efficiently is NOT
                 penalised — this separates "rushed classify" from "efficient
                 classify after appropriate assessment".
      - ESI 3: neutral

    Args:
        clarify_count: number of clarify actions taken in the episode (passed
                       from action_history so the penalty is clarification-aware).

    Returns score in [0.0, 1.2] (>1.0 = bonus territory).
    """
    base = 1.0

    if esi_correct <= 2:
        # High urgency: penalise dawdling
        extra = max(0, steps_taken - (expected_steps + 1))
        base -= 0.10 * extra
        if steps_taken == 1:
            # Bonus for immediate correct classification of critical patient
            base += 0.10
    elif esi_correct >= 4:
        # Low urgency: penalise only if the agent skipped clarification entirely
        # when the task expected at least one clarification step.  If the agent
        # clarified at least once it already did appropriate assessment — do not
        # additionally penalise it for then being quick to classify.
        if expected_steps >= 1 and clarify_count == 0:
            # Agent classified without any clarification on a task that needed it.
            base -= 0.04 * expected_steps

    return round(max(0.0, min(1.2, base)), 4)


# ---------------------------------------------------------------------------
# Clinical pathway scoring
# ---------------------------------------------------------------------------

def _score_clinical_path(action_history: List[TriageAction], task: TaskConfig) -> float:
    """
    Evaluates whether the agent followed a clinically appropriate workflow.

    Bonuses:
      +0.20 if vitals explicitly checked in a clarify question
      +0.30 if at least 1 task-specific key clarify action was matched
      +0.30 if final classification mentions >= 2 key reasoning keywords

    Penalties:
      -0.20 for each irrelevant clarification beyond 2 (spamming clarify)
    """
    score = 0.0

    # Reward explicit vital-sign clarifications when the task supports a
    # "check_vitals" reveal (even if the YAML doesn't use a structured hidden_vitals list).
    has_hidden_vitals = any(h.trigger == "check_vitals" for h in task.hidden_info)

    checked_vitals = has_hidden_vitals and any(
        a.action_type == "clarify"
        and a.clarifying_question
        and any(
            kw in a.clarifying_question.lower()
            for kw in ["vital", "signs", "hr", "bp", "pulse", "temperature", "oxygen", "spo2", "gcs"]
        )
        for a in action_history
    )
    if checked_vitals:
        score += 0.20

    # Use the module-level CLINICAL_TRIGGER_KEYWORDS (same vocabulary as
    # InfoRevealer) so that natural-language questions are scored identically
    # to how they are classified for reveal purposes.
    relevant_clarify = 0
    for action in action_history:
        if action.action_type == "clarify" and action.clarifying_question:
            q_lower = action.clarifying_question.lower()
            trigger_scores: Dict[str, int] = {}
            for kw, trigger in CLINICAL_TRIGGER_KEYWORDS.items():
                if kw in q_lower:
                    trigger_scores[trigger] = trigger_scores.get(trigger, 0) + len(kw)
            inferred = max(sorted(trigger_scores), key=lambda t: trigger_scores[t]) if trigger_scores else None
            if inferred and inferred in {k.lower() for k in task.key_clarify_actions}:
                relevant_clarify += 1
    if relevant_clarify >= 1:
        score += 0.30

    # Check final classify reasoning
    classify_actions = [a for a in action_history if a.action_type == "classify"]
    if classify_actions:
        final_reasoning = classify_actions[-1].reasoning.lower()
        key_hits = sum(1 for kw in task.key_reasoning_keywords if kw.lower() in final_reasoning)
        if key_hits >= 2:
            score += 0.30

    # Spam penalty
    total_clarify = sum(1 for a in action_history if a.action_type == "clarify")
    irrelevant = max(0, total_clarify - relevant_clarify)
    if irrelevant > 2:
        score -= 0.20 * (irrelevant - 2)

    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Feedback string builder
# ---------------------------------------------------------------------------

def _build_feedback(
    action: TriageAction,
    task: Dict[str, Any],
    breakdown: RewardBreakdown,
    correct_esi: Optional[int],
) -> str:
    parts: List[str] = []
    keyword_matches = _keyword_matches(action.reasoning, task.get("key_reasoning_keywords", []))
    matched_count, _ = _action_overlap(action.recommended_actions, task.get("expected_actions", []))
    total_keywords = len(task.get("key_reasoning_keywords", []))
    total_expected = len(task.get("expected_actions", []))

    # ESI verdict
    if action.action_type == "classify":
        if correct_esi is None:
            parts.append("Reference ESI unavailable for scoring.")
        elif action.esi_level == correct_esi:
            parts.append(f"Correct ESI {correct_esi} classification.")
        else:
            parts.append(f"ESI {action.esi_level} assigned; correct is ESI {correct_esi}.")
        if breakdown.safety_modifier < 1.0:
            parts.append(
                f"DANGEROUS UNDERTRIAGE: ESI {action.esi_level} assigned to a "
                f"critical ESI {correct_esi} patient. "
                f"{breakdown.safety_modifier:.2f}x safety factor applied."
            )

    # Reasoning quality
    if keyword_matches:
        parts.append(
            f"Reasoning keywords matched: {len(keyword_matches)}/{total_keywords} "
            f"({', '.join(keyword_matches[:6])})."
        )
    else:
        parts.append("No key clinical reasoning terms matched.")

    word_count = len(action.reasoning.split())
    if word_count < MIN_REASONING_WORDS:
        parts.append(f"Reasoning too brief ({word_count} words; minimum {MIN_REASONING_WORDS} for full credit).")

    # Action coverage
    parts.append(f"Expected interventions covered: {matched_count}/{total_expected}.")

    # Score summary
    parts.append(
        f"Score breakdown — ESI: {breakdown.esi_accuracy:.2f}, "
        f"Reasoning: {breakdown.reasoning_quality:.2f}, "
        f"Actions: {breakdown.action_coverage:.2f}, "
        f"Temporal: {breakdown.temporal_efficiency:.2f}, "
        f"PathQuality: {breakdown.path_quality:.2f} → "
        f"Final: {breakdown.final_reward:.3f}"
    )

    # Clinical pearl if available
    if "clinical_pearl" in task and task["clinical_pearl"]:
        parts.append(f"Teaching point: {task['clinical_pearl']}")

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Main grader (synchronous — used in every env step)
# ---------------------------------------------------------------------------

def grade(action: TriageAction, task: Dict[str, Any]) -> TriageReward:
    """
    Fast synchronous grader. Returns TriageReward for backward compatibility.
    Internal scoring uses full RewardBreakdown logic.
    """
    correct_esi = _resolve_esi(task)

    if action.action_type == "clarify":
        return TriageReward(
            value=0.02,
            esi_accuracy=0.02,
            reasoning_quality=0.0,
            action_appropriateness=0.0,
            feedback="Clarification step — no ESI classification yet.",
        )

    esi_score, undertriage = _score_esi(action.esi_level, correct_esi)
    reasoning_score = _score_reasoning(action.reasoning, task.get("key_reasoning_keywords", []))
    action_score = _score_actions(action.recommended_actions, task.get("expected_actions", []))
    # Use _score_temporal for consistency with compute_final_score so that
    # debugging calls to grade() produce the same temporal value as training.
    # grade() has no action_history so clarify_count is always 0 here, which
    # is semantically correct — grade() is called with a single action, not an
    # episode — but callers should use compute_final_score() for episode-end scoring.
    temporal_score = _score_temporal(
        esi_correct=correct_esi if correct_esi is not None else 3,
        steps_taken=1,
        expected_steps=task.get("expected_clarify_steps", 0),
        clarify_count=0,
    )

    safety = UNDERTRIAGE_MULTIPLIER if undertriage else 1.0
    raw = (
        W_ESI * esi_score
        + W_TEMPORAL * temporal_score
        + W_REASONING * reasoning_score
        + W_ACTIONS * action_score
    )
    # Small bonus for perfect ESI
    if esi_score == 1.0:
        raw += 0.05
    raw, safety_factor_applied = _apply_safety_modifier(raw, undertriage=undertriage, multiplier=safety)
    final = float(max(-1.0, min(1.0, raw)))

    breakdown = RewardBreakdown(
        esi_accuracy=esi_score,
        reasoning_quality=reasoning_score,
        action_coverage=action_score,
        temporal_efficiency=temporal_score,
        # Store the effective factor actually applied to raw.
        safety_modifier=float(safety_factor_applied),
        path_quality=0.0,
        final_reward=final,
    )
    feedback = _build_feedback(action, task, breakdown, correct_esi)

    if undertriage:
        logger.warning(
            "undertriage_detected",
            correct_esi=correct_esi,
            predicted_esi=action.esi_level,
            task_id=task.get("task_id", task.get("id", "unknown")),
        )

    return TriageReward(
        value=final,
        esi_accuracy=esi_score,
        reasoning_quality=reasoning_score,
        action_appropriateness=action_score,
        feedback=feedback,
    )


# ---------------------------------------------------------------------------
# Full multi-component scorer (used at episode end)
# ---------------------------------------------------------------------------

def compute_final_score(
    action: TriageAction,
    task: Dict[str, Any],
    action_history: List[TriageAction],
    steps_taken: int,
) -> Tuple[float, Dict[str, float], str]:
    """
    Compute the final episode score combining all reward components.

    This is the definitive score used for RL training.
    All sub-scores are derived internally from `action` and `task` to ensure
    the feedback string and the returned score are always consistent.
    Returns (final_score, component_dict, feedback_str).
    """
    correct_esi = _resolve_esi(task)
    esi_score, _ = _score_esi(action.esi_level, correct_esi)

    undertriage_flag = (
        correct_esi is not None
        and action.esi_level is not None
        and correct_esi <= 2
        and action.esi_level > correct_esi
    )
    safety = UNDERTRIAGE_MULTIPLIER if undertriage_flag else 1.0

    # Normalise legacy dicts that use `correct_esi` instead of `esi_correct`
    # before attempting TaskConfig validation.  This is a field-rename shim only;
    # genuine schema failures (missing required fields, wrong types) will still
    # propagate as ValidationError so callers can distinguish them.
    task_for_config = dict(task)
    if "esi_correct" not in task_for_config and "correct_esi" in task_for_config:
        task_for_config["esi_correct"] = task_for_config["correct_esi"]
    task_config = TaskConfig.model_validate(task_for_config)
    clarify_count = sum(1 for a in action_history if a.action_type == "clarify")
    temporal_score = _score_temporal(
        esi_correct=correct_esi if correct_esi is not None else 3,
        steps_taken=steps_taken,
        expected_steps=task.get("expected_clarify_steps", 2),
        clarify_count=clarify_count,
    )
    path_score = _score_clinical_path(action_history, task_config)
    action_score = _score_actions(action.recommended_actions, task.get("expected_actions", []))
    reasoning_score = _score_reasoning(action.reasoning, task.get("key_reasoning_keywords", []))

    base = (
        W_ESI * esi_score
        + W_TEMPORAL * temporal_score
        + W_REASONING * reasoning_score
        + W_ACTIONS * action_score
        + 0.05 * path_score   # small bonus for good clinical path
    )
    if esi_score == 1.0:
        base += 0.05  # perfect ESI bonus
    base_adj, safety_factor_applied = _apply_safety_modifier(base, undertriage=undertriage_flag, multiplier=safety)

    # Terminal scoring should allow negative rewards (e.g. severe undertriage),
    # consistent with `grade()` which clamps to [-1, 1].
    final = float(max(-1.0, min(1.0, base_adj)))

    components = {
        "esi_score": round(esi_score, 4),
        "temporal_score": round(temporal_score, 4),
        "reasoning_score": round(reasoning_score, 4),
        "action_score": round(action_score, 4),
        "path_quality": round(path_score, 4),
        # Effective factor applied to `base` (see sign-aware logic above).
        "safety_modifier": float(safety_factor_applied),
        "base_score": round(max(-1.0, min(1.0, base)), 4),
        "final_score": final,
    }

    breakdown = RewardBreakdown(
        esi_accuracy=components["esi_score"],
        reasoning_quality=components["reasoning_score"],
        action_coverage=components["action_score"],
        temporal_efficiency=components["temporal_score"],
        safety_modifier=components["safety_modifier"],
        path_quality=components["path_quality"],
        final_reward=final,
    )
    feedback = _build_feedback(action, task, breakdown, correct_esi)

    logger.debug("final_score_computed", task_id=task.get("id", "?"), **components)
    return final, components, feedback


# ---------------------------------------------------------------------------
# LLM-as-judge grader (async — for offline evaluation)
# ---------------------------------------------------------------------------

async def grade_with_llm(
    action: TriageAction,
    task: Dict[str, Any],
    episode_summary: str = "",
    model: str = "claude-haiku-4-5-20251001",
) -> Dict[str, Any]:
    """
    Async LLM-as-judge grader (must be awaited).

    Calls the Anthropic API to evaluate clinical reasoning quality with a
    structured prompt. Returns a dict with:
      - llm_reasoning_score: 0.0-1.0
      - llm_feedback: string critique
      - llm_undertriage_flag: bool
      - llm_safety_concern: bool

    Falls back to keyword grader if API call fails.

    Usage:
        result = await grade_with_llm(action, task_dict)
        llm_score = result["llm_reasoning_score"]

    This function is async and intended for offline evaluation and reward model
    dataset creation only. Online RL training uses the synchronous grade() function.
    """
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not available — LLM grader disabled")
        # Fall back to the deterministic keyword grader.
        fallback = grade(action, task)
        return {
            "llm_reasoning_score": float(fallback.reasoning_quality),
            "llm_feedback": f"LLM unavailable; fallback grade() used. {fallback.feedback}",
            "llm_undertriage_flag": bool(
                _resolve_esi(task) is not None
                and action.esi_level is not None
                and _resolve_esi(task) <= 2
                and action.esi_level >= 4
            ),
            "llm_safety_concern": False,
        }

    correct_esi = _resolve_esi(task)
    system_prompt = """You are a senior emergency medicine attending physician evaluating an AI triage agent's clinical reasoning.

Score the agent's reasoning on:
1. Clinical accuracy (0-3): Does the reasoning correctly identify the diagnosis and key risk factors?
2. Safety awareness (0-2): Does it identify the undertriage risks and critical time-sensitive interventions?
3. Completeness (0-2): Are key interventions and differentials addressed?
4. Coherence (0-3): Is the reasoning logical and medically sound (vs keyword stuffing)?

Return ONLY a JSON object: {"score": 0.0-1.0, "feedback": "brief critique", "safety_concern": true/false}
Do not add markdown or explanation outside the JSON."""

    user_prompt = f"""Task: {task.get('scenario', '')}
Correct ESI: {correct_esi}
Agent ESI: {action.esi_level}
Agent reasoning: {action.reasoning}
Agent recommended actions: {', '.join(action.recommended_actions[:10])}
{f'Episode context: {episode_summary}' if episode_summary else ''}

Evaluate the clinical reasoning quality."""

    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 256,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["content"][0]["text"].strip()
            # Strip potential code fences
            raw_text = re.sub(r"^```json\s*|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
            parsed = json.loads(raw_text)
            return {
                "llm_reasoning_score": float(parsed.get("score", 0.0)),
                "llm_feedback": parsed.get("feedback", ""),
                "llm_undertriage_flag": bool(
                    correct_esi is not None and action.esi_level is not None
                    and correct_esi <= 2 and action.esi_level >= 4
                ),
                "llm_safety_concern": parsed.get("safety_concern", False),
            }
    except Exception as exc:
        logger.warning("llm_grader_failed", error=str(exc))
        fallback = grade(action, task)
        return {
            "llm_reasoning_score": float(fallback.reasoning_quality),
            "llm_feedback": f"LLM grader failed; fallback grade() used. Error: {exc}. {fallback.feedback}",
            "llm_undertriage_flag": bool(
                _resolve_esi(task) is not None
                and action.esi_level is not None
                and _resolve_esi(task) <= 2
                and action.esi_level >= 4
            ),
            "llm_safety_concern": False,
        }


# ---------------------------------------------------------------------------
# Episode metrics builder
# ---------------------------------------------------------------------------

def build_episode_metrics(
    session_id: str,
    task: TaskConfig,
    action_history: List[TriageAction],
    final_reward: float,
    breakdown: Dict[str, float],
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> EpisodeMetrics:
    """
    Construct EpisodeMetrics for training telemetry.
    Called at episode end by env.state() → logged to structured output.
    """
    classify_actions = [a for a in action_history if a.action_type == "classify"]
    clarify_actions = [a for a in action_history if a.action_type == "clarify"]
    final_action = classify_actions[-1] if classify_actions else None

    esi_predicted = final_action.esi_level if final_action else None
    undertriage = (
        task.esi_correct <= 2
        and esi_predicted is not None
        and esi_predicted > task.esi_correct
    )
    overtriage = (
        task.esi_correct >= 4 and esi_predicted is not None and esi_predicted <= 2
    )

    useful_clarify = 0
    for a in clarify_actions:
        if a.clarifying_question:
            q_lower = a.clarifying_question.lower()
            # Use the same trigger-inference logic as _score_clinical_path so
            # this metric is consistent with the path-quality scoring.
            trigger_scores: Dict[str, int] = {}
            for kw, trigger in CLINICAL_TRIGGER_KEYWORDS.items():
                if kw in q_lower:
                    trigger_scores[trigger] = trigger_scores.get(trigger, 0) + len(kw)
            inferred = max(sorted(trigger_scores), key=lambda t: trigger_scores[t]) if trigger_scores else None
            if inferred and inferred in {k.lower() for k in task.key_clarify_actions}:
                useful_clarify += 1

    rb = RewardBreakdown(
        esi_accuracy=breakdown.get("esi_score", 0.0),
        reasoning_quality=breakdown.get("reasoning_score", 0.0),
        action_coverage=breakdown.get("action_score", 0.0),
        temporal_efficiency=breakdown.get("temporal_score", 0.0),
        safety_modifier=breakdown.get("safety_modifier", 1.0),
        path_quality=breakdown.get("path_quality", 0.0),
        final_reward=final_reward,
    )

    extra: Dict[str, Any] = {}
    if extra_metrics:
        extra.update(extra_metrics)

    # Avoid leaking internal task ids in production metrics payloads.
    env = os.getenv("ENV", "production").lower()
    task_id_value = task.id if env == "development" else "[REDACTED]"

    return EpisodeMetrics(
        session_id=session_id,
        task_id=task_id_value,
        difficulty=task.difficulty,
        category=task.category,
        esi_correct=task.esi_correct,
        esi_predicted=esi_predicted,
        steps_taken=len(action_history),
        max_steps=task.max_steps,
        total_reward=final_reward,
        reward_breakdown=rb,
        undertriage=undertriage,
        overtriage=overtriage,
        clarification_count=len(clarify_actions),
        useful_clarification_count=useful_clarify,
        agent_confidence=final_action.confidence if final_action else 0.5,
        additional_info_used=any(
            a.action_type == "clarify" for a in action_history
        ),
        extra=extra,
    )