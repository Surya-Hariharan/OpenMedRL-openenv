"""
triagerl.reward.path_quality
============================
Clinical pathway quality scorer.

Evaluates whether the agent followed a clinically appropriate information-
gathering workflow before classifying, based on:
  1. Whether the agent explicitly checked vitals when the task had a
     ``check_vitals`` hidden layer.
  2. Whether the agent's clarify questions matched the task's expected
     clarify triggers (``key_clarify_actions``).
  3. Whether the final classification reasoning mentioned ≥ 2 of the task's
     key reasoning keywords.
  4. Penalty for irrelevant/spammy clarify actions beyond a tolerance.

Design contract
---------------
*  Single source of truth: ``KEYWORD_TO_TRIGGER`` is imported from
   ``triagerl.core.constants``.  No local copy, no duplication.
*  Pure function: no I/O, no logging, no side-effects.
*  Inputs are plain Python types + ``TaskConfig`` (frozen Pydantic model).
*  Returns a single float ∈ [0.0, 1.0].

Keyword disambiguation algorithm
---------------------------------
The same length-weighted trigger inference used by ``InfoRevealer`` is
applied here so that path-quality scoring is consistent with the actual
reveal decisions made during the episode.

For each clarify question, keyword scores are accumulated per trigger
(``score[trigger] += len(keyword)``).  The trigger with the highest
cumulative score wins.  Ties broken alphabetically.  This is the same
algorithm as ``InfoRevealer.infer_trigger()`` — guaranteed by sharing
the same ``KEYWORD_TO_TRIGGER`` import.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional

from triagerl.core.constants import KEYWORD_TO_TRIGGER
from triagerl.core.models import TriageAction
from triagerl.tasks import TaskConfig


# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

# Bonus awarded when the agent explicitly checked vitals in a task that
# had a ``check_vitals`` hidden layer.
BONUS_VITALS_CHECKED: float = 0.20

# Bonus for each relevant clarify action matched (awarded if ≥ 1 matched).
BONUS_RELEVANT_CLARIFY: float = 0.30

# Bonus when the final reasoning mentions ≥ ``KEYWORD_HIT_THRESHOLD`` keywords.
BONUS_REASONING_KEYWORDS: float = 0.30
KEYWORD_HIT_THRESHOLD: int = 2

# Penalty per irrelevant clarification beyond ``IRRELEVANT_CLARIFY_TOLERANCE``.
SPAM_PENALTY_PER_EXCESS: float = 0.20
IRRELEVANT_CLARIFY_TOLERANCE: int = 2

# Vital-sign tokens used to detect explicit vital checking in a question
# (matched against the question text, not against KEYWORD_TO_TRIGGER —
# these are clinical terms, not trigger identifiers).
_VITAL_CHECK_TERMS: FrozenSet[str] = frozenset({
    "vital", "vitals", "signs", "hr", "bp", "pulse",
    "temperature", "oxygen", "spo2", "gcs",
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_trigger(question_lower: str) -> Optional[str]:
    """
    Infer the best-matching trigger from a lowercase clarify question.

    Uses the same length-weighted algorithm as ``InfoRevealer.infer_trigger``
    (identical because both import the same ``KEYWORD_TO_TRIGGER``).

    Returns the trigger name, or None if no keyword matched.
    """
    scores: Dict[str, int] = {}
    for keyword, trigger in KEYWORD_TO_TRIGGER.items():
        if keyword in question_lower:
            scores[trigger] = scores.get(trigger, 0) + len(keyword)

    if not scores:
        return None

    # Deterministic tie-breaking via alphabetical sort over trigger names.
    return max(sorted(scores), key=lambda t: scores[t])


def _question_checks_vitals(question: str) -> bool:
    """
    Return True if the question explicitly mentions vital sign terms.

    This is intentionally broader than the trigger-inference algorithm —
    "pulse ox" and "spo2" both count, even if the inferred trigger is
    ``ask_history`` for some edge-case question.
    """
    lowered = question.lower()
    return any(term in lowered for term in _VITAL_CHECK_TERMS)


# ---------------------------------------------------------------------------
# Primary scorer
# ---------------------------------------------------------------------------

def score_clinical_path(
    action_history: List[TriageAction],
    task: TaskConfig,
) -> float:
    """
    Score the clinical pathway quality for an episode.

    Parameters
    ----------
    action_history : list[TriageAction]
        Ordered list of all actions taken in the episode, including both
        clarify and classify actions.
    task : TaskConfig
        The task configuration (frozen, read-only).

    Returns
    -------
    float
        Path quality score ∈ [0.00, 1.00], rounded to 4 decimal places.

    Scoring breakdown
    -----------------
    +0.20  vitals explicitly checked when task has a check_vitals layer
    +0.30  ≥ 1 clarify action matched a task-expected trigger
    +0.30  final classify reasoning mentions ≥ 2 key reasoning keywords
    -0.20  per irrelevant clarification beyond IRRELEVANT_CLARIFY_TOLERANCE (2)
    """
    score = 0.0

    # ── Partition action history ──────────────────────────────────────────────
    clarify_actions  = [a for a in action_history if a.action_type == "clarify"]
    classify_actions = [a for a in action_history if a.action_type == "classify"]

    # ── Bonus 1: vitals checked ───────────────────────────────────────────────
    # Only meaningful when the task actually has a check_vitals reveal layer.
    has_vitals_layer = any(h.trigger == "check_vitals" for h in task.hidden_info)
    if has_vitals_layer:
        vitals_checked = any(
            a.clarifying_question and _question_checks_vitals(a.clarifying_question)
            for a in clarify_actions
        )
        if vitals_checked:
            score += BONUS_VITALS_CHECKED

    # ── Bonus 2: relevant clarify actions ────────────────────────────────────
    # Count clarify actions that inferred a trigger matching one of the task's
    # key_clarify_actions.  key_clarify_actions is a list of trigger names
    # (e.g. ["ask_history", "check_vitals"]).
    expected_triggers: FrozenSet[str] = frozenset(
        k.lower() for k in task.key_clarify_actions
    )
    relevant_clarify  = 0
    irrelevant_clarify = 0

    for action in clarify_actions:
        if not action.clarifying_question:
            irrelevant_clarify += 1
            continue

        inferred = _infer_trigger(action.clarifying_question.lower())

        if inferred and inferred in expected_triggers:
            relevant_clarify += 1
        else:
            irrelevant_clarify += 1

    if relevant_clarify >= 1:
        score += BONUS_RELEVANT_CLARIFY

    # ── Bonus 3: final reasoning keyword coverage ─────────────────────────────
    if classify_actions:
        final_reasoning = classify_actions[-1].reasoning.lower()
        keyword_hits = sum(
            1 for kw in task.key_reasoning_keywords
            if kw.lower() in final_reasoning
        )
        if keyword_hits >= KEYWORD_HIT_THRESHOLD:
            score += BONUS_REASONING_KEYWORDS

    # ── Penalty: spam clarification ───────────────────────────────────────────
    # Irrelevant clarifications beyond the tolerance incur a per-excess penalty.
    # The tolerance of 2 allows some exploratory questions without punishment.
    excess_irrelevant = max(0, irrelevant_clarify - IRRELEVANT_CLARIFY_TOLERANCE)
    if excess_irrelevant > 0:
        score -= SPAM_PENALTY_PER_EXCESS * excess_irrelevant

    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Useful-clarify counter (used by episode metrics builder)
# ---------------------------------------------------------------------------

def count_useful_clarifications(
    action_history: List[TriageAction],
    task: TaskConfig,
) -> int:
    """
    Count the number of clarify actions that matched a task-expected trigger.

    This is the same logic as ``score_clinical_path``'s relevant-clarify
    counting, extracted as a standalone function so that
    ``build_episode_metrics`` can import it without pulling in the full
    path scorer.

    Parameters
    ----------
    action_history : list[TriageAction]
        Full episode action history.
    task : TaskConfig
        Task configuration.

    Returns
    -------
    int
        Number of clarify actions whose inferred trigger matched one of
        ``task.key_clarify_actions``.
    """
    expected_triggers: FrozenSet[str] = frozenset(
        k.lower() for k in task.key_clarify_actions
    )
    count = 0
    for action in action_history:
        if action.action_type != "clarify" or not action.clarifying_question:
            continue
        inferred = _infer_trigger(action.clarifying_question.lower())
        if inferred and inferred in expected_triggers:
            count += 1
    return count