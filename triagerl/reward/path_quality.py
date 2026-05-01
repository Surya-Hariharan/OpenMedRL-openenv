"""
triagerl.reward.path_quality
============================
Clinical pathway quality scorer.

Fixes vs previous version
--------------------------
1. CRITICAL: No longer re-infers triggers independently from question text.
   Previous version used _infer_trigger() on the clarifying question to
   decide if a clarify action was "relevant". This caused systematic
   divergence from what InfoRevealer actually revealed, because InfoRevealer's
   implementation was in the opaque medical_triage_env package — never
   verified to be identical to the local re-inference code.

   FIX: score_clinical_path() now accepts an explicit list of
   ActualClarifyRecord objects, each containing the question and the ACTUAL
   trigger that the environment revealed (taken directly from the step()
   return payload). No re-inference. Ground truth from the env, not a
   post-hoc approximation.

2. count_useful_clarifications() updated to use ActualClarifyRecord.
   Previous version re-inferred triggers, producing a metric inconsistent
   with actual episode events.

3. _question_checks_vitals() retained for the vitals-bonus check only,
   where it is used to detect whether the agent explicitly asked about vitals
   (a question-level check, not a trigger-inference step).

Design contract
---------------
* Pure function — no I/O, no logging, no side-effects.
* All trigger classification comes from the env reveal payloads, not
  from keyword re-inference.
* Returns float ∈ [0.0, 1.0].
"""
from __future__ import annotations

from typing import FrozenSet, List, NamedTuple, Optional, Sequence
import math
import re
from collections import Counter

from triagerl.core.constants import VALID_TRIGGERS
from triagerl.tasks.schema import TaskConfig


# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

BONUS_VITALS_CHECKED:          float = 0.20
BONUS_RELEVANT_CLARIFY:        float = 0.30
BONUS_REASONING_KEYWORDS:      float = 0.30
KEYWORD_HIT_THRESHOLD:         int   = 2
SPAM_PENALTY_PER_EXCESS:       float = 0.30
IRRELEVANT_CLARIFY_TOLERANCE:  int   = 2
LOW_DIVERSITY_THRESHOLD:       float = 0.50
LOW_DIVERSITY_PENALTY:         float = 0.12
SHALLOW_MIN_WORDS:             int   = 18
SHALLOW_STRUCTURE_BONUS:       float = 0.10
SHALLOW_PENALTY:               float = 0.25
KEYWORD_DENSITY_THRESHOLD:     float = 0.18
KEYWORD_DENSITY_PENALTY:       float = 1.00
REPETITION_START_COUNT:        int   = 3
REPETITION_EXPONENT:           float = 1.35

# Vital-sign terms used to detect explicit vital checking in question text.
# This is a question-level check (not trigger inference).
_VITAL_CHECK_TERMS: FrozenSet[str] = frozenset({
    "vital", "vitals", "signs", "hr", "bp", "pulse",
    "temperature", "oxygen", "spo2", "gcs",
})


# ---------------------------------------------------------------------------
# ActualClarifyRecord
# ---------------------------------------------------------------------------

class ActualClarifyRecord(NamedTuple):
    """
    Record of a single clarify action and the trigger the env actually fired.

    FIX: This replaces the post-hoc trigger inference approach. The trigger
    field is populated by the env's step() handler from the InfoRevealer
    reveal payload, not computed from the question text.

    Fields
    ------
    question : str
        The clarifying question the agent asked.
    trigger : str | None
        The trigger that InfoRevealer actually fired, from the reveal
        payload's "trigger" key. None if no information was revealed
        (empty payload) or if the payload lacked a valid trigger key.
    """
    question: str
    trigger:  Optional[str]   # None = no reveal or unknown trigger


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _question_checks_vitals(question: str) -> bool:
    """
    Return True if the question explicitly mentions vital sign terms.

    Used only for the vitals-bonus check (Bonus 1), not for trigger
    classification. This is intentionally broader than trigger inference.
    """
    lowered = question.lower()
    return any(term in lowered for term in _VITAL_CHECK_TERMS)


def _has_reasoning_structure(reasoning: str) -> bool:
    """
    Lightweight structure heuristic for clinical reasoning.

    We treat reasoning as structured if it uses at least one of the common
    explanatory markers or if it has multiple sentence clauses.
    """
    lowered = reasoning.lower()
    markers = (
        "because", "therefore", "due to", "suggests", "overall",
        "however", "since", "risk", "concern", "plan",
    )
    if any(marker in lowered for marker in markers):
        return True
    if lowered.count(".") + lowered.count(";") >= 2:
        return True
    return False


# ---------------------------------------------------------------------------
# Primary scorer
# ---------------------------------------------------------------------------

def score_clinical_path(
    clarify_records: Sequence[ActualClarifyRecord],
    final_reasoning: str,
    task: TaskConfig,
) -> float:
    """
    Score the clinical pathway quality for an episode.

    FIX: Signature changed. Now receives ActualClarifyRecord objects
    (question + actual env trigger) instead of a raw action_history from
    which triggers were re-inferred. This eliminates divergence between
    path quality scores and actual episode events.

    Parameters
    ----------
    clarify_records : sequence of ActualClarifyRecord
        One record per clarify action. trigger field is the actual trigger
        from the env's reveal payload (None if no reveal occurred).
    final_reasoning : str
        The reasoning text from the final classify action.
    task : TaskConfig
        Frozen task configuration.

    Returns
    -------
    float
        Positive values reward good pathway structure; negative values are
        allowed to strongly penalise keyword stuffing.

    Scoring breakdown
    -----------------
    +0.20  vitals explicitly checked when task has a check_vitals layer
    +0.30  ≥ 1 clarify action had a trigger matching task's expected triggers
    +0.30  final reasoning mentions ≥ KEYWORD_HIT_THRESHOLD key reasoning keywords
    -0.20  per irrelevant clarification beyond IRRELEVANT_CLARIFY_TOLERANCE
    """
    score = 0.0

    # ── Bonus 1: vitals checked (question-text level) ─────────────────────────
    has_vitals_layer = any(h.trigger == "check_vitals" for h in task.hidden_info)
    if has_vitals_layer:
        vitals_checked = any(
            _question_checks_vitals(r.question) for r in clarify_records
        )
        if vitals_checked:
            score += BONUS_VITALS_CHECKED

    # ── Bonus 2: relevant clarify actions (actual trigger, not re-inferred) ───
    expected_triggers: FrozenSet[str] = frozenset(
        k.lower() for k in task.key_clarify_actions
    )
    relevant_count   = 0
    irrelevant_count = 0

    for record in clarify_records:
        if record.trigger is not None and record.trigger in expected_triggers:
            relevant_count += 1
        else:
            # trigger is None (no reveal) or trigger not in expected set
            irrelevant_count += 1

    if relevant_count >= 1:
        score += BONUS_RELEVANT_CLARIFY

    # ── Bonus 3: final reasoning keyword coverage ─────────────────────────────
    reasoning_lower = final_reasoning.lower()
    keyword_hits = sum(
        1 for kw in task.key_reasoning_keywords
        if kw.lower() in reasoning_lower
    )
    if keyword_hits >= KEYWORD_HIT_THRESHOLD:
        score += BONUS_REASONING_KEYWORDS

    # ── Anti-gaming: keyword-stuffing / low-diversity penalty
    # Penalise unnaturally repetitive or dense use of keywords in final
    # reasoning. This is a small penalty so it does not dominate the
    # path score but provides a gradient against stuffing.
    tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", final_reasoning.lower()) if len(t) > 2]
    sig_tokens = [t for t in tokens if t not in {"the", "and", "for", "with", "that", "this", "from", "are", "was", "were"}]
    unique_ratio = (len(set(sig_tokens)) / len(sig_tokens)) if sig_tokens else 1.0

    task_keywords = [kw.lower() for kw in task.key_reasoning_keywords if kw]
    key_hits = sum(reasoning_lower.count(kw) for kw in task_keywords)
    keyword_density = key_hits / max(1, len(sig_tokens))

    if len(sig_tokens) < SHALLOW_MIN_WORDS:
        score -= SHALLOW_PENALTY
    if not _has_reasoning_structure(final_reasoning):
        score -= SHALLOW_STRUCTURE_BONUS

    # Low diversity is a strong anti-stuffing signal only when the text is
    # already keyword-dense. This avoids punishing long but legitimate
    # reasoning that uses generic filler phrases.
    if keyword_density > KEYWORD_DENSITY_THRESHOLD and unique_ratio < LOW_DIVERSITY_THRESHOLD:
        score -= LOW_DIVERSITY_PENALTY * (LOW_DIVERSITY_THRESHOLD - unique_ratio + 0.5)

    if keyword_density > KEYWORD_DENSITY_THRESHOLD:
        score -= KEYWORD_DENSITY_PENALTY * (keyword_density - KEYWORD_DENSITY_THRESHOLD)

    # Exponential repetition penalty is applied only to task-reasoning tokens.
    # This keeps generic filler from being over-penalised while still
    # clamping down on repeated medical keywords.
    counts = Counter(sig_tokens)
    key_token_set = {
        tok
        for kw in task_keywords
        for tok in re.findall(r"[a-zA-Z0-9]+", kw)
        if len(tok) > 2
    }
    key_repetition_severity = 0.0
    for token, count in counts.items():
        if token in key_token_set and count >= 2:
            key_repetition_severity += math.pow(count - 1, 1.7)
    if key_repetition_severity > 0.0:
        score -= 0.0024 * key_repetition_severity

    # ── Penalty: spam clarification ───────────────────────────────────────────
    excess_irrelevant = max(0, irrelevant_count - IRRELEVANT_CLARIFY_TOLERANCE)
    if excess_irrelevant > 0:
        score -= SPAM_PENALTY_PER_EXCESS * excess_irrelevant

    return round(score, 4)


# ---------------------------------------------------------------------------
# Useful clarification counter
# ---------------------------------------------------------------------------

def count_useful_clarifications(
    clarify_records: Sequence[ActualClarifyRecord],
    task: TaskConfig,
) -> int:
    """
    Count clarify actions whose actual env trigger matched a task-expected trigger.

    FIX: Uses actual trigger from env payload, not re-inferred from question text.

    Parameters
    ----------
    clarify_records : sequence of ActualClarifyRecord
    task : TaskConfig

    Returns
    -------
    int
    """
    expected_triggers: FrozenSet[str] = frozenset(
        k.lower() for k in task.key_clarify_actions
    )
    useful = 0
    for record in clarify_records:
        trigger = getattr(record, "trigger", None)
        if trigger is not None and trigger in expected_triggers:
            useful += 1
    return useful