"""
triagerl.reward.grader
======================
Episode-end scoring orchestrator.

This module is the single entry point for all classification reward
computation.  It assembles the output of the component modules into
the final scalar and structured breakdown used by RL training.

Public API
----------
    compute_final_score(action, task, action_history, steps_taken)
        → (final_score, ComponentDict, feedback_str)

    build_episode_metrics(session_id, task, action_history, final_reward,
                          breakdown, extra_metrics)
        → EpisodeMetrics

    grade(action, task)
        → TriageReward   [legacy single-action scorer for debugging]

Design contract
---------------
*  No I/O, no logging.  Callers log the returned values.
*  No magic numbers — all weights/bonuses are named constants here.
*  Inputs to ``compute_final_score`` are plain types (TriageAction, dict,
   list, int).  ``TaskConfig`` is only used where a frozen Pydantic model
   is required (path_quality, episode_metrics).
*  ``build_episode_metrics`` is the only function that reads ``os.environ``
   (for task-id redaction) — documented explicitly.

Reward formula (compute_final_score)
-------------------------------------
    base = W_ESI   × esi_score
         + W_TEMPORAL × temporal_score
         + W_REASONING × reasoning_score
         + W_ACTIONS × action_score
         + W_PATH × path_score

    if esi_score == ESI_PERFECT_SCORE:
        base += PERFECT_ESI_BONUS

    adjusted, safety_factor = apply_safety_modifier(base, undertriage, multiplier)
    final = clamp(adjusted, -1.0, 1.0)

Component weights
-----------------
    W_ESI       = 0.70   (correct classification is the primary task)
    W_TEMPORAL  = 0.10   (urgency-aware speed)
    W_REASONING = 0.10   (clinical reasoning quality)
    W_ACTIONS   = 0.10   (intervention coverage)
    W_PATH      = 0.05   (pathway bonus — small, additive, not weighted into sum)

Note: W_PATH is intentionally outside the weight sum (not required to sum
to 1.0).  It is a bonus for good clinical workflow, not a primary objective.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from triagerl.core.models import (
    EpisodeMetrics,
    RewardBreakdown,
    TriageAction,
    TriageReward,
)
from triagerl.core.types import RewardComponent
from triagerl.tasks import TaskConfig

from .components import (
    action_overlap,
    keyword_matches,
    score_actions,
    score_esi,
    score_reasoning,
    score_temporal,
)
from .path_quality import count_useful_clarifications, score_clinical_path
from .safety import UNDERTRIAGE_MULTIPLIER, apply_safety_modifier, is_undertriage


# ---------------------------------------------------------------------------
# Named constants — no magic numbers below this line
# ---------------------------------------------------------------------------

# Component weights (sum of W_ESI + W_TEMPORAL + W_REASONING + W_ACTIONS = 1.0)
W_ESI:       float = 0.70
W_TEMPORAL:  float = 0.10
W_REASONING: float = 0.10
W_ACTIONS:   float = 0.10

# Path quality bonus — added to base AFTER weighted sum (not part of weight budget)
W_PATH:      float = 0.05

# Perfect-ESI bonus — added to base when esi_score == 1.0
PERFECT_ESI_BONUS: float = 0.05

# Reward clamp bounds
REWARD_MIN: float = -1.0
REWARD_MAX: float =  1.0

# ESI score value for perfect classification (imported from components for DRY)
from .components import ESI_SCORE_PERFECT  # noqa: E402

# grade() legacy constants (single-action debug utility)
_GRADE_CLARIFY_STUB_REWARD:  float = 0.02
_GRADE_STUB_ESI_ACCURACY:    float = 0.02

# Task-id redaction environment variable
_TASK_ID_REDACTED: str = "[REDACTED]"
_DEV_ENV_VALUE:    str = "development"


# ---------------------------------------------------------------------------
# Internal resolver (legacy dict compatibility shim)
# ---------------------------------------------------------------------------

def _resolve_esi(task: Dict[str, Any]) -> Optional[int]:
    """
    Extract the correct ESI from a task dict.

    Supports both ``esi_correct`` (canonical) and ``correct_esi`` (legacy
    field name used in some older serialised task dicts).
    """
    v = task.get("esi_correct", task.get("correct_esi"))
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _normalise_task_dict(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of ``task`` with ``esi_correct`` populated from
    ``correct_esi`` if the canonical key is absent.

    This preserves backward compatibility for callers that still pass
    legacy-keyed dicts without modifying the caller's copy.
    """
    if "esi_correct" not in task and "correct_esi" in task:
        task = dict(task)
        task["esi_correct"] = task["correct_esi"]
    return task


# ---------------------------------------------------------------------------
# Feedback string builder (internal)
# ---------------------------------------------------------------------------

def _build_feedback(
    action: TriageAction,
    task: Dict[str, Any],
    breakdown: RewardBreakdown,
    correct_esi: Optional[int],
) -> str:
    """
    Produce a human-readable feedback string for debugging and dataset creation.

    This string is NOT used in the reward signal — only for logging and
    reward-model training data.
    """
    parts: List[str] = []

    keywords      = task.get("key_reasoning_keywords", [])
    expected_acts = task.get("expected_actions", [])

    kw_matched    = keyword_matches(action.reasoning, keywords)
    act_matched, _ = action_overlap(action.recommended_actions, expected_acts)

    # ── ESI verdict ───────────────────────────────────────────────────────────
    if action.action_type == "classify":
        if correct_esi is None:
            parts.append("Reference ESI unavailable for scoring.")
        elif action.esi_level == correct_esi:
            parts.append(f"Correct ESI {correct_esi} classification.")
        else:
            parts.append(
                f"ESI {action.esi_level} assigned; correct is ESI {correct_esi}."
            )
        if breakdown.safety_modifier < REWARD_MAX:
            parts.append(
                f"DANGEROUS UNDERTRIAGE: ESI {action.esi_level} assigned to a "
                f"critical ESI {correct_esi} patient. "
                f"{breakdown.safety_modifier:.2f}× safety factor applied."
            )

    # ── Reasoning ─────────────────────────────────────────────────────────────
    if kw_matched:
        parts.append(
            f"Reasoning keywords matched: {len(kw_matched)}/{len(keywords)} "
            f"({', '.join(kw_matched[:6])})."
        )
    else:
        parts.append("No key clinical reasoning terms matched.")

    word_count = len(action.reasoning.split())
    from .components import REASONING_MIN_WORDS  # local import — avoids top-level cycle risk
    if word_count < REASONING_MIN_WORDS:
        parts.append(
            f"Reasoning too brief ({word_count} words; "
            f"minimum {REASONING_MIN_WORDS} for full credit)."
        )

    # ── Actions ───────────────────────────────────────────────────────────────
    parts.append(
        f"Expected interventions covered: {act_matched}/{len(expected_acts)}."
    )

    # ── Score summary ─────────────────────────────────────────────────────────
    parts.append(
        f"Score breakdown — "
        f"ESI: {breakdown.esi_accuracy:.2f}, "
        f"Reasoning: {breakdown.reasoning_quality:.2f}, "
        f"Actions: {breakdown.action_coverage:.2f}, "
        f"Temporal: {breakdown.temporal_efficiency:.2f}, "
        f"PathQuality: {breakdown.path_quality:.2f} → "
        f"Final: {breakdown.final_reward:.3f}"
    )

    # ── Clinical pearl ────────────────────────────────────────────────────────
    pearl = task.get("clinical_pearl", "")
    if pearl:
        parts.append(f"Teaching point: {pearl}")

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# compute_final_score — primary episode-end scorer
# ---------------------------------------------------------------------------

ComponentDict = Dict[str, float]
"""
Keys correspond to ``triagerl.core.types.RewardComponent`` string values.
All values are rounded to 4 decimal places except ``final_score``.
"""


def compute_final_score(
    action: TriageAction,
    task: Dict[str, Any],
    action_history: List[TriageAction],
    steps_taken: int,
) -> Tuple[float, ComponentDict, str]:
    """
    Compute the definitive episode-end score for a classify action.

    This is the score used for RL training gradients.  All sub-scores
    are derived from the same inputs as the returned feedback string,
    guaranteeing consistency.

    Parameters
    ----------
    action : TriageAction
        The terminal classify action.
    task : dict
        Task configuration dict.  Supports both ``esi_correct`` and the
        legacy ``correct_esi`` field name.
    action_history : list[TriageAction]
        Full episode action history (includes the classify action itself).
    steps_taken : int
        Total number of steps taken (1-indexed: clarify + classify count).

    Returns
    -------
    (final_score, component_dict, feedback_str)
        final_score   : float ∈ [-1.0, 1.0]
        component_dict: ComponentDict (see type alias above)
        feedback_str  : human-readable grader summary
    """
    task = _normalise_task_dict(task)
    correct_esi = _resolve_esi(task)

    # ── Per-component scores ──────────────────────────────────────────────────
    esi_score, _undertriage_from_esi = score_esi(action.esi_level, correct_esi)

    # Undertriage is re-derived explicitly here (not from score_esi) so that
    # the safety modifier logic is self-contained and not coupled to esi_score.
    undertriage_flag = (
        correct_esi is not None
        and action.esi_level is not None
        and is_undertriage(correct_esi, action.esi_level)
    )
    safety_multiplier = UNDERTRIAGE_MULTIPLIER if undertriage_flag else 1.0

    clarify_count = sum(1 for a in action_history if a.action_type == "clarify")

    temporal_score = score_temporal(
        esi_correct=correct_esi if correct_esi is not None else 3,
        steps_taken=steps_taken,
        expected_clarify_steps=task.get("expected_clarify_steps", 2),
        clarify_count=clarify_count,
    )

    # Path quality requires TaskConfig (frozen Pydantic model).
    task_config = TaskConfig.model_validate(task)
    path_score  = score_clinical_path(action_history, task_config)

    action_score   = score_actions(
        action.recommended_actions,
        task.get("expected_actions", []),
    )
    reasoning_score = score_reasoning(
        action.reasoning,
        task.get("key_reasoning_keywords", []),
    )

    # ── Weighted base ─────────────────────────────────────────────────────────
    base = (
        W_ESI       * esi_score
        + W_TEMPORAL  * temporal_score
        + W_REASONING * reasoning_score
        + W_ACTIONS   * action_score
        + W_PATH      * path_score
    )

    # Perfect-ESI bonus (additive, outside weight budget)
    if esi_score == ESI_SCORE_PERFECT:
        base += PERFECT_ESI_BONUS

    # ── Safety modifier ───────────────────────────────────────────────────────
    base_adj, safety_factor = apply_safety_modifier(
        base,
        undertriage=undertriage_flag,
        multiplier=safety_multiplier,
    )

    final = float(max(REWARD_MIN, min(REWARD_MAX, base_adj)))

    # ── Component dict ────────────────────────────────────────────────────────
    components: ComponentDict = {
        RewardComponent.ESI_SCORE.value:       round(esi_score,      4),
        RewardComponent.TEMPORAL_SCORE.value:  round(temporal_score, 4),
        RewardComponent.REASONING_SCORE.value: round(reasoning_score, 4),
        RewardComponent.ACTION_SCORE.value:    round(action_score,   4),
        RewardComponent.PATH_QUALITY.value:    round(path_score,     4),
        RewardComponent.SAFETY_MODIFIER.value: float(safety_factor),
        RewardComponent.BASE_SCORE.value:      round(
            max(REWARD_MIN, min(REWARD_MAX, base)), 4
        ),
        RewardComponent.FINAL_SCORE.value:     final,
    }

    # ── Feedback ──────────────────────────────────────────────────────────────
    breakdown = RewardBreakdown(
        esi_accuracy=components[RewardComponent.ESI_SCORE.value],
        reasoning_quality=components[RewardComponent.REASONING_SCORE.value],
        action_coverage=components[RewardComponent.ACTION_SCORE.value],
        temporal_efficiency=components[RewardComponent.TEMPORAL_SCORE.value],
        safety_modifier=components[RewardComponent.SAFETY_MODIFIER.value],
        path_quality=components[RewardComponent.PATH_QUALITY.value],
        final_reward=final,
    )
    feedback = _build_feedback(action, task, breakdown, correct_esi)

    return final, components, feedback


# ---------------------------------------------------------------------------
# build_episode_metrics — telemetry for training dashboards
# ---------------------------------------------------------------------------

def build_episode_metrics(
    session_id: str,
    task: TaskConfig,
    action_history: List[TriageAction],
    final_reward: float,
    breakdown: ComponentDict,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> EpisodeMetrics:
    """
    Construct ``EpisodeMetrics`` for end-of-episode telemetry.

    This function reads ``os.environ["ENV"]`` to determine whether to
    redact the task_id in the returned model.  This is its only side-effect.

    Parameters
    ----------
    session_id : str
        The env session UUID.
    task : TaskConfig
        Frozen task configuration.
    action_history : list[TriageAction]
        Full episode action history.
    final_reward : float
        Terminal step reward (from the last step in the episode).
    breakdown : ComponentDict
        Component scores from ``compute_final_score`` (or zeros for
        non-classify terminal episodes).
    extra_metrics : dict | None
        Arbitrary extra fields to store in ``EpisodeMetrics.extra``.

    Returns
    -------
    EpisodeMetrics
    """
    classify_actions = [a for a in action_history if a.action_type == "classify"]
    clarify_actions  = [a for a in action_history if a.action_type == "clarify"]
    final_action     = classify_actions[-1] if classify_actions else None

    esi_predicted = final_action.esi_level if final_action else None

    undertriage = (
        task.esi_correct <= 2
        and esi_predicted is not None
        and esi_predicted > task.esi_correct
    )
    overtriage = (
        task.esi_correct >= 4
        and esi_predicted is not None
        and esi_predicted <= 2
    )

    useful_clarify = count_useful_clarifications(action_history, task)

    rb = RewardBreakdown(
        esi_accuracy=breakdown.get(RewardComponent.ESI_SCORE.value,       0.0),
        reasoning_quality=breakdown.get(RewardComponent.REASONING_SCORE.value, 0.0),
        action_coverage=breakdown.get(RewardComponent.ACTION_SCORE.value,  0.0),
        temporal_efficiency=breakdown.get(RewardComponent.TEMPORAL_SCORE.value, 0.0),
        safety_modifier=breakdown.get(RewardComponent.SAFETY_MODIFIER.value, 1.0),
        path_quality=breakdown.get(RewardComponent.PATH_QUALITY.value,     0.0),
        final_reward=final_reward,
    )

    extra: Dict[str, Any] = {}
    if extra_metrics:
        extra.update(extra_metrics)

    env_name   = os.getenv("ENV", "production").lower()
    task_id_v  = task.id if env_name == _DEV_ENV_VALUE else _TASK_ID_REDACTED

    return EpisodeMetrics(
        session_id=session_id,
        task_id=task_id_v,
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
        agent_confidence=(final_action.confidence if final_action else 0.5),
        additional_info_used=any(
            a.action_type == "clarify" for a in action_history
        ),
        extra=extra,
    )


# ---------------------------------------------------------------------------
# grade() — legacy single-action scorer (backward compatibility)
# ---------------------------------------------------------------------------

def grade(action: TriageAction, task: Dict[str, Any]) -> TriageReward:
    """
    Fast synchronous grader for a single action.

    Intended for debugging and offline analysis only.  For RL training use
    ``compute_final_score`` which incorporates the full episode context
    (temporal score, clinical path quality, clarification count).

    For clarify actions, returns a stub reward — no ESI scoring is possible
    without a classification.

    Parameters
    ----------
    action : TriageAction
    task : dict

    Returns
    -------
    TriageReward (backward-compatible scalar reward model)
    """
    task = _normalise_task_dict(task)
    correct_esi = _resolve_esi(task)

    if action.action_type == "clarify":
        return TriageReward(
            value=_GRADE_CLARIFY_STUB_REWARD,
            esi_accuracy=_GRADE_STUB_ESI_ACCURACY,
            reasoning_quality=0.0,
            action_appropriateness=0.0,
            feedback="Clarification step — no ESI classification yet.",
        )

    esi_score, undertriage = score_esi(action.esi_level, correct_esi)

    reasoning_score = score_reasoning(
        action.reasoning,
        task.get("key_reasoning_keywords", []),
    )
    action_score = score_actions(
        action.recommended_actions,
        task.get("expected_actions", []),
    )
    # grade() has no action_history, so clarify_count=0 is correct — it is
    # called with a single action, not an episode trajectory.
    temporal_score = score_temporal(
        esi_correct=correct_esi if correct_esi is not None else 3,
        steps_taken=1,
        expected_clarify_steps=task.get("expected_clarify_steps", 0),
        clarify_count=0,
    )

    safety_multiplier = UNDERTRIAGE_MULTIPLIER if undertriage else 1.0

    base = (
        W_ESI       * esi_score
        + W_TEMPORAL  * temporal_score
        + W_REASONING * reasoning_score
        + W_ACTIONS   * action_score
    )
    if esi_score == ESI_SCORE_PERFECT:
        base += PERFECT_ESI_BONUS

    base_adj, safety_factor = apply_safety_modifier(
        base, undertriage=undertriage, multiplier=safety_multiplier
    )
    final = float(max(REWARD_MIN, min(REWARD_MAX, base_adj)))

    breakdown = RewardBreakdown(
        esi_accuracy=esi_score,
        reasoning_quality=reasoning_score,
        action_coverage=action_score,
        temporal_efficiency=temporal_score,
        safety_modifier=float(safety_factor),
        path_quality=0.0,   # not computed in single-action grade()
        final_reward=final,
    )
    feedback = _build_feedback(action, task, breakdown, correct_esi)

    return TriageReward(
        value=final,
        esi_accuracy=esi_score,
        reasoning_quality=reasoning_score,
        action_appropriateness=action_score,
        feedback=feedback,
    )