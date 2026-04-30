"""
triagerl.core.types
===================
Canonical enum definitions for the TriageRL system.

Rules
-----
*  No imports from any other triagerl module.
*  No business logic — purely declarative type definitions.
*  All enums use ``str`` mixin so values serialize cleanly to JSON
   and can be compared directly to string literals without .value access.
*  These enums are the single source of truth for state machine phase
   labels and valid action type strings across the entire system.

Usage
-----
    from triagerl.core.types import PhaseState, ActionType, RewardComponent

    phase = PhaseState.ASSESSMENT
    assert phase == "assessment"          # True — str mixin
    assert phase.value == "assessment"   # also True

    action = ActionType.CLASSIFY
    payload = {"action_type": action}    # serializes as "classify"
"""
from __future__ import annotations

from enum import Enum


# ---------------------------------------------------------------------------
# Phase state machine
# ---------------------------------------------------------------------------

class PhaseState(str, Enum):
    """
    Episode phase labels for the triage state machine.

    Transitions
    -----------
    ASSESSMENT → INTERVENTION | DISPOSITION
    INTERVENTION → DISPOSITION
    DISPOSITION → COMPLETED

    The environment enforces valid transitions.  An agent cannot jump from
    ASSESSMENT directly to COMPLETED — it must pass through DISPOSITION
    (i.e. emit a classify action).

    Attributes
    ----------
    ASSESSMENT
        The agent is gathering information.  Only ``clarify`` actions are
        fully rewarded in this phase.  Premature classification from this
        phase incurs a temporal penalty when the task expected clarification.

    INTERVENTION
        Optional phase (not used in all tasks).  Represents a window where
        the agent can request immediate stabilisation actions before formal
        triage disposition.  Reserved for future task types.

    DISPOSITION
        The agent has emitted a ``classify`` action and the episode is
        awaiting final grading.  No further clarification is permitted.

    COMPLETED
        Terminal phase.  Reward has been computed and logged.  The episode
        loop exits after entering this phase.
    """
    ASSESSMENT   = "assessment"
    INTERVENTION = "intervention"
    DISPOSITION  = "disposition"
    COMPLETED    = "completed"


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """
    Valid action type strings for ``TriageAction.action_type``.

    These are the only two action types the environment accepts.  Any other
    string in ``action_type`` must be rejected by the action validator.

    Attributes
    ----------
    CLARIFY
        Ask a focused clinical question to reveal hidden information.
        Triggers the ``InfoRevealer`` pipeline and returns a shaping reward.
        Only valid when the episode phase is ASSESSMENT.

    CLASSIFY
        Assign an ESI level (1–5) and finalise the triage disposition.
        Triggers the full grading pipeline (ESI, temporal, reasoning,
        action coverage, path quality, safety modifier).
        Valid in ASSESSMENT or DISPOSITION phases.
    """
    CLARIFY  = "clarify"
    CLASSIFY = "classify"


# ---------------------------------------------------------------------------
# Reward component identifiers
# ---------------------------------------------------------------------------

class RewardComponent(str, Enum):
    """
    Named identifiers for each sub-score in the reward breakdown.

    Used as keys in ``ComponentDict`` (``grader.py``) and as field names
    in training telemetry.  String mixin ensures safe JSON serialisation
    and dict key lookup without ``.value``.

    Attributes
    ----------
    ESI_SCORE
        Accuracy of the ESI level assignment.  Range: [-0.15, 1.00].
    TEMPORAL_SCORE
        Urgency-aware efficiency score.  Range: [0.00, 1.20].
    REASONING_SCORE
        Clinical reasoning quality (keyword coverage + anti-gaming).
        Range: [0.00, 1.00].
    ACTION_SCORE
        Fraction of expected interventions mentioned.  Range: [0.00, 1.00].
    PATH_QUALITY
        Clinical pathway quality (did the agent follow the right workflow).
        Range: [0.00, 1.00].
    SAFETY_MODIFIER
        Undertriage safety multiplier applied to the base score.
        Range: (0.0, 1.0].  Value of 1.0 means no undertriage occurred.
    BASE_SCORE
        Weighted sum before safety modifier.  Range: [-1.00, 1.00].
    FINAL_SCORE
        Clamped post-modifier score used for RL gradient.
        Range: [-1.00, 1.00].
    """
    ESI_SCORE       = "esi_score"
    TEMPORAL_SCORE  = "temporal_score"
    REASONING_SCORE = "reasoning_score"
    ACTION_SCORE    = "action_score"
    PATH_QUALITY    = "path_quality"
    SAFETY_MODIFIER = "safety_modifier"
    BASE_SCORE      = "base_score"
    FINAL_SCORE     = "final_score"


# ---------------------------------------------------------------------------
# Difficulty tiers (used by curriculum sampling and metrics grouping)
# ---------------------------------------------------------------------------

class DifficultyTier(str, Enum):
    """
    Task difficulty classification used for curriculum learning and reporting.

    Attributes
    ----------
    EASY
        Textbook presentations with no masking factors.
        Expected: agent classifies in 0–1 clarify steps.
    MEDIUM
        Moderate masking factors or ambiguous presentations.
        Expected: agent clarifies once before classifying.
    HARD
        Multiple simultaneous masking factors, atypical presentations,
        or high-stakes edge cases.
        Expected: agent clarifies 1–2 times with targeted questions.
    """
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ---------------------------------------------------------------------------
# Clinical category identifiers
# ---------------------------------------------------------------------------

class ClinicalCategory(str, Enum):
    """
    Task clinical category.  Matches ``category`` field in ``tasks.yaml``.

    Used for per-category metric aggregation and curriculum sampling.
    """
    CARDIOVASCULAR = "cardiovascular"
    NEUROLOGICAL   = "neurological"
    INFECTIOUS     = "infectious"
    RESPIRATORY    = "respiratory"
    ABDOMINAL      = "abdominal"