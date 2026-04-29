"""
triagerl.core.types
===================
Enumerations and type aliases for the TriageRL system.

Design rules
------------
*  No business logic.  Enums carry identity and ordering only.
*  No imports from any other triagerl module — this file sits at the bottom
   of the dependency graph.
*  All string-valued enums use lowercase values so they serialise cleanly
   to JSON without a custom encoder.
*  Every enum documents what each member means in clinical and RL terms so
   that task authors and researchers can use this file as a reference.
"""
from __future__ import annotations

from enum import Enum, IntEnum


# ---------------------------------------------------------------------------
# Phase state machine values
# ---------------------------------------------------------------------------

class PhaseState(str, Enum):
    """
    The episode phase observed by the agent at each step.

    Transitions (enforced by env/phase.py — see PhaseStateMachine):

        ASSESSMENT  ──clarify──►  ASSESSMENT   (while under clarify budget)
        ASSESSMENT  ──clarify──►  DISPOSITION  (once clarify budget is met)
        ASSESSMENT  ──classify─►  COMPLETED    (immediate classification)
        DISPOSITION ──classify─►  COMPLETED    (normal terminal transition)
        COMPLETED   ── (any) ───►  (raises RuntimeError — terminal)

    The INTERVENTION phase is reserved for a planned intermediate workup
    phase (e.g. ordering targeted investigations) that has no action type,
    reward logic, or task YAML support yet.  It is modelled here so the
    type system is complete, but PhaseStateMachine will never transition
    into it during active training.

    Serialisation note
    ------------------
    Values are lowercase strings so that model_dump() JSON is human-readable
    without a custom encoder.  The existing TriagePhase enum in the original
    codebase used identical values — this enum is a drop-in replacement.
    """

    ASSESSMENT     = "assessment"
    """
    Initial information-gathering phase.
    The agent may issue clarify actions to unlock hidden information layers.
    Vital drift is active from the configured starts_at_step onward.
    """

    INTERVENTION   = "intervention"
    """
    Planned intermediate phase for targeted workup (NOT YET ACTIVE).
    Reserved: PhaseStateMachine will never produce this value in training.
    Exists so the type system is complete and forward-compatible.
    """

    DISPOSITION    = "disposition"
    """
    Decision-ready phase.
    The agent is expected to issue a classify action.
    Clarify actions are still accepted but incur a redundancy penalty.
    """

    COMPLETED      = "completed"
    """
    Terminal phase.  Episode is over.
    Any further action raises RuntimeError.
    Observations in this phase have phase == COMPLETED and done == True.
    """


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """
    The two valid action types an agent may emit.

    CLARIFY  — ask a natural-language question to unlock a hidden information
               layer.  Costs one step.  May reveal history, vitals, or exam
               findings depending on the keywords in the question.

    CLASSIFY — assign an ESI level (1–5) with clinical reasoning and a list
               of recommended interventions.  Always terminal unless the
               environment rejects the action for schema reasons.
    """

    CLARIFY  = "clarify"
    CLASSIFY = "classify"


# ---------------------------------------------------------------------------
# ESI level
# ---------------------------------------------------------------------------

class ESILevel(IntEnum):
    """
    Emergency Severity Index levels.

    Using IntEnum so that numeric comparisons (esi > 2, abs(pred - correct))
    work without conversion.  Values match the clinical ESI scale exactly.

    Level definitions (for reference — authoritative source is the task YAML):
        1 = Immediate life-saving intervention required
        2 = High-risk situation / severe distress / altered mental status
        3 = Urgent but stable; requires multiple resources
        4 = Less urgent; requires one resource
        5 = Non-urgent; no resources required
    """

    IMMEDIATE   = 1
    EMERGENT    = 2
    URGENT      = 3
    LESS_URGENT = 4
    NON_URGENT  = 5

    @classmethod
    def is_critical(cls, level: int) -> bool:
        """Return True if the level is clinically critical (ESI 1 or 2)."""
        return level <= 2

    @classmethod
    def is_valid(cls, level: int) -> bool:
        """Return True if level is in the valid ESI range [1, 5]."""
        return 1 <= level <= 5


# ---------------------------------------------------------------------------
# Difficulty tier
# ---------------------------------------------------------------------------

class DifficultyTier(str, Enum):
    """
    Task difficulty classification.

    Used by the curriculum sampler (tasks/loader.py) and by episode metrics
    for per-tier reward analysis.

    EASY   — classic presentations with minimal or no confounders.
             Agent should classify correctly with zero or one clarify action.

    MEDIUM — atypical features or masking comorbidities.
             Requires at least one targeted clarify action; confounder
             awareness is necessary for correct ESI assignment.

    HARD   — multiple simultaneous masking factors; high undertriage risk.
             Requires correct ordering of clarify actions and active
             resistance to confounders.  Safety penalties are highest here.
    """

    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ---------------------------------------------------------------------------
# Trigger sentinel
# ---------------------------------------------------------------------------

class TriggerSentinel(str, Enum):
    """
    Special return value from InfoRevealer.infer_trigger() when no keyword
    in the clarifying question matches any entry in KEYWORD_TO_TRIGGER.

    CLARIFY is the only sentinel value.  It is intentionally *not* a member
    of VALID_TRIGGERS so that a HiddenInfoItem with trigger=CLARIFY.value
    would fail schema validation — preventing task authors from accidentally
    creating a layer that any vague question unlocks.
    """

    CLARIFY = "clarify"


# ---------------------------------------------------------------------------
# Category
# ---------------------------------------------------------------------------

class ClinicalCategory(str, Enum):
    """
    The five clinical categories used in the task corpus.
    Used by the curriculum sampler and episode metrics for per-category
    reward analysis.
    """

    CARDIOVASCULAR = "cardiovascular"
    NEUROLOGICAL   = "neurological"
    INFECTIOUS     = "infectious"
    RESPIRATORY    = "respiratory"
    ABDOMINAL      = "abdominal"


# ---------------------------------------------------------------------------
# Reward component names
# ---------------------------------------------------------------------------

class RewardComponent(str, Enum):
    """
    Named reward component keys.

    Used as dict keys in the component breakdown returned by grader.py so
    that consumers can reference components by name rather than magic strings.

        components[RewardComponent.ESI_SCORE]    → float
        components[RewardComponent.FINAL_SCORE]  → float
    """

    ESI_SCORE      = "esi_score"
    TEMPORAL_SCORE = "temporal_score"
    REASONING_SCORE = "reasoning_score"
    ACTION_SCORE   = "action_score"
    PATH_QUALITY   = "path_quality"
    SAFETY_MODIFIER = "safety_modifier"
    BASE_SCORE     = "base_score"
    FINAL_SCORE    = "final_score"