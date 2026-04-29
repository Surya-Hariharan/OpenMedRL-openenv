"""
triagerl.core.models
====================
Pydantic v2 data models for the TriageRL system.

Design rules
------------
*  No business logic.  Validators enforce structural invariants only
   (range clamping, field presence) — no clinical scoring, no reward
   computation, no environment transitions.
*  All models use ``model_config = ConfigDict(frozen=True)`` for models
   that represent immutable domain objects (TaskConfig, HiddenInfoItem).
   Episode-state models (TriageObservation, PatientPresentation) are mutable
   because the environment builds them incrementally each step.
*  Import only from the Python standard library and Pydantic.  Zero
   dependencies on any other triagerl module.
*  Field ordering follows the information flow: what the agent *sees* first
   appears first in the model.

Compatibility note
------------------
The public interface (field names, validator behaviour, model names) is
intentionally identical to the original models.py so that existing
serialised observations, task YAMLs, and client code continue to work
without modification.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from triagerl.core.types import DifficultyTier, PhaseState


# ===========================================================================
# Vital signs
# ===========================================================================

class VitalSigns(BaseModel):
    """
    Current patient vital signs snapshot.

    Fields are Optional because partial observability hides some vitals until
    the agent performs a check_vitals clarify action.  Once revealed they
    remain visible for all subsequent steps.

    Validators clamp physiologically impossible values rather than raising,
    because vital drift can push values to the boundary and clamping is
    safer than crashing an in-progress episode.
    """

    heart_rate:               Optional[int]   = Field(default=None, description="bpm")
    blood_pressure_systolic:  Optional[int]   = Field(default=None, description="mmHg")
    blood_pressure_diastolic: Optional[int]   = Field(default=None, description="mmHg")
    respiratory_rate:         Optional[int]   = Field(default=None, description="breaths/min")
    oxygen_saturation:        Optional[float] = Field(default=None, description="%")
    temperature:              Optional[float] = Field(default=None, description="°C")
    gcs:                      Optional[int]   = Field(default=None, description="Glasgow Coma Scale 3–15")

    @field_validator("gcs")
    @classmethod
    def clamp_gcs(cls, v: Optional[int]) -> Optional[int]:
        """Clamp GCS to [3, 15] — anatomically impossible values are rejected silently."""
        if v is not None:
            return max(3, min(15, v))
        return v

    @field_validator("oxygen_saturation")
    @classmethod
    def clamp_spo2(cls, v: Optional[float]) -> Optional[float]:
        """Clamp SpO2 to [50.0, 100.0] and round to 1 decimal place."""
        if v is not None:
            return round(max(50.0, min(100.0, v)), 1)
        return v


# ===========================================================================
# Lab and imaging findings
# ===========================================================================

class LabResult(BaseModel):
    """A single lab value, optionally hidden until a relevant clarify action."""

    name:            str
    value:           str
    unit:            str
    reference_range: str
    critical:        bool = False
    hidden:          bool = Field(
        default=False,
        description="True = only surfaced in observation after a relevant clarify action.",
    )

    model_config = ConfigDict(frozen=True)


class ImagingFinding(BaseModel):
    """A revealed imaging or bedside investigation finding."""

    modality: str  = Field(description="e.g. 'ECG', 'CXR', 'CT head', 'Bedside echo'")
    finding:  str  = Field(description="Plain-text description of the finding.")
    critical: bool = False

    model_config = ConfigDict(frozen=True)


# ===========================================================================
# Patient presentation (delivered to the agent each step)
# ===========================================================================

class PatientPresentation(BaseModel):
    """
    Complete patient snapshot delivered to the agent at every step.

    ``additional_info`` is ``None`` until the agent performs at least one
    successful clarify action — this is the primary partial-observability
    mechanism.  Once revealed it accumulates across steps.

    ``revealed_labs`` and ``revealed_imaging`` start empty and grow as the
    agent asks targeted questions that match relevant trigger keywords.
    """

    patient_id:           str
    age:                  int
    sex:                  str
    chief_complaint:      str
    symptoms:             List[str]       = Field(default_factory=list)
    vitals:               VitalSigns
    medical_history:      List[str]       = Field(default_factory=list)
    current_medications:  List[str]       = Field(default_factory=list)
    allergies:            List[str]       = Field(default_factory=list)
    triage_arrival_mode:  str             = "ambulatory"
    time_of_onset:        str
    additional_info:      Optional[str]   = Field(
        default=None,
        description="Hidden until first successful clarify.  Aggregated pipe-separated string.",
    )
    revealed_labs:        List[LabResult]       = Field(default_factory=list)
    revealed_imaging:     List[ImagingFinding]  = Field(default_factory=list)
    confounders_visible:  List[str]             = Field(default_factory=list)


# ===========================================================================
# Triage action (agent output)
# ===========================================================================

class TriageAction(BaseModel):
    """
    Agent action at each environment step.

    action_type
        "clarify"  — ask one natural-language question to unlock a hidden
                     information layer.  Costs one step.
        "classify" — assign an ESI level with reasoning.  Terminal action.

    clarifying_question
        Free-text question (action_type == "clarify").  Parsed by
        InfoRevealer.infer_trigger() to determine which hidden layer to
        unlock.  Direct injection of trigger token strings (e.g.
        "ask_history") is detected and blocked.

    reasoning
        Clinical reasoning chain (both action types, mandatory for
        "classify").  Graded for keyword coverage and logical coherence.
        Must be ≥ 30 words for full reasoning credit.

    recommended_actions
        List of recommended clinical interventions.  Graded by token-level
        overlap with task.expected_actions.  Required for ESI 1–2 cases.

    confidence
        Agent self-reported confidence in [0.0, 1.0].  Used for calibration
        metrics only — does not affect reward.

    intervention_given
        Reserved for the planned "intervene" action type.  Always None in
        current training.
    """

    action_type:          str
    esi_level:            Optional[int]   = None
    clarifying_question:  Optional[str]   = None
    reasoning:            str             = ""
    recommended_actions:  List[str]       = Field(default_factory=list)
    confidence:           float           = Field(default=0.5, ge=0.0, le=1.0)
    intervention_given:   Optional[str]   = None  # reserved

    @field_validator("esi_level")
    @classmethod
    def validate_esi(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (1 <= v <= 5):
            raise ValueError(f"ESI level must be 1–5, got {v}.")
        return v

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        normalised = v.strip().lower()
        if normalised not in {"clarify", "classify"}:
            raise ValueError(
                f"action_type must be 'clarify' or 'classify', got {v!r}."
            )
        return normalised


# ===========================================================================
# Triage observation (environment output — delivered to the agent)
# ===========================================================================

class TriageObservation(BaseModel):
    """
    Complete environment observation returned after each reset() or step().

    task_id
        Opaque per-episode case reference (e.g. "case-3f8a21b0").
        Intentionally NOT the internal task id — prevents task-id
        memorisation hacks where an agent learns the correct ESI for a
        specific task id rather than reasoning from clinical features.

    deterioration_signal
        Float in [0.0, 1.0] estimating how much the patient has deteriorated
        since step 0.  Derived from publicly visible vitals only — not a
        cheat code.  Intended to drive urgency-aware behaviour in RL agents.

    time_pressure_note
        Human-readable urgency reminder injected when deterioration is high
        and steps are low.  None when no special urgency applies.

    clarification_history
        Rolling last-5 clarify questions asked this episode (truncated to 100
        chars each).  Provides the agent with a minimal episode memory so
        that multi-step reasoning is possible without full conversation
        history threading.
    """

    task_id:                   str
    step_number:               int
    max_steps:                 int
    phase:                     PhaseState      = PhaseState.ASSESSMENT
    patient:                   PatientPresentation
    additional_info_revealed:  bool
    clarification_history:     List[str]       = Field(default_factory=list)
    deterioration_signal:      float           = Field(default=0.0, ge=0.0, le=1.0)
    time_pressure_note:        Optional[str]   = None


# ===========================================================================
# Reward models
# ===========================================================================

class RewardBreakdown(BaseModel):
    """
    Granular per-component reward breakdown.

    All component scores are in their natural range before weighting:
        esi_accuracy        ∈ [-0.15, 1.00]
        reasoning_quality   ∈ [0.00, 1.00]
        action_coverage     ∈ [0.00, 1.00]
        temporal_efficiency ∈ [0.00, 1.20]   (>1 = speed bonus for ESI 1)
        path_quality        ∈ [0.00, 1.00]
        safety_modifier     ∈ [0.00, 1.00]   (1.0 = no penalty, 0.25 = undertriage)
        final_reward        ∈ [-1.00, 1.00]  (clamped scalar for RL)

    field ``feedback`` is a human-readable summary string for debugging and
    reward model dataset creation.  It is not used in the reward signal.
    """

    esi_accuracy:        float = 0.0
    reasoning_quality:   float = 0.0
    action_coverage:     float = 0.0
    temporal_efficiency: float = 0.0
    safety_modifier:     float = 1.0
    path_quality:        float = 0.0
    final_reward:        float = 0.0
    feedback:            str   = ""


class TriageReward(BaseModel):
    """
    Backward-compatible scalar reward model.

    Used by legacy callers (e.g. the synchronous grade() debug utility).
    New code should consume RewardBreakdown directly.
    """

    value:                   float
    esi_accuracy:            float
    reasoning_quality:       float
    action_appropriateness:  float
    feedback:                str


# ===========================================================================
# Episode-level metrics (telemetry — logged at episode end)
# ===========================================================================

class EpisodeMetrics(BaseModel):
    """
    Metrics captured at episode end for training telemetry dashboards.

    Training scripts use this to compute:
        - mean reward per difficulty tier
        - undertriage rate (primary safety KPI)
        - clarification efficiency
        - confidence calibration error

    task_id is redacted to "[REDACTED]" in production logs (env=production).
    Set ENV=development to log real task ids during local debugging.

    extra
        Arbitrary additional metrics dict for experiment-specific logging
        without schema changes.  Keys should be snake_case strings.
    """

    session_id:                  str
    task_id:                     str
    difficulty:                  str
    category:                    str
    esi_correct:                 int
    esi_predicted:               Optional[int]
    steps_taken:                 int
    max_steps:                   int
    total_reward:                float
    reward_breakdown:            RewardBreakdown
    undertriage:                 bool  = False
    overtriage:                  bool  = False
    clarification_count:         int   = 0
    useful_clarification_count:  int   = 0
    agent_confidence:            float = 0.5
    deterioration_at_classify:   float = 0.0
    additional_info_used:        bool  = False
    extra:                       Dict[str, Any] = Field(default_factory=dict)


# ===========================================================================
# Hidden information item (task configuration — consumed by InfoRevealer)
# ===========================================================================

class HiddenInfoItem(BaseModel):
    """
    One layer of hidden information, unlocked by a specific trigger string.

    trigger
        Must be one of the values in core.constants.VALID_TRIGGERS.
        The value "clarify" is explicitly rejected — see constants.py for
        the rationale.

    data
        Arbitrary key/value payload revealed when the trigger fires.
        The InfoRevealer merges this dict into the episode's revealed_info
        and additionally extracts structured labs and imaging findings by
        scanning key names for known medical tokens.

    reveal_label
        Human-readable name for this layer (e.g. "Bilateral BP measurement").
        Used in debug logs and dataset generation; not shown to the agent.

    clinical_priority
        1 = critical to know (missing it causes undertriage),
        2 = helpful (improves reasoning quality),
        3 = nice-to-have (minor path quality bonus).
        Used by the curriculum designer; not used in reward computation.
    """

    trigger:           str
    data:              Dict[str, Any]
    reveal_label:      str = ""
    clinical_priority: int = 1

    model_config = ConfigDict(frozen=True)

    @field_validator("trigger")
    @classmethod
    def validate_trigger(cls, v: str) -> str:
        # Import here to avoid a circular dependency at module level.
        # constants.py has no triagerl imports, so this is safe.
        from triagerl.core.constants import VALID_TRIGGERS
        if v not in VALID_TRIGGERS:
            raise ValueError(
                f"HiddenInfoItem.trigger must be one of {sorted(VALID_TRIGGERS)}, "
                f"got {v!r}.  "
                f"Using 'clarify' as a trigger is prohibited — it would cause any "
                f"vague question with no recognisable keyword to unlock this layer."
            )
        return