"""
triagerl.core.models
====================
Canonical Pydantic v2 data models for the TriageRL system.

Rules
-----
*  No imports from ``medical_triage_env`` — this module is the replacement.
*  No business logic.  Models validate structure and types only.
*  All models are frozen (immutable) by default to prevent accidental
   mutation in multi-step episode loops.
*  Validation errors must surface explicitly — no silent coercions that
   could corrupt reward computation.
*  All fields have explicit types and clear docstrings — this layer is
   the contract between the environment, reward, and training stacks.

Model hierarchy
---------------
    VitalSigns                      ← observation primitive
    PatientPresentation             ← wraps VitalSigns + demographics
    TriageAction                    ← agent output (clarify | classify)
    TriageObservation               ← env → agent observation
    RewardBreakdown                 ← per-component scores (grader output)
    EpisodeMetrics                  ← full episode telemetry
    TriageReward                    ← legacy single-step reward (grade() only)

Validation philosophy
---------------------
*  ESI levels are constrained to [1, 5].  Out-of-range values raise
   immediately — a reward function that receives ESI 6 is already broken.
*  Confidence must be [0.0, 1.0] — no silent clamping.
*  VitalSigns allows None for fields that may be unrecorded, because
   partial observability is a core feature of the benchmark.
*  RewardBreakdown final_reward is explicitly clamped to [-1.0, 1.0]
   by the grader before construction — the model validates but does not
   re-clamp, to surface grader bugs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from triagerl.core.types import ActionType, PhaseState, RewardComponent


# ---------------------------------------------------------------------------
# Primitive: VitalSigns
# ---------------------------------------------------------------------------

class VitalSigns(BaseModel):
    """
    Structured vital sign readings for a single observation.

    All fields are Optional because the benchmark exercises partial
    observability — some vitals may be unavailable at the time of triage.
    The environment reveals additional vitals through the ``check_vitals``
    hidden info layer.

    Validation
    ----------
    *  heart_rate, blood_pressure_*, respiratory_rate must be positive
       if provided.
    *  oxygen_saturation must be in (0.0, 100.0] if provided.
    *  temperature must be in [30.0, 45.0] (°C) if provided.
    *  gcs must be in [3, 15] if provided.
    """
    model_config = ConfigDict(frozen=True)

    heart_rate:                Optional[float] = Field(
        default=None,
        description="Heart rate in beats per minute.",
    )
    blood_pressure_systolic:   Optional[float] = Field(
        default=None,
        description="Systolic blood pressure in mmHg.",
    )
    blood_pressure_diastolic:  Optional[float] = Field(
        default=None,
        description="Diastolic blood pressure in mmHg.",
    )
    respiratory_rate:          Optional[float] = Field(
        default=None,
        description="Respiratory rate in breaths per minute.",
    )
    oxygen_saturation:         Optional[float] = Field(
        default=None,
        description="Peripheral oxygen saturation (SpO2) as a percentage.",
    )
    temperature:               Optional[float] = Field(
        default=None,
        description="Body temperature in degrees Celsius.",
    )
    gcs:                       Optional[int]   = Field(
        default=None,
        description="Glasgow Coma Scale total score (3–15).",
    )

    @field_validator("heart_rate", "blood_pressure_systolic",
                     "blood_pressure_diastolic", "respiratory_rate",
                     mode="before")
    @classmethod
    def _positive_or_none(cls, v: Any) -> Optional[float]:
        if v is None:
            return v
        v = float(v)
        if v <= 0:
            raise ValueError(f"Vital sign value must be positive, got {v!r}.")
        return v

    @field_validator("oxygen_saturation", mode="before")
    @classmethod
    def _validate_spo2(cls, v: Any) -> Optional[float]:
        if v is None:
            return v
        v = float(v)
        if not (0.0 < v <= 100.0):
            raise ValueError(
                f"oxygen_saturation must be in (0.0, 100.0], got {v!r}."
            )
        return v

    @field_validator("temperature", mode="before")
    @classmethod
    def _validate_temperature(cls, v: Any) -> Optional[float]:
        if v is None:
            return v
        v = float(v)
        if not (30.0 <= v <= 45.0):
            raise ValueError(
                f"temperature must be in [30.0, 45.0] °C, got {v!r}."
            )
        return v

    @field_validator("gcs", mode="before")
    @classmethod
    def _validate_gcs(cls, v: Any) -> Optional[int]:
        if v is None:
            return v
        v = int(v)
        if not (3 <= v <= 15):
            raise ValueError(f"gcs must be in [3, 15], got {v!r}.")
        return v


# ---------------------------------------------------------------------------
# Primitive: PatientPresentation
# ---------------------------------------------------------------------------

class PatientPresentation(BaseModel):
    """
    Full patient presentation as delivered to the triage agent.

    This is the core of the agent's observation: demographics, chief
    complaint, symptoms, history, and current vital signs.

    The ``vitals`` field may be partially populated on the initial
    observation, with additional vitals revealed after a ``check_vitals``
    clarify action.
    """
    model_config = ConfigDict(frozen=True)

    patient_id:           str             = Field(
        description="Opaque patient identifier for this episode.",
    )
    age:                  int             = Field(
        ge=0, le=120,
        description="Patient age in years.",
    )
    sex:                  str             = Field(
        description="Patient biological sex ('male' | 'female' | 'other').",
    )
    chief_complaint:      str             = Field(
        description="The presenting complaint in the patient's own words.",
    )
    scenario_description: str             = Field(
        default="",
        description="Clinical scenario narrative for context.",
    )
    symptoms:             List[str]       = Field(
        default_factory=list,
        description="Observed and reported symptoms.",
    )
    medical_history:      List[str]       = Field(
        default_factory=list,
        description="Relevant past medical history.",
    )
    current_medications:  List[str]       = Field(
        default_factory=list,
        description="Current medications and doses.",
    )
    allergies:            List[str]       = Field(
        default_factory=list,
        description="Known drug and substance allergies.",
    )
    triage_arrival_mode:  str             = Field(
        default="ambulatory",
        description="Mode of arrival: 'ambulatory' | 'ambulance' | 'wheelchair'.",
    )
    time_of_onset:        str             = Field(
        default="",
        description="Patient-reported symptom onset (free text).",
    )
    vitals:               VitalSigns      = Field(
        description="Current or most recent vital sign readings.",
    )

    @field_validator("sex", mode="before")
    @classmethod
    def _normalise_sex(cls, v: Any) -> str:
        lowered = str(v).lower().strip()
        if lowered not in ("male", "female", "other"):
            # Accept m/f shorthand gracefully.
            if lowered in ("m",):
                return "male"
            if lowered in ("f",):
                return "female"
            raise ValueError(
                f"sex must be 'male', 'female', or 'other', got {v!r}."
            )
        return lowered


# ---------------------------------------------------------------------------
# Action: TriageAction
# ---------------------------------------------------------------------------

class TriageAction(BaseModel):
    """
    Structured action output from the triage agent.

    The ``action_type`` field discriminates between the two legal actions:

    classify
        The agent assigns a definitive ESI level (1–5).  ``esi_level``
        is required.  ``clarifying_question`` must be None.

    clarify
        The agent requests additional clinical information.
        ``clarifying_question`` is required.  ``esi_level`` must be None.

    Validation enforces this discriminated union — a classify action
    without ``esi_level`` raises immediately.

    All fields except ``action_type`` are optional at the model level
    so that the cross-field validator can produce a single, clear error
    message rather than multiple cryptic field errors.
    """
    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,   # serialize enums as strings
    )

    action_type:          ActionType      = Field(
        description="'classify' to assign ESI, 'clarify' to ask a question.",
    )
    esi_level:            Optional[int]   = Field(
        default=None,
        ge=1, le=5,
        description="ESI level 1–5. Required when action_type='classify'.",
    )
    clarifying_question:  Optional[str]   = Field(
        default=None,
        description="The question to ask. Required when action_type='clarify'.",
    )
    reasoning:            str             = Field(
        default="",
        description="Clinical reasoning chain. Expected ≥ 30 words for full credit.",
    )
    recommended_actions:  List[str]       = Field(
        default_factory=list,
        description="Proposed clinical interventions (e.g. '12-lead ECG', 'IV access').",
    )
    confidence:           float           = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Agent's self-reported confidence ∈ [0.0, 1.0].",
    )

    @model_validator(mode="after")
    def _validate_action_discriminant(self) -> "TriageAction":
        """
        Enforce the classify/clarify discriminated union at the model level.

        Rules
        -----
        classify: esi_level must be set; clarifying_question must be None.
        clarify:  clarifying_question must be set; esi_level must be None.
        """
        action = self.action_type
        esi    = self.esi_level
        cq     = self.clarifying_question

        if action == ActionType.CLASSIFY:
            if esi is None:
                raise ValueError(
                    "action_type='classify' requires esi_level (1–5)."
                )
            if cq is not None:
                raise ValueError(
                    "action_type='classify' must not include clarifying_question."
                )

        elif action == ActionType.CLARIFY:
            if not cq or not cq.strip():
                raise ValueError(
                    "action_type='clarify' requires a non-empty clarifying_question."
                )
            if esi is not None:
                raise ValueError(
                    "action_type='clarify' must not include esi_level."
                )

        return self

    @field_validator("clarifying_question", mode="before")
    @classmethod
    def _strip_question(cls, v: Any) -> Optional[str]:
        if v is None:
            return v
        stripped = str(v).strip()
        return stripped if stripped else None

    @field_validator("reasoning", mode="before")
    @classmethod
    def _strip_reasoning(cls, v: Any) -> str:
        return str(v or "").strip()


# ---------------------------------------------------------------------------
# Observation: TriageObservation
# ---------------------------------------------------------------------------

class TriageObservation(BaseModel):
    """
    The structured observation passed from the environment to the agent
    at each step.

    This is what the agent sees — the environment's view of the current
    episode state.

    Fields
    ------
    task_ref : str
        Opaque task reference (not the real task_id in production — the
        actual id is redacted to prevent the agent from memorising answers).
    patient : PatientPresentation
        Current patient presentation, possibly enriched by prior clarify
        actions.
    step_number : int
        Current step count (1-indexed).  Starts at 1 after ``reset()``.
    max_steps : int
        Hard episode step limit.  The episode is terminated with a timeout
        penalty when ``step_number > max_steps``.
    phase : PhaseState
        Current episode phase.
    clarification_history : list[str]
        Ordered list of clarifying questions asked so far (agent questions
        only, not the revealed answers).
    additional_info_revealed : bool
        True when at least one ``clarify`` action has successfully revealed
        hidden information.
    deterioration_signal : float
        Abstract signal in [0.0, 1.0] encoding how rapidly the patient is
        deteriorating.  Derived from vital drift; does not expose raw values.
        Intended to motivate timely classification without leaking the answer.
    revealed_info : dict[str, Any]
        Accumulated key-value pairs revealed by prior clarify actions.
        Empty until the first successful reveal.
    """
    model_config = ConfigDict(frozen=True, use_enum_values=True)

    task_ref:                  str                    = Field(
        description="Opaque task reference (redacted in production).",
    )
    patient:                   PatientPresentation    = Field(
        description="Current patient presentation.",
    )
    step_number:               int                    = Field(
        ge=1,
        description="Current step count (1-indexed).",
    )
    max_steps:                 int                    = Field(
        ge=1,
        description="Hard step limit for this episode.",
    )
    phase:                     PhaseState             = Field(
        description="Current episode phase.",
    )
    clarification_history:     List[str]              = Field(
        default_factory=list,
        description="Ordered list of clarifying questions asked so far.",
    )
    additional_info_revealed:  bool                   = Field(
        default=False,
        description="True when at least one clarify action revealed hidden info.",
    )
    deterioration_signal:      float                  = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Abstract deterioration signal ∈ [0.0, 1.0].",
    )
    revealed_info:             Dict[str, Any]         = Field(
        default_factory=dict,
        description="Accumulated key-value pairs revealed by clarify actions.",
    )


# ---------------------------------------------------------------------------
# Reward: RewardBreakdown
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """
    Per-component reward breakdown for a single classify action.

    Produced by ``triagerl.reward.grader.compute_final_score()`` and
    consumed by training telemetry and the LLM judge.

    All component scores are in their natural ranges (pre-clamping).
    The ``final_reward`` is post-safety-modifier and clamped to [-1.0, 1.0].

    Ranges
    ------
    esi_accuracy        ∈ [-0.15, 1.00]
    reasoning_quality   ∈ [0.00,  1.00]
    action_coverage     ∈ [0.00,  1.00]
    temporal_efficiency ∈ [0.00,  1.20]   (>1.0 = speed bonus)
    path_quality        ∈ [0.00,  1.00]
    safety_modifier     ∈ (0.00,  1.00]   (1.0 = no undertriage)
    final_reward        ∈ [-1.00, 1.00]
    """
    model_config = ConfigDict(frozen=True)

    esi_accuracy:        float = Field(
        description="ESI classification accuracy score.",
    )
    reasoning_quality:   float = Field(
        description="Clinical reasoning quality (keyword coverage).",
    )
    action_coverage:     float = Field(
        description="Fraction of expected interventions covered.",
    )
    temporal_efficiency: float = Field(
        description="Urgency-weighted episode speed score.",
    )
    path_quality:        float = Field(
        default=0.0,
        description="Clinical pathway quality bonus.",
    )
    safety_modifier:     float = Field(
        default=1.0,
        description="Undertriage safety multiplier (1.0 = safe).",
    )
    final_reward:        float = Field(
        ge=-1.0, le=1.0,
        description="Clamped post-modifier reward used for RL gradient.",
    )

    @field_validator("safety_modifier", mode="before")
    @classmethod
    def _validate_safety_modifier(cls, v: Any) -> float:
        v = float(v)
        if not (0.0 < v <= 1.0):
            raise ValueError(
                f"safety_modifier must be in (0.0, 1.0], got {v!r}."
            )
        return v

    def as_component_dict(self) -> Dict[str, float]:
        """
        Return a flat dict keyed by ``RewardComponent`` string values.

        Useful for constructing ``ComponentDict`` in the grader without
        manually mapping field names.
        """
        return {
            RewardComponent.ESI_SCORE.value:       self.esi_accuracy,
            RewardComponent.REASONING_SCORE.value: self.reasoning_quality,
            RewardComponent.ACTION_SCORE.value:    self.action_coverage,
            RewardComponent.TEMPORAL_SCORE.value:  self.temporal_efficiency,
            RewardComponent.PATH_QUALITY.value:    self.path_quality,
            RewardComponent.SAFETY_MODIFIER.value: self.safety_modifier,
            RewardComponent.FINAL_SCORE.value:     self.final_reward,
        }


# ---------------------------------------------------------------------------
# Telemetry: EpisodeMetrics
# ---------------------------------------------------------------------------

class EpisodeMetrics(BaseModel):
    """
    Full episode telemetry emitted at the end of each episode.

    Used for:
    *  Training dashboard logging (W&B, TensorBoard, structlog)
    *  Offline evaluation aggregation
    *  Dataset curation for reward model training

    The ``task_id`` field is redacted to ``[REDACTED]`` in production
    builds to prevent task leakage into training logs.
    See ``grader.build_episode_metrics()`` for the redaction logic.

    Classification quality
    ----------------------
    undertriage : bool
        True when the agent assigned a higher ESI than the correct level
        for a critical patient (ESI ≤ 2).  This is the primary safety flag.
    overtriage : bool
        True when the agent assigned ESI ≤ 2 for a low-acuity patient
        (correct ESI ≥ 4).  Wastes resources but is less dangerous.

    Clarification quality
    ---------------------
    clarification_count : int
        Total number of clarify actions taken.
    useful_clarification_count : int
        Number of clarify actions that matched a task-expected trigger.
        ``useful_clarification_count / clarification_count`` gives the
        clarification precision signal.
    """
    model_config = ConfigDict(frozen=True)

    # ── Identity ──────────────────────────────────────────────────────────────
    session_id:                  str                  = Field(
        description="Unique session UUID for this episode.",
    )
    task_id:                     str                  = Field(
        description="Task identifier (redacted in production).",
    )
    difficulty:                  str                  = Field(
        description="Task difficulty tier: 'easy' | 'medium' | 'hard'.",
    )
    category:                    str                  = Field(
        description="Clinical category: cardiovascular | neurological | etc.",
    )

    # ── Ground truth ──────────────────────────────────────────────────────────
    esi_correct:                 int                  = Field(
        ge=1, le=5,
        description="Ground-truth ESI level for this task.",
    )
    esi_predicted:               Optional[int]        = Field(
        default=None,
        ge=1, le=5,
        description="Agent-assigned ESI level (None if no classify action).",
    )

    # ── Episode structure ─────────────────────────────────────────────────────
    steps_taken:                 int                  = Field(
        ge=1,
        description="Total steps taken in the episode.",
    )
    max_steps:                   int                  = Field(
        ge=1,
        description="Hard step limit for this episode.",
    )

    # ── Reward signal ─────────────────────────────────────────────────────────
    total_reward:                float                = Field(
        ge=-1.0, le=1.0,
        description="Terminal step reward (final graded score).",
    )
    reward_breakdown:            RewardBreakdown      = Field(
        description="Per-component score breakdown.",
    )

    # ── Safety flags ──────────────────────────────────────────────────────────
    undertriage:                 bool                 = Field(
        default=False,
        description="True when a critical patient was undertriaged.",
    )
    overtriage:                  bool                 = Field(
        default=False,
        description="True when a low-acuity patient was overtriaged.",
    )

    # ── Clarification quality ─────────────────────────────────────────────────
    clarification_count:         int                  = Field(
        default=0, ge=0,
        description="Total clarify actions taken.",
    )
    useful_clarification_count:  int                  = Field(
        default=0, ge=0,
        description="Clarify actions that matched a task-expected trigger.",
    )

    # ── Agent metadata ────────────────────────────────────────────────────────
    agent_confidence:            float                = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Agent-reported confidence on the final classify action.",
    )
    additional_info_used:        bool                 = Field(
        default=False,
        description="True when the agent used at least one clarify action.",
    )

    # ── Extension slot ────────────────────────────────────────────────────────
    extra:                       Dict[str, Any]       = Field(
        default_factory=dict,
        description="Arbitrary extra fields for experimental tracking.",
    )

    @model_validator(mode="after")
    def _validate_clarification_counts(self) -> "EpisodeMetrics":
        if self.useful_clarification_count > self.clarification_count:
            raise ValueError(
                f"useful_clarification_count ({self.useful_clarification_count}) "
                f"cannot exceed clarification_count ({self.clarification_count})."
            )
        return self


# ---------------------------------------------------------------------------
# Legacy: TriageReward (backward-compatible single-step reward)
# ---------------------------------------------------------------------------

class TriageReward(BaseModel):
    """
    Single-step reward returned by ``triagerl.reward.grader.grade()``.

    Exists for backward compatibility with offline debugging and dataset
    analysis scripts that call ``grade(action, task)`` directly.

    For RL training, use ``compute_final_score()`` which returns a richer
    ``ComponentDict`` incorporating episode context (clarify count,
    clinical path quality, etc.).

    Fields
    ------
    value : float
        Final scalar reward ∈ [-1.0, 1.0].
    esi_accuracy : float
        Raw ESI score component.
    reasoning_quality : float
        Reasoning keyword coverage score.
    action_appropriateness : float
        Expected action coverage score.
    feedback : str
        Human-readable grader commentary.
    """
    model_config = ConfigDict(frozen=True)

    value:                  float  = Field(ge=-1.0, le=1.0)
    esi_accuracy:           float  = Field(default=0.0)
    reasoning_quality:      float  = Field(default=0.0)
    action_appropriateness: float  = Field(default=0.0)
    feedback:               str    = Field(default="")