"""
triagerl.core.models
====================
Canonical Pydantic v2 data models for the TriageRL system.

Fixes vs previous version
--------------------------
1. RewardBreakdown.safety_modifier: range changed [0.0, 1.0] from (0.0, 1.0].
   Previous strict > 0.0 caused ValidationError when apply_safety_modifier
   returned effective=0.0 on the raw=0.0 undertriage edge case, crashing
   the grader silently during training.

2. RewardBreakdown.base_score removed. It was clamped pre-modifier and
   created a misleading artefact in training logs — looked like the
   modifier was not applied. Replaced with pre_clamp_score (post-modifier,
   pre-clamp) so debugging is unambiguous.

3. EpisodeMetrics.classification_made added — required by eval_split.py
   summarize() logic and missing from previous version.

Rules
-----
* No imports from medical_triage_env.
* No business logic — data structures and validation only.
* All models frozen (immutable) to prevent mutation in episode loops.
* Validation errors surface explicitly — no silent coercions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from triagerl.core.types import ActionType, PhaseState, RewardComponent


# ---------------------------------------------------------------------------
# VitalSigns
# ---------------------------------------------------------------------------

class VitalSigns(BaseModel):
    """
    Structured vital sign readings for a single observation step.

    All fields are Optional — partial observability is a core benchmark
    feature. Additional vitals are revealed through the check_vitals
    hidden info layer after a clarify action.
    """
    model_config = ConfigDict(frozen=True)

    heart_rate:               Optional[float] = Field(default=None)
    blood_pressure_systolic:  Optional[float] = Field(default=None)
    blood_pressure_diastolic: Optional[float] = Field(default=None)
    respiratory_rate:         Optional[float] = Field(default=None)
    oxygen_saturation:        Optional[float] = Field(default=None)
    temperature:              Optional[float] = Field(default=None)
    gcs:                      Optional[int]   = Field(default=None)

    @field_validator(
        "heart_rate",
        "blood_pressure_systolic",
        "blood_pressure_diastolic",
        "respiratory_rate",
        mode="before",
    )
    @classmethod
    def _positive_or_none(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        v = float(v)
        if v <= 0:
            raise ValueError(f"Vital sign must be positive, got {v!r}.")
        return v

    @field_validator("oxygen_saturation", mode="before")
    @classmethod
    def _spo2(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        v = float(v)
        if not (0.0 < v <= 100.0):
            raise ValueError(
                f"oxygen_saturation must be in (0.0, 100.0], got {v!r}."
            )
        return v

    @field_validator("temperature", mode="before")
    @classmethod
    def _temperature(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        v = float(v)
        if not (30.0 <= v <= 45.0):
            raise ValueError(
                f"temperature must be in [30.0, 45.0] °C, got {v!r}."
            )
        return v

    @field_validator("gcs", mode="before")
    @classmethod
    def _gcs(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        v = int(v)
        if not (3 <= v <= 15):
            raise ValueError(f"gcs must be in [3, 15], got {v!r}.")
        return v


# ---------------------------------------------------------------------------
# PatientPresentation
# ---------------------------------------------------------------------------

class PatientPresentation(BaseModel):
    """Full patient presentation as delivered to the triage agent."""
    model_config = ConfigDict(frozen=True)

    patient_id:           str        = Field()
    age:                  int        = Field(ge=0, le=120)
    sex:                  str        = Field()
    chief_complaint:      str        = Field()
    scenario_description: str        = Field(default="")
    symptoms:             List[str]  = Field(default_factory=list)
    medical_history:      List[str]  = Field(default_factory=list)
    current_medications:  List[str]  = Field(default_factory=list)
    allergies:            List[str]  = Field(default_factory=list)
    triage_arrival_mode:  str        = Field(default="ambulatory")
    time_of_onset:        str        = Field(default="")
    vitals:               VitalSigns = Field()

    @field_validator("sex", mode="before")
    @classmethod
    def _normalise_sex(cls, v: Any) -> str:
        s = str(v).lower().strip()
        if s == "m":
            return "male"
        if s == "f":
            return "female"
        if s not in ("male", "female", "other"):
            raise ValueError(f"sex must be 'male'/'female'/'other', got {v!r}.")
        return s


# ---------------------------------------------------------------------------
# TriageAction
# ---------------------------------------------------------------------------

class TriageAction(BaseModel):
    """
    Structured action emitted by the triage agent.

    Discriminated union enforced by model_validator:
      classify → esi_level required; clarifying_question absent.
      clarify  → clarifying_question required; esi_level absent.
    """
    model_config = ConfigDict(frozen=True, use_enum_values=True)

    action_type:         ActionType    = Field()
    esi_level:           Optional[int] = Field(default=None, ge=1, le=5)
    clarifying_question: Optional[str] = Field(default=None, max_length=2000)
    reasoning:           str           = Field(default="", max_length=8000)
    recommended_actions: List[str]     = Field(default_factory=list)
    confidence:          float         = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _enforce_discriminant(self) -> "TriageAction":
        if self.action_type == ActionType.CLASSIFY:
            if self.esi_level is None:
                raise ValueError("action_type='classify' requires esi_level (1-5).")
            if self.clarifying_question is not None:
                raise ValueError(
                    "action_type='classify' must not include clarifying_question."
                )
        elif self.action_type == ActionType.CLARIFY:
            cq = (self.clarifying_question or "").strip()
            if not cq:
                raise ValueError(
                    "action_type='clarify' requires a non-empty clarifying_question."
                )
            if self.esi_level is not None:
                raise ValueError(
                    "action_type='clarify' must not include esi_level."
                )
        return self

    @field_validator("clarifying_question", mode="before")
    @classmethod
    def _strip_cq(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator("reasoning", mode="before")
    @classmethod
    def _strip_reasoning(cls, v: Any) -> str:
        return str(v or "").strip()


# ---------------------------------------------------------------------------
# TriageObservation
# ---------------------------------------------------------------------------

class TriageObservation(BaseModel):
    """Observation passed from environment to agent at each step."""
    model_config = ConfigDict(frozen=True, use_enum_values=True)

    task_ref:                 str                 = Field()
    patient:                  PatientPresentation = Field()
    step_number:              int                 = Field(ge=1)
    max_steps:                int                 = Field(ge=1)
    phase:                    PhaseState          = Field()
    clarification_history:    List[str]           = Field(default_factory=list)
    additional_info_revealed: bool                = Field(default=False)
    deterioration_signal:     float               = Field(default=0.0, ge=0.0, le=1.0)
    revealed_info:            Dict[str, Any]      = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# RewardBreakdown
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """
    Per-component reward breakdown for a classify action.

    FIX: safety_modifier range is now [0.0, 1.0], not (0.0, 1.0].
    When apply_safety_modifier returns effective=0.0 (raw=0.0 with
    undertriage), RewardBreakdown no longer raises ValidationError.

    FIX: pre_clamp_score replaces base_score. It stores the post-modifier,
    pre-clamp value so debugging shows the actual penalty magnitude before
    clamping to [-1, 1]. The old base_score was pre-modifier and created
    the false impression that the safety modifier was not applied.

    Ranges
    ------
    esi_accuracy        ∈ [-0.15, 1.00]
    reasoning_quality   ∈ [0.00,  1.00]
    action_coverage     ∈ [0.00,  1.00]
    temporal_efficiency ∈ [0.00,  1.00]
    path_quality        ∈ [0.00,  1.00]
    safety_modifier     ∈ [0.00,  1.00]   ← was (0.00, 1.00]
    pre_clamp_score     any float          ← replaces base_score
    final_reward        ∈ [-1.00, 1.00]
    """
    model_config = ConfigDict(frozen=True)

    esi_accuracy:        float = Field()
    reasoning_quality:   float = Field(default=0.0)
    action_coverage:     float = Field(default=0.0)
    temporal_efficiency: float = Field(default=0.0)
    path_quality:        float = Field(default=0.0)
    safety_modifier:     float = Field(default=1.0)
    pre_clamp_score:     float = Field(default=0.0)
    final_reward:        float = Field(ge=-1.0, le=1.0)

    @field_validator("safety_modifier", mode="before")
    @classmethod
    def _safety_modifier_range(cls, v: Any) -> float:
        v = float(v)
        # FIX: >= 0.0 (was > 0.0 — caused crash on raw=0.0 undertriage)
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"safety_modifier must be in [0.0, 1.0], got {v!r}."
            )
        return v

    def as_component_dict(self) -> Dict[str, float]:
        """Flat dict keyed by RewardComponent string values."""
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
# EpisodeMetrics
# ---------------------------------------------------------------------------

class EpisodeMetrics(BaseModel):
    """Full episode telemetry emitted at the end of each episode."""
    model_config = ConfigDict(frozen=True)

    session_id:                 str             = Field()
    task_id:                    str             = Field()
    difficulty:                 str             = Field()
    category:                   str             = Field()
    esi_correct:                int             = Field(ge=1, le=5)
    esi_predicted:              Optional[int]   = Field(default=None, ge=1, le=5)
    steps_taken:                int             = Field(ge=1)
    max_steps:                  int             = Field(ge=1)
    total_reward:               float           = Field(ge=-1.0, le=1.0)
    reward_breakdown:           RewardBreakdown = Field()
    undertriage:                bool            = Field(default=False)
    overtriage:                 bool            = Field(default=False)
    clarification_count:        int             = Field(default=0, ge=0)
    useful_clarification_count: int             = Field(default=0, ge=0)
    agent_confidence:           float           = Field(default=0.5, ge=0.0, le=1.0)
    additional_info_used:       bool            = Field(default=False)
    classification_made:        bool            = Field(default=False)
    extra:                      Dict[str, Any]  = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_counts(self) -> "EpisodeMetrics":
        if self.useful_clarification_count > self.clarification_count:
            raise ValueError(
                f"useful_clarification_count ({self.useful_clarification_count}) "
                f"cannot exceed clarification_count ({self.clarification_count})."
            )
        return self


# ---------------------------------------------------------------------------
# TriageReward  (legacy single-step scorer — grade() only)
# ---------------------------------------------------------------------------

class TriageReward(BaseModel):
    """Backward-compatible single-step reward returned by grade()."""
    model_config = ConfigDict(frozen=True)

    value:                  float = Field(ge=-1.0, le=1.0)
    esi_accuracy:           float = Field(default=0.0)
    reasoning_quality:      float = Field(default=0.0)
    action_appropriateness: float = Field(default=0.0)
    feedback:               str   = Field(default="")