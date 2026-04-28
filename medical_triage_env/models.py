"""
Pydantic models for the Medical Triage RL Environment.

Key additions over v1:
  - LabResult and ImagingFinding in PatientPresentation
  - TriagePhase enum for multi-phase episodes
  - TriageAction supports 'intervene' action type (beyond clarify/classify)
  - TriageObservation carries deterioration_signal for RL curriculum awareness
  - RewardBreakdown is granular — 6 named components for interpretable training
  - EpisodeMetrics captures rollout-level statistics for training logs
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TriagePhase(str, Enum):
        """
        Three-phase episode model:
            ASSESSMENT     - agent gathers information (clarify actions allowed)
            INTERVENTION   - [PLANNED, NOT YET IMPLEMENTED] intermediate phase
                             for targeted workup; no action type, reward logic,
                             or task YAML support exists yet. The phase can be
                             returned by _observation_phase() but no step logic
                             transitions into it during active training.
            CLASSIFICATION - agent assigns ESI level and reasoning (classify action)
            COMPLETED      - terminal state
        """

        ASSESSMENT = "assessment"
        INTERVENTION = "intervention"   # planned scaffolding — not yet active
        CLASSIFICATION = "classification"
        COMPLETED = "completed"


class ActionType(str, Enum):
    CLARIFY = "clarify"
    CLASSIFY = "classify"


class DifficultyTier(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# Vital signs (unchanged structure, extended docstring)
# ---------------------------------------------------------------------------

class VitalSigns(BaseModel):
    """
    Current patient vital signs — may drift between steps.

    Fields are Optional because partial observability hides some vitals
    until the agent performs a relevant clarify action.
    """
    heart_rate: Optional[int] = None                    # bpm
    blood_pressure_systolic: Optional[int] = None       # mmHg
    blood_pressure_diastolic: Optional[int] = None      # mmHg
    respiratory_rate: Optional[int] = None              # breaths/min
    oxygen_saturation: Optional[float] = None           # %
    temperature: Optional[float] = None                 # °C
    gcs: Optional[int] = None                           # Glasgow Coma Scale 3-15

    @field_validator("gcs")
    @classmethod
    def clamp_gcs(cls, v: Optional[int]) -> Optional[int]:
        if v is not None:
            return max(3, min(15, v))
        return v

    @field_validator("oxygen_saturation")
    @classmethod
    def clamp_spo2(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            return round(max(50.0, min(100.0, v)), 1)
        return v


# ---------------------------------------------------------------------------
# Lab and imaging findings
# ---------------------------------------------------------------------------

class LabResult(BaseModel):
    """A revealed lab result."""
    name: str
    value: str
    unit: str
    reference_range: str
    critical: bool = False


class ImagingFinding(BaseModel):
    """A revealed imaging result (X-ray, CT, ECG finding, etc.)."""
    modality: str               # "ECG", "CXR", "CT head", "Bedside echo"
    finding: str                # Plain text description
    critical: bool = False


# ---------------------------------------------------------------------------
# Patient presentation
# ---------------------------------------------------------------------------

class PatientPresentation(BaseModel):
    """
    Full patient snapshot delivered to the agent each step.

    additional_info is None until the agent performs a relevant clarify action —
    this is the primary partial observability mechanism.
    Labs and imaging are only populated after relevant reveals.
    """
    patient_id: str
    age: int
    sex: str
    chief_complaint: str
    symptoms: List[str] = Field(default_factory=list)
    vitals: VitalSigns
    medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    triage_arrival_mode: str = "ambulatory"
    time_of_onset: str
    additional_info: Optional[str] = None
    revealed_labs: List[LabResult] = Field(default_factory=list)
    revealed_imaging: List[ImagingFinding] = Field(default_factory=list)
    confounders_visible: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class TriageAction(BaseModel):
    """
    Agent action at each step.

    action_type:
      "clarify"   — ask for more information (uses one step, may reveal hidden_info)
      "classify"  — assign ESI level and reasoning (terminal if successful)

    clarifying_question: free-text question asked by the agent (logged for
                         ReasoningPathGrader evaluation)
    reasoning:          clinical reasoning chain (graded for keyword coverage
                         and logical coherence)
    recommended_actions: list of interventions the agent recommends
    confidence:         0.0-1.0 agent self-reported confidence (used for
                         calibration metrics in training logs)
    """
    action_type: str
    esi_level: Optional[int] = None
    clarifying_question: Optional[str] = None
    reasoning: str = ""
    recommended_actions: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # TODO: populate when the planned 'intervene' action type is implemented.
    intervention_given: Optional[str] = None

    @field_validator("esi_level")
    @classmethod
    def validate_esi(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (1 <= v <= 5):
            raise ValueError(f"ESI level must be 1-5, got {v}")
        return v


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

class TriageObservation(BaseModel):
    """
    Complete environment observation returned to the agent after each step.

    deterioration_signal: float 0.0-1.0 indicating how much the patient has
                          deteriorated since step 0 (for RL curriculum awareness,
                          not a cheat — derived from publicly visible vitals only).
    """
    task_id: str
    step_number: int
    max_steps: int
    phase: TriagePhase = TriagePhase.ASSESSMENT
    patient: PatientPresentation
    additional_info_revealed: bool
    clarification_history: List[str] = Field(default_factory=list)
    deterioration_signal: float = Field(default=0.0, ge=0.0, le=1.0)
    time_pressure_note: Optional[str] = None    # e.g. "Patient deteriorating — 2 steps remain"


# ---------------------------------------------------------------------------
# Rewards — granular breakdown for interpretable RL training
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """
    Granular reward components.

    All components are in [-1, 1] before weighting.
    final_reward is the scalar the RL algorithm receives.

    Components:
      esi_accuracy        — correct ESI level classification
      reasoning_quality   — clinical reasoning keyword coverage + length
      action_coverage     — fraction of expected interventions mentioned
      temporal_efficiency — speed relative to ESI urgency
      safety_modifier     — penalty multiplier for dangerous undertriage
      path_quality        — bonus for following correct clinical workflow
    """
    esi_accuracy: float = 0.0
    reasoning_quality: float = 0.0
    action_coverage: float = 0.0
    temporal_efficiency: float = 0.0
    safety_modifier: float = 1.0        # multiplier: 1.0 = no penalty, 0.25 = undertriage
    path_quality: float = 0.0
    final_reward: float = 0.0
    feedback: str = ""


# Backward-compatible alias
class TriageReward(BaseModel):
    value: float
    esi_accuracy: float
    reasoning_quality: float
    action_appropriateness: float
    feedback: str


# ---------------------------------------------------------------------------
# Episode-level metrics for training telemetry
# ---------------------------------------------------------------------------

class EpisodeMetrics(BaseModel):
    """
    Metrics captured at episode end — logged to structured telemetry.

    Used by training scripts to compute:
      - mean reward per difficulty tier
      - undertriage rate (critical KPI for safety alignment)
      - clarification efficiency (did the agent ask useful questions?)
      - calibration error (confidence vs actual accuracy)
    """
    session_id: str
    task_id: str
    difficulty: str
    category: str
    esi_correct: int
    esi_predicted: Optional[int]
    steps_taken: int
    max_steps: int
    total_reward: float
    reward_breakdown: RewardBreakdown
    undertriage: bool = False           # esi_predicted > esi_correct when esi_correct <= 2
    overtriage: bool = False            # esi_predicted <= 2 when esi_correct >= 4
    clarification_count: int = 0
    useful_clarification_count: int = 0
    agent_confidence: float = 0.5
    deterioration_at_classify: float = 0.0
    additional_info_used: bool = False
    extra: Dict[str, Any] = Field(default_factory=dict)