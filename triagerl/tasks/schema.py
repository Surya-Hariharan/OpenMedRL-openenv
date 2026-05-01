from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, field_validator

from triagerl.core.constants import VALID_TRIGGERS


class HiddenInfoItem(BaseModel):
    trigger: str
    data: Dict[str, Any]
    reveal_label: str = ""
    clinical_priority: int = 1

    model_config = ConfigDict(frozen=True)

    @field_validator("trigger")
    @classmethod
    def validate_trigger(cls, v: str) -> str:
        if v not in VALID_TRIGGERS:
            raise ValueError(
                f"HiddenInfoItem.trigger must be one of {sorted(VALID_TRIGGERS)}, got {v!r}."
            )
        return v


class VitalDrift(BaseModel):
    per_step: Dict[str, float] = Field(default_factory=dict)
    noise_sigma: Dict[str, float] = Field(default_factory=dict)
    starts_at_step: int = 1

    model_config = ConfigDict(frozen=True)


class LabResult(BaseModel):
    name: str
    value: str
    unit: str
    reference_range: str
    critical: bool = False
    hidden: bool = False

    model_config = ConfigDict(frozen=True)


class ImagingFinding(BaseModel):
    finding: str
    modality: str = "CXR"
    description: str = ""
    critical: bool = False
    hidden: bool = False

    model_config = ConfigDict(frozen=True)


class PatientInfo(BaseModel):
    patient_id: str
    age: int
    sex: str
    symptoms: List[str] = Field(default_factory=list)
    medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    time_of_onset: str
    triage_arrival_mode: str = "ambulatory"

    model_config = ConfigDict(frozen=True)


class TaskConfig(BaseModel):
    id: str
    category: str
    difficulty: str
    esi_correct: int
    chief_complaint: str
    scenario: str
    initial_vitals: Dict[str, Any]
    patient_info: PatientInfo
    initial_labs: List[LabResult] = Field(default_factory=list)
    hidden_info: List[HiddenInfoItem] = Field(default_factory=list)
    confounders: List[str] = Field(default_factory=list)
    vital_drift: VitalDrift
    expected_clarify_steps: int
    key_clarify_actions: List[str] = Field(default_factory=list)
    expected_actions: List[str] = Field(default_factory=list)
    key_reasoning_keywords: List[str] = Field(default_factory=list)
    expected_severity: str
    max_steps: int
    why_difficulty: str
    undertriage_consequence: str = ""
    clinical_pearl: str = ""

    model_config = ConfigDict(frozen=True)
