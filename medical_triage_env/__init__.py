"""
triagerl.core
=============
Pure data layer — no I/O, no environment logic, no reward logic.

Import from here rather than from sub-modules to insulate callers from
internal renames.

    from triagerl.core import (
        KEYWORD_TO_TRIGGER, VALID_TRIGGERS,          # constants
        PhaseState, ActionType, DifficultyTier,       # types
        TriageAction, TriageObservation, VitalSigns,  # models
    )
"""
from triagerl.core.constants import KEYWORD_TO_TRIGGER, VALID_TRIGGERS
from triagerl.core.types import ActionType, DifficultyTier, ESILevel, PhaseState
from triagerl.core.models import (
    EpisodeMetrics,
    ImagingFinding,
    LabResult,
    PatientPresentation,
    RewardBreakdown,
    TriageAction,
    TriageObservation,
    TriageReward,
    VitalSigns,
)

__all__ = [
    # constants
    "KEYWORD_TO_TRIGGER",
    "VALID_TRIGGERS",
    # types
    "ActionType",
    "DifficultyTier",
    "ESILevel",
    "PhaseState",
    # models
    "EpisodeMetrics",
    "ImagingFinding",
    "LabResult",
    "PatientPresentation",
    "RewardBreakdown",
    "TriageAction",
    "TriageObservation",
    "TriageReward",
    "VitalSigns",
]