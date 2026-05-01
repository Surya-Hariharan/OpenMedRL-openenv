"""
triagerl.env
============
Environment layer — pure orchestration, no reward logic, no I/O.

Public surface
--------------
    MedicalTriageEnv       — OpenEnv-compatible episode loop
    PhaseStateMachine      — explicit FSM for phase transitions
    InvalidTransitionError — raised on illegal phase moves
    EpisodeState           — mutable episode data container
    InfoRevealer           — progressive information disclosure
    VitalDriftEngine       — stochastic vital deterioration
    compute_deterioration_signal — urgency estimation from visible vitals
    compute_time_pressure_note   — human-readable urgency note

Reward injection
----------------
To override the default production reward logic (which delegates to
``triagerl.graders``), pass a custom ``RewardCallback`` implementation to
``MedicalTriageEnv``:

    class MyRewardCallback:
        def on_clarify(self, *, question, reveal_payload, ...): ...
        def on_classify(self, *, action, task, ...): ...
        def on_timeout(self, *, current_reward, task, ...): ...

    env = MedicalTriageEnv(task_config, reward_callback=MyRewardCallback())

Dependency graph (no cycles)
-----------------------------
    types.py / models.py / constants.py / tasks.py
          ↓
    env/episode.py
    env/phase.py
    env/drift.py
    env/revealer.py
          ↓
    env/triage_env.py  (imports all of the above + graders.py via callback)
"""
from .drift import VitalDriftEngine, compute_deterioration_signal, compute_time_pressure_note
from .episode import EpisodeState
from .phase import InvalidTransitionError, PhaseStateMachine
from .revealer import InfoRevealer
from .triage_env import (
    ClarifyRewardResult,
    ClassifyRewardResult,
    DefaultRewardCallback,
    MedicalTriageEnv,
    RewardCallback,
    StepInfo,
)

__all__ = [
    # Core env
    "MedicalTriageEnv",
    "StepInfo",
    # Phase machine
    "PhaseStateMachine",
    "InvalidTransitionError",
    # Episode state
    "EpisodeState",
    # Sub-components
    "InfoRevealer",
    "VitalDriftEngine",
    # Pure functions
    "compute_deterioration_signal",
    "compute_time_pressure_note",
    # Reward protocol
    "RewardCallback",
    "DefaultRewardCallback",
    "ClarifyRewardResult",
    "ClassifyRewardResult",
]