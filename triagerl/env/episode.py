"""
triagerl.env.episode
====================
Pure data snapshot of one triage episode.

Design contract
---------------
*  ``EpisodeState`` is a dataclass, not a Pydantic model, because it is
   internal mutable engine state — not an API schema.  Pydantic models are
   reserved for the external-facing observation and action types.
*  No business logic.  Validators enforce structural invariants (list
   types, immutability of ``session_id``/``task_id``) only.
*  No I/O, no logging, no reward logic.
*  Copyable with ``dataclasses.replace()`` for snapshotting in unit tests.

Relationship to other modules
------------------------------
    EpisodeState ←── triage_env.py (owned; mutated per step)
    EpisodeState ──► phase.py      (reads PhaseStateMachine)
    EpisodeState ──► models.py     (TriageAction, LabResult, ImagingFinding)

The ``action_history``, ``revealed_labs``, and ``revealed_imaging`` lists
are grown in-place by the env orchestrator.  They are never replaced with
new list objects so that external callers holding a reference see updates
automatically.  ``revealed_info`` is a dict that accumulates trigger payloads.
"""
from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from triagerl.core.models import TriageAction
from triagerl.tasks.schema import LabResult, ImagingFinding


# ---------------------------------------------------------------------------
# Constants for zero-reward / no-score sentinel values
# ---------------------------------------------------------------------------

_EMPTY_COMPONENTS: Dict[str, float] = {
    "esi_score":       0.0,
    "temporal_score":  0.0,
    "reasoning_score": 0.0,
    "action_score":    0.0,
    "path_quality":    0.0,
    "safety_modifier": 1.0,
    "final_score":     0.0,
}


# ---------------------------------------------------------------------------
# EpisodeState
# ---------------------------------------------------------------------------

@dataclass
class EpisodeState:
    """
    All mutable state for a single triage episode.

    Owned exclusively by ``MedicalTriageEnv``.  Never passed to external
    callers directly — the env builds a ``TriageObservation`` from it each
    step.

    Parameters (set at construction / reset time)
    ----------------------------------------------
    session_id : str
        Unique UUID for this environment instance (stable across resets).
    task_id : str
        The internal task identifier.  Never surfaced to the agent.
    public_task_ref : str
        Opaque per-episode case reference (e.g. ``"case-3f8a21b0"``).
        Rotated on every reset to prevent task-id memorisation.

    Mutable episode fields
    ----------------------
    step : int
        Number of steps taken since the last reset().  0 after reset.
    done : bool
        True once the episode has terminated.
    current_vitals : dict
        Live vital-sign snapshot.  Updated by VitalDriftEngine each step
        and by InfoRevealer when check_vitals is triggered.
    initial_visible_vitals : dict
        Snapshot taken at reset time (after hidden vitals are masked).
        Used by the deterioration signal computation.  Read-only after
        reset — never mutated again.
    revealed_info : dict
        Accumulated key/value pairs unlocked via clarify actions.
    revealed_labs : list[LabResult]
        Structured lab results that have been surfaced so far.
    revealed_imaging : list[ImagingFinding]
        Structured imaging findings that have been surfaced so far.
    additional_info_revealed : bool
        Flipped to True the first time any clarify trigger succeeds.
    action_history : list[TriageAction]
        Ordered log of all actions taken this episode.
    episode_rewards : list[float]
        Ordered list of per-step scalar rewards (for return computation).
    last_score_components : dict[str, float]
        Component breakdown from the most recent classify scoring.
        Keyed by ``RewardComponent`` string values.  Empty dict until the
        first classify action.
    last_feedback : str
        Human-readable grader feedback from the last classify or terminal
        event.  Empty string until first classify or timeout.
    """

    # ── Identity (stable across the env lifetime) ─────────────────────────
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id:    str = ""

    # ── Per-episode identity (rotated on reset) ────────────────────────────
    public_task_ref: str = field(default_factory=lambda: f"case-{uuid.uuid4().hex[:8]}")

    # ── Lifecycle ──────────────────────────────────────────────────────────
    step: int  = 0
    done: bool = False

    # ── Vitals ─────────────────────────────────────────────────────────────
    current_vitals:         Dict[str, Any] = field(default_factory=dict)
    initial_visible_vitals: Dict[str, Any] = field(default_factory=dict)

    # ── Revealed information ───────────────────────────────────────────────
    revealed_info:           Dict[str, Any]      = field(default_factory=dict)
    revealed_labs:           List[LabResult]      = field(default_factory=list)
    revealed_imaging:        List[ImagingFinding] = field(default_factory=list)
    additional_info_revealed: bool               = False

    # ── History ────────────────────────────────────────────────────────────
    action_history:  List[TriageAction] = field(default_factory=list)
    episode_rewards: List[float]        = field(default_factory=list)

    # ── Scoring state (populated by classify grading) ─────────────────────
    last_score_components: Dict[str, float] = field(
        default_factory=lambda: dict(_EMPTY_COMPONENTS)
    )
    last_feedback: str = ""

    # ------------------------------------------------------------------
    # Convenience accessors (computed properties — no mutation)
    # ------------------------------------------------------------------

    @property
    def clarify_count(self) -> int:
        """Number of clarify actions taken this episode."""
        return sum(1 for a in self.action_history if a.action_type == "clarify")

    @property
    def classify_count(self) -> int:
        """Number of classify actions taken this episode (0 or 1 in practice)."""
        return sum(1 for a in self.action_history if a.action_type == "classify")

    @property
    def classification_made(self) -> bool:
        """True if at least one classify action has been recorded."""
        return self.classify_count > 0

    @property
    def cumulative_reward(self) -> float:
        """Sum of all episode rewards so far."""
        return float(sum(self.episode_rewards))

    @property
    def terminal_reward(self) -> Optional[float]:
        """The final step reward, or None if no steps have been taken."""
        return float(self.episode_rewards[-1]) if self.episode_rewards else None

    @property
    def shaping_return(self) -> float:
        """Sum of all non-terminal (shaping) step rewards."""
        if len(self.episode_rewards) <= 1:
            return 0.0
        return float(sum(self.episode_rewards[:-1]))

    @property
    def clarification_history_window(self) -> List[str]:
        """
        Rolling last-5 clarify question strings (≤100 chars each).

        Used to populate ``TriageObservation.clarification_history`` so the
        agent has a minimal episode memory without full conversation threading.
        """
        clarify_actions = [a for a in self.action_history if a.action_type == "clarify"]
        return [
            f"Step {i + 1}: {(a.clarifying_question or 'clarify')[:100]}..."
            for i, a in enumerate(clarify_actions)
        ][-5:]

    @property
    def last_classify_action(self) -> Optional[TriageAction]:
        """The most recent classify action, or None."""
        for action in reversed(self.action_history):
            if action.action_type == "classify":
                return action
        return None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        rotate_public_ref: bool = True,
    ) -> None:
        """
        Reset all per-episode mutable fields in-place.

        ``session_id`` and ``task_id`` are intentionally preserved — they
        are stable env-level identifiers, not episode-level identifiers.

        Parameters
        ----------
        rotate_public_ref : bool
            When True (default), generate a new opaque ``public_task_ref``.
            Pass False only in unit tests that need a stable reference.
        """
        if rotate_public_ref:
            self.public_task_ref = f"case-{uuid.uuid4().hex[:8]}"

        self.step                    = 0
        self.done                    = False
        self.current_vitals          = {}
        self.initial_visible_vitals  = {}
        self.revealed_info           = {}
        self.revealed_labs           = []
        self.revealed_imaging        = []
        self.additional_info_revealed = False
        self.action_history          = []
        self.episode_rewards         = []
        self.last_score_components   = dict(_EMPTY_COMPONENTS)
        self.last_feedback           = ""

    # ------------------------------------------------------------------
    # Snapshot (for debugging / test assertions)
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a shallow-copy dict of all episode state fields.

        Intended for test assertions and debug logging.  Not a serialisation
        format — use ``TriageObservation.model_dump()`` for external APIs.
        """
        return {
            "session_id":              self.session_id,
            "task_id":                 self.task_id,
            "public_task_ref":         self.public_task_ref,
            "step":                    self.step,
            "done":                    self.done,
            "clarify_count":           self.clarify_count,
            "classify_count":          self.classify_count,
            "additional_info_revealed": self.additional_info_revealed,
            "cumulative_reward":       self.cumulative_reward,
            "terminal_reward":         self.terminal_reward,
            "shaping_return":          self.shaping_return,
            "revealed_trigger_count":  len(self.revealed_info),
            "last_feedback":           self.last_feedback,
            "last_score_components":   deepcopy(self.last_score_components),
        }