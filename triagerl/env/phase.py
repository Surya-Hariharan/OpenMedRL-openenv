"""
triagerl.env.phase
==================
Explicit finite-state machine for the triage episode lifecycle.

Design contract
---------------
*  The only source of truth for valid transitions.  No other module may
   mutate ``PhaseState`` directly — all transitions go through
   ``PhaseStateMachine.transition()``.
*  No reward logic.  No I/O.  No logging side-effects.
*  Raises ``InvalidTransitionError`` on illegal moves instead of silently
   ignoring them — this surfaces bugs in orchestration code immediately.
*  ``PhaseStateMachine`` is intentionally not a dataclass: it owns a mutable
   ``current`` field and a transition history that the ``EpisodeState``
   snapshot copies at will.

State diagram
-------------

    ┌─────────────┐  clarify (budget remaining)  ┌─────────────┐
    │  ASSESSMENT │ ─────────────────────────────►│  ASSESSMENT │
    └─────────────┘                               └─────────────┘
           │
           │ clarify (budget exhausted)
           ▼
    ┌─────────────┐
    │ DISPOSITION │◄──── classify (from ASSESSMENT, skip disposition)
    └─────────────┘
           │
           │ classify
           ▼
    ┌─────────────┐
    │  COMPLETED  │
    └─────────────┘

    COMPLETED ──(any)──► InvalidTransitionError

Notes
-----
*  INTERVENTION is modelled in PhaseState (types.py) for forward
   compatibility but is never entered by this machine.  If a future action
   type triggers it, add ``ActionType.INTERVENE`` handling here.
*  Clarify budget exhaustion is evaluated *before* calling transition() by
   the orchestrator (triage_env.py) which compares ``clarify_count`` against
   ``task.expected_clarify_steps``.  This keeps the machine itself stateless
   with respect to task configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, List, Tuple

from triagerl.core.types import ActionType, PhaseState


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class InvalidTransitionError(RuntimeError):
    """Raised when an action attempts an illegal phase transition.

    Attributes
    ----------
    from_phase : PhaseState
        The current phase when the illegal action was received.
    action_type : ActionType
        The action that triggered the error.
    message : str
        Human-readable description of the violation.
    """

    def __init__(
        self,
        from_phase: PhaseState,
        action_type: ActionType,
        message: str = "",
    ) -> None:
        self.from_phase = from_phase
        self.action_type = action_type
        super().__init__(
            message
            or (
                f"Illegal transition: action='{action_type.value}' "
                f"from phase='{from_phase.value}'. "
                "Episode may be in a terminal state."
            )
        )


# ---------------------------------------------------------------------------
# Transition table (pure data — referenced at class level)
# ---------------------------------------------------------------------------

# Maps (current_phase, action_type) → (next_phase_if_budget_ok, next_phase_if_budget_exhausted)
# None in the tuple means "same as budget_ok".
#
# Clarify budget exhaustion: when clarify_count >= expected_clarify_steps, the
# machine advances to DISPOSITION regardless of the budget slot below.  The
# orchestrator is responsible for passing the correct ``budget_exhausted`` flag.
_TRANSITION_TABLE: dict[
    tuple[PhaseState, ActionType],
    tuple[PhaseState, PhaseState],  # (budget_ok, budget_exhausted)
] = {
    # From ASSESSMENT ──────────────────────────────────────────────────────
    (PhaseState.ASSESSMENT, ActionType.CLARIFY):   (PhaseState.ASSESSMENT, PhaseState.DISPOSITION),
    (PhaseState.ASSESSMENT, ActionType.CLASSIFY):  (PhaseState.COMPLETED,  PhaseState.COMPLETED),

    # From DISPOSITION ─────────────────────────────────────────────────────
    # Clarify is still *accepted* in DISPOSITION (penalised by reward layer)
    # but does not advance phase further — already at disposition.
    (PhaseState.DISPOSITION, ActionType.CLARIFY):  (PhaseState.DISPOSITION, PhaseState.DISPOSITION),
    (PhaseState.DISPOSITION, ActionType.CLASSIFY): (PhaseState.COMPLETED,   PhaseState.COMPLETED),

    # COMPLETED is terminal; any action raises InvalidTransitionError below.
}

# Phases that are legal starting points for any action.
_ACTIONABLE_PHASES: FrozenSet[PhaseState] = frozenset({
    PhaseState.ASSESSMENT,
    PhaseState.DISPOSITION,
})


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

@dataclass
class PhaseStateMachine:
    """
    Explicit finite-state machine for one triage episode.

    Parameters
    ----------
    initial : PhaseState
        Starting phase.  Almost always ``PhaseState.ASSESSMENT``.

    Attributes
    ----------
    current : PhaseState
        The active phase.  Read-only from the outside — mutated only by
        ``transition()``.
    history : list[tuple[PhaseState, ActionType, PhaseState]]
        Ordered log of (from_phase, action, to_phase) transitions for
        debugging and telemetry.  Never influences behaviour.
    """

    initial: PhaseState = PhaseState.ASSESSMENT
    current: PhaseState = field(init=False)
    history: List[Tuple[PhaseState, ActionType, PhaseState]] = field(
        default_factory=list, init=False
    )

    def __post_init__(self) -> None:
        self.current = self.initial

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def transition(
        self,
        action_type: ActionType,
        *,
        clarify_budget_exhausted: bool = False,
    ) -> PhaseState:
        """
        Advance the machine by one action and return the new phase.

        Parameters
        ----------
        action_type : ActionType
            The action the agent just submitted.
        clarify_budget_exhausted : bool
            ``True`` when the number of clarify actions taken so far
            (BEFORE this action is counted) equals or exceeds the task's
            ``expected_clarify_steps``.  The orchestrator must compute this.

        Returns
        -------
        PhaseState
            The phase the episode is now in.  Always equals ``self.current``
            after the call.

        Raises
        ------
        InvalidTransitionError
            If the current phase is COMPLETED, or if the (phase, action)
            pair is not in the transition table.
        """
        if self.current is PhaseState.COMPLETED:
            raise InvalidTransitionError(
                self.current,
                action_type,
                "Episode is already in COMPLETED phase. "
                "Call reset() before submitting further actions.",
            )

        key = (self.current, action_type)
        if key not in _TRANSITION_TABLE:
            raise InvalidTransitionError(
                self.current,
                action_type,
                f"No transition defined for (phase={self.current.value!r}, "
                f"action={action_type.value!r}). "
                f"Valid actions from this phase: "
                f"{[a.value for (p, a) in _TRANSITION_TABLE if p is self.current]}",
            )

        budget_ok_target, exhausted_target = _TRANSITION_TABLE[key]
        next_phase = exhausted_target if clarify_budget_exhausted else budget_ok_target

        self.history.append((self.current, action_type, next_phase))
        self.current = next_phase
        return self.current

    def reset(self) -> None:
        """Return the machine to its initial phase and clear history."""
        self.current = self.initial
        self.history.clear()

    # ------------------------------------------------------------------
    # Predicates (read-only, no side-effects)
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """True when the episode has ended — no further actions accepted."""
        return self.current is PhaseState.COMPLETED

    @property
    def accepts_clarify(self) -> bool:
        """True when the current phase can accept a clarify action."""
        return (self.current, ActionType.CLARIFY) in _TRANSITION_TABLE

    @property
    def accepts_classify(self) -> bool:
        """True when the current phase can accept a classify action."""
        return (self.current, ActionType.CLASSIFY) in _TRANSITION_TABLE

    @property
    def observation_phase(self) -> PhaseState:
        """
        The phase value to surface in ``TriageObservation``.

        INTERVENTION is never produced by this machine.  If a future
        refactor adds it, update this property to map it to DISPOSITION
        for backward-compatible agent prompting.
        """
        return self.current

    def __repr__(self) -> str:
        return (
            f"PhaseStateMachine(current={self.current.value!r}, "
            f"transitions={len(self.history)})"
        )