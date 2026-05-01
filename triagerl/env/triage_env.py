"""
triagerl.env.triage_env
=======================
Thin orchestrator for the medical triage RL environment.

Responsibility
--------------
Coordinates ``PhaseStateMachine``, ``EpisodeState``, ``InfoRevealer``,
``VitalDriftEngine``, and the grader layer into the OpenEnv interface:

    observation  =  env.reset(task_id, seed)
    obs, r, done, info  =  env.step(action)
    snapshot  =  env.state()

What this module does NOT contain
----------------------------------
*  No reward computation — all scoring delegated to ``graders.py``.
*  No FastAPI / HTTP — that belongs in the server layer (env.py).
*  No logging side-effects beyond DEBUG — INFO/WARNING emitted by graders
   and session store, not here.
*  No global state — every instance is fully self-contained.

Separation of concerns summary
-------------------------------
    triage_env.py   — orchestrates episode lifecycle
    phase.py        — enforces legal phase transitions
    episode.py      — owns all mutable episode state (data only)
    revealer.py     — processes clarify questions, returns reveal payload
    drift.py        — applies vital drift, computes deterioration signal
    graders.py      — all reward computation (not imported here directly;
                       accessed via the ``RewardCallback`` protocol)
    models.py       — Pydantic schemas for observations, actions, vitals

RewardCallback protocol
-----------------------
To keep the env layer free of any grading import, reward logic is injected
via a ``RewardCallback`` protocol.  The default implementation in this
module wraps ``graders.compute_final_score`` and the clarify shaping rules
that previously lived in the monolithic env.py.

This makes the env fully testable with a mock reward function and enables
future reward-model swaps without touching env code.

OpenEnv interface
-----------------
    reset(task_id, seed) → TriageObservation
    step(action)         → (TriageObservation, float, bool, StepInfo)
    state()              → dict   [serialisable snapshot for session store]

StepInfo keys
-------------
    raw_score              float   — pre-shaping score (classify) or shaping reward (clarify)
    step                   int     — current step number
    grader_feedback        str     — human-readable grader text
    additional_info_revealed bool  — True after first successful clarify
    score_components       dict    — per-component breakdown (classify only)
    workflow_phase         str     — current PhaseState value
    revealed_trigger_count int     — number of triggers unlocked so far
    task_ref               str     — opaque per-episode case reference
    episode_metrics        dict    — only present when done=True
    classification_made    bool    — only present when done=True
"""
from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

from triagerl.core.models import (
    PatientPresentation,
    TriageAction,
    TriageObservation,
    VitalSigns,
)
from triagerl.tasks.schema import LabResult, ImagingFinding
from triagerl.core.types import ActionType, PhaseState
from triagerl.tasks import TaskConfig

from .drift import VitalDriftEngine, compute_deterioration_signal, compute_time_pressure_note
from .episode import EpisodeState
from .phase import InvalidTransitionError, PhaseStateMachine
from .revealer import InfoRevealer


# ---------------------------------------------------------------------------
# StepInfo TypedDict-style alias (plain dict for JSON serialisability)
# ---------------------------------------------------------------------------

StepInfo = Dict[str, Any]


# ---------------------------------------------------------------------------
# RewardCallback protocol (dependency-injection seam)
# ---------------------------------------------------------------------------

class ClarifyRewardResult:
    """
    Structured return type for clarify shaping so the protocol is explicit.

    Attributes
    ----------
    reward : float
        Shaping reward for this clarify step.
    terminate_episode : bool
        True when the clarify loop should be terminated (e.g. looping too
        much).  The env will set done=True and apply a terminal penalty.
    feedback : str
        Human-readable reason (surfaced in StepInfo["grader_feedback"]).
    """
    __slots__ = ("reward", "terminate_episode", "feedback")

    def __init__(
        self,
        reward: float,
        terminate_episode: bool = False,
        feedback: str = "",
    ) -> None:
        self.reward = reward
        self.terminate_episode = terminate_episode
        self.feedback = feedback


class ClassifyRewardResult:
    """
    Structured return type for terminal classification scoring.

    Attributes
    ----------
    final_score : float
        Clamped terminal reward ∈ [-1.0, 1.0].
    components : dict[str, float]
        Per-component breakdown (esi_score, temporal_score, …).
    feedback : str
        Human-readable grader summary.
    episode_metrics : dict
        Serialisable episode-level telemetry dict for logging.
    """
    __slots__ = ("final_score", "components", "feedback", "episode_metrics")

    def __init__(
        self,
        final_score: float,
        components: Dict[str, float],
        feedback: str,
        episode_metrics: Dict[str, Any],
    ) -> None:
        self.final_score     = final_score
        self.components      = components
        self.feedback        = feedback
        self.episode_metrics = episode_metrics


class RewardCallback(Protocol):
    """
    Protocol for injectable reward logic.

    Implementations must be thread-safe and side-effect-free with respect to
    episode state — the env passes all necessary context as arguments.
    """

    def on_clarify(
        self,
        *,
        question: str,
        reveal_payload: Dict[str, Any],
        clarify_count: int,
        task: TaskConfig,
        action_history: List[TriageAction],
        clarify_penalty_dampening: float,
    ) -> ClarifyRewardResult:
        """Compute shaping reward for a clarify action."""
        ...

    def on_classify(
        self,
        *,
        action: TriageAction,
        task: TaskConfig,
        action_history: List[TriageAction],
        steps_taken: int,
        session_id: str,
        episode_rewards: List[float],
    ) -> ClassifyRewardResult:
        """Compute terminal reward for a classify action."""
        ...

    def on_timeout(
        self,
        *,
        current_reward: float,
        task: TaskConfig,
        action_history: List[TriageAction],
        session_id: str,
        episode_rewards: List[float],
    ) -> ClassifyRewardResult:
        """Compute penalty reward when max_steps is reached without classify."""
        ...


# ---------------------------------------------------------------------------
# Default reward callback (wraps existing graders.py)
# ---------------------------------------------------------------------------

class DefaultRewardCallback:
    """
    Production reward callback that delegates to ``triagerl.graders``.

    Separated from the env so that:
      * Tests can inject a mock without importing graders.
      * Grader changes never require env changes.
      * The clarify shaping constants are in one place.
    """

    # Clarify shaping constants (previously inline in env.py)
    _BASE_RELEVANT_REWARD:    float = 0.03
    _BASE_IRRELEVANT_REWARD:  float = 0.01
    _NO_REVEAL_PENALTY:       float = -0.01
    _INJECTION_PENALTY:       float = -0.02   # trigger token in question
    _NO_HIDDEN_INFO_PENALTY:  float = -0.01   # task has no hidden info
    _OVER_BUDGET_SOFT:        float = -0.02   # clarify_count > expected + 1
    _OVER_BUDGET_HARD_BASE:   float = -0.05   # multiplied by dampening
    _TIMEOUT_PENALTY:         float = -0.10

    # Direct trigger token strings (same set as InfoRevealer).
    _INJECTION_TOKENS = frozenset({"ask_history", "check_vitals", "examine_patient"})

    def on_clarify(
        self,
        *,
        question: str,
        reveal_payload: Dict[str, Any],
        clarify_count: int,
        task: TaskConfig,
        action_history: List[TriageAction],
        clarify_penalty_dampening: float,
    ) -> ClarifyRewardResult:
        """
        Shaping rewards for clarify actions.

        Reward logic (order of application):
          1. Base reward from reveal success (+0.03 relevant, +0.01 other, -0.01 none).
          2. Penalty if question contains a direct trigger token (-0.02).
          3. Penalty if task has no hidden info (-0.01).
          4. Soft penalty if clarify_count > expected_clarify_steps + 1 (-0.02).
          5. Hard penalty + optional termination if clarify_count >
             max(expected_clarify_steps + 2, 4), scaled by dampening.
        """
        reward = 0.0

        # 1. Reveal success
        if reveal_payload:
            trigger = str(reveal_payload.get("trigger", ""))
            reward = (
                self._BASE_RELEVANT_REWARD
                if trigger in task.key_clarify_actions
                else self._BASE_IRRELEVANT_REWARD
            )
        else:
            reward = self._NO_REVEAL_PENALTY

        # 2. Injection penalty
        if any(tok in question.lower() for tok in self._INJECTION_TOKENS):
            reward += self._INJECTION_PENALTY

        # 3. No-hidden-info penalty
        if not task.hidden_info:
            reward += self._NO_HIDDEN_INFO_PENALTY

        # 4. Over-budget soft penalty
        if clarify_count > task.expected_clarify_steps + 1:
            reward += self._OVER_BUDGET_SOFT

        # 5. Over-budget hard penalty + termination
        terminate = False
        feedback  = ""
        hard_threshold = max(task.expected_clarify_steps + 2, 4)
        if clarify_count > hard_threshold:
            reward += self._OVER_BUDGET_HARD_BASE * clarify_penalty_dampening
            if clarify_penalty_dampening >= 0.5:
                terminate = True
                feedback  = (
                    "Clarification loop terminated due to low-value repeated questioning."
                )

        return ClarifyRewardResult(
            reward=reward,
            terminate_episode=terminate,
            feedback=feedback,
        )

    def on_classify(
        self,
        *,
        action: TriageAction,
        task: TaskConfig,
        action_history: List[TriageAction],
        steps_taken: int,
        session_id: str,
        episode_rewards: List[float],
    ) -> ClassifyRewardResult:
        from triagerl.reward.grader import build_episode_metrics, compute_final_score  # noqa: PLC0415

        final_score, components, feedback = compute_final_score(
            action=action,
            task=task.model_dump(),
            action_history=action_history,
            steps_taken=steps_taken,
        )

        rewards = list(episode_rewards)
        episode_return_sum = float(sum(rewards)) + final_score
        terminal_reward    = final_score
        shaping_return     = float(sum(rewards))

        classify_made = True
        stalled       = False

        metrics = build_episode_metrics(
            session_id=session_id,
            task=task,
            action_history=action_history,
            final_reward=terminal_reward,
            breakdown=components,
            extra_metrics={
                "episode_return_sum":  episode_return_sum,
                "shaping_return":      shaping_return,
                "classification_made": classify_made,
                "stalled_on_critical": stalled,
            },
        )

        payload = metrics.model_dump()
        payload["classification_made"]       = classify_made
        payload["ended_without_classification"] = False
        payload["stalled_on_critical"]       = stalled

        return ClassifyRewardResult(
            final_score=final_score,
            components=components,
            feedback=feedback,
            episode_metrics=payload,
        )

    def on_timeout(
        self,
        *,
        current_reward: float,
        task: TaskConfig,
        action_history: List[TriageAction],
        session_id: str,
        episode_rewards: List[float],
    ) -> ClassifyRewardResult:
        from triagerl.reward.grader import build_episode_metrics  # noqa: PLC0415

        timeout_reward = max(-1.0, current_reward - 0.10)
        feedback       = "Episode reached max steps before disposition."

        classify_made = any(a.action_type == "classify" for a in action_history)
        stalled       = (
            not classify_made
            and task.esi_correct <= 2
        )

        # Zero-component breakdown for timeout (no classify was scored).
        components: Dict[str, float] = {
            "esi_score":       0.0,
            "temporal_score":  0.0,
            "reasoning_score": 0.0,
            "action_score":    0.0,
            "path_quality":    0.0,
            "safety_modifier": 1.0,
            "final_score":     timeout_reward,
        }

        rewards = list(episode_rewards)
        episode_return_sum = float(sum(rewards)) + timeout_reward
        shaping_return     = float(sum(rewards))

        metrics = build_episode_metrics(
            session_id=session_id,
            task=task,
            action_history=action_history,
            final_reward=timeout_reward,
            breakdown=components,
            extra_metrics={
                "episode_return_sum":  episode_return_sum,
                "shaping_return":      shaping_return,
                "classification_made": classify_made,
                "stalled_on_critical": stalled,
            },
        )

        payload = metrics.model_dump()
        payload["classification_made"]          = classify_made
        payload["ended_without_classification"] = not classify_made
        payload["stalled_on_critical"]          = stalled

        return ClassifyRewardResult(
            final_score=timeout_reward,
            components=components,
            feedback=feedback,
            episode_metrics=payload,
        )


# ---------------------------------------------------------------------------
# MedicalTriageEnv
# ---------------------------------------------------------------------------

class MedicalTriageEnv:
    """
    OpenEnv-compatible medical triage environment.

    Thin orchestrator — contains zero reward computation, zero Pydantic
    schema definitions, and zero HTTP concerns.  All domain logic is
    delegated to focused sub-modules.

    Parameters
    ----------
    task_config : TaskConfig
        Validated task configuration (pre-loaded by the caller).
    seed : int | None
        RNG seed for reproducible drift trajectories.
    clarify_penalty_dampening : float
        Multiplier [0.0, 1.0] for the hard clarify-loop penalty.  1.0 =
        full penalty; 0.0 = no penalty and no loop termination.  Used
        during curriculum warm-up to avoid killing early exploratory runs.
    reward_callback : RewardCallback | None
        Dependency-injected reward logic.  Defaults to
        ``DefaultRewardCallback`` which wraps the production graders.
    session_id : str | None
        Stable env-level UUID.  Generated if not provided.
    """

    def __init__(
        self,
        task_config: TaskConfig,
        seed: Optional[int] = None,
        clarify_penalty_dampening: float = 1.0,
        reward_callback: Optional[RewardCallback] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self._task:                    TaskConfig      = task_config
        self._seed:                    Optional[int]   = seed
        self._clarify_dampening:       float           = float(clarify_penalty_dampening)
        self._reward_cb:               RewardCallback  = reward_callback or DefaultRewardCallback()

        # Stable env-level identity — does not rotate on reset.
        self._session_id: str = session_id or str(uuid.uuid4())
        # Sub-modules and episode state are potentially expensive to
        # construct (they may validate models or create RNGs). Defer
        # actual construction to `reset()` to keep __init__ lightweight
        # so callers can instantiate many env objects cheaply.
        self._revealer: Optional[InfoRevealer] = None
        self._drifter: Optional[VitalDriftEngine] = None
        self._phase: Optional[PhaseStateMachine] = None
        self._state: Optional[EpisodeState] = None

    # ------------------------------------------------------------------
    # Internal initializer (idempotent)
    # ------------------------------------------------------------------
    def _ensure_components(self) -> None:
        """
        Lazily construct sub-modules and episode state if not already present.
        This is safe to call from `reset()`, `step()`, and other public
        entrypoints to guarantee the env is ready.
        """
        if self._revealer is None:
            self._revealer = InfoRevealer(self._task, seed=self._seed)
        if self._drifter is None:
            self._drifter = VitalDriftEngine(self._task.vital_drift, seed=self._seed)
        if self._phase is None:
            self._phase = PhaseStateMachine(initial=PhaseState.ASSESSMENT)
        if self._state is None:
            self._state = EpisodeState(
                session_id=self._session_id,
                task_id=self._task.id,
            )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def task_id(self) -> str:
        return self._task.id

    @property
    def task_config(self) -> TaskConfig:
        return self._task

    @property
    def current_step(self) -> int:
        self._ensure_components()
        assert self._state is not None
        return self._state.step

    @property
    def done(self) -> bool:
        self._ensure_components()
        assert self._state is not None
        return self._state.done

    @property
    def episode_rewards(self) -> List[float]:
        self._ensure_components()
        assert self._state is not None
        return list(self._state.episode_rewards)

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> TriageObservation:
        """
        Start a new episode.

        Parameters
        ----------
        seed : int | None
            When provided, re-seeds the drift and reveal RNGs for
            deterministic replay.  When None, RNGs advance freely so
            each episode produces a distinct vital trajectory.

        Returns
        -------
        TriageObservation
            Initial observation with step_number=0.
        """
        # Ensure sub-modules and state exist before performing the reset.
        self._ensure_components()

        effective_seed = seed if seed is not None else self._seed

        # Reset sub-modules.
        assert self._revealer is not None
        assert self._drifter is not None
        assert self._phase is not None

        self._revealer.reset(seed=seed)   # None = advance freely; int = replay
        self._drifter.reset(seed=seed)
        self._phase.reset()

        # Reset episode state.
        assert self._state is not None
        self._state.reset(rotate_public_ref=True)

        # Initialise vitals from task config (masking hidden vitals).
        raw_vitals = self._revealer.get_initial_vitals()
        clamped    = VitalSigns.model_validate(raw_vitals).model_dump()
        self._state.current_vitals         = clamped
        self._state.initial_visible_vitals = deepcopy(clamped)

        # Surface visible initial labs.
        for lab in self._task.initial_labs:
            if getattr(lab, "hidden", False):
                continue
            try:
                self._state.revealed_labs.append(
                    LabResult(
                        name=lab.name,
                        value=lab.value,
                        unit=lab.unit,
                        reference_range=lab.reference_range,
                        critical=bool(lab.critical),
                    )
                )
            except Exception:
                continue

        return self._build_observation()

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(
        self,
        action: TriageAction,
    ) -> Tuple[TriageObservation, float, bool, StepInfo]:
        """
        Advance the episode by one action.

        Parameters
        ----------
        action : TriageAction
            Validated agent action.  The action_type must be "clarify" or
            "classify"; any other value raises ValueError.

        Returns
        -------
        observation : TriageObservation
        reward : float
        done : bool
        info : StepInfo

        Raises
        ------
        RuntimeError
            If the episode is already in COMPLETED phase.
        ValueError
            If action_type is invalid.
        InvalidTransitionError
            If the phase machine rejects the transition (should never occur
            for valid action_types but guards against future bugs).
        """
        # Guarantee env components exist.
        self._ensure_components()

        assert self._state is not None
        if self._state.done:
            raise RuntimeError(
                "Episode is already finished. Call reset() to start a new episode."
            )

        # Normalise and append to history BEFORE computing budget state.
        action = TriageAction(
            action_type          = (action.action_type or "").strip().lower(),
            esi_level            = action.esi_level,
            clarifying_question  = action.clarifying_question,
            reasoning            = action.reasoning,
            recommended_actions  = list(action.recommended_actions or []),
            confidence           = action.confidence,
        )

        action_type = ActionType(action.action_type)  # validated by TriageAction validator

        self._state.step += 1
        self._state.action_history.append(action)

        reward:     float            = 0.0
        raw_reward: float            = 0.0
        components: Dict[str, float] = {}
        episode_metrics_payload: Optional[Dict[str, Any]] = None

        # ── Clarify branch ─────────────────────────────────────────────
        if action_type is ActionType.CLARIFY:
            reward, episode_metrics_payload = self._handle_clarify(action)
            raw_reward = reward

        # ── Classify branch ────────────────────────────────────────────
        elif action_type is ActionType.CLASSIFY:
            reward, components, episode_metrics_payload = self._handle_classify(action)
            raw_reward = reward

        else:
            # This branch is unreachable if TriageAction validator is intact.
            raise ValueError(
                f"Unknown action_type {action.action_type!r}. "
                "Must be 'clarify' or 'classify'."
            )

        # ── Phase transition ────────────────────────────────────────────
        clarify_budget_exhausted = (
            self._state.clarify_count >= max(1, self._task.expected_clarify_steps)
        )
        try:
            self._phase.transition(
                action_type,
                clarify_budget_exhausted=clarify_budget_exhausted,
            )
        except InvalidTransitionError:
            # Phase machine disagreement with done flag is a bug — re-raise.
            raise

        # ── Vital drift (only on non-terminal, non-final steps) ────────
        if (
            not self._state.done
            and self._state.step < self._task.max_steps
        ):
            drifted = self._drifter.apply(
                self._state.current_vitals,
                self._state.step,
            )
            self._state.current_vitals = VitalSigns.model_validate(drifted).model_dump()

        # ── Timeout handling ───────────────────────────────────────────
        if self._state.step >= self._task.max_steps and not self._state.done:
            timeout_result = self._reward_cb.on_timeout(
                current_reward=reward,
                task=self._task,
                action_history=self._state.action_history,
                session_id=self._session_id,
                episode_rewards=self._state.episode_rewards,
            )
            reward                  = timeout_result.final_score
            components              = timeout_result.components
            episode_metrics_payload = timeout_result.episode_metrics
            self._state.done        = True
            self._state.last_feedback = timeout_result.feedback

        # ── Sync phase to terminal if done ─────────────────────────────
        if self._state.done and not self._phase.is_terminal:
            # Force the machine to COMPLETED in case we short-circuited
            # via timeout or loop termination.
            try:
                self._phase.transition(
                    ActionType.CLASSIFY,
                    clarify_budget_exhausted=True,
                )
            except InvalidTransitionError:
                pass  # May already be at COMPLETED if classify was processed.
            # Ensure it is terminal regardless.
            self._phase.current = PhaseState.COMPLETED

        reward = float(reward)
        self._state.episode_rewards.append(reward)

        obs = self._build_observation()

        info: StepInfo = {
            "raw_score":               raw_reward,
            "step":                    self._state.step,
            "grader_feedback":         self._state.last_feedback,
            "additional_info_revealed": self._state.additional_info_revealed,
            "score_components":        dict(components),
            "workflow_phase":          self._phase.current.value,
            "revealed_trigger_count":  len(self._revealer.revealed_trigger_keys()),
            "task_ref":                self._state.public_task_ref,
        }

        if self._state.done:
            info["episode_metrics"]     = episode_metrics_payload or {}
            info["classification_made"] = self._state.classification_made

        return obs, reward, self._state.done, info

    # ------------------------------------------------------------------
    # state() — for session store serialisation
    # ------------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        """
        Return a serialisable snapshot of the env state.

        Used by the session store to persist / restore the env between
        HTTP requests.  Keys intentionally mirror the legacy env.py
        ``state()`` output for backward compatibility.
        """
        # Ensure components exist to build a serialisable snapshot.
        self._ensure_components()

        assert self._state is not None

        return {
            "session_id":            self._session_id,
            "task_ref":              self._state.public_task_ref,
            "step":                  self._state.step,
            "done":                  self._state.done,
            "episode_rewards":       list(self._state.episode_rewards),
            "cumulative_score":      round(self._state.cumulative_reward, 2),
            "additional_info_revealed": self._state.additional_info_revealed,
            "workflow_phase":        self._phase.current.value,
            "revealed_triggers":     self._revealer.revealed_trigger_keys(),
            "revealed_labs":         [l.model_dump() for l in self._state.revealed_labs],
            "revealed_imaging":      [i.model_dump() for i in self._state.revealed_imaging],
            "action_history_summary": [
                {
                    "step":        i + 1,
                    "action_type": a.action_type,
                    "esi_level":   a.esi_level,
                    "confidence":  a.confidence,
                }
                for i, a in enumerate(self._state.action_history)
            ],
            "max_steps": self._task.max_steps,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _handle_clarify(
        self,
        action: TriageAction,
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        Process a clarify action: run InfoRevealer, apply payload, compute shaping.

        Returns (reward, episode_metrics_payload).
        ``episode_metrics_payload`` is only non-None when ``terminate_episode``
        is True (loop termination).
        """
        question       = action.clarifying_question or "clarify"
        reveal_payload = self._revealer.process_clarify(question, self._state.step)

        if reveal_payload:
            self._state.additional_info_revealed = True
            self._apply_reveal_payload(reveal_payload)

        result = self._reward_cb.on_clarify(
            question=question,
            reveal_payload=reveal_payload,
            clarify_count=self._state.clarify_count,
            task=self._task,
            action_history=self._state.action_history,
            clarify_penalty_dampening=self._clarify_dampening,
        )

        self._state.last_feedback = result.feedback

        episode_metrics_payload: Optional[Dict[str, Any]] = None
        if result.terminate_episode:
            self._state.done = True
            # Build degenerate episode metrics for loop-terminated episodes.
            from triagerl.reward.grader import build_episode_metrics  # noqa: PLC0415

            metrics = build_episode_metrics(
                session_id=self._session_id,
                task=self._task,
                action_history=self._state.action_history,
                final_reward=result.reward,
                breakdown=self._state.last_score_components,
                extra_metrics={
                    "episode_return_sum":  self._state.cumulative_reward + result.reward,
                    "shaping_return":      self._state.cumulative_reward,
                    "classification_made": False,
                    "stalled_on_critical": self._task.esi_correct <= 2,
                    "termination_reason":  "clarify_loop",
                },
            )
            episode_metrics_payload = metrics.model_dump()
            episode_metrics_payload["classification_made"]          = False
            episode_metrics_payload["ended_without_classification"] = True
            episode_metrics_payload["stalled_on_critical"]          = self._task.esi_correct <= 2

        return result.reward, episode_metrics_payload

    def _handle_classify(
        self,
        action: TriageAction,
    ) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        """
        Process a classify action: compute terminal reward via RewardCallback.

        Returns (final_score, components, episode_metrics_payload).
        """
        result = self._reward_cb.on_classify(
            action=action,
            task=self._task,
            action_history=self._state.action_history,
            steps_taken=self._state.step,
            session_id=self._session_id,
            episode_rewards=self._state.episode_rewards,
        )

        self._state.done                 = True
        self._state.last_score_components = dict(result.components)
        self._state.last_feedback        = result.feedback

        return result.final_score, result.components, result.episode_metrics

    def _apply_reveal_payload(self, payload: Dict[str, Any]) -> None:
        """
        Merge a RevealPayload from InfoRevealer into EpisodeState.

        Handles three sub-payloads:
          1. ``revealed``      → merged into ``state.revealed_info``.
          2. ``labs``          → appended to ``state.revealed_labs`` (deduped).
          3. ``imaging``       → appended to ``state.revealed_imaging`` (deduped).
          4. ``vitals_update`` → structured BP update (from InfoRevealer parser).
          5. Fallback          → ``revealed.vitals`` dict if present.
        """
        revealed = payload.get("revealed", {})
        if isinstance(revealed, dict):
            self._state.revealed_info.update(revealed)

        # Labs
        for lab_dict in payload.get("labs", []):
            try:
                lab = LabResult.model_validate(lab_dict)
            except Exception:
                continue
            duplicate = any(
                e.name == lab.name and e.value == lab.value
                for e in self._state.revealed_labs
            )
            if not duplicate:
                self._state.revealed_labs.append(lab)

        # Imaging
        for img_dict in payload.get("imaging", []):
            try:
                img = ImagingFinding.model_validate(img_dict)
            except Exception:
                continue
            duplicate = any(
                e.finding == img.finding
                for e in self._state.revealed_imaging
            )
            if not duplicate:
                self._state.revealed_imaging.append(img)

        # Structured vitals update (preferred path — set by InfoRevealer._extract_bp_update)
        vitals_update = payload.get("vitals_update")
        if isinstance(vitals_update, dict) and vitals_update:
            self._state.current_vitals.update(vitals_update)
            self._state.current_vitals = VitalSigns.model_validate(
                self._state.current_vitals
            ).model_dump()
            return  # Don't also apply raw vitals dict.

        # Fallback: raw vitals dict inside revealed (legacy path)
        raw_vitals = None
        if isinstance(revealed, dict):
            raw_vitals = revealed.get("vitals")
        if isinstance(raw_vitals, dict):
            self._state.current_vitals.update(raw_vitals)
            self._state.current_vitals = VitalSigns.model_validate(
                self._state.current_vitals
            ).model_dump()

    def _build_observation(self) -> TriageObservation:
        """
        Construct the current ``TriageObservation`` from EpisodeState.

        This is the only place where EpisodeState fields are translated into
        the external-facing Pydantic schema.  No business logic here — all
        decisions about what to show/hide were made in reset()/step().
        """
        state = self._state

        # ── Additional info text ───────────────────────────────────────
        # Avoid iterating over an unbounded revealed_info dict; cap to
        # a small number of items for formatting so observation
        # construction stays O(1)-ish even if reveal payloads are large.
        if not state.additional_info_revealed:
            additional_info_str: Optional[str] = None
        elif state.revealed_info:
            MAX_REVEALED_ITEMS = 10
            items = list(state.revealed_info.items())
            displayed = items[:MAX_REVEALED_ITEMS]
            parts = [f"{k.replace('_', ' ')}: {v}" for k, v in displayed]
            raw = " | ".join(parts)
            if len(items) > MAX_REVEALED_ITEMS:
                raw += f" | ... (+{len(items)-MAX_REVEALED_ITEMS} more)"
            additional_info_str = raw[:297] + "..." if len(raw) > 300 else raw
        else:
            additional_info_str = "Additional history was elicited on clarification."

        # ── Patient presentation ───────────────────────────────────────
        # Avoid an unnecessary deep copy of the task patient_info dump; the
        # dict returned by model_dump is fresh and we'll only mutate a few
        # shallow fields below.
        patient_dict = self._task.patient_info.model_dump()
        patient_dict["chief_complaint"]     = self._task.chief_complaint
        patient_dict["vitals"]              = state.current_vitals
        patient_dict["additional_info"]     = additional_info_str
        # Cap revealed labs/imaging list lengths to avoid expensive dumps
        # on unexpectedly large payloads while preserving first-order info.
        MAX_REVEALED_LIST = 10
        patient_dict["revealed_labs"] = [l.model_dump() for l in state.revealed_labs[:MAX_REVEALED_LIST]]
        if len(state.revealed_labs) > MAX_REVEALED_LIST:
            patient_dict["revealed_labs_count"] = len(state.revealed_labs)

        patient_dict["revealed_imaging"] = [i.model_dump() for i in state.revealed_imaging[:MAX_REVEALED_LIST]]
        if len(state.revealed_imaging) > MAX_REVEALED_LIST:
            patient_dict["revealed_imaging_count"] = len(state.revealed_imaging)
        patient_dict["confounders_visible"] = list(self._task.confounders)

        patient = PatientPresentation.model_validate(patient_dict)

        # ── Deterioration signal ───────────────────────────────────────
        det_signal = compute_deterioration_signal(
            current_vitals=state.current_vitals,
            baseline_vitals=state.initial_visible_vitals,
        )

        steps_remaining = max(0, self._task.max_steps - state.step)
        time_note = compute_time_pressure_note(
            deterioration_signal=det_signal,
            steps_remaining=steps_remaining,
            is_done=state.done,
        )

        return TriageObservation(
            task_ref=state.public_task_ref,
            step_number=max(1, min(state.step, self._task.max_steps)),
            max_steps=self._task.max_steps,
            phase=self._phase.observation_phase,
            patient=patient,
            additional_info_revealed=state.additional_info_revealed,
            clarification_history=state.clarification_history_window,
            deterioration_signal=det_signal,
        )