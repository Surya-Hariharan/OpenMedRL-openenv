"""
triagerl.reward.shaping
=======================
Clarify-step reward shaping for RL training.

Responsibility
--------------
Compute per-step shaping rewards for ``clarify`` actions.  Shaping rewards
are small, non-terminal signals designed to:

  1. Reward genuinely informative clarifications (+).
  2. Penalise reward-hacking attempts (direct trigger injection) (−).
  3. Penalise wasteful clarification on tasks with no hidden info (−).
  4. Penalise excessive clarification loops (−, graduated).
  5. Signal loop-termination when the agent is stuck (structural flag).

Design contract
---------------
*  Pure function: ``compute_clarify_shaping`` takes only plain data and
   returns a ``ClarifyShapingResult`` dataclass.  No I/O, no side-effects.
*  The ``terminate_episode`` flag on ``ClarifyShapingResult`` is advisory —
   the env orchestrator decides whether to honour it based on the dampening
   setting.  The shaping function does not know about ``done`` state.
*  All thresholds are named module-level constants.  No magic numbers.

Shaping rules (applied in order, all additive)
----------------------------------------------
BASE REWARD (from reveal success):
    +SHAPING_RELEVANT     if reveal trigger is in task.key_clarify_actions
    +SHAPING_IRRELEVANT   if reveal occurred but trigger is not key
    +SHAPING_NO_REVEAL    if no information was revealed (empty payload)

PENALTIES (always applied after base, cumulative):
    +PENALTY_INJECTION    if question contains a direct trigger token
    +PENALTY_NO_HIDDEN    if task has no hidden_info at all
    +PENALTY_SOFT_LOOP    if clarify_count > expected_clarify_steps + 1
    +PENALTY_HARD_LOOP    if clarify_count > max(expected_clarify_steps + 2, 4)
                          (scaled by clarify_penalty_dampening)
    → terminate_episode = True  if hard loop AND dampening ≥ 0.5

Component breakdown
-------------------
``ClarifyShapingResult.components`` exposes each sub-reward for training
dashboards and gradient debugging:
    "base_reward"    — reveal success/failure base
    "injection"      — direct-trigger injection penalty
    "no_hidden"      — no-hidden-info efficiency penalty
    "soft_loop"      — over-budget soft penalty
    "hard_loop"      — over-budget hard penalty (pre-dampening)
    "total"          — sum of all components (= result.reward)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional

from triagerl.core.models import TriageAction
from triagerl.tasks import TaskConfig


# ---------------------------------------------------------------------------
# Named constants — single source of truth
# ---------------------------------------------------------------------------

# Base rewards from reveal quality
SHAPING_RELEVANT:    float = 0.03   # trigger ∈ task.key_clarify_actions
SHAPING_IRRELEVANT:  float = 0.01   # reveal occurred, trigger not key
SHAPING_NO_REVEAL:   float = -0.01  # clarify produced nothing

# Penalties (all non-positive)
PENALTY_INJECTION:   float = -0.02  # direct trigger token in question text
PENALTY_NO_HIDDEN:   float = -0.01  # task has no hidden_info
PENALTY_SOFT_LOOP:   float = -0.02  # clarify_count > expected + 1
PENALTY_HARD_LOOP:   float = -0.05  # clarify_count > hard threshold (pre-dampening)

# Hard loop threshold relative to expected clarify steps
HARD_LOOP_EXCESS:    int   = 2      # expected + HARD_LOOP_EXCESS = hard threshold
HARD_LOOP_FLOOR:     int   = 4      # minimum hard threshold regardless of expected steps

# Dampening threshold: terminate episode only when dampening is this high.
# Below this, the loop penalty is applied but episode continues.
TERMINATE_DAMPENING_THRESHOLD: float = 0.5

# Direct trigger token strings (same frozenset as constants.VALID_TRIGGERS
# — imported directly to guarantee consistency without circular imports).
_DIRECT_TRIGGER_TOKENS: FrozenSet[str] = frozenset({
    "ask_history",
    "check_vitals",
    "examine_patient",
})


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClarifyShapingResult:
    """
    Structured result from ``compute_clarify_shaping``.

    Attributes
    ----------
    reward : float
        Net shaping reward for this clarify step.  Summed from all
        components; may be positive or negative.
    terminate_episode : bool
        True when the clarify-loop hard threshold was breached and
        ``clarify_penalty_dampening ≥ TERMINATE_DAMPENING_THRESHOLD``.
        The env orchestrator decides whether to act on this flag.
    feedback : str
        Human-readable explanation of the termination reason (non-empty
        only when ``terminate_episode`` is True).
    components : dict[str, float]
        Per-sub-reward breakdown for telemetry.  Keys:
        "base_reward", "injection", "no_hidden", "soft_loop",
        "hard_loop", "total".
    """
    reward:             float
    terminate_episode:  bool              = False
    feedback:           str               = ""
    components:         Dict[str, float]  = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Primary shaping function
# ---------------------------------------------------------------------------

def compute_clarify_shaping(
    *,
    question: str,
    reveal_payload: Dict[str, Any],
    clarify_count: int,
    task: TaskConfig,
    clarify_penalty_dampening: float = 1.0,
) -> ClarifyShapingResult:
    """
    Compute the shaping reward for a single clarify action.

    Parameters
    ----------
    question : str
        The agent's clarifying question text.
    reveal_payload : dict
        The payload returned by ``InfoRevealer.process_clarify()``.
        An empty dict means no reveal occurred.
    clarify_count : int
        The number of clarify actions taken INCLUDING this one (i.e.
        ``len([a for a in action_history if a.action_type == "clarify"])``
        after appending the current action).
    task : TaskConfig
        Frozen task configuration.
    clarify_penalty_dampening : float
        Multiplier ∈ [0.0, 1.0] for the hard loop penalty.  1.0 = full
        penalty with termination; 0.0 = no penalty, no termination.
        Used during curriculum warm-up to protect early exploration.

    Returns
    -------
    ClarifyShapingResult
        See class docstring for field descriptions.
    """
    base_reward:  float = 0.0
    injection:    float = 0.0
    no_hidden:    float = 0.0
    soft_loop:    float = 0.0
    hard_loop_v:  float = 0.0
    terminate:    bool  = False
    feedback:     str   = ""

    # ── 1. Base reward from reveal ────────────────────────────────────────────
    if reveal_payload:
        trigger = str(reveal_payload.get("trigger", ""))
        # key_clarify_actions is a list of trigger names as strings.
        expected_triggers = {k.lower() for k in task.key_clarify_actions}
        base_reward = (
            SHAPING_RELEVANT
            if trigger.lower() in expected_triggers
            else SHAPING_IRRELEVANT
        )
    else:
        base_reward = SHAPING_NO_REVEAL

    # ── 2. Direct injection penalty ───────────────────────────────────────────
    # Prevents reward hacking where the agent literally types "ask_history"
    # to try to force a trigger match.
    if any(tok in question.lower() for tok in _DIRECT_TRIGGER_TOKENS):
        injection = PENALTY_INJECTION

    # ── 3. No-hidden-info efficiency penalty ─────────────────────────────────
    # If the task has no hidden information, any clarify action is wasteful
    # regardless of whether it asks a clinically sensible question.
    if not task.hidden_info:
        no_hidden = PENALTY_NO_HIDDEN

    # ── 4. Soft over-budget penalty ──────────────────────────────────────────
    # Applies once the agent has gone one step beyond the expected budget.
    soft_threshold = task.expected_clarify_steps + 1
    if clarify_count > soft_threshold:
        soft_loop = PENALTY_SOFT_LOOP

    # ── 5. Hard over-budget penalty (with optional termination) ──────────────
    hard_threshold = max(
        task.expected_clarify_steps + HARD_LOOP_EXCESS,
        HARD_LOOP_FLOOR,
    )
    if clarify_count > hard_threshold:
        hard_loop_v = PENALTY_HARD_LOOP * clarify_penalty_dampening
        if clarify_penalty_dampening >= TERMINATE_DAMPENING_THRESHOLD:
            terminate = True
            feedback  = (
                "Clarification loop terminated: agent asked too many low-value "
                f"questions ({clarify_count} clarify actions vs "
                f"{task.expected_clarify_steps} expected)."
            )

    # ── Assemble ──────────────────────────────────────────────────────────────
    total = base_reward + injection + no_hidden + soft_loop + hard_loop_v

    components: Dict[str, float] = {
        "base_reward": round(base_reward, 4),
        "injection":   round(injection,   4),
        "no_hidden":   round(no_hidden,   4),
        "soft_loop":   round(soft_loop,   4),
        "hard_loop":   round(hard_loop_v, 4),
        "total":       round(total,       4),
    }

    return ClarifyShapingResult(
        reward=round(total, 4),
        terminate_episode=terminate,
        feedback=feedback,
        components=components,
    )


# ---------------------------------------------------------------------------
# Phase-transition helper (used by the env orchestrator)
# ---------------------------------------------------------------------------

def should_advance_to_disposition(
    clarify_count: int,
    expected_clarify_steps: int,
) -> bool:
    """
    Return True when the agent has met or exceeded the expected clarify budget.

    This is the single decision point for ASSESSMENT → DISPOSITION transition
    based on clarification count.  The phase machine itself is
    ``clarify_budget_exhausted`` — this function provides the value.

    Parameters
    ----------
    clarify_count : int
        Number of clarify actions taken so far (including the current one).
    expected_clarify_steps : int
        Number of clarify steps the task designer expected.

    Returns
    -------
    bool
    """
    return clarify_count >= max(1, expected_clarify_steps)


# ---------------------------------------------------------------------------
# Timeout penalty (re-exported from env reward logic for symmetry)
# ---------------------------------------------------------------------------

TIMEOUT_PENALTY: float = -0.10
"""
Penalty applied to the last step reward when the episode hits max_steps
without a classify action.  Applied by the env orchestrator, not the grader.
"""

TIMEOUT_FEEDBACK: str = "Episode reached max steps before triage disposition."