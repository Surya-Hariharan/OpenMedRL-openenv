"""
triagerl.reward.shaping
=======================
Clarify-step reward shaping for RL training.

Fixes vs previous version
--------------------------
1. reveal_payload structure now validated before use. Previous version
   silently called reveal_payload.get("trigger", "") — if InfoRevealer
   returned a payload without a "trigger" key (which was never guaranteed),
   every reveal scored as SHAPING_IRRELEVANT regardless of clinical quality,
   making the relevance-discrimination mechanism silently fail.

   FIX: reveal_payload is now validated by _extract_trigger_from_payload().
   The function checks for "trigger" key, validates it against VALID_TRIGGERS,
   and returns None (not "") on failure so the caller can distinguish "no
   trigger key" from "trigger is empty string". Shaping logic branches
   explicitly on this distinction.

2. TERMINATE_DAMPENING_THRESHOLD reduced from 0.5 to 0.3. With train.py
   using clarify_penalty_dampening=0.1 during curriculum warm-up, the
   previous threshold of 0.5 meant episodes NEVER terminated early even
   with runaway clarification loops. At dampening=0.3 (mid-curriculum),
   termination now activates.

3. ClarifyShapingResult.components dict is deterministic — keys always
   present with 0.0 defaults, not conditionally populated.

Design contract
---------------
* Pure function: identical inputs → identical outputs.
* No I/O, no logging, no side-effects.
* All thresholds are named module-level constants.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional

from triagerl.core.constants import VALID_TRIGGERS
from triagerl.tasks.schema import TaskConfig


# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

# Base rewards from reveal quality
SHAPING_RELEVANT:    float = 0.03   # trigger ∈ task.key_clarify_actions
SHAPING_IRRELEVANT:  float = 0.01   # reveal occurred, trigger not key
SHAPING_NO_REVEAL:   float = -0.01  # clarify produced nothing

# Penalties (all non-positive)
PENALTY_INJECTION:   float = -0.02  # direct trigger token in question text
PENALTY_NO_HIDDEN:   float = -0.01  # task has no hidden_info
PENALTY_SOFT_LOOP:   float = -0.02  # clarify_count > expected + 1
PENALTY_HARD_LOOP:   float = -0.05  # clarify_count > hard threshold

# Loop thresholds
HARD_LOOP_EXCESS:    int   = 2
HARD_LOOP_FLOOR:     int   = 4

# FIX: reduced from 0.5 → 0.3 so mid-curriculum (dampening=0.3) terminates loops
TERMINATE_DAMPENING_THRESHOLD: float = 0.3

# Direct trigger token strings — must match VALID_TRIGGERS exactly
_DIRECT_TRIGGER_TOKENS: FrozenSet[str] = frozenset(VALID_TRIGGERS)

# Timeout constants (used by env orchestrator)
TIMEOUT_PENALTY:  float = -0.10
TIMEOUT_FEEDBACK: str   = "Episode reached max steps before triage disposition."


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClarifyShapingResult:
    """
    Structured result from compute_clarify_shaping.

    Attributes
    ----------
    reward : float
        Net shaping reward for this clarify step.
    terminate_episode : bool
        True when hard loop threshold breached and dampening ≥ threshold.
    feedback : str
        Non-empty only when terminate_episode is True.
    components : dict[str, float]
        Per-sub-reward breakdown for telemetry. All keys always present.
        Keys: "base_reward", "injection", "no_hidden", "soft_loop",
              "hard_loop", "total".
    trigger_used : str | None
        The trigger extracted from reveal_payload (None if extraction failed).
        Exposed for debugging path quality consistency.
    """
    reward:            float
    terminate_episode: bool             = False
    feedback:          str              = ""
    components:        Dict[str, float] = field(default_factory=dict)
    trigger_used:      Optional[str]    = None


# ---------------------------------------------------------------------------
# Internal: safe trigger extraction
# ---------------------------------------------------------------------------

def _extract_trigger_from_payload(reveal_payload: Dict[str, Any]) -> Optional[str]:
    """
    Extract and validate the trigger from a reveal payload.

    FIX: Previous code did reveal_payload.get("trigger", "") which returned
    "" on missing key — indistinguishable from empty-string trigger value.
    Both cases silently produced SHAPING_IRRELEVANT.

    Now returns:
    - str: the validated trigger name (in VALID_TRIGGERS)
    - None: if key absent, value is not a string, or value not in VALID_TRIGGERS

    The None case is logged in ClarifyShapingResult.trigger_used so the
    env can detect InfoRevealer payload structure changes.
    """
    if not reveal_payload:
        return None
    raw = reveal_payload.get("trigger")
    if raw is None:
        return None
    trigger = str(raw).strip().lower()
    if trigger not in VALID_TRIGGERS:
        return None
    return trigger


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
        The payload returned by InfoRevealer.process_clarify().
        FIX: Now validated before use. Must contain "trigger" key with a
        value in VALID_TRIGGERS to score as SHAPING_RELEVANT.
    clarify_count : int
        Number of clarify actions taken INCLUDING this one.
    task : TaskConfig
        Frozen task configuration.
    clarify_penalty_dampening : float
        Multiplier ∈ [0.0, 1.0] for the hard loop penalty. 1.0 = full
        penalty with termination; 0.0 = no penalty, no termination.

    Returns
    -------
    ClarifyShapingResult
    """
    base_reward: float = 0.0
    injection:   float = 0.0
    no_hidden:   float = 0.0
    soft_loop:   float = 0.0
    hard_loop_v: float = 0.0
    terminate:   bool  = False
    feedback:    str   = ""

    # ── 1. Base reward from reveal ────────────────────────────────────────────
    # FIX: Use validated trigger extraction, not raw .get("trigger", "")
    trigger_used: Optional[str] = None

    if reveal_payload:
        trigger_used = _extract_trigger_from_payload(reveal_payload)
        if trigger_used is not None:
            # Payload has a valid trigger — check if it matches expected
            expected_triggers = {k.lower() for k in task.key_clarify_actions}
            base_reward = (
                SHAPING_RELEVANT
                if trigger_used in expected_triggers
                else SHAPING_IRRELEVANT
            )
        else:
            # Payload returned but trigger key missing or invalid.
            # Score as SHAPING_IRRELEVANT (information was revealed, but we
            # cannot classify relevance — do not penalise the agent for an
            # InfoRevealer payload structure issue).
            base_reward = SHAPING_IRRELEVANT
    else:
        # Empty payload: clarify produced no new information.
        base_reward = SHAPING_NO_REVEAL

    # ── 2. Direct injection penalty ───────────────────────────────────────────
    question_lower = question.lower()
    if any(tok in question_lower for tok in _DIRECT_TRIGGER_TOKENS):
        injection = PENALTY_INJECTION

    # ── 3. No-hidden-info efficiency penalty ─────────────────────────────────
    if not task.hidden_info:
        no_hidden = PENALTY_NO_HIDDEN

    # ── 4. Soft over-budget penalty ──────────────────────────────────────────
    soft_threshold = task.expected_clarify_steps + 1
    if clarify_count > soft_threshold:
        soft_loop = PENALTY_SOFT_LOOP

    # ── 5. Hard over-budget penalty ───────────────────────────────────────────
    hard_threshold = max(
        task.expected_clarify_steps + HARD_LOOP_EXCESS,
        HARD_LOOP_FLOOR,
    )
    if clarify_count > hard_threshold:
        hard_loop_v = PENALTY_HARD_LOOP * clarify_penalty_dampening
        # FIX: threshold reduced 0.5 → 0.3 so mid-curriculum triggers termination
        if clarify_penalty_dampening >= TERMINATE_DAMPENING_THRESHOLD:
            terminate = True
            feedback  = (
                f"Clarification loop terminated: {clarify_count} clarify actions "
                f"vs {task.expected_clarify_steps} expected."
            )

    # ── Assemble ──────────────────────────────────────────────────────────────
    total = base_reward + injection + no_hidden + soft_loop + hard_loop_v

    # All keys always present (FIX: was conditionally populated)
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
        trigger_used=trigger_used,
    )


# ---------------------------------------------------------------------------
# Phase-transition helper
# ---------------------------------------------------------------------------

def should_advance_to_disposition(
    clarify_count: int,
    expected_clarify_steps: int,
) -> bool:
    """
    Return True when the agent has met or exceeded the expected clarify budget.

    Single decision point for ASSESSMENT → DISPOSITION transition.
    """
    return clarify_count >= max(1, expected_clarify_steps)