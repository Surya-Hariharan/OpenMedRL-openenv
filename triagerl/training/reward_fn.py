"""
triagerl.training.reward_fn
============================
GRPO reward function for TriageRL.

Fixes vs previous version (train.py inline reward)
---------------------------------------------------
1. CRITICAL FIX: Task is extracted deterministically from the prompt, not
   selected randomly. Previous code did `random.choice(task_ids)` for every
   completion, completely disconnecting the reward from the prompt that
   generated it. This made the gradient signal pure noise.

   FIX: extract_task_id_from_prompt() retrieves the task_id that was
   embedded by build_prompt() at dataset generation time. The same task is
   always used for both prompt generation and reward computation.

2. CRITICAL FIX: Reward is deterministic. Given the same prompt + completion,
   the reward function always returns the same value. This is required for
   GRPO's group-relative advantage estimation — rewards must be consistent
   across the N generations per prompt.

   Previous code used random.choice() inside the reward function, meaning
   4 generations of the same prompt were each scored against different tasks.
   Group-relative advantages were meaningless.

3. Multi-step episode simulation: The reward function reconstructs the full
   episode from the prompt's step context. If the prompt is from step 2
   (after a clarify action), the env is replayed to that state before
   scoring the completion.

4. Structured error handling: Parse failures → -1.0, schema failures → -0.8,
   env step failures → -0.5. Distinct penalties enable diagnostic analysis
   of training failure modes in logs.

5. Component logging: Returns per-component breakdown as metadata for
   training dashboard visibility. GRPO requires List[float] rewards — the
   breakdown is logged separately via a thread-local buffer.

Public API
----------
    build_reward_fn(env_factory, task_map, log_components) → reward_fn
    triage_grpo_reward(prompts, completions, **kwargs) → List[float]
"""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from triagerl.core.models import TriageAction
from triagerl.reward.grader import compute_final_score
from triagerl.reward.path_quality import ActualClarifyRecord
from triagerl.tasks.schema import TaskConfig
from triagerl.training.dataset import (
    extract_task_id_from_prompt,
    extract_sample_seed_from_prompt,
)

logger = logging.getLogger(__name__)

# Thread-local storage for per-step component breakdowns (for logging only)
_component_log: threading.local = threading.local()


# ---------------------------------------------------------------------------
# JSON extraction utilities
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
_DICT_RE       = re.compile(r"\{.*\}", re.DOTALL)


def extract_action_json(completion: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Extract a JSON object from a completion string.

    Tries markdown code block first, then raw dict extraction.

    Returns
    -------
    (parsed_dict, error_message)
    parsed_dict is None on failure; error_message is "" on success.
    """
    block_match = _JSON_BLOCK_RE.search(completion)
    if block_match:
        content = block_match.group(1)
    else:
        dict_match = _DICT_RE.search(completion)
        if dict_match:
            content = dict_match.group(0)
        else:
            return None, "No JSON object found in completion."

    try:
        return json.loads(content), ""
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"


# ---------------------------------------------------------------------------
# Episode replay helper
# ---------------------------------------------------------------------------

# Clarify templates stratified by trigger type.
# Three templates per trigger ensure distribution diversity within an epoch
# while remaining clinically appropriate for that information category.
_CLARIFY_TEMPLATES_BY_TRIGGER: Dict[str, List[str]] = {
    "ask_history": [
        "Any key past medical history, medications, and medication adherence?",
        "What relevant medical history, surgical history, and current medications are on file?",
        "Are there any prior cardiac events, chronic conditions, or allergies I should know about?",
    ],
    "check_vitals": [
        "Please provide repeat vital signs including oxygen saturation and GCS.",
        "What are the most recent vital signs — specifically BP, HR, SpO2, and respiratory rate?",
        "Can you confirm the current blood pressure, heart rate, oxygen saturation, and GCS?",
    ],
    "examine_patient": [
        "What are the focused physical examination findings including any red flags?",
        "What does the physical examination reveal? Any signs of shock, altered mentation, or acute distress?",
        "Please describe the relevant physical findings, skin signs, and any examination red flags.",
    ],
}

# Fallback pool used when the task's primary trigger cannot be determined.
_CLARIFY_TEMPLATES_FALLBACK: List[str] = [
    tmpl
    for templates in _CLARIFY_TEMPLATES_BY_TRIGGER.values()
    for tmpl in templates
]


def _replay_to_step(
    env: Any,
    step_context: str,
    seed: int,
    task_id: Optional[str] = None,
    task_map: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, List[ActualClarifyRecord]]:
    """
    Replay clarify actions to reconstruct mid-episode env state.

    If the prompt was generated from step 2 (one clarify already taken),
    we re-execute that same clarify action to restore env state before
    scoring the classify completion.

    Template selection is now trigger-stratified: the task's first
    ``key_clarify_action`` determines which trigger pool to draw from,
    ensuring the replay action is clinically appropriate for the task.

    Returns
    -------
    (current_obs, clarify_records)
    """
    import random

    clarify_records: List[ActualClarifyRecord] = []

    # Detect if prompt is mid-episode by step context string
    if "step 2" not in step_context and "clarification already made" not in step_context:
        # Initial observation — no replay needed
        return env, clarify_records

    # Select template pool based on task's expected primary trigger.
    pool = _CLARIFY_TEMPLATES_FALLBACK
    if task_id and task_map:
        task = task_map.get(task_id)
        if task is not None:
            primary_trigger = (
                task.key_clarify_actions[0].lower()
                if task.key_clarify_actions
                else None
            )
            if primary_trigger and primary_trigger in _CLARIFY_TEMPLATES_BY_TRIGGER:
                pool = _CLARIFY_TEMPLATES_BY_TRIGGER[primary_trigger]

    # Need to replay one clarify step to restore env state
    rng = random.Random(seed)
    cq  = rng.choice(pool)

    try:
        clarify_action = TriageAction(
            action_type="clarify",
            clarifying_question=cq,
            reasoning="Reconstructing episode state for reward computation.",
            recommended_actions=[],
            confidence=0.45,
        )
        step_result = env.step(clarify_action)
        if isinstance(step_result, tuple):
            next_obs, _, _, info = step_result
        else:
            next_obs = step_result
            info = {}

        # Extract actual trigger from env step info/reveal payload
        reveal_payload = info.get("reveal_payload", {}) if isinstance(info, dict) else {}
        trigger = reveal_payload.get("trigger") if reveal_payload else None

        clarify_records.append(
            ActualClarifyRecord(question=cq, trigger=trigger)
        )
        return next_obs, clarify_records

    except Exception as e:
        logger.warning("Episode replay failed: %s", e)
        return env, clarify_records


# ---------------------------------------------------------------------------
# Per-completion reward computation
# ---------------------------------------------------------------------------

def _score_completion(
    prompt: str,
    completion: str,
    task_map: Dict[str, TaskConfig],
    env_factory: Callable[[str], Any],
) -> Tuple[float, str, Optional[Dict[str, float]]]:
    """
    Score a single completion against its bound task.

    Returns
    -------
    (reward, error_label, component_dict)
    error_label is "" on success, descriptive string on failure.
    component_dict is None on failure, breakdown dict on success.
    """
    # ── Extract task_id from prompt — deterministic, no randomness ────────────
    task_id = extract_task_id_from_prompt(prompt)
    if task_id is None:
        logger.error("Prompt missing task_id sentinel — cannot score completion.")
        return -1.0, "missing_task_id", None

    task = task_map.get(task_id)
    if task is None:
        logger.error("task_id %r not in task_map.", task_id)
        return -1.0, "unknown_task_id", None

    # ── Parse completion ───────────────────────────────────────────────────────
    parsed_dict, parse_err = extract_action_json(completion)
    if parsed_dict is None:
        return -1.0, f"parse_failure:{parse_err[:80]}", None

    # ── Validate action schema ─────────────────────────────────────────────────
    try:
        action = TriageAction.model_validate(parsed_dict)
    except Exception as e:
        return -0.8, f"schema_failure:{str(e)[:80]}", None

    # ── Only classify actions receive the full reward signal ───────────────────
    # Clarify actions during GRPO should receive a small positive reward to
    # encourage exploration, but not the full grading pipeline (which requires
    # a classification). The shaping reward for clarify is handled by the env
    # and returned as part of the episode trajectory — here we only score the
    # terminal action.
    if action.action_type == "clarify":
        # Small negative reward to prevent clarify-only exploits while keeping
        # magnitude low so clarifying isn't overly penalized.
        return -0.01, "clarify_action", None

    # ── Build env and replay to match prompt step ──────────────────────────────
    # Extract step context from prompt for mid-episode replay (do this before
    # resetting the env so we can use it to compute a fallback replay seed).
    step_ctx_match = re.search(r"__STEP_CTX__:([^\n]+)", prompt)
    step_context = step_ctx_match.group(1).strip() if step_ctx_match else ""

    try:
        env = env_factory(task_id)
        # Prefer explicit sample seed embedded in prompt for deterministic
        # replay. Fall back to old hash-based seed if sentinel missing.
        sample_seed = extract_sample_seed_from_prompt(prompt)
        if sample_seed is not None:
            env.reset(seed=sample_seed)
            replay_seed = int(sample_seed) & 0xFFFFFFFF
        else:
            env.reset()
            replay_seed = hash(task_id + step_context) & 0xFFFFFFFF
    except Exception as e:
        return -0.5, f"env_reset_failure:{str(e)[:80]}", None

    # Replay clarify actions using the same seed used when the prompt was
    # generated. Pass task_id and task_map so the replay selects a
    # trigger-appropriate clarify template instead of a random one.
    _, clarify_records = _replay_to_step(
        env,
        step_context,
        seed=replay_seed,
        task_id=task_id,
        task_map=task_map,
    )
    steps_taken = len(clarify_records) + 1  # clarify steps + this classify step

    # ── Score the classify action ──────────────────────────────────────────────
    try:
        # compute_final_score expects a plain dict for the task; TaskConfig
        # objects must be serialized to avoid attribute/dict mismatch.
        task_dict = task.model_dump() if hasattr(task, "model_dump") else dict(task)

        final_score, breakdown, _ = compute_final_score(
            action=action,
            task=task_dict,
            clarifies=clarify_records,
            steps_taken=steps_taken,
        )
    except Exception as e:
        logger.exception("compute_final_score failed for task %s: %s", task_id, e)
        return -0.5, f"grader_failure:{str(e)[:80]}", None

    # compute_final_score returns a ComponentDict (plain dict) as the
    # second value. Use it directly for logging/telemetry.
    component_dict = breakdown if isinstance(breakdown, dict) else dict(breakdown)
    return float(final_score), "", component_dict


# ---------------------------------------------------------------------------
# Reward function factory — primary public API
# ---------------------------------------------------------------------------

def build_reward_fn(
    task_map: Dict[str, TaskConfig],
    env_factory: Callable[[str], Any],
    log_components: bool = True,
) -> Callable[[List[str], List[str]], List[float]]:
    """
    Build a GRPO-compatible reward function bound to a fixed task map.

    Parameters
    ----------
    task_map : dict[str, TaskConfig]
        Maps task_id → TaskConfig. Must be pre-loaded and immutable.
        Passed by reference — no copy made — so do not mutate after calling.
    env_factory : callable
        Function (task_id: str) → env instance. Must be stateless/thread-safe
        — called once per completion, not reused across calls.
    log_components : bool
        If True, logs per-component breakdowns at DEBUG level.

    Returns
    -------
    reward_fn : callable matching GRPOTrainer reward_funcs signature
        reward_fn(prompts: List[str], completions: List[str], **kwargs)
            → List[float]
    """
    def reward_fn(
        prompts: List[str],
        completions: List[str],
        **kwargs: Any,
    ) -> List[float]:
        """
        GRPO reward function.

        DETERMINISTIC: same prompt + completion → same reward, always.
        No randomness. task_id extracted from prompt sentinel.

        Parameters
        ----------
        prompts : list[str]
            One prompt per completion (from dataset).
        completions : list[str]
            Model-generated completions to score.

        Returns
        -------
        list[float] — one reward per completion ∈ [-1.0, 1.0]
        """
        if len(prompts) != len(completions):
            raise ValueError(
                f"prompts ({len(prompts)}) and completions ({len(completions)}) "
                "must have the same length."
            )

        rewards: List[float] = []
        for prompt, completion in zip(prompts, completions):
            reward, error_label, components = _score_completion(
                prompt=prompt,
                completion=completion,
                task_map=task_map,
                env_factory=env_factory,
            )
            rewards.append(reward)

            if log_components:
                if error_label:
                    logger.debug(
                        "reward=%.3f error=%s task=%s",
                        reward,
                        error_label,
                        extract_task_id_from_prompt(prompt) or "UNKNOWN",
                    )
                elif components:
                    logger.debug(
                        "reward=%.3f esi=%.3f reason=%.3f actions=%.3f "
                        "temporal=%.3f path=%.3f safety=%.3f task=%s",
                        reward,
                        components.get("esi_score", 0.0),
                        components.get("reasoning_score", 0.0),
                        components.get("action_score", 0.0),
                        components.get("temporal_score", 0.0),
                        components.get("path_quality", 0.0),
                        components.get("safety_modifier", 1.0),
                        extract_task_id_from_prompt(prompt) or "UNKNOWN",
                    )

        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Module-level convenience function (used if reward_fn is passed directly)
# ---------------------------------------------------------------------------

def triage_grpo_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs: Any,
) -> List[float]:
    """
    Stateless module-level reward function for GRPOTrainer.

    Requires _GLOBAL_TASK_MAP and _GLOBAL_ENV_FACTORY to be set by
    training setup before use. Use build_reward_fn() for the
    production path — this function exists as a fallback for frameworks
    that do not support closures as reward functions.
    """
    global _GLOBAL_TASK_MAP, _GLOBAL_ENV_FACTORY
    if _GLOBAL_TASK_MAP is None or _GLOBAL_ENV_FACTORY is None:
        raise RuntimeError(
            "triage_grpo_reward requires _GLOBAL_TASK_MAP and "
            "_GLOBAL_ENV_FACTORY to be set. Use build_reward_fn() instead."
        )
    fn = build_reward_fn(_GLOBAL_TASK_MAP, _GLOBAL_ENV_FACTORY, log_components=True)
    return fn(prompts, completions, **kwargs)


# Module-level globals for stateless reward function path
_GLOBAL_TASK_MAP:     Optional[Dict[str, TaskConfig]]  = None
_GLOBAL_ENV_FACTORY:  Optional[Callable[[str], Any]]   = None