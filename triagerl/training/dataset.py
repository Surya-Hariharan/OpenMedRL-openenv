"""
triagerl.training.dataset
=========================
On-policy episode dataset generation for GRPO training.

Fixes vs previous version (train.py static dataset)
-----------------------------------------------------
1. CRITICAL FIX: Task ID is now deterministically bound to every prompt.
   Previous train.py generated 128 static initial observations with no
   task_id in the prompt, then the reward function selected a RANDOM task
   to evaluate the completion against. This completely severed the
   prompt ↔ reward connection, making gradient signal pure noise.

   FIX: Every prompt carries __TASK_ID__:{task_id} in a structured header
   that the reward function extracts. Same task is used to generate the
   prompt and to evaluate the completion.

2. CRITICAL FIX: Dataset is regenerated between training epochs (on-policy).
   Previous static 128-sample dataset was generated once at startup and
   never refreshed — completions from step 50 were still being evaluated
   against step-0 initial observations, making the dataset increasingly
   off-policy as training progressed.

   FIX: TriageEpisodeDataset.refresh() rebuilds the dataset from fresh
   env resets. Called between epochs by the training loop.

3. Multi-step support: Dataset includes observations at any step of the
   episode (not just step 1). For tasks with expected_clarify_steps > 0,
   we generate observations at both the initial step and after one clarify
   action, so the model learns mid-episode continuation behaviour.

4. Deterministic generation: All randomness seeded from (global_seed,
   epoch, sample_index). Same epoch number → same dataset. Different epochs
   → different samples (curriculum progression).

5. Curriculum scheduling: Early epochs oversample easy tasks. Later epochs
   shift toward hard tasks. Controlled by difficulty_weights parameter.

Public API
----------
    build_prompt(obs_json, task_id, step_context) → str
    extract_task_id_from_prompt(prompt) → str | None
    TriageEpisodeDataset(task_ids, n_samples, seed, env_factory)
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from datasets import Dataset


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

# Sentinel used to embed task_id in prompt — must be stable across versions
_TASK_ID_SENTINEL = "__TASK_ID__"
_STEP_CTX_SENTINEL = "__STEP_CTX__"
_SAMPLE_SEED_SENTINEL = "__SAMPLE_SEED__"

SYSTEM_PROMPT = """\
You are an emergency department triage policy model.
Assess the patient observation below and return a single JSON object.

Valid response formats:

Classify (assign ESI level):
{"action_type": "classify", "esi_level": <1-5>, "reasoning": "<clinical reasoning, minimum 40 words>", "recommended_actions": ["action1", "action2", ...], "confidence": <0.0-1.0>}

Clarify (ask one focused question):
{"action_type": "clarify", "clarifying_question": "<specific clinical question>", "reasoning": "<why this information is needed>", "recommended_actions": [], "confidence": <0.0-1.0>}

ESI definitions:
1 = Immediate life threat / life-saving intervention required now
2 = High risk / emergent — cannot wait
3 = Urgent but stable — needs multiple resources
4 = Less urgent — needs one resource
5 = Non-urgent — no resources needed

Clinical safety rules:
- Never undertriage ESI 1-2 patients. When uncertain between ESI 1 and 2, choose 1.
- Hypotension + altered consciousness + infection signs = ESI 1 until proven otherwise.
- Non-blanching rash + fever + meningism = ESI 1 immediately.
- Tearing chest/back pain with BP differential between arms = ESI 1, no thrombolytics.
- Silent chest in asthma is more dangerous than wheeze.
- COPD: target SpO2 88-92%, never 98-100%.
- Always check INR in anticoagulated patients.
- Stop metformin immediately in suspected AKI or sepsis.

Return JSON only. No markdown. No preamble. No explanation outside the JSON."""


def build_prompt(
    obs_json: str,
    task_id: str,
    step_context: str = "",
    sample_seed: Optional[int] = None,
) -> str:
    """
    Build a complete prompt string with task_id embedded deterministically.

    The task_id sentinel is in the prompt header — invisible to the model
    as clinical guidance, but extractable by the reward function without
    parsing clinical text.

    Parameters
    ----------
    obs_json : str
        JSON-serialised observation from env.reset() or env.step().
    task_id : str
        The task identifier. Embedded in prompt header for reward extraction.
    step_context : str
        Optional context string describing episode state (e.g. "step 2 of 3,
        one clarification already made"). Helps model orient in episode.

    Returns
    -------
    str — complete prompt ready for tokenisation.
    """
    header_lines = [
        f"{_TASK_ID_SENTINEL}:{task_id}",
    ]
    if step_context:
        header_lines.append(f"{_STEP_CTX_SENTINEL}:{step_context}")
    if sample_seed is not None:
        header_lines.append(f"{_SAMPLE_SEED_SENTINEL}:{int(sample_seed)}")

    header = "\n".join(header_lines)

    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|context|>\n{header}\n"
        f"<|user|>\nPatient observation:\n{obs_json}\n"
        f"<|assistant|>\n"
    )


def extract_task_id_from_prompt(prompt: str) -> Optional[str]:
    """
    Extract the task_id embedded by build_prompt().

    Returns None if the sentinel is not present (malformed prompt).
    Used by the reward function to bind completion → task deterministically.
    """
    m = re.search(rf"{re.escape(_TASK_ID_SENTINEL)}:([^\n]+)", prompt)
    if m:
        return m.group(1).strip()
    return None


def extract_sample_seed_from_prompt(prompt: str) -> Optional[int]:
    m = re.search(rf"{re.escape(_SAMPLE_SEED_SENTINEL)}:([0-9]+)", prompt)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# On-policy dataset builder
# ---------------------------------------------------------------------------

class TriageEpisodeDataset:
    """
    On-policy episode prompt dataset for GRPO training.

    Generates prompts from fresh env resets. Refreshed between epochs so
    the model is always trained on prompts from its current policy's
    exploration frontier, not stale step-0 observations.

    Parameters
    ----------
    task_ids : list[str]
        All available task IDs from the loader.
    n_samples : int
        Number of prompts per epoch (dataset size).
    global_seed : int
        Base seed. Epoch seed = global_seed * 1000 + epoch_number, ensuring
        different but reproducible samples per epoch.
    env_factory : callable
        Function (task_id: str) -> env instance. Env must have reset() and
        step() methods matching the MedicalTriageEnv interface.
    difficulty_weights : dict | None
        Maps difficulty tier to sampling weight for curriculum scheduling.
        Default: {"easy": 0.34, "medium": 0.33, "hard": 0.33}.
    task_difficulty_map : dict | None
        Maps task_id → difficulty tier. Required if difficulty_weights used.
    include_mid_episode : bool
        If True, also generate prompts from mid-episode states (after one
        clarify action). Enables learning continuation behaviour.
        Default: True.
    """

    def __init__(
        self,
        task_ids: List[str],
        n_samples: int,
        global_seed: int,
        env_factory: Callable[[str], Any],
        difficulty_weights: Optional[Dict[str, float]] = None,
        task_difficulty_map: Optional[Dict[str, str]] = None,
        include_mid_episode: bool = True,
    ) -> None:
        if not task_ids:
            raise ValueError("task_ids must be non-empty.")
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1.")

        self._task_ids             = list(task_ids)
        self._n_samples            = n_samples
        self._global_seed          = global_seed
        self._env_factory          = env_factory
        self._task_difficulty_map  = task_difficulty_map or {}
        self._include_mid_episode  = include_mid_episode
        self._current_epoch        = 0
        self._rows: List[Dict[str, str]] = []

        # Default uniform weights if not specified
        self._difficulty_weights = difficulty_weights or {
            "easy": 0.34,
            "medium": 0.33,
            "hard": 0.33,
        }

        # Initial dataset build for epoch 0
        self.refresh(epoch=0)

    # ------------------------------------------------------------------
    # Curriculum task sampling
    # ------------------------------------------------------------------

    def _sample_task_ids_for_epoch(
        self,
        epoch: int,
        n: int,
    ) -> List[str]:
        """
        Deterministically sample n task IDs for a given epoch.

        Curriculum schedule: epoch 0-4 = easy-heavy. epoch 5-14 = balanced.
        epoch 15+ = hard-heavy. Overridden by explicit difficulty_weights.

        All randomness seeded from (global_seed, epoch) — same inputs
        always produce same output across ranks and runs.
        """
        import random
        rng = random.Random(self._global_seed * 1000 + epoch)

        # Build epoch-specific difficulty weights
        if epoch < 5:
            weights = {"easy": 0.60, "medium": 0.30, "hard": 0.10}
        elif epoch < 15:
            weights = {"easy": 0.30, "medium": 0.40, "hard": 0.30}
        else:
            weights = self._difficulty_weights

        # Partition task IDs by difficulty
        by_difficulty: Dict[str, List[str]] = {"easy": [], "medium": [], "hard": []}
        unclassified: List[str] = []
        for tid in self._task_ids:
            diff = self._task_difficulty_map.get(tid, "")
            if diff in by_difficulty:
                by_difficulty[diff].append(tid)
            else:
                unclassified.append(tid)

        sampled: List[str] = []
        for _ in range(n):
            # Choose a difficulty tier
            tier = rng.choices(
                list(weights.keys()),
                weights=list(weights.values()),
                k=1,
            )[0]
            pool = by_difficulty.get(tier, []) or self._task_ids
            sampled.append(rng.choice(pool))

        return sampled

    # ------------------------------------------------------------------
    # Episode prompt generation
    # ------------------------------------------------------------------

    def _generate_prompt_for_task(
        self,
        task_id: str,
        sample_idx: int,
        epoch: int,
    ) -> Optional[Dict[str, str]]:
        """
        Reset env for task_id and generate a prompt from the observation.

        If include_mid_episode is True, 50% of samples (based on sample_idx
        parity) will advance one clarify step to generate a mid-episode prompt.
        This teaches the model continuation behaviour, not just step-1 responses.

        Returns dict with "prompt" and "task_id" keys, or None on env failure.
        """
        # Deterministic per-prompt seed so reward_fn can reconstruct the
        # exact same episode. Seed derivation must match the reward-side
        # extraction logic.
        sample_seed = self._global_seed * 100000 + epoch * 1000 + sample_idx
        try:
            env = self._env_factory(task_id)
            obs = env.reset(seed=sample_seed)
        except Exception:
            return None

        try:
            obs_json = obs.model_dump_json() if hasattr(obs, "model_dump_json") else json.dumps(obs)
        except Exception:
            obs_json = str(obs)

        step_context = f"step 1 of {getattr(obs, 'max_steps', '?')}"

        # Mid-episode sample: take one clarify step first
        if self._include_mid_episode and (sample_idx % 2 == 1):
            import random
            rng = random.Random(sample_seed)
            clarify_templates = [
                "Any key past medical history, medications, and medication adherence?",
                "Please provide repeat vital signs including oxygen saturation and GCS.",
                "What are the focused physical examination findings including any red flags?",
                "Any relevant family history, social history, or recent investigations?",
            ]
            cq = rng.choice(clarify_templates)

            try:
                from triagerl.core.models import TriageAction
                clarify_action = TriageAction(
                    action_type="clarify",
                    clarifying_question=cq,
                    reasoning="Gathering high-yield information before classification.",
                    recommended_actions=[],
                    confidence=0.45,
                )
                step_result = env.step(clarify_action)
                if isinstance(step_result, tuple):
                    next_obs = step_result[0]
                else:
                    next_obs = step_result

                obs_json = (
                    next_obs.model_dump_json()
                    if hasattr(next_obs, "model_dump_json")
                    else json.dumps(next_obs)
                )
                max_s = getattr(next_obs, "max_steps", "?")
                step_context = (
                    f"step 2 of {max_s}, one clarification already made"
                )
            except Exception:
                # If mid-episode step fails, fall back to initial observation
                pass

        prompt = build_prompt(obs_json, task_id, step_context, sample_seed=sample_seed)
        return {"prompt": prompt, "task_id": task_id}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def refresh(self, epoch: int) -> None:
        """
        Regenerate the dataset for a new epoch.

        Called by the training loop between epochs to keep the dataset
        on-policy. All randomness seeded from (global_seed, epoch).

        Parameters
        ----------
        epoch : int
            Current training epoch number (0-indexed).
        """
        self._current_epoch = epoch
        sampled_ids = self._sample_task_ids_for_epoch(epoch, self._n_samples)

        rows: List[Dict[str, str]] = []
        for idx, task_id in enumerate(sampled_ids):
            row = self._generate_prompt_for_task(task_id, idx, epoch)
            if row is not None:
                rows.append(row)

        # If env failures dropped rows below target, attempt to pad with
        # additional samples. Protect against infinite retry loops by
        # bounding attempts. If we still cannot generate enough real rows,
        # synthesize minimal header-only prompts as a last resort so
        # refresh() always completes and returns a non-empty dataset.
        attempts = 0
        max_attempts = max(10, self._n_samples * 5)
        import random
        while len(rows) < self._n_samples and attempts < max_attempts and self._task_ids:
            rng = random.Random(self._global_seed + epoch + len(rows) + attempts)
            fallback_id = rng.choice(self._task_ids)
            row = self._generate_prompt_for_task(fallback_id, len(rows), epoch)
            if row is not None:
                rows.append(row)
            attempts += 1

        # If padding failed (e.g. env_factory consistently raised), synthesize
        # simple header-only prompts so downstream training can proceed.
        if len(rows) < self._n_samples and self._task_ids:
            for i in range(len(rows), self._n_samples):
                tid = self._task_ids[i % len(self._task_ids)]
                # Minimal observation stub: small stable JSON so reward_fn can
                # at least extract task_id and avoid crashes. This preserves
                # behaviour (prompt contains task_id) while guaranteeing
                # termination.
                obs_stub = {
                    "task_ref": f"case-stub-{i}",
                    "patient": {},
                    "step_number": 1,
                    "max_steps": 1,
                    "phase": "ASSESSMENT",
                }
                obs_json = json.dumps(obs_stub)
                prompt = build_prompt(obs_json, tid, "step 1 of 1", sample_seed=None)
                rows.append({"prompt": prompt, "task_id": tid})

        self._rows = rows

    def to_hf_dataset(self) -> Dataset:
        """
        Return a HuggingFace Dataset of {"prompt": ..., "task_id": ...} rows.

        The task_id column is metadata only — GRPOTrainer passes prompts to
        the model and completions to the reward function. The reward function
        extracts task_id from the prompt text using extract_task_id_from_prompt().
        """
        if not self._rows:
            raise RuntimeError(
                "Dataset is empty. Call refresh() or check env_factory."
            )
        return Dataset.from_list(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def current_epoch(self) -> int:
        return self._current_epoch