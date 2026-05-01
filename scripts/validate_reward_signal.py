#!/usr/bin/env python3
import json
import statistics
import sys
from typing import List
from pathlib import Path

# Ensure repo root is on sys.path so local `triagerl` package imports resolve
repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from triagerl.tasks.loader import load_all_tasks, get_task
from triagerl.training.dataset import TriageEpisodeDataset, extract_task_id_from_prompt
from triagerl.training.reward_fn import build_reward_fn
from triagerl.env.triage_env import MedicalTriageEnv


# Simple env_factory used by dataset and reward function
def _env_factory(task_id: str) -> MedicalTriageEnv:
    task = get_task(task_id)
    return MedicalTriageEnv(task_config=task, clarify_penalty_dampening=1.0)


def make_good_completion(prompt: str, task_map) -> str:
    tid = extract_task_id_from_prompt(prompt)
    task = task_map[tid]
    reasoning = (
        "".join([k + " " for k in (task.key_reasoning_keywords or [])[:6]])
        + ("Detailed clinical reasoning. " * 40)
    )
    action = {
        "action_type": "classify",
        "esi_level": int(task.esi_correct),
        "reasoning": reasoning,
        "recommended_actions": (task.expected_actions or [])[:4],
        "confidence": 0.95,
    }
    return json.dumps(action)


def make_bad_completion(prompt: str, task_map) -> str:
    tid = extract_task_id_from_prompt(prompt)
    task = task_map[tid]
    # Predict a worse (higher number) ESI (undertriage) where possible
    pred = min(5, int(task.esi_correct) + 2)
    action = {
        "action_type": "classify",
        "esi_level": pred,
        "reasoning": "Brief, non-specific.",
        "recommended_actions": [],
        "confidence": 0.2,
    }
    return json.dumps(action)


def make_overclarify_completion(prompt: str) -> str:
    action = {
        "action_type": "clarify",
        "clarifying_question": "Can you provide more details? " * 8,
        "reasoning": "",
        "recommended_actions": [],
        "confidence": 0.5,
    }
    return json.dumps(action)


def make_keyword_stuffing_completion(prompt: str, task_map) -> str:
    tid = extract_task_id_from_prompt(prompt)
    task = task_map[tid]
    kws = (task.key_reasoning_keywords or [])[:4]
    # Short text but repeats keywords heavily
    reasoning = " ".join([" ".join(kws)] * 30)
    action = {
        "action_type": "classify",
        "esi_level": int(task.esi_correct),
        "reasoning": reasoning,
        "recommended_actions": (task.expected_actions or [])[:2],
        "confidence": 0.9,
    }
    return json.dumps(action)


def stats(name: str, values: List[float]):
    print(f"\n{name} — n={len(values)} mean={statistics.mean(values):.4f} std={statistics.pstdev(values):.4f} min={min(values):.4f} max={max(values):.4f}")


def main():
    # Explicit start marker to ensure script execution is visible
    print("VALIDATION_RUN_START", flush=True)

    task_map, task_ids = load_all_tasks()

    dataset_obj = TriageEpisodeDataset(
        task_ids=task_ids,
        n_samples=40,
        global_seed=2026,
        env_factory=_env_factory,
        include_mid_episode=True,
    )

    # Use epoch 0 dataset
    dataset_obj.refresh(epoch=0)
    rows = list(dataset_obj._rows)  # use internal rows to avoid HF dependency
    prompts = [r["prompt"] for r in rows]

    reward_fn = build_reward_fn(task_map=task_map, env_factory=_env_factory, log_components=False)

    good_rewards = []
    bad_rewards = []
    clarify_rewards = []
    stuffed_rewards = []

    # Determinism check: collect one-run rewards and then re-run for equality
    deterministic_ok = True

    for prompt in prompts:
        good_c = make_good_completion(prompt, task_map)
        bad_c = make_bad_completion(prompt, task_map)
        clar_c = make_overclarify_completion(prompt)
        stuffed_c = make_keyword_stuffing_completion(prompt, task_map)

        r_good = reward_fn([prompt], [good_c])[0]
        r_bad = reward_fn([prompt], [bad_c])[0]
        r_clar = reward_fn([prompt], [clar_c])[0]
        r_stuffed = reward_fn([prompt], [stuffed_c])[0]

        good_rewards.append(r_good)
        bad_rewards.append(r_bad)
        clarify_rewards.append(r_clar)
        stuffed_rewards.append(r_stuffed)

    # Re-run determinism check for a subset
    for i, prompt in enumerate(prompts[:10]):
        good_c = make_good_completion(prompt, task_map)
        r1 = reward_fn([prompt], [good_c])[0]
        r2 = reward_fn([prompt], [good_c])[0]
        if r1 != r2:
            deterministic_ok = False
            print(f"Determinism failure at sample {i}: {r1} != {r2}")

    print("\n=== Reward distribution summary ===")
    stats("Good policy", good_rewards)
    stats("Bad policy", bad_rewards)
    stats("Over-clarify (clarify actions)", clarify_rewards)
    stats("Keyword-stuffing", stuffed_rewards)

    print(f"\nDeterministic rewards (same inputs) => {deterministic_ok}")

    # Compare means
    mean_good = statistics.mean(good_rewards)
    mean_bad = statistics.mean(bad_rewards)
    mean_clar = statistics.mean(clarify_rewards)
    mean_stuffed = statistics.mean(stuffed_rewards)

    print("\nMean comparison:")
    print(f" Good {mean_good:.4f} vs Bad {mean_bad:.4f} vs Clarify {mean_clar:.4f} vs Stuffed {mean_stuffed:.4f}")

    # Simple signal checks
    if mean_good <= mean_bad:
        print("\nWARNING: Good policy mean <= Bad policy mean — reward might be misleading.")
    else:
        print("\nOK: Good policy mean > Bad policy mean")

    # Check for constant distribution
    def is_constant(vals):
        return all(v == vals[0] for v in vals)

    if is_constant(good_rewards) or is_constant(bad_rewards):
        print("\nWARNING: Some reward distribution is constant — no learning signal.")

    # Check for reward plateaus: small std dev relative to range
    def plateau(vals):
        rng = max(vals) - min(vals)
        sd = statistics.pstdev(vals)
        return sd < 0.05 and rng < 0.1

    if plateau(good_rewards):
        print("\nNOTICE: Good policy rewards show low variance (possible plateau).")
    if plateau(bad_rewards):
        print("\nNOTICE: Bad policy rewards show low variance (possible plateau).")

    # Detect reward hacking by keyword-stuffing giving higher than good
    if statistics.mean(stuffed_rewards) >= statistics.mean(good_rewards):
        print("\nALERT: Keyword-stuffing achieves rewards >= good policy — potential reward hacking.")

    print("\nDone.")

    # Also write a concise summary to a local file as a fallback for capturing results
    try:
        with open("reward_validation_internal.txt", "w", encoding="utf-8") as fh:
            fh.write("=== Reward validation summary\n")
            fh.write(f"n_episodes={len(prompts)}\n")
            fh.write(f"mean_good={mean_good:.6f}\n")
            fh.write(f"mean_bad={mean_bad:.6f}\n")
            fh.write(f"mean_clarify={mean_clar:.6f}\n")
            fh.write(f"mean_stuffed={mean_stuffed:.6f}\n")
            fh.write(f"deterministic_ok={deterministic_ok}\n")
    except Exception:
        pass

    # Explicit end marker
    print("VALIDATION_RUN_END", flush=True)


if __name__ == '__main__':
    main()
