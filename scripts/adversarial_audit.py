from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from triagerl.env.triage_env import MedicalTriageEnv
from triagerl.tasks.loader import get_task, load_all_tasks
from triagerl.training.dataset import TriageEpisodeDataset, extract_task_id_from_prompt
from triagerl.training.reward_fn import build_reward_fn, _score_completion
from triagerl.core.models import TriageAction


def env_factory(task_id: str) -> MedicalTriageEnv:
    return MedicalTriageEnv(task_config=get_task(task_id), clarify_penalty_dampening=1.0)


def make_json(action: dict[str, Any]) -> str:
    return json.dumps(action)


def make_good(prompt: str, task_map: dict[str, Any]) -> str:
    tid = extract_task_id_from_prompt(prompt)
    task = task_map[tid]
    reasoning = ("because " * 20) + ("clinical reasoning. " * 20)
    return make_json({
        "action_type": "classify",
        "esi_level": int(task.esi_correct),
        "reasoning": reasoning,
        "recommended_actions": list(task.expected_actions or [])[:4],
        "confidence": 0.95,
    })


def make_stuffed(prompt: str, task_map: dict[str, Any]) -> str:
    tid = extract_task_id_from_prompt(prompt)
    task = task_map[tid]
    kws = list(task.key_reasoning_keywords or [])[:4]
    reasoning = " ".join([" ".join(kws)] * 30)
    return make_json({
        "action_type": "classify",
        "esi_level": int(task.esi_correct),
        "reasoning": reasoning,
        "recommended_actions": list(task.expected_actions or [])[:2],
        "confidence": 0.9,
    })


def make_bad(prompt: str, task_map: dict[str, Any]) -> str:
    tid = extract_task_id_from_prompt(prompt)
    task = task_map[tid]
    pred = min(5, int(task.esi_correct) + 2)
    return make_json({
        "action_type": "classify",
        "esi_level": pred,
        "reasoning": "brief",
        "recommended_actions": [],
        "confidence": 0.2,
    })


def probe_reward_fallbacks(prompt: str, task_map: dict[str, Any], reward_fn):
    cases = {
        "malformed": "{",
        "empty": "",
        "clarify": make_json({"action_type": "clarify", "clarifying_question": "Any more details?", "reasoning": "", "recommended_actions": [], "confidence": 0.5}),
        "good": make_good(prompt, task_map),
        "stuffed": make_stuffed(prompt, task_map),
        "bad": make_bad(prompt, task_map),
        "long": make_good(prompt, task_map)[:-1] + ',"reasoning":"' + ('x ' * 5000) + '"}',
    }
    print("REWARD_CASES")
    for name, completion in cases.items():
        try:
            reward = reward_fn([prompt], [completion])[0]
            print(name, reward)
        except Exception as exc:
            print(name, "EXC", type(exc).__name__, str(exc))


def probe_env_resets(task_map: dict[str, Any]):
    task_id = next(iter(task_map.keys()))
    env = env_factory(task_id)
    results = []
    for i in range(20):
        try:
            obs = env.reset(seed=i)
            results.append((obs.task_ref, obs.step_number, obs.max_steps))
        except Exception as exc:
            results.append(("EXC", type(exc).__name__, str(exc)))
    print("RESET_RESULTS", results[:5])
    print("RESET_COUNT", len(results))


def probe_max_steps(task_map: dict[str, Any]):
    task_id = next(iter(task_map.keys()))
    env = env_factory(task_id)
    obs = env.reset(seed=123)
    actions = []
    done = False
    info = None
    reward = None
    for _ in range(getattr(obs, "max_steps", 8) + 3):
        try:
            action = TriageAction(
                action_type="clarify",
                clarifying_question="Any more details?",
                reasoning="Need more information.",
                recommended_actions=[],
                confidence=0.5,
            )
            obs, reward, done, info = env.step(action)
            actions.append((reward, done, info.get("workflow_phase") if isinstance(info, dict) else None))
            if done:
                break
        except Exception as exc:
            actions.append(("EXC", type(exc).__name__, str(exc)))
            break
    print("MAX_STEPS_TRACE", actions[-5:])


def probe_dataset(task_map: dict[str, Any]):
    task_ids = list(task_map.keys())
    ds = TriageEpisodeDataset(task_ids=task_ids, n_samples=120, global_seed=2026, env_factory=env_factory, include_mid_episode=True)
    ds.refresh(epoch=0)
    rows = list(ds._rows)
    task_ids_extracted = [extract_task_id_from_prompt(r["prompt"]) for r in rows]
    prompt_ok = all(tid in task_map for tid in task_ids_extracted)
    duplicate_prompts = len({r["prompt"] for r in rows}) != len(rows)
    print("DATASET", {
        "n": len(rows),
        "prompt_ok": prompt_ok,
        "duplicates": duplicate_prompts,
        "missing_task_ids": sum(1 for tid in task_ids_extracted if tid is None),
    })


def probe_determinism(task_map: dict[str, Any], reward_fn):
    task_ids = list(task_map.keys())
    ds = TriageEpisodeDataset(task_ids=task_ids, n_samples=20, global_seed=2026, env_factory=env_factory, include_mid_episode=True)
    ds.refresh(epoch=0)
    prompt = ds._rows[0]["prompt"]
    good = make_good(prompt, task_map)
    r1 = reward_fn([prompt], [good])[0]
    r2 = reward_fn([prompt], [good])[0]
    print("DETERMINISM", r1, r2, r1 == r2)


def main() -> None:
    print("AUDIT_START")
    task_map, _ = load_all_tasks()
    reward_fn = build_reward_fn(task_map=task_map, env_factory=env_factory, log_components=False)
    task_ids = list(task_map.keys())
    ds = TriageEpisodeDataset(task_ids=task_ids, n_samples=12, global_seed=2026, env_factory=env_factory, include_mid_episode=True)
    ds.refresh(epoch=0)
    prompt = ds._rows[0]["prompt"]

    probe_reward_fallbacks(prompt, task_map, reward_fn)
    probe_env_resets(task_map)
    probe_max_steps(task_map)
    probe_dataset(task_map)
    probe_determinism(task_map, reward_fn)
    print("AUDIT_END")


if __name__ == "__main__":
    main()
