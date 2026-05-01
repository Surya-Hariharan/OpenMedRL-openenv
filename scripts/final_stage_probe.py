from __future__ import annotations

import json
import random
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from triagerl.core.models import TriageAction
from triagerl.env.triage_env import MedicalTriageEnv
from triagerl.tasks.loader import get_task, load_all_tasks
from triagerl.training.dataset import TriageEpisodeDataset, extract_task_id_from_prompt
from triagerl.training.reward_fn import build_reward_fn


def env_factory(task_id: str) -> MedicalTriageEnv:
    return MedicalTriageEnv(task_config=get_task(task_id), clarify_penalty_dampening=1.0)


def classify_json(esi: int, reasoning: str, task):
    return json.dumps({
        "action_type": "classify",
        "esi_level": esi,
        "reasoning": reasoning,
        "recommended_actions": list(task.expected_actions or [])[:4],
        "confidence": 0.9,
    })


def clarify_json(text: str):
    return json.dumps({
        "action_type": "clarify",
        "clarifying_question": text,
        "reasoning": "",
        "recommended_actions": [],
        "confidence": 0.5,
    })


def main() -> None:
    task_map, task_ids = load_all_tasks()
    reward_fn = build_reward_fn(task_map=task_map, env_factory=env_factory, log_components=False)
    ds = TriageEpisodeDataset(task_ids=task_ids, n_samples=12, global_seed=2026, env_factory=env_factory, include_mid_episode=True)
    ds.refresh(epoch=0)
    prompt = ds._rows[0]["prompt"]
    tid = extract_task_id_from_prompt(prompt)
    task = task_map[tid]

    print("EDGE_REWARDS")
    cases = {
        "good": classify_json(int(task.esi_correct), ("because " * 10) + ("clinical reasoning. " * 20), task),
        "esi2": classify_json(2, "brief reasoning", task),
        "stuffed": classify_json(int(task.esi_correct), " ".join((task.key_reasoning_keywords or [])[:4] * 30), task),
        "minimal": classify_json(int(task.esi_correct), "ok", task),
        "long": classify_json(int(task.esi_correct), "clinical " * 4000, task),
        "badjson": "{",
        "empty": "",
        "clarify": clarify_json("Can you provide more details?"),
    }
    for name, completion in cases.items():
        try:
            reward = reward_fn([prompt], [completion])[0]
            print(name, reward)
        except Exception as exc:
            print(name, "EXC", type(exc).__name__, str(exc))

    print("ENV_INVALID_ACTIONS")
    env = env_factory(task_ids[0])
    obs = env.reset(seed=123)
    print("reset", obs.task_ref, obs.step_number, obs.max_steps)
    try:
        env.step(TriageAction(action_type="clarify", clarifying_question="Any more details?", reasoning="Need more.", recommended_actions=[], confidence=0.5))
        print("clarify_step_ok")
    except Exception as exc:
        print("clarify_step_exc", type(exc).__name__, str(exc))
    try:
        env.step(TriageAction(action_type="classify", esi_level=2, reasoning="brief", recommended_actions=[], confidence=0.9))
        print("classify_step_ok")
    except Exception as exc:
        print("classify_step_exc", type(exc).__name__, str(exc))
    try:
        env.step(TriageAction(action_type="invalid", reasoning="x", recommended_actions=[], confidence=0.1))
        print("invalid_action_ok")
    except Exception as exc:
        print("invalid_action_exc", type(exc).__name__, str(exc))

    print("STRESS_RESET")
    env = env_factory(task_ids[0])
    ok = 0
    for i in range(100):
        try:
            o = env.reset(seed=i)
            if o.step_number >= 1 and o.task_ref:
                ok += 1
        except Exception as exc:
            print("reset_exc", i, type(exc).__name__, str(exc))
            break
    print("reset_ok", ok)

    print("PIPELINE_CHECK")
    try:
        env = env_factory(tid)
        obs = env.reset(seed=42)
        print("pipeline_reset_ok", obs.task_ref, obs.step_number)
    except Exception as exc:
        print("pipeline_reset_exc", type(exc).__name__, str(exc))

    print("DONE")


if __name__ == "__main__":
    main()
