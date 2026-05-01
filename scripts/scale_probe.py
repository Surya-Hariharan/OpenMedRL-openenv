from __future__ import annotations

import sys
from pathlib import Path
repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from triagerl.env.triage_env import MedicalTriageEnv
from triagerl.tasks.loader import load_all_tasks, get_task
from triagerl.training.dataset import TriageEpisodeDataset, extract_task_id_from_prompt


def env_factory(task_id: str) -> MedicalTriageEnv:
    return MedicalTriageEnv(task_config=get_task(task_id), clarify_penalty_dampening=1.0)


def main() -> None:
    task_map, task_ids = load_all_tasks()
    print("RESET_100_START")
    env = env_factory(task_ids[0])
    ok = 0
    for i in range(100):
        try:
            obs = env.reset(seed=i)
            if obs.step_number >= 1 and obs.task_ref:
                ok += 1
        except Exception as exc:
            print("RESET_FAIL", i, type(exc).__name__, str(exc))
            break
    print("RESET_100_OK", ok)

    print("DATASET_1000_START")
    ds = TriageEpisodeDataset(task_ids=task_ids, n_samples=1000, global_seed=2026, env_factory=env_factory, include_mid_episode=True)
    ds.refresh(epoch=0)
    rows = list(ds._rows)
    prompts_ok = all(extract_task_id_from_prompt(r["prompt"]) in task_map for r in rows)
    uniq = len({r["prompt"] for r in rows})
    print("DATASET_1000", len(rows), uniq, prompts_ok)


if __name__ == "__main__":
    main()
