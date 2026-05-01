from __future__ import annotations

import json
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from triagerl.env.triage_env import MedicalTriageEnv
from triagerl.tasks.loader import get_task, load_all_tasks
from triagerl.core.models import TriageAction


def main() -> None:
    task_map, task_ids = load_all_tasks()
    task = get_task(task_ids[0])
    env = MedicalTriageEnv(task_config=task, clarify_penalty_dampening=1.0)
    obs = env.reset(seed=123)
    action = TriageAction(
        action_type="classify",
        esi_level=int(task.esi_correct),
        reasoning="Clinical reasoning with enough detail to validate the terminal path.",
        recommended_actions=list(task.expected_actions or [])[:2],
        confidence=0.9,
    )
    next_obs, reward, done, info = env.step(action)
    print(json.dumps({
        "reset_task_ref": obs.task_ref,
        "step_number": obs.step_number,
        "reward": reward,
        "done": done,
        "workflow_phase": info.get("workflow_phase") if isinstance(info, dict) else None,
        "next_step_number": next_obs.step_number,
    }, default=str))


if __name__ == "__main__":
    main()
