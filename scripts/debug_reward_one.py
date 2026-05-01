from pathlib import Path
import sys
repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from triagerl.tasks.loader import load_all_tasks
from triagerl.training.dataset import TriageEpisodeDataset, extract_task_id_from_prompt
from triagerl.training.reward_fn import _score_completion
from triagerl.env.triage_env import MedicalTriageEnv

# env factory
from triagerl.tasks.loader import get_task

def _env_factory(task_id: str) -> MedicalTriageEnv:
    task = get_task(task_id)
    return MedicalTriageEnv(task_config=task, clarify_penalty_dampening=1.0)

# prepare dataset
task_map, task_ids = load_all_tasks()
dataset = TriageEpisodeDataset(task_ids=task_ids, n_samples=4, global_seed=2026, env_factory=_env_factory, include_mid_episode=True)
dataset.refresh(epoch=0)
rows = list(dataset._rows)

prompt = rows[0]['prompt']
print('Prompt snippet:', prompt[:200])

# make a good completion
import json
from triagerl.tasks.loader import get_task
from triagerl.training.dataset import extract_task_id_from_prompt

tid = extract_task_id_from_prompt(prompt)
task = task_map[tid]
reasoning = ("".join([k + " " for k in (task.key_reasoning_keywords or [])[:6]]) + ("Detailed clinical reasoning. " * 40))
action = {
    "action_type": "classify",
    "esi_level": int(task.esi_correct),
    "reasoning": reasoning,
    "recommended_actions": (task.expected_actions or [])[:4],
    "confidence": 0.95,
}
completion = json.dumps(action)

print('Scoring...')
reward, err, comp = _score_completion(prompt, completion, task_map, _env_factory)
print('result:', reward, err)
print('components:', comp)
