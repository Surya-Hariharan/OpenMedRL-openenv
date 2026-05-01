from pathlib import Path
import json
import sys
repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from triagerl.tasks.loader import load_all_tasks, get_task
from triagerl.training.dataset import TriageEpisodeDataset, extract_task_id_from_prompt
from triagerl.training.reward_fn import _score_completion
from triagerl.env.triage_env import MedicalTriageEnv


def _env_factory(task_id: str) -> MedicalTriageEnv:
    return MedicalTriageEnv(task_config=get_task(task_id), clarify_penalty_dampening=1.0)


def make_good_completion(prompt: str, task_map) -> str:
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
    return json.dumps(action)


def make_stuffed_completion(prompt: str, task_map) -> str:
    tid = extract_task_id_from_prompt(prompt)
    task = task_map[tid]
    kws = (task.key_reasoning_keywords or [])[:4]
    reasoning = " ".join([" ".join(kws)] * 30)
    action = {
        "action_type": "classify",
        "esi_level": int(task.esi_correct),
        "reasoning": reasoning,
        "recommended_actions": (task.expected_actions or [])[:2],
        "confidence": 0.9,
    }
    return json.dumps(action)


task_map, task_ids = load_all_tasks()
dataset = TriageEpisodeDataset(task_ids=task_ids, n_samples=1, global_seed=2026, env_factory=_env_factory, include_mid_episode=True)
dataset.refresh(epoch=0)
prompt = dataset._rows[0]['prompt']

for label, completion_fn in [('good', make_good_completion), ('stuffed', make_stuffed_completion)]:
    completion = completion_fn(prompt, task_map)
    reward, err, components = _score_completion(prompt, completion, task_map, _env_factory)
    print(label, 'reward=', reward, 'err=', err)
    print(label, 'components=', components)
