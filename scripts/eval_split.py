from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional, Tuple

from triagerl.env.triage_env import MedicalTriageEnv
from triagerl.core.models import TriageAction, TriageObservation
from triagerl.tasks.loader import load_all_tasks


@dataclass
class EpisodeResult:
    task_id: str
    steps: int
    terminal_reward: float
    episode_return_sum: float
    classified: bool
    undertriage: bool
    overtriage: bool


class HackathonBaselinePolicy:
    """
    Deterministic baseline that is:
    - non-cheating (doesn't read correct_esi)
    - stable/reproducible for demos
    - good enough to show metric movement when you improve prompts/models
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def __call__(self, obs: TriageObservation) -> TriageAction:
        # If we haven't revealed anything yet and we still have budget, ask a focused question.
        if (not obs.additional_info_revealed) and obs.step_number <= 2:
            return TriageAction(
                action_type="clarify",
                clarifying_question="Any key past history, medications/anticoagulants, and any repeat vitals or focused exam findings?",
                reasoning="Collecting missing high-yield data to avoid unsafe triage errors.",
                recommended_actions=[],
                confidence=0.45,
            )

        v = obs.patient.vitals
        sbp = v.blood_pressure_systolic
        rr = v.respiratory_rate
        spo2 = v.oxygen_saturation
        gcs = v.gcs
        temp = v.temperature
        hr = v.heart_rate

        shock = (sbp is not None and sbp < 90) or (gcs is not None and gcs <= 12)
        severe_hypoxia = (spo2 is not None and spo2 < 88)
        resp_distress = (rr is not None and rr >= 30) or (spo2 is not None and spo2 < 92)
        fever_toxic = (temp is not None and temp >= 39.0 and hr is not None and hr >= 115)

        if shock or severe_hypoxia:
            esi = 1
            conf = 0.65
        elif resp_distress or fever_toxic or obs.deterioration_signal >= 0.6:
            esi = 2
            conf = 0.6
        elif obs.deterioration_signal >= 0.35:
            esi = 3
            conf = 0.55
        else:
            # Default to moderate/low acuity split
            esi = 3 if self.rng.random() < 0.7 else 4
            conf = 0.52

        return TriageAction(
            action_type="classify",
            esi_level=esi,
            reasoning="Decision based on vitals, oxygenation, neurologic status, and deterioration trend.",
            recommended_actions=["continuous monitoring", "senior review if worsening"],
            confidence=conf,
        )


def run_episode(task_id: str, policy: HackathonBaselinePolicy, seed: Optional[int] = None) -> EpisodeResult:
    env = MedicalTriageEnv(task_id, seed=seed)
    obs = env.reset(seed=seed)

    total = 0.0
    terminal = 0.0

    while True:
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        total += float(reward)
        if done:
            terminal = float(reward)
            ep = info.get("episode_metrics", {}) or {}
            return EpisodeResult(
                task_id=task_id,
                steps=int(info.get("step", env.current_step)),
                terminal_reward=terminal,
                episode_return_sum=total,
                classified=bool(ep.get("classification_made", False)),
                undertriage=bool(ep.get("undertriage", False)),
                overtriage=bool(ep.get("overtriage", False)),
            )


def split_tasks(task_ids: List[str], seed: int, test_n: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    ids = list(task_ids)
    rng.shuffle(ids)
    test = ids[: max(0, min(test_n, len(ids)))]
    train = ids[len(test) :]
    return train, test


def summarize(results: List[EpisodeResult]) -> Dict[str, float]:
    if not results:
        return {}
    return {
        "episodes": float(len(results)),
        "mean_terminal_reward": round(mean(r.terminal_reward for r in results), 4),
        "mean_episode_return_sum": round(mean(r.episode_return_sum for r in results), 4),
        "undertriage_rate": round(mean(1.0 if r.undertriage else 0.0 for r in results), 4),
        "overtriage_rate": round(mean(1.0 if r.overtriage else 0.0 for r in results), 4),
        "classification_rate": round(mean(1.0 if r.classified else 0.0 for r in results), 4),
        "mean_steps": round(mean(r.steps for r in results), 3),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hackathon-ready deterministic eval split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-n", type=int, default=5)
    p.add_argument("--episodes-per-task", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks, task_ids = load_all_tasks()
    train_ids, test_ids = split_tasks(task_ids, seed=args.seed, test_n=args.test_n)

    policy = HackathonBaselinePolicy(seed=args.seed)

    def run_split(ids: List[str]) -> List[EpisodeResult]:
        out: List[EpisodeResult] = []
        for tid in ids:
            for k in range(args.episodes_per_task):
                out.append(run_episode(tid, policy=policy, seed=(args.seed * 1000 + k)))
        return out

    train_res = run_split(train_ids)
    test_res = run_split(test_ids)

    payload = {
        "train": summarize(train_res),
        "test": summarize(test_res),
        "split": {"train": train_ids, "test": test_ids},
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

