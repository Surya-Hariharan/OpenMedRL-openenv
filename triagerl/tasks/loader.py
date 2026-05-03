from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from pydantic import ValidationError

from triagerl.logs.logger import get_logger
from triagerl.tasks.schema import TaskConfig

logger = get_logger(__name__)


class _TaskLoader:
    def __init__(self) -> None:
        self._cache: Dict[str, TaskConfig] = {}
        self._ids: List[str] = []
        self._by_category: Dict[str, List[str]] = {}
        self._by_difficulty: Dict[str, List[str]] = {}
        self._load()

    def _corpus_path(self) -> Path:
        return Path(__file__).resolve().parent / "corpus" / "tasks.yaml"

    def _load(self) -> None:
        corpus_path = self._corpus_path()
        if not corpus_path.exists():
            raise FileNotFoundError(f"Task corpus not found: {corpus_path}")

        logger.info("loading_tasks_from_file", path=str(corpus_path))
        try:
            data = yaml.safe_load(corpus_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            logger.error("yaml_parse_error", error=str(exc))
            raise ValueError(f"Failed to parse YAML: {exc}") from exc

        if not isinstance(data, dict) or "tasks" not in data:
            raise ValueError("YAML must have a top-level 'tasks' list")

        for raw in data["tasks"]:
            try:
                cfg = TaskConfig.model_validate(raw)
            except ValidationError as exc:
                logger.error("task_validation_failed", task_id=raw.get("id", "?"), error=str(exc))
                raise ValueError(f"Task {raw.get('id')} failed validation: {exc}") from exc

            self._cache[cfg.id] = cfg
            self._ids.append(cfg.id)
            self._by_category.setdefault(cfg.category, []).append(cfg.id)
            self._by_difficulty.setdefault(cfg.difficulty, []).append(cfg.id)

        task_aliases = {
            "classic-mi": "classic-stemi",
            "meningitis-suspect": "meningococcal-meningitis",
            "masked-sepsis": "masked-urosepsis",
        }

        for alias, canonical in task_aliases.items():
            if canonical in self._cache and alias not in self._cache:
                self._cache[alias] = self._cache[canonical]
                self._ids.append(alias)

        logger.info(
            "tasks_loaded",
            count=len(self._cache),
            categories=list(self._by_category.keys()),
            difficulties=list(self._by_difficulty.keys()),
        )

    def get(self, task_id: str) -> TaskConfig:
        if task_id not in self._cache:
            raise KeyError(f"Unknown task_id '{task_id}'. Available: {self._ids}")
        return self._cache[task_id]

    def all(self) -> Tuple[Dict[str, TaskConfig], List[str]]:
        return dict(self._cache), list(self._ids)

    def sample(
        self,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        exclude: Optional[List[str]] = None,
        rng: Optional[random.Random] = None,
    ) -> TaskConfig:
        pool = list(self._ids)
        if category:
            pool = [t for t in pool if self._cache[t].category == category]
        if difficulty:
            pool = [t for t in pool if self._cache[t].difficulty == difficulty]
        if exclude:
            pool = [t for t in pool if t not in exclude]
        if not pool:
            raise ValueError(f"No tasks match category={category} difficulty={difficulty}")
        r = rng or random
        return self._cache[r.choice(pool)]

    def next(self, current_id: Optional[str]) -> TaskConfig:
        if current_id is None or current_id not in self._cache:
            return self._cache[self._ids[0]]
        idx = self._ids.index(current_id)
        return self._cache[self._ids[(idx + 1) % len(self._ids)]]


_loader = _TaskLoader()


def get_task(task_id: str) -> TaskConfig:
    return _loader.get(task_id)


def get_task_list() -> List[str]:
    _, ids = _loader.all()
    return ids


def get_next_task(current_task_id: Optional[str]) -> TaskConfig:
    return _loader.next(current_task_id)


def load_all_tasks() -> Tuple[Dict[str, TaskConfig], List[str]]:
    return _loader.all()


def sample_task(
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    exclude: Optional[List[str]] = None,
    rng: Optional[random.Random] = None,
) -> TaskConfig:
    return _loader.sample(category=category, difficulty=difficulty, exclude=exclude, rng=rng)


def get_curriculum_batch(
    batch_size: int,
    difficulty_weights: Optional[Dict[str, float]] = None,
    rng: Optional[random.Random] = None,
) -> List[TaskConfig]:
    if difficulty_weights is None:
        difficulty_weights = {"easy": 0.2, "medium": 0.5, "hard": 0.3}

    r = rng or random
    difficulties = list(difficulty_weights.keys())
    weights = [difficulty_weights[d] for d in difficulties]
    batch = []
    for _ in range(batch_size):
        chosen_difficulty = r.choices(difficulties, weights=weights, k=1)[0]
        try:
            task = _loader.sample(difficulty=chosen_difficulty, rng=r)
            batch.append(task)
        except ValueError:
            batch.append(_loader.sample(rng=r))
    return batch
