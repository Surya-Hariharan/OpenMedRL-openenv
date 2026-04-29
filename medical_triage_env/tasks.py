from __future__ import annotations

from triagerl.tasks.loader import (
    _TaskLoader,
    get_curriculum_batch,
    get_next_task,
    get_task,
    get_task_list,
    load_all_tasks,
    sample_task,
)
from triagerl.tasks.schema import HiddenInfoItem, LabResult, PatientInfo, TaskConfig, VitalDrift

TASK_LIST = get_task_list()

__all__ = [
    "TaskConfig",
    "HiddenInfoItem",
    "VitalDrift",
    "LabResult",
    "PatientInfo",
    "_TaskLoader",
    "get_task",
    "get_task_list",
    "get_next_task",
    "load_all_tasks",
    "sample_task",
    "get_curriculum_batch",
    "TASK_LIST",
]
