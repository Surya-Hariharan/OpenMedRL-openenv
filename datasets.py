"""Lightweight shim for HuggingFace `datasets.Dataset.from_list` used
in offline validation. This file is only intended for local validation and
testing when the `datasets` package is not installed. It should not be
used in production training runs that rely on the real library.
"""
from typing import List, Dict, Any


class Dataset:
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    @classmethod
    def from_list(cls, rows: List[Dict[str, Any]]):
        return cls(rows)

    def __len__(self):
        return len(self._rows)

    def to_list(self):
        return list(self._rows)
