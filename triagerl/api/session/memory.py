from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Any, Dict


class InMemorySessionStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: Dict[str, Any] = {}

    def get(self, session_id: str, default: Any = None) -> Any:
        with self._lock:
            return deepcopy(self._sessions.get(session_id, default))

    def set(self, session_id: str, value: Any) -> None:
        with self._lock:
            self._sessions[session_id] = deepcopy(value)

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
