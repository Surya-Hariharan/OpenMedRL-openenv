from __future__ import annotations

import os
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from .memory import InMemorySessionStore
from .redis import RedisSessionStore


@runtime_checkable
class SessionStore(Protocol):
    def get(self, session_id: str, default: Any = None) -> Any:
        ...

    def set(self, session_id: str, value: Any) -> None:
        ...

    def delete(self, session_id: str) -> None:
        ...

    def clear(self) -> None:
        ...


_STORE_LOCK = RLock()
_STORE: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _STORE

    with _STORE_LOCK:
        if _STORE is None:
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                _STORE = RedisSessionStore(redis_url)
            else:
                _STORE = InMemorySessionStore()
        return _STORE


def reset_session_store() -> None:
    global _STORE

    with _STORE_LOCK:
        _STORE = None
