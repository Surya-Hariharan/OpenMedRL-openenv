from __future__ import annotations

import time
from copy import deepcopy
from threading import RLock
from typing import Any, Dict, Optional, Tuple

# Sessions expire after this many seconds of inactivity to prevent unbounded growth.
DEFAULT_TTL_SECONDS: int = 3600  # 1 hour


class InMemorySessionStore:
    """
    Thread-safe in-process session store with TTL eviction.

    Each set() call resets the TTL for that session.  Expired sessions are
    evicted lazily on get() and proactively on every set() call (the eviction
    pass is O(n) but sessions are typically short-lived and small in number).
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._lock = RLock()
        self._sessions: Dict[str, Tuple[Any, float]] = {}  # value, expiry_timestamp
        self._ttl = ttl_seconds

    def _is_expired(self, expiry: float) -> bool:
        return time.monotonic() > expiry

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [sid for sid, (_, exp) in self._sessions.items() if now > exp]
        for sid in expired:
            del self._sessions[sid]

    def get(self, session_id: str, default: Any = None) -> Any:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return default
            value, expiry = entry
            if self._is_expired(expiry):
                del self._sessions[session_id]
                return default
            return deepcopy(value)

    def set(self, session_id: str, value: Any) -> None:
        with self._lock:
            self._evict_expired()
            expiry = time.monotonic() + self._ttl
            self._sessions[session_id] = (deepcopy(value), expiry)

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    @property
    def active_count(self) -> int:
        """Number of non-expired sessions currently held."""
        with self._lock:
            now = time.monotonic()
            return sum(1 for _, (_, exp) in self._sessions.items() if now <= exp)
