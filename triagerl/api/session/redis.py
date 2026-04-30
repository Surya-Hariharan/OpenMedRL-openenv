from __future__ import annotations

import pickle
from typing import Any

try:
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None


class RedisSessionStore:
    def __init__(self, url: str, *, key_prefix: str = "triagerl:session:") -> None:
        if redis is None:
            raise RuntimeError("redis package is required for RedisSessionStore")
        self._client = redis.Redis.from_url(url)
        self._key_prefix = key_prefix

    def _key(self, session_id: str) -> str:
        return f"{self._key_prefix}{session_id}"

    def get(self, session_id: str, default: Any = None) -> Any:
        raw_value = self._client.get(self._key(session_id))
        if raw_value is None:
            return default
        return pickle.loads(raw_value)

    def set(self, session_id: str, value: Any) -> None:
        self._client.set(self._key(session_id), pickle.dumps(value))

    def delete(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))

    def clear(self) -> None:
        keys = list(self._client.scan_iter(match=f"{self._key_prefix}*"))
        if keys:
            self._client.delete(*keys)
