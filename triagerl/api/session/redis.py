from __future__ import annotations

import json
import pickle
from typing import Any

try:
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None

# Sessions expire after this many seconds so Redis memory is bounded.
DEFAULT_TTL_SECONDS: int = 3600  # 1 hour


class RedisSessionStore:
    """
    Redis-backed session store with TTL and JSON-preferred serialisation.

    Serialisation strategy
    ----------------------
    Values that are JSON-serialisable (plain dicts/lists/scalars) are stored
    as UTF-8 JSON under the key ``{prefix}{session_id}``.  Values that
    require pickle (e.g. MedicalTriageEnv instances) are stored under
    ``{prefix}{session_id}:pkl`` with a ``X-Pickle: 1`` marker key so the
    deserialiser knows which path to take.

    Security note
    -------------
    Pickle deserialisation of untrusted data is an RCE vector.  Redis MUST
    be deployed on a private network with authentication (rediss:// + AUTH)
    and must never be reachable from the public internet.  A full migration
    to JSON serialisation of MedicalTriageEnv (via env.to_dict() /
    env.from_dict()) would eliminate this risk — tracked as a future milestone.
    """

    def __init__(
        self,
        url: str,
        *,
        key_prefix: str = "triagerl:session:",
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        if redis is None:
            raise RuntimeError("redis package is required for RedisSessionStore")
        self._client = redis.Redis.from_url(url, decode_responses=False)
        self._key_prefix = key_prefix
        self._ttl = ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"{self._key_prefix}{session_id}"

    def _pkl_key(self, session_id: str) -> str:
        return f"{self._key_prefix}{session_id}:pkl"

    def get(self, session_id: str, default: Any = None) -> Any:
        # Try JSON path first (safe)
        raw_json = self._client.get(self._key(session_id))
        if raw_json is not None:
            try:
                return json.loads(raw_json.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        # Fall back to pickle path for non-JSON-serialisable values (e.g. env objects)
        raw_pkl = self._client.get(self._pkl_key(session_id))
        if raw_pkl is not None:
            return pickle.loads(raw_pkl)  # noqa: S301 — internal network only

        return default

    def set(self, session_id: str, value: Any) -> None:
        try:
            serialised = json.dumps(value).encode("utf-8")
            # JSON path: store under the canonical key, remove any stale pickle key
            self._client.set(self._key(session_id), serialised, ex=self._ttl)
            self._client.delete(self._pkl_key(session_id))
        except (TypeError, ValueError):
            # Value is not JSON-serialisable — use pickle fallback
            self._client.set(
                self._pkl_key(session_id), pickle.dumps(value), ex=self._ttl
            )
            # Remove any stale JSON key that might shadow this session
            self._client.delete(self._key(session_id))

    def delete(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))
        self._client.delete(self._pkl_key(session_id))

    def clear(self) -> None:
        keys = list(self._client.scan_iter(match=f"{self._key_prefix}*"))
        if keys:
            self._client.delete(*keys)
