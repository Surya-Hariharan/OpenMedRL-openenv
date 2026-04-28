"""Session lifecycle management for medical triage episodes."""
from __future__ import annotations

import asyncio
import os
import pickle
import threading
import time
from typing import Dict, List, Optional, Protocol, TYPE_CHECKING

from fastapi import HTTPException

from .logs import get_logger

if TYPE_CHECKING:
    from .env import MedicalTriageEnv

logger = get_logger(__name__)


class SessionStore(Protocol):
    def create(self, env: "MedicalTriageEnv") -> str: ...
    def get(self, session_id: str) -> "MedicalTriageEnv": ...
    def save(self, env: "MedicalTriageEnv") -> None: ...
    def destroy(self, session_id: str) -> None: ...
    def list_active(self) -> List[str]: ...
    def clear(self) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


class InMemorySessionStore:
    """In-memory episode sessions with TTL eviction.

    NOTE: This is per-process memory, so it is unsafe with multiple workers.
    Use `RedisSessionStore` in multi-worker deployments.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, MedicalTriageEnv] = {}
        self._last_access: Dict[str, float] = {}
        self._ttl_seconds: int = 30 * 60
        self._max_sessions: int = 200
        self._sweep_task: asyncio.Task | None = None
        self._lock = threading.RLock()

        logger.info(
            "session_manager_initialized",
            ttl_seconds=self._ttl_seconds,
            max_sessions=self._max_sessions,
        )

    def create(self, env: "MedicalTriageEnv") -> str:
        """Register a new environment and return its session id."""
        session_id = env.session_id
        with self._lock:
            self._evict_expired_sessions_locked()
            self._sessions[session_id] = env
            self._last_access[session_id] = time.time()
            self._evict_lru_if_needed_locked()
        logger.info(
            "session_created",
            session_id=session_id,
            task_id=env.task_id,
            active_sessions=len(self._sessions),
        )
        return session_id

    def get(self, session_id: str) -> "MedicalTriageEnv":
        with self._lock:
            self._evict_expired_sessions_locked()
            env = self._sessions.get(session_id)
            if env is None:
                logger.warning("session_not_found", session_id=session_id)
                raise HTTPException(
                    status_code=404,
                    detail=f"Session {session_id} not found or expired",
                )
            self._last_access[session_id] = time.time()
            return env

    def save(self, env: "MedicalTriageEnv") -> None:
        """Persist updated env state (no-op beyond in-memory overwrite)."""
        session_id = env.session_id
        with self._lock:
            self._evict_expired_sessions_locked()
            if session_id not in self._sessions:
                # Mirror semantics of get(): updating a missing session is a 404.
                logger.warning("session_not_found_on_save", session_id=session_id)
                raise HTTPException(
                    status_code=404,
                    detail=f"Session {session_id} not found or expired",
                )
            self._sessions[session_id] = env
            self._last_access[session_id] = time.time()

    def destroy(self, session_id: str) -> None:
        with self._lock:
            env = self._sessions.pop(session_id, None)
            self._last_access.pop(session_id, None)
        if env is not None:
            logger.info(
                "session_destroyed",
                session_id=session_id,
                task_id=env.task_id,
            )

    def list_active(self) -> List[str]:
        with self._lock:
            return list(self._sessions.keys())

    def clear(self) -> None:
        """Clear all sessions and access timestamps (primarily for tests)."""
        with self._lock:
            self._sessions.clear()
            self._last_access.clear()

    def _evict_lru_if_needed_locked(self) -> None:
        """Evict least recently used sessions if capacity is exceeded."""
        over = len(self._sessions) - self._max_sessions
        if over <= 0:
            return
        ordered = sorted(self._last_access.items(), key=lambda kv: kv[1])
        for session_id, _ in ordered[:over]:
            self._sessions.pop(session_id, None)
            self._last_access.pop(session_id, None)
            logger.info("session_evicted_lru", session_id=session_id)

    def _evict_expired_sessions_locked(self) -> None:
        now = time.time()
        expired = [
            session_id
            for session_id, last_seen in self._last_access.items()
            if now - last_seen > self._ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)
            self._last_access.pop(session_id, None)
        for session_id in expired:
            logger.info("session_expired", session_id=session_id, ttl_seconds=self._ttl_seconds)

    async def _ttl_sweep_loop(self) -> None:
        interval_seconds = 300
        logger.info("ttl_sweep_started", interval_seconds=interval_seconds)
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                # Avoid blocking the event loop with a synchronous lock.
                # Note: `start()` is a no-op for in-memory store; kept for backward compatibility.
                await asyncio.to_thread(self._evict_expired_sessions_locked)
            except asyncio.CancelledError:
                logger.info("ttl_sweep_cancelled")
                break
            except Exception as exc:
                logger.error("ttl_sweep_error", error=str(exc), error_type=type(exc).__name__)

    def start(self) -> None:
        # No background sweep: avoids duplicate/ghost tasks under reload.
        return None

    def stop(self) -> None:
        return None


class RedisSessionStore:
    """Redis-backed session store shared across processes/workers.

    Stores a pickled `MedicalTriageEnv` blob under a session key, with Redis TTL.
    This avoids per-process singleton issues under Uvicorn/Gunicorn multi-worker.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        key_prefix: str = "medical-triage-env:session:",
        ttl_seconds: int = 30 * 60,
    ) -> None:
        try:
            import redis  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "RedisSessionStore requires the 'redis' package. "
                "Install it (e.g. `pip install redis`) or disable REDIS_URL."
            ) from exc

        self._redis = redis.Redis.from_url(redis_url, decode_responses=False)
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds

        logger.info(
            "redis_session_store_initialized",
            ttl_seconds=self._ttl_seconds,
            key_prefix=self._key_prefix,
        )

    def _key(self, session_id: str) -> str:
        return f"{self._key_prefix}{session_id}"

    def _dump(self, env: "MedicalTriageEnv") -> bytes:
        # Using pickle is acceptable here because the data is internal server state
        # and never deserialized from untrusted user input.
        return pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)

    def _load(self, payload: bytes) -> "MedicalTriageEnv":
        return pickle.loads(payload)

    def create(self, env: "MedicalTriageEnv") -> str:
        session_id = env.session_id
        self._redis.setex(self._key(session_id), self._ttl_seconds, self._dump(env))
        logger.info("session_created", session_id=session_id, task_id=env.task_id, backend="redis")
        return session_id

    def get(self, session_id: str) -> "MedicalTriageEnv":
        payload = self._redis.get(self._key(session_id))
        if payload is None:
            logger.warning("session_not_found", session_id=session_id, backend="redis")
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found or expired",
            )
        # Touch TTL on access to match in-memory behavior.
        self._redis.expire(self._key(session_id), self._ttl_seconds)
        return self._load(payload)

    def save(self, env: "MedicalTriageEnv") -> None:
        session_id = env.session_id
        key = self._key(session_id)
        if not self._redis.exists(key):
            logger.warning("session_not_found_on_save", session_id=session_id, backend="redis")
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found or expired",
            )
        self._redis.setex(key, self._ttl_seconds, self._dump(env))

    def destroy(self, session_id: str) -> None:
        self._redis.delete(self._key(session_id))
        logger.info("session_destroyed", session_id=session_id, backend="redis")

    def list_active(self) -> List[str]:
        # SCAN to avoid blocking Redis on large keyspaces.
        ids: List[str] = []
        pattern = f"{self._key_prefix}*"
        for key in self._redis.scan_iter(match=pattern, count=200):
            try:
                # key is bytes
                key_s = key.decode("utf-8", errors="ignore")
            except Exception:
                continue
            ids.append(key_s.replace(self._key_prefix, "", 1))
        return ids

    def clear(self) -> None:
        for sid in self.list_active():
            self.destroy(sid)

    def start(self) -> None:
        # No background sweep needed; Redis TTL handles eviction.
        return None

    def stop(self) -> None:
        return None


_STORE: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """Singleton store instance per process (backend can be Redis shared).

    Production code should always obtain the store through this function.

    Test isolation
    --------------
    Because this is a module-level singleton, tests that create sessions will
    leak state into subsequent tests unless they reset the store between runs.
    Use `reset_session_store()` in a pytest fixture::

        import pytest
        from medical_triage_env.session import reset_session_store

        @pytest.fixture(autouse=True)
        def isolate_session_store():
            reset_session_store()   # fresh store before each test
            yield
            reset_session_store()   # clean up after
    """
    global _STORE
    if _STORE is not None:
        return _STORE

    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis_url:
        _STORE = RedisSessionStore(redis_url)
    else:
        _STORE = InMemorySessionStore()
        logger.warning(
            "in_memory_session_store_in_use",
            note="Multi-worker deployments will lose sessions across workers. Set REDIS_URL to enable Redis-backed sessions.",
        )
    return _STORE


def reset_session_store() -> None:
    """Tear down and nullify the module-level session store singleton.

    Stops any background tasks, clears all in-flight sessions, and resets
    ``_STORE`` to ``None`` so the next call to :func:`get_session_store`
    creates a fresh instance.

    Primarily intended for test isolation — see the fixture example in
    :func:`get_session_store`.
    """
    global _STORE
    if _STORE is not None:
        try:
            _STORE.stop()
        except Exception:  # pragma: no cover — best-effort teardown
            pass
        try:
            _STORE.clear()
        except Exception:  # pragma: no cover
            pass
        _STORE = None
