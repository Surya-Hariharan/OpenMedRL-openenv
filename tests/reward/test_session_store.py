"""
Tests for triagerl.api.session.memory (InMemorySessionStore)

Covers: basic get/set/delete, TTL expiry, eviction, active_count.
"""
import time
import pytest

from triagerl.api.session.memory import InMemorySessionStore


class TestInMemorySessionStore:
    def test_set_and_get(self):
        store = InMemorySessionStore(ttl_seconds=60)
        store.set("s1", {"env": "data"})
        result = store.get("s1")
        assert result == {"env": "data"}

    def test_get_missing_returns_default(self):
        store = InMemorySessionStore(ttl_seconds=60)
        assert store.get("missing") is None
        assert store.get("missing", "fallback") == "fallback"

    def test_delete(self):
        store = InMemorySessionStore(ttl_seconds=60)
        store.set("s1", "value")
        store.delete("s1")
        assert store.get("s1") is None

    def test_clear(self):
        store = InMemorySessionStore(ttl_seconds=60)
        store.set("s1", 1)
        store.set("s2", 2)
        store.clear()
        assert store.get("s1") is None
        assert store.get("s2") is None

    def test_ttl_expiry(self):
        store = InMemorySessionStore(ttl_seconds=1)
        store.set("s1", "expires-soon")
        assert store.get("s1") == "expires-soon"
        # Manually expire by patching the stored timestamp
        sid = "s1"
        value, _ = store._sessions[sid]
        store._sessions[sid] = (value, time.monotonic() - 1.0)  # already expired
        assert store.get("s1") is None

    def test_expired_sessions_evicted_on_set(self):
        store = InMemorySessionStore(ttl_seconds=60)
        store.set("s1", "v1")
        # Force expire s1
        value, _ = store._sessions["s1"]
        store._sessions["s1"] = (value, time.monotonic() - 1.0)
        # Trigger eviction via a new set
        store.set("s2", "v2")
        assert "s1" not in store._sessions

    def test_active_count(self):
        store = InMemorySessionStore(ttl_seconds=60)
        store.set("s1", 1)
        store.set("s2", 2)
        assert store.active_count == 2
        # Force expire s1
        value, _ = store._sessions["s1"]
        store._sessions["s1"] = (value, time.monotonic() - 1.0)
        assert store.active_count == 1

    def test_set_returns_deep_copy(self):
        store = InMemorySessionStore(ttl_seconds=60)
        original = {"key": [1, 2, 3]}
        store.set("s1", original)
        original["key"].append(4)
        result = store.get("s1")
        assert result["key"] == [1, 2, 3], "Store should hold a deep copy, not a reference"

    def test_get_returns_deep_copy(self):
        store = InMemorySessionStore(ttl_seconds=60)
        store.set("s1", {"mutable": [1, 2]})
        r1 = store.get("s1")
        r1["mutable"].append(3)
        r2 = store.get("s1")
        assert r2["mutable"] == [1, 2], "get() should return a deep copy, not the stored reference"
