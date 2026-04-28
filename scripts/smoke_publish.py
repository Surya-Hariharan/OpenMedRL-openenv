from __future__ import annotations

import json
import sys
from typing import Any, Dict

import httpx


def assert_no_task_list_leak() -> None:
    """
    Verify /reset doesn't leak internal task ids in the error body.
    """
    with httpx.Client(base_url="http://localhost:8000", timeout=10.0) as c:
        r = c.post("/reset", json={"task_id": "definitely-not-a-real-task-id"})
        if r.status_code != 404:
            raise RuntimeError(f"Expected 404 for invalid task_id, got {r.status_code}: {r.text[:200]}")
        body = r.text.lower()
        if "available" in body or "tasks" in body:
            raise RuntimeError(f"/reset error body looks like it may leak corpus: {r.text[:400]}")


def assert_state_does_not_leak_task_id() -> None:
    with httpx.Client(base_url="http://localhost:8000", timeout=10.0) as c:
        reset = c.post("/reset", json={})
        reset.raise_for_status()
        data = reset.json()
        sid = data["info"]["session_id"]
        if "task_id" in data.get("info", {}):
            raise RuntimeError(f"/reset leaked task_id: {data['info'].get('task_id')}")
        if "task_ref" not in data.get("info", {}):
            raise RuntimeError("/reset missing task_ref")
        st = c.get("/state", params={"session_id": sid})
        st.raise_for_status()
        payload: Dict[str, Any] = st.json()
        if "task_id" in payload:
            raise RuntimeError(f"/state leaked task_id: {payload.get('task_id')}")
        if "task_ref" not in payload:
            raise RuntimeError("/state missing task_ref")


def assert_basic_step() -> None:
    with httpx.Client(base_url="http://localhost:8000", timeout=10.0) as c:
        reset = c.post("/reset", json={})
        reset.raise_for_status()
        data = reset.json()
        sid = data["info"]["session_id"]
        step = c.post(
            "/step",
            json={
                "session_id": sid,
                "action": {
                    "action_type": "clarify",
                    "clarifying_question": "Any key past history or repeat vitals?",
                    "reasoning": "Gather key information before triage decision.",
                    "recommended_actions": [],
                    "confidence": 0.3,
                },
            },
        )
        step.raise_for_status()
        out = step.json()
        if "observation" not in out or "reward" not in out or "done" not in out:
            raise RuntimeError(f"/step response missing keys: {json.dumps(out)[:300]}")


def assert_tasks_endpoint_redacts() -> None:
    with httpx.Client(base_url="http://localhost:8000", timeout=10.0) as c:
        r = c.get("/tasks")
        r.raise_for_status()
        payload = r.json()
        tasks = payload.get("tasks", None)
        if not isinstance(tasks, list):
            raise RuntimeError(f"/tasks returned unexpected payload: {json.dumps(payload)[:200]}")
        if len(tasks) != 0:
            raise RuntimeError("/tasks should return [] in production mode")


def main() -> None:
    try:
        assert_no_task_list_leak()
        assert_state_does_not_leak_task_id()
        assert_basic_step()
        assert_tasks_endpoint_redacts()
        print(json.dumps({"ok": True, "smoke": "publish"}, indent=2))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()

