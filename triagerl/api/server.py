from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, ValidationError

from triagerl.api.session.store import get_session_store
from triagerl.core.models import TriageAction
from triagerl.env.triage_env import MedicalTriageEnv
from triagerl.tasks.loader import get_task, get_task_list


app = FastAPI(title="triagerl")


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    action: dict


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "tasks": len(get_task_list())}


@app.get("/tasks")
def tasks() -> dict:
    return {"task_ids": get_task_list()}


@app.post("/reset")
def reset(req: ResetRequest) -> dict:
    try:
        task = get_task(req.task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    session_id = str(uuid4())
    env = MedicalTriageEnv(task_config=task, session_id=session_id)
    observation = env.reset()

    get_session_store().set(session_id, env)

    return {
        "observation": observation.model_dump(),
        "info": {
            "session_id": session_id,
            "task_id": task.id,
        },
    }


@app.post("/step")
def step(req: StepRequest) -> dict:
    session_store = get_session_store()
    env = session_store.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id '{req.session_id}'")

    try:
        action = TriageAction.model_validate(req.action)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        observation, reward, done, info = env.step(action)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    session_store.set(req.session_id, env)

    return {
        "observation": observation.model_dump(),
        "reward": float(reward),
        "done": bool(done),
        "info": info,
    }


def main() -> None:
    import uvicorn

    uvicorn.run("triagerl.api.server:app", host="0.0.0.0", port=8000)
