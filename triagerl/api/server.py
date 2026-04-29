from __future__ import annotations

from medical_triage_env.env import app


def main() -> None:
    import uvicorn

    uvicorn.run("triagerl.api.server:app", host="0.0.0.0", port=8000)
