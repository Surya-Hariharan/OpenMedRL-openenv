from __future__ import annotations

from fastapi import FastAPI


app = FastAPI(title="triagerl")


def main() -> None:
    import uvicorn

    uvicorn.run("triagerl.api.server:app", host="0.0.0.0", port=8000)
