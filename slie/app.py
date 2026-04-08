from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi import Body
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from slie.env import SLIEEnvironment
from slie.models import (
    ResetRequest,
    ResetResponse,
    SLIEAction,
    StateResponse,
    StepResponse,
)

app = FastAPI(
    title="SLIE", version="1.1.0", description="Sign Language Interaction Environment"
)

# Single global environment instance — single-threaded, one episode at a time
env = SLIEEnvironment()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "environment": "slie", "version": "1.1.0"}


@app.post("/reset", response_model=ResetResponse)
def reset(request: Annotated[ResetRequest | None, Body()] = None) -> ResetResponse:
    payload = request or ResetRequest()
    try:
        return env.reset(task_id=payload.task_id, episode_seed=payload.episode_seed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse)
def step(action: SLIEAction) -> StepResponse:
    # FIX: catch RuntimeError from env.step() and return HTTP 400 (not 500)
    try:
        return env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    return env.get_state()


app.mount("/", StaticFiles(directory=".", html=True), name="static")
