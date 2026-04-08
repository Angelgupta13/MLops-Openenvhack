from __future__ import annotations
import json
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from openenv_state import OPENENV_STATE, OpenEnvState
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import MLOpsAction, MLOpsObservation, MLOpsState
from mlops_environment import MLOpsEnvironment

app = FastAPI(
    title="MLOps Pipeline Debugger",
    description="OpenEnv environment: AI agent diagnoses broken ML training runs.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_http_env: Optional[MLOpsEnvironment] = None


class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None


class StepResponse(BaseModel):
    observation: MLOpsObservation
    reward: float
    done: bool
    info: Dict[str, Any]


@app.get("/health")
async def health():
    return {"status": "ok", "environment": "mlops_debug_env", "version": "1.0.0"}


@app.get("/openenv/state", response_model=OpenEnvState)
def openenv_state():
    # Expose the current OpenEnv-like state persisted in memory/state.json
    return OPENENV_STATE


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "task_id": "easy",
                "name": "Config Error Diagnosis",
                "difficulty": "easy",
                "max_steps": 20,
            },
            {
                "task_id": "medium",
                "name": "Data Leakage Detection",
                "difficulty": "medium",
                "max_steps": 30,
            },
            {
                "task_id": "hard",
                "name": "Silent Evaluation Bug",
                "difficulty": "hard",
                "max_steps": 40,
            },
        ]
    }


@app.post("/reset", response_model=MLOpsObservation)
async def reset(req: ResetRequest):
    global _http_env
    _http_env = MLOpsEnvironment(task_id=req.task_id)
    return _http_env.reset(seed=req.seed)


@app.post("/step", response_model=StepResponse)
async def step(action: MLOpsAction):
    if _http_env is None:
        raise HTTPException(400, "Call /reset first.")
    obs, reward, done, info = _http_env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=MLOpsState)
async def state():
    if _http_env is None:
        raise HTTPException(400, "Call /reset first.")
    return _http_env.state


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    env: Optional[MLOpsEnvironment] = None
    try:
        while True:
            msg = json.loads(await websocket.receive_text())
            method = msg.get("method")
            if method == "reset":
                env = MLOpsEnvironment(task_id=msg.get("task_id", "easy"))
                obs = env.reset(seed=msg.get("seed"))
                await websocket.send_text(
                    json.dumps({"method": "reset", "observation": obs.model_dump()})
                )
            elif method == "step":
                if env is None:
                    await websocket.send_text(json.dumps({"error": "Call reset first"}))
                    continue
                action = MLOpsAction(**msg.get("action", {}))
                obs, reward, done, info = env.step(action)
                await websocket.send_text(
                    json.dumps(
                        {
                            "method": "step",
                            "observation": obs.model_dump(),
                            "reward": reward,
                            "done": done,
                            "info": info,
                        }
                    )
                )
            elif method == "state":
                if env is None:
                    await websocket.send_text(json.dumps({"error": "Call reset first"}))
                    continue
                await websocket.send_text(
                    json.dumps({"method": "state", "state": env.state.model_dump()})
                )
    except WebSocketDisconnect:
        pass
