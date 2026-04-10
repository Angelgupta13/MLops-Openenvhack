from __future__ import annotations
import json
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
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
    task_id: Optional[str] = "easy"
    seed: Optional[int] = None
    task: Optional[str] = None  # Support both task_id and task


class StepResponse(BaseModel):
    observation: MLOpsObservation
    reward: float
    done: bool
    info: Dict[str, Any]


@app.post("/reset", response_model=MLOpsObservation)
async def reset(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    task_id = body.get("task_id") or body.get("task") or "easy"
    seed = body.get("seed")
    global _http_env
    _http_env = MLOpsEnvironment(task_id=task_id)
    return _http_env.reset(seed=seed)


@app.get("/")
async def root():
    return {
        "message": "MLOps Pipeline Debugger API",
        "version": "1.0.0",
        "docs": "This is an OpenEnv-compatible RL environment",
        "endpoints": {
            "GET /": "This message",
            "GET /health": "Health check",
            "GET /tasks": "List available tasks",
            "GET /openenv/state": "OpenEnv state",
            "POST /reset": "Start a new episode",
            "POST /step": "Take an action",
            "GET /state": "Get current state",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok", "environment": "mlops_debug_env", "version": "1.0.0"}


@app.get("/openenv/state", response_model=OpenEnvState)
def openenv_state():
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


@app.post("/step", response_model=StepResponse)
async def step(request: Request):
    if _http_env is None:
        raise HTTPException(400, "Call /reset first.")

    # Get raw body as dict
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Handle various input formats
    action = None
    if "action_type" in body:
        action = MLOpsAction(**body)
    elif "action" in body:
        action = MLOpsAction(**body["action"])
    elif "message" in body:
        action = MLOpsAction(action_type=body["message"])

    if action is None or action.action_type is None:
        raise HTTPException(422, "Field required: action_type")

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
