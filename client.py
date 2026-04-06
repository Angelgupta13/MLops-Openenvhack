"""MLOps Pipeline Debugger — Python client"""
from __future__ import annotations
from typing import Any, Dict, Optional
import httpx

from models import MLOpsAction, MLOpsObservation, MLOpsState


class StepResult:
    def __init__(self, observation, reward, done, info):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info
    def __repr__(self):
        return f"StepResult(reward={self.reward:.4f}, done={self.done})"


class MLOpsDebugEnv:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def reset(self, task_id: str = "easy", seed: Optional[int] = None) -> MLOpsObservation:
        r = await self._client.post("/reset", json={"task_id": task_id, "seed": seed})
        r.raise_for_status()
        return MLOpsObservation(**r.json())

    async def step(self, action: MLOpsAction) -> StepResult:
        r = await self._client.post("/step", json=action.model_dump(exclude_none=True))
        r.raise_for_status()
        d = r.json()
        return StepResult(MLOpsObservation(**d["observation"]), d["reward"], d["done"], d["info"])

    async def state(self) -> MLOpsState:
        r = await self._client.get("/state")
        r.raise_for_status()
        return MLOpsState(**r.json())

    def sync(self) -> "SyncMLOpsDebugEnv":
        return SyncMLOpsDebugEnv(self.base_url)


class SyncMLOpsDebugEnv:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.Client] = None

    def __enter__(self):
        self._client = httpx.Client(base_url=self.base_url, timeout=30.0)
        return self

    def __exit__(self, *args):
        if self._client:
            self._client.close()

    def reset(self, task_id: str = "easy", seed: Optional[int] = None) -> MLOpsObservation:
        r = self._client.post("/reset", json={"task_id": task_id, "seed": seed})
        r.raise_for_status()
        return MLOpsObservation(**r.json())

    def step(self, action: MLOpsAction) -> StepResult:
        r = self._client.post("/step", json=action.model_dump(exclude_none=True))
        r.raise_for_status()
        d = r.json()
        return StepResult(MLOpsObservation(**d["observation"]), d["reward"], d["done"], d["info"])

    def state(self) -> MLOpsState:
        r = self._client.get("/state")
        r.raise_for_status()
        return MLOpsState(**r.json())
