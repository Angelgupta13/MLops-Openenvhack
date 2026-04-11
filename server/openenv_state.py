from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel


class OpenEnvState(BaseModel):
    run_id: str
    task_id: str
    seed: int
    step_count: int
    max_steps: int
    scores: Dict[str, float]
    end_score: float
    rewards: List[float]
    artifacts_read: List[str]
    timestamp: str


# Global current state (mutable during run)
OPENENV_STATE: OpenEnvState = OpenEnvState(
    run_id="",
    task_id="",
    seed=0,
    step_count=0,
    max_steps=30,
    scores={"easy": 0.01, "medium": 0.01, "hard": 0.01},
    end_score=0.01,
    rewards=[],
    artifacts_read=[],
    timestamp=datetime.utcnow().isoformat(),
)
