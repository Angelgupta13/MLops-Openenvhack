# Architecture

## System Overview

```
Agent (inference.py)
    │
    │  POST /reset, POST /step
    ▼
FastAPI Server (app.py)
    │
    │  reset(), step()
    ▼
MLOpsEnvironment (mlops_environment.py)
    │
    ├── ArtifactGenerator (artifact_generator.py)
    │   └── BUG_CATALOGUE: 9 bug specs across 3 tiers
    │   └── Procedural generation: config, logs, stats, code, eval, model card
    │
    ├── Sanity Check Engine (artifact_generator.py)
    │   └── 8 computed diagnostics grounded in generated artifacts
    │
    ├── Grader (_handle_submit)
    │   └── 4-component scoring: category + file + field + fix
    │
    └── Models (models.py)
        └── MLOpsAction, MLOpsObservation, MLOpsState, ArtifactMeta
```

## Data Flow

### Episode Lifecycle

```
1. reset(task_id, seed)
   ├── Random(seed) selects bug from task pool
   ├── ArtifactGenerator creates 6 consistent artifacts with planted fault
   └── Returns: MLOpsObservation with task description + artifact metadata

2. step(action) × N
   ├── read_* actions → return artifact content (reward: +0.02 new, -0.02 duplicate)
   ├── run_sanity_check → compute diagnostic from artifacts (reward: +0.01 new)
   ├── query_artifact → return specific field via dot notation
   └── submit_diagnosis → grade against ground truth (terminal)

3. Grading (_handle_submit)
   ├── Compare 4 components against BugSpec ground truth
   ├── Apply hard task penalty if score < 0.70
   └── Return: score ∈ (0.01, 0.99), breakdown, ground truth
```

### Determinism Guarantees

- `random.Random(seed)` for bug selection and artifact variation
- `np.random.RandomState(seed)` for numeric distributions
- No external state, no network calls during generation
- Same (task_id, seed) always produces identical episode

## Component Responsibilities

### app.py — API Layer
- FastAPI server on port 7860
- REST endpoints: `/reset`, `/step`, `/state`, `/health`, `/tasks`
- WebSocket endpoint: `/ws` for streaming interaction
- Stateless request handling; delegates to MLOpsEnvironment

### mlops_environment.py — Core Logic
- Episode state management (step count, artifacts read, score)
- Action routing to handlers
- Grading logic with 4-component scoring
- `grade_task()` standalone grader for OpenEnv validation

### artifact_generator.py — Content Generation
- `BugSpec` dataclass: category, file, field, gold_fix, difficulty
- `BUG_CATALOGUE`: 9 bug specifications
- `ArtifactGenerator`: produces 6 artifacts per episode
- `run_sanity_check()`: 8 computed diagnostic checks

### models.py — Data Models
- `MLOpsAction`: 8 action types with typed parameters
- `MLOpsObservation`: full agent observation per step
- `MLOpsState`: internal state for debugging/RL harness
- `ArtifactMeta`: artifact metadata (name, description, size hint)

### inference.py — Baseline Agent
- LLM-powered agent using Gemini via OpenAI-compatible API
- Investigation phase: reads artifacts, runs sanity checks
- Diagnosis phase: submits structured diagnosis
- Fallback logic for unparseable LLM output
- Rate limiting with exponential backoff

### client.py — Client Library
- `MLOpsDebugEnv`: async httpx client
- `SyncMLOpsDebugEnv`: synchronous wrapper
- Context manager support for connection lifecycle

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/tasks` | List available tasks |
| POST | `/reset` | Start new episode |
| POST | `/step` | Execute action |
| GET | `/state` | Current episode state |
| GET | `/openenv/state` | OpenEnv framework state |
| WS | `/ws` | WebSocket interface |

## Reward Architecture

The reward function has two layers:

**Per-step (dense):** Encourages systematic investigation
- New artifact read: +0.02 (explore broadly)
- Duplicate read: -0.02 (don't brute force)
- New sanity check: +0.01 (use diagnostics)

**Terminal (graded):** Evaluates diagnosis quality
- 4 independent components sum to max 1.0
- Keyword/substring matching (no LLM judge)
- Hard task asymmetric penalty (1.5x on missed components)

This two-layer design means an agent that investigates thoroughly but diagnoses wrong still earns per-step rewards, while an agent that submits immediately with a lucky guess earns terminal reward but misses exploration bonuses.
