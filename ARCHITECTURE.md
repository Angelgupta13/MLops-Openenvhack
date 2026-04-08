# Backend Architecture

## Project Structure

```
MLops-Openenvhack/
├── app.py                 # FastAPI server - main entry point
├── inference.py           # Baseline LLM agent for evaluation
├── models.py              # Pydantic models (Action, Observation, State)
├── mlops_environment.py   # Core environment logic
├── artifact_generator.py  # Procedural bug/artifact generation
├── client.py              # Python client library
├── openenv.yaml           # OpenEnv specification
├── Dockerfile             # Container configuration
├── requirements.txt       # Python dependencies
└── README.md             # Documentation
```

## How It Works

### 1. Server (app.py)
- Runs FastAPI on port 7860
- Provides REST endpoints:
  - `GET /health` - Health check
  - `POST /reset` - Initialize new task
  - `POST /step` - Execute action
  - `GET /state` - Get current state
  - `GET /tasks` - List available tasks
  - `GET /openenv/state` - OpenEnv state

### 2. Environment (mlops_environment.py)
- Manages task state
- Processes actions through `_handle_*` methods
- Generates rewards based on agent behavior
- Tracks artifacts read and sanity checks

### 3. Artifact Generator (artifact_generator.py)
- Procedurally generates training artifacts with planted bugs
- Creates realistic: logs, configs, preprocessing code, eval results
- Supports 9 bug types across 3 difficulty levels

### 4. Inference Agent (inference.py)
- LLM-powered agent using OpenAI API
- Reads artifacts, runs sanity checks
- Submits diagnosis with confidence scoring
- Implements rate limiting and fallback

## API Flow

```
Client -> app.py (FastAPI)
           |
           +-> mlops_environment.py (core logic)
                    |
                    +-> artifact_generator.py (bug generation)
                    |
                    +-> models.py (data validation)
                    |
                    +-> Returns Observation, Reward, Done, Info
```

## Task Flow

```
1. Client POST /reset with task_id (easy/medium/hard)
2. Environment generates artifacts with planted bug
3. Client POST /step with action
4. Environment processes action, returns observation
5. Agent investigates until diagnosis submitted
6. Grader scores against planted bug (0.0 - 1.0)
```

## Data Models

### Action Types
- read_config, read_logs, check_dataset_stats
- inspect_preprocessing, read_eval_results
- run_sanity_check, query_artifact
- submit_diagnosis

### Reward Structure
- +0.02 per new artifact read
- -0.02 per duplicate read
- +0.01 per new sanity check
- Terminal: +0.15 category + 0.25 file + 0.30 field + 0.30 fix
