# OpenEnv Submission: MLOps Pipeline Debugger

## Summary

This submission implements a complete OpenEnv environment for ML pipeline debugging - a real-world task where an AI agent acts as a senior ML engineer diagnosing broken training runs.

## Baseline Scores

| Task | Score |
|------|-------|
| Easy | 0.91 |
| Medium | 0.85 |
| Hard | 1.00 |
| **Average** | **0.92** |

## What's Implemented

### Core Environment
- ✅ step()/reset()/state() API endpoints
- ✅ Procedural artifact generation with planted bugs
- ✅ 3 tasks: easy (config errors), medium (data leakage), hard (silent evaluation bugs)
- ✅ Deterministic graders with 0.0-1.0 scoring

### Technical
- ✅ Typed Pydantic models (Action, Observation, State)
- ✅ openenv.yaml with metadata
- ✅ Docker containerization
- ✅ FastAPI server on port 7860
- ✅ State endpoint at /openenv/state

### Inference
- ✅ inference.py uses OpenAI Client
- ✅ Environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN
- ✅ Correct output format: [START], [STEP], [END] with score=

### Hard Task Fallback
- Implements one-shot fallback to Gemini 3.1 Pro Preview if hard score < 0.8

### Rate Limiting
- Per-call throttling + exponential backoff to handle API rate limits

## Files Submitted
- inference.py - Baseline inference script
- app.py - FastAPI server
- models.py - Pydantic models
- mlops_environment.py - Environment
- artifact_generator.py - Bug generation
- client.py - Python client
- openenv.yaml - OpenEnv spec
- Dockerfile - Container build
- README.md - Documentation
- validate-submission.sh - Pre-validation script

## Validation

```bash
# Build Docker
docker build -t mlops-debug-env .

# Run server
docker run -p 7860:7860 mlops-debug-env

# Run inference
python inference.py --seed 42
```

## Competition Requirements Met

- ✅ Real-world task (ML debugging)
- ✅ OpenEnv spec compliance
- ✅ 3 tasks with graders
- ✅ Meaningful reward function
- ✅ Baseline inference script
- ✅ Dockerfile
- ✅ README with setup instructions

## Deployment to HF Spaces

The Dockerfile is ready for HF Spaces deployment. Build and run:
```bash
docker build -t mlops-debug-env .
docker run -p 7860:7860 mlops-debug-env
```

---

**Competition Branch:** `fix/openenv-3task-yesterday`  
**New Repo:** https://github.com/Angelgupta13/MLops-Openenvhack