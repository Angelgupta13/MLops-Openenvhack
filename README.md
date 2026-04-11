---
title: MLOps Pipeline Debugger
emoji: 🔧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# MLOps Pipeline Debugger

[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0.0-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.11](https://img.shields.io/badge/python-3.11-green)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

An **OpenEnv-compatible RL environment** where an AI agent acts as a senior ML engineer diagnosing a broken training run. Built for the **Meta PyTorch Hackathon x Scaler School of Technology**.

---

## The Real-World Problem

Every ML team has experienced it: a training job finishes overnight and something is wrong. Loss exploded to NaN. Validation accuracy is suspiciously perfect at epoch 1. Test performance is catastrophically below validation with no error thrown.

A senior engineer must systematically investigate — reading logs, checking configs, inspecting preprocessing code, running sanity checks — to find the root cause. **This is the #1 time sink in production ML operations**, and it's a skill that separates junior from senior ML engineers.

This environment simulates that investigation workflow. It's not a toy problem — it models the **actual top-3 failure modes** from production ML pipelines:

| Failure Mode | Real-World Frequency | Environment Task |
|---|---|---|
| Hyperparameter misconfiguration | ~40% of training failures | Task 1 (Easy) |
| Data leakage / preprocessing bugs | ~35% of silent accuracy inflation | Task 2 (Medium) |
| Silent evaluation pipeline bugs | ~25% of post-deployment incidents | Task 3 (Hard) |

---

## How It Works

At `reset()`, a complete set of **6 realistic training artifacts** is procedurally generated with one planted fault. The agent investigates using **8 structured actions** and submits a diagnosis. The grader checks against ground truth — **fully deterministic, no LLM judge**.

```
reset(task_id="hard", seed=42)
    │
    ├── Generates: config.yaml, train.log, dataset_stats.json,
    │              preprocessing.py, eval_results.json, model_card.json
    │
    ├── Plants: one bug from the task's 3-bug pool
    │
    └── Agent investigates → submits diagnosis → grader scores [0.01, 0.99]
```

**9 distinct bug types across 3 difficulty tiers. Every episode can have a different bug. Scores vary continuously based on diagnosis precision.**

---

## Procedural Artifact Generation

Every episode generates 6 internally-consistent training artifacts from scratch:

| Artifact | Contents | Role in Investigation |
|---|---|---|
| `config.yaml` | Model arch, optimizer, LR, batch size, scheduler | Check hyperparameters |
| `train.log` | Epoch-by-epoch loss/accuracy/gradient norms | Identify symptom patterns |
| `dataset_stats.json` | Split sizes, class distribution, overlap counts | Detect data issues |
| `preprocessing.py` | Full sklearn/PyTorch pipeline code | Find pipeline bugs |
| `eval_results.json` | Final val/test metrics with hardware info | Quantify metric gaps |
| `model_card.json` | Architecture summary, tokenizer version | Cross-reference versions |

Artifacts are **internally consistent** — config matches logs, dataset stats match preprocessing code — except for the one planted fault. An agent must read multiple artifacts and correlate signals across them to locate the bug.

---

## Action Space (8 actions)

```python
class MLOpsAction(BaseModel):
    action_type: Literal[
        "read_config",           # Full training configuration
        "read_logs",             # Training logs (filterable: keyword or "epoch:N-M")
        "check_dataset_stats",   # Split sizes, class distribution, overlap counts
        "inspect_preprocessing", # Full preprocessing pipeline code
        "read_eval_results",     # Final val/test metrics
        "run_sanity_check",      # Computed diagnostic check (8 types)
        "query_artifact",        # Specific field from any artifact (dot notation)
        "submit_diagnosis",      # Final answer — triggers grading
    ]
```

**Sanity check types** (computed diagnostics, not just artifact reads):
`label_consistency` | `data_leakage` | `gradient_norms` | `class_balance` | `feature_statistics` | `encoder_version_match` | `loss_trajectory` | `metric_gap_analysis`

---

## Observation Space

```python
class MLOpsObservation(BaseModel):
    task_id: str                          # easy | medium | hard
    task_description: str                 # Full task brief with investigation strategy
    run_id: str                           # Unique run identifier
    run_summary: Dict[str, Any]           # Model, dataset, training status
    available_artifacts: List[ArtifactMeta]  # What can be read (name, description, size)
    artifacts_read: List[str]             # Investigation progress tracking
    last_action_result: Dict[str, Any]    # Full content of last action
    step_count: int
    max_steps: int
    done: bool
    messages: List[str]                   # System warnings (duplicate reads, etc.)
```

---

## Tasks & Difficulty Progression

### Task 1 — Config Error Diagnosis `(easy)` | 20 steps max

**Bug pool (one picked randomly per episode):**
- `exploding_lr` — `learning_rate: 50.0` causes loss to diverge to NaN by epoch 3
- `wrong_optimizer` — `SGD(momentum=0.99)` causes loss oscillation with no convergence
- `batch_size_overflow` — `batch_size: 4096` exceeds dataset size, trivial overfitting

**Signal strength:** High. Symptoms visible immediately in training logs.

### Task 2 — Data Leakage Detection `(medium)` | 30 steps max

**Bug pool:**
- `data_leakage_scaler` — `StandardScaler.fit_transform(X_full)` called before train/val split
- `data_leakage_overlap` — `train_test_split(random_state=None)` produces overlapping splits
- `wrong_split_ratio` — `test_size=0.8` trains on 20% and evaluates on 80%

**Signal strength:** Medium. Requires correlating val accuracy anomaly in logs with preprocessing code.

### Task 3 — Silent Evaluation Bug `(hard)` | 40 steps max

**Bug pool:**
- `label_encoder_mismatch` — Train/eval use different `LabelEncoder.fit()` orderings
- `silent_metric_swap` — `val_accuracy` and `test_accuracy` assignments swapped in eval code
- `tokenizer_version_drift` — Training uses tokenizer v2, eval uses v1 (847 tokens map to `[UNK]`)

**Signal strength:** Low. Training logs look completely normal. Only the val/test metric gap is suspicious — no errors, no warnings, no exceptions. Requires reasoning about what's *absent*.

**Asymmetric penalty:** Missing a silent evaluation bug is penalized 1.5x — mirroring real incident severity weighting where silent production bugs are far more costly than loud training failures.

---

## Reward Design

**Dense per-step rewards** (not sparse — provides learning signal throughout the episode):

```
Investigation phase:
  +0.02  First time reading an artifact     (rewards systematic exploration)
  -0.02  Re-reading same artifact+filter    (penalizes brute force)
  +0.01  Running a new sanity check         (rewards diagnostic reasoning)

Diagnosis grading (4 independent components):
  +0.15  Correct failure_category           (what kind of bug?)
  +0.25  Correct root_cause_file            (which file contains it?)
  +0.30  Correct root_cause_field           (which parameter/function?)
  +0.30  Correct proposed_fix               (keyword overlap with gold fix)

Task 3 modifier:
  If score < 0.70 → additional 0.5x penalty on missed components
  (silent bugs reaching production are more costly than loud failures)
```

**Why dense rewards?** Sparse terminal-only rewards make it impossible to distinguish "investigated well but diagnosed wrong" from "didn't investigate at all." Our per-step rewards incentivize thorough investigation, penalize lazy repetition, and the 4-component terminal grading provides partial credit for partially-correct diagnoses.

**Score spectrum:**
```
No investigation, wrong diagnosis  →  0.01
Category only correct              →  0.10–0.15
Category + file correct            →  0.35–0.40
Category + file + field correct    →  0.65
Perfect diagnosis                  →  0.90–0.99
```

---

## Baseline Scores

| Task | Baseline (Qwen2.5-72B) | Optimized (Gemini 2.5 Flash) |
|---|---|---|
| Easy | ~0.42 | ~0.91 |
| Medium | ~0.28 | ~0.85 |
| Hard | ~0.15 | ~0.92 |

The baseline agent (no task-specific prompting) struggles significantly on medium and hard tasks, confirming meaningful difficulty progression.

---

## Setup & Usage

### Docker (recommended)

```bash
docker build -t mlops-debug-env .
docker run -p 7860:7860 mlops-debug-env
curl http://localhost:7860/health
```

### Local Python

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Python Client

```python
from client import MLOpsDebugEnv
from models import MLOpsAction

with MLOpsDebugEnv(base_url="http://localhost:7860").sync() as env:
    obs = env.reset(task_id="hard", seed=1)

    # Investigate
    r = env.step(MLOpsAction(action_type="read_eval_results"))
    r = env.step(MLOpsAction(action_type="run_sanity_check",
                             sanity_check_type="metric_gap_analysis"))
    r = env.step(MLOpsAction(action_type="inspect_preprocessing"))

    # Diagnose
    r = env.step(MLOpsAction(
        action_type="submit_diagnosis",
        failure_category="label_mismatch",
        root_cause_file="preprocessing.py",
        root_cause_field="LabelEncoder.fit_order",
        diagnosis="Train and eval use different LabelEncoder orderings",
        proposed_fix="Use single LabelEncoder instance across both pipelines"
    ))
    print(f"Score: {r.info['score']}")
```

### Inference Script

```bash
export GEMINI_API_KEY="your_key"
export ENV_BASE_URL="http://localhost:7860"
python inference.py                    # all 3 tasks
python inference.py --task easy --seed 42
```

**Output format (OpenEnv standard):**
```
[START] task=easy env=mlops-debug-env model=gemini-2.5-flash
[STEP] step=1 action=read_logs reward=0.02 done=false error=null
[STEP] step=2 action=run_sanity_check reward=0.01 done=false error=null
[STEP] step=3 action=read_config reward=0.02 done=false error=null
[STEP] step=4 action=submit_diagnosis reward=0.91 done=true error=null
[END] success=true steps=4 score=0.9100 rewards=0.02,0.01,0.02,0.91
```

---

## Design Decisions

**Why MLOps debugging?** Config errors, data leakage, and silent eval bugs are the actual top-3 failure modes in production ML. Every ML team at every company deals with these. This isn't a synthetic benchmark — it models a real workflow.

**Why procedural generation?** Fixed bug scenarios would let agents memorize answers. Our seed-based generation produces different bug instances, model configs, and artifact contents per episode while maintaining internal consistency.

**Why deterministic grading?** LLM-as-judge introduces variance and bias. Our grader uses substring/keyword matching against planted ground truth — zero subjectivity, reproducible to 4 decimal places.

**Why asymmetric penalties?** In production, a loud training crash (Task 1) is caught immediately. A silent evaluation bug (Task 3) can serve wrong predictions for weeks before anyone notices. The 1.5x penalty on Task 3 mirrors this real-world cost asymmetry.

**Why 8 sanity check types?** Real ML debugging involves running diagnostic scripts — not just reading files. Our computed sanity checks (gradient norm analysis, data leakage detection, metric gap analysis) simulate the diagnostic tools a senior engineer would use.

---

## Project Structure

```
MLops-Openenvhack/
├── app.py                  # FastAPI server (REST + WebSocket)
├── mlops_environment.py    # Core environment: reset/step/grading
├── artifact_generator.py   # Procedural artifact + bug generation
├── models.py               # Pydantic models (Action, Observation, State)
├── inference.py             # LLM baseline agent
├── client.py               # Python client library (async + sync)
├── openenv_state.py        # Global state singleton
├── openenv.yaml            # OpenEnv specification
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
└── server/                 # HF Space deployment copy
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes (for inference) | — | Gemini API key for baseline agent |
| `MODEL_NAME` | No | `gemini-2.5-flash` | LLM model identifier |
| `API_BASE_URL` | No | Gemini endpoint | OpenAI-compatible API base URL |
| `ENV_BASE_URL` | No | `http://localhost:7860` | Environment server URL |

---

## License

MIT
