# MLOps Pipeline Debugger

[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0.0-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.11](https://img.shields.io/badge/python-3.11-green)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Latest Baseline Scores

| Task | Score |
|------|-------|
| Easy | 0.91 |
| Medium | 0.85 |
| Hard | 1.00 |
| **Average** | **0.92** |

*Tested with Gemini 2.5 Flash + Gemini 3.1 Pro Preview fallback for hard task*

An **OpenEnv-compatible reinforcement learning environment** where an AI agent acts as a senior ML engineer diagnosing a broken training run.

---

## What Is This?

Every ML team has experienced it: a training job finishes overnight and something is wrong. Loss exploded to NaN. Validation accuracy is suspiciously perfect at epoch 1. Test performance is catastrophically below validation with no error thrown. An engineer must systematically investigate — reading logs, checking configs, inspecting preprocessing code, running sanity checks — to find the root cause.

This environment simulates that investigation. At `reset()`, a complete set of realistic training artifacts is **procedurally generated** with one planted fault. The agent investigates using 8 targeted actions and submits a structured diagnosis. The grader checks against the planted ground truth — **fully deterministic, no LLM judge needed**.

**9 distinct bug types across 3 tasks. Every episode can have a different bug. Scores vary continuously 0.0 → 1.0 based on diagnosis precision.**

---

## Environment Design

### Procedural Artifact Generation

Every episode generates 6 realistic training artifacts from scratch:

| Artifact | Contents |
|---|---|
| `config.yaml` | Model arch, optimizer, LR, batch size, scheduler, augmentation |
| `train.log` | Epoch-by-epoch loss/accuracy/gradient norms with realistic timestamps |
| `dataset_stats.json` | Split sizes, class distribution, overlap counts, feature statistics |
| `preprocessing.py` | Full sklearn/PyTorch preprocessing pipeline code |
| `eval_results.json` | Final val/test metrics with hardware info |
| `model_card.json` | Architecture summary, tokenizer version, preprocessing config |

Artifacts are **internally consistent** — config matches logs, dataset stats match preprocessing code — except for the one planted fault. A real ML engineer would need to read multiple artifacts and correlate signals to locate it.

---

## Action Space

```python
class MLOpsAction(BaseModel):
    action_type: Literal[
        "read_config",          # Full config.yaml
        "read_logs",            # Training logs (filterable: keyword or "epoch:N-M")
        "check_dataset_stats",  # Split sizes, class distribution, overlap counts
        "inspect_preprocessing",# Full preprocessing pipeline code
        "read_eval_results",    # Final val/test metrics
        "run_sanity_check",     # Computed diagnostic (see types below)
        "query_artifact",       # Specific field from any artifact (dot notation)
        "submit_diagnosis",     # Final answer — triggers grading
    ]
    
    # Sanity check types:
    # label_consistency | data_leakage | gradient_norms | class_balance
    # feature_statistics | encoder_version_match | loss_trajectory | metric_gap_analysis
    
    # submit_diagnosis fields:
    # failure_category | root_cause_file | root_cause_field | diagnosis | proposed_fix
```

---

## Observation Space

```python
class MLOpsObservation(BaseModel):
    task_id: str                          # easy | medium | hard
    task_description: str                 # Full task brief with investigation strategy
    run_id: str                           # Unique run identifier
    run_summary: Dict[str, Any]           # Model, dataset, training status
    available_artifacts: List[ArtifactMeta]  # What can be read
    artifacts_read: List[str]             # Investigation progress
    last_action_result: Dict[str, Any]    # Full content of last action
    step_count: int
    max_steps: int
    done: bool
    messages: List[str]                   # System warnings (duplicate reads, etc.)
```

---

## Tasks

### Task 1 — Config Error Diagnosis `(easy)`

**Bug pool (one picked randomly per episode):**
- `exploding_lr` — `learning_rate: 50.0` causes loss → NaN by epoch 3
- `wrong_optimizer` — `SGD(momentum=0.99)` causes oscillation with no convergence
- `batch_size_overflow` — `batch_size: 4096` exceeds dataset size, val accuracy 99.9% trivially

**Signal:** Visible immediately in training logs. Loss curve or accuracy values are obviously wrong.

**Optimal strategy:** `read_logs` → `run_sanity_check(loss_trajectory)` → `read_config` → `submit_diagnosis`

Max steps: **20** | Expected baseline score: ~0.42

---

### Task 2 — Data Leakage Detection `(medium)`

**Bug pool:**
- `data_leakage_scaler` — `StandardScaler.fit_transform(X_full)` called before train/val split
- `data_leakage_overlap` — `train_test_split(random_state=None)` produces non-deterministic overlapping splits
- `wrong_split_ratio` — `test_size=0.8` trains on 20% and evaluates on 80% (inverted)

**Signal:** Val accuracy suspiciously high from epoch 1 in logs; val/test gap in eval results; sample overlap count in dataset stats.

**Optimal strategy:** `read_logs` → `read_eval_results` → `run_sanity_check(data_leakage)` → `inspect_preprocessing` → `submit_diagnosis`

Max steps: **30** | Expected baseline score: ~0.28

---

### Task 3 — Silent Evaluation Bug `(hard)`

**Bug pool:**
- `label_encoder_mismatch` — Train/eval use different `LabelEncoder.fit()` orderings → silent wrong predictions
- `silent_metric_swap` — `val_accuracy` and `test_accuracy` assignments are swapped in eval code
- `tokenizer_version_drift` — Training uses tokenizer v2, eval uses v1 → 847 tokens map to `[UNK]`

**Signal:** Training logs look completely normal. Only the val/test metric gap in eval results is suspicious — no errors, no warnings, no exceptions.

**Asymmetric penalty:** Missing a silent evaluation bug (which would affect production predictions) is penalized 1.5× — mirroring real incident severity weighting.

**Optimal strategy:** `read_eval_results` → `run_sanity_check(metric_gap_analysis)` → `inspect_preprocessing` → `run_sanity_check(label_consistency OR encoder_version_match)` → `submit_diagnosis`

Max steps: **40** | Expected baseline score: ~0.15

---

## Reward Function

**Dense per-step rewards** (not sparse):

```
+0.02  First time reading an artifact (rewards systematic exploration)
-0.02  Reading same artifact with same filter again (penalizes brute force)
+0.01  Running a new sanity check (rewards diagnostic reasoning)

At submit_diagnosis:
+0.15  Correct failure_category  (config_error / data_leakage / evaluation_bug / ...)
+0.25  Correct root_cause_file   (exact match)
+0.30  Correct root_cause_field  (substring match, case-insensitive)
+0.30  Correct proposed_fix      (keyword overlap with gold fix)

Task 3 modifier: if score < 0.70, additional 0.5× penalty on missed components
```

**Score spectrum** (verified):
```
All wrong            → 0.00
Category only        → 0.10–0.15
Category + file      → 0.35–0.40
Category + file + field → 0.65
Perfect diagnosis    → 0.90–1.00
```

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
# Sync usage
from client import MLOpsDebugEnv
from models import MLOpsAction

with MLOpsDebugEnv(base_url="http://localhost:7860").sync() as env:
    obs = env.reset(task_id="hard", seed=1)
    print(obs.task_description)

    # Investigate systematically
    r = env.step(MLOpsAction(action_type="read_eval_results"))
    print(r.observation.last_action_result["content"])

    r = env.step(MLOpsAction(
        action_type="run_sanity_check",
        sanity_check_type="metric_gap_analysis"
    ))
    # Reveals val/test gap anomaly

    r = env.step(MLOpsAction(action_type="inspect_preprocessing"))
    # Shows the buggy pipeline code

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

---

## Baseline Inference Script

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_BASE_URL="http://localhost:7860"

python inference.py          # all 3 tasks, seed=42
python inference.py --task easy --seed 42
```

**Output format:**
```
[START] task=easy env=mlops-debug-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=read_logs reward=0.02 done=false error=null
[STEP] step=2 action=run_sanity_check reward=0.01 done=false error=null
[STEP] step=3 action=read_config reward=0.02 done=false error=null
[STEP] step=4 action=submit_diagnosis reward=0.95 done=true error=null
[END] success=true steps=4 rewards=0.02,0.01,0.02,0.95
```

**Baseline scores** (Qwen2.5-72B-Instruct, seed=42):

| Task | Score | Notes |
|---|---|---|
| easy | ~0.42 | Gets category right, struggles with exact field name |
| medium | ~0.28 | Often identifies leakage but misidentifies exact mechanism |
| hard | ~0.15 | Silent bugs with normal training logs are genuinely hard |

---

## Why This Environment

**Real problem.** Every ML team at every company has debugging broken training runs as a core workflow. The three bug categories in this environment — config errors, data leakage, silent evaluation bugs — are the actual top-3 failure modes in production ML pipelines.

**Deterministic grading.** The planted bug is ground truth. Diagnosis matching is substring/keyword matching against known-correct answers. Zero subjectivity, zero LLM-as-judge, reproducible across runs.

**Genuinely hard for frontier models.** Task 3 (silent evaluation bugs) requires reasoning about what's *absent* — no error signals, normal training logs — and tracing backwards from a metric anomaly to a pipeline version mismatch. State-of-the-art models score ~0.15 without careful prompting.

**Seed-based reproducibility.** `reset(seed=42)` always produces the same bug, same artifacts, same grading. Baseline scores are reproducible to 4 decimal places.

---

## Environment Variables

| Variable | Description |
|---|---|
| `API_BASE_URL` | LLM API endpoint (OpenAI-compatible) |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face / API token |
| `ENV_BASE_URL` | Environment server URL (default: `http://localhost:7860`) |

---

## License

MIT — see LICENSE
