MLOps Pipeline Debugging Environment
One line: Agent acts as an ML engineer debugging a broken training run — NaN loss, data leakage, silent metric corruption — using logs, config files, and dataset stats.
Why this wins on every axis:
It's directly Meta + HF's world. HF hosts models. Meta trains models. Every judge has personally dealt with a broken training run. You don't need to explain why this matters — they've lived it at 3am.
The grading is perfectly deterministic. You plant exactly one root cause per episode (wrong learning rate, train/val data leak, wrong label encoding). The agent's submitted diagnosis is either correct or wrong. Partial credit for getting the category right but wrong fix.
It's genuinely agentic. Agent can't solve it in one shot. It must read_training_logs() → check_dataset_stats() → inspect_model_config() → run_validation_subset() → submit_diagnosis(). Multi-step, exploratory, exactly what RL environments should reward.
Nobody else will build it. Code review: 30 teams. Email triage: 50 teams. ML pipeline debugging: maybe 2-3, and they'll do it badly because the failure modes require real ML knowledge.
3 Tasks:

Easy: loss=nan from step 1. Agent checks config, finds learning_rate=100.0 instead of 1e-4. Single file, obvious signal.
Medium: Val accuracy is 99% on epoch 1 (suspiciously perfect). Agent must detect train/val split was done after tokenization, causing data leakage. Requires checking dataset preprocessing order.
Hard: Model trains fine, validation is fine, but a silent bug — evaluation uses label_encoder_v1 while training used label_encoder_v2 (different class ordering). No error thrown. Agent must trace the full preprocessing pipeline across 3 files and identify the version mismatch.

Reward function is genuinely interesting:
+0.15  correct failure category (config / data / evaluation)
+0.25  correct root cause file identified  
+0.30  correct root cause variable/parameter identified
+0.30  correct fix proposed
-0.02  per redundant duplicate query (penalize brute force)
Dense signal at every step. Asymmetric on hard task (missing a silent metric bug is 2× more penalized than a false flag — mirrors real model deployment risk).
Good. Let me walk you through exactly what gets built — no vague handwaving.

---

## The Core Mental Model

You are simulating this exact real-world scenario:

> An ML engineer opens their terminal Monday morning. Training job finished overnight. Something is wrong. They don't know what. They have to figure it out from artifacts — logs, configs, dataset stats, metric curves — and write a diagnosis.

The **environment generates a broken training run at `reset()`**. The agent investigates using tools. It submits a diagnosis. The grader checks against the planted bug.

---

## What Exists Inside the Environment

When `reset()` is called, the environment procedurally generates:

```
training_run/
├── config.yaml          ← hyperparams, model arch, optimizer settings
├── train.log            ← epoch-by-epoch loss/accuracy + timestamps
├── dataset_stats.json   ← class distribution, split sizes, feature stats
├── preprocessing.py     ← data pipeline code (text, not executable)
├── eval_results.json    ← final metrics on val/test sets
└── model_card.json      ← architecture summary, param count
```

All of these are **synthetically generated text files** with one bug planted. The agent reads them like a real engineer reads artifacts.

---

## The 8 Actions

```python
class MLOpsAction(BaseModel):
    action_type: Literal[
        "read_config",         # Get full config.yaml contents
        "read_logs",           # Get training logs, optionally filtered by epoch/keyword
        "check_dataset_stats", # Get split sizes, class distribution, feature ranges
        "inspect_preprocessing", # Read the preprocessing pipeline code
        "read_eval_results",   # Get validation/test metrics
        "run_sanity_check",    # Agent specifies what to check — env returns computed result
        "query_artifact",      # Fetch any specific field from any artifact
        "submit_diagnosis",    # Agent's final answer — triggers grading
    ]
    
    # Fields per action
    log_filter: Optional[str]       # e.g. "epoch 1", "nan", "warning"
    artifact_path: Optional[str]    # e.g. "config.yaml:learning_rate"
    sanity_check_type: Optional[str] # e.g. "label_overlap", "class_balance"
    
    # submit_diagnosis fields
    failure_category: Optional[str] # "config_error|data_leakage|evaluation_bug|preprocessing_bug"
    root_cause_file: Optional[str]  # which file contains the bug
    root_cause_field: Optional[str] # which param/variable is wrong
    diagnosis: Optional[str]        # natural language explanation
    proposed_fix: Optional[str]     # what change fixes it
```

---

## The 3 Tasks — Exactly

### Task 1 — Exploding Gradient (Easy)

**What the environment generates:**
- `config.yaml` contains `learning_rate: 50.0` (should be `1e-4`)
- `train.log` shows loss going `2.31 → 8.94 → 847.2 → nan` by epoch 3
- Dataset stats are normal, eval results show `loss: nan, accuracy: 0.0`

**What the agent needs to do:**
1. `read_logs()` → sees loss exploding
2. `read_config()` → spots `learning_rate: 50.0`
3. `submit_diagnosis(failure_category="config_error", root_cause_file="config.yaml", root_cause_field="learning_rate", proposed_fix="Set learning_rate to 1e-4")`

**Grading:** Correct category (+0.15) + correct file (+0.25) + correct field (+0.30) + correct fix (+0.30) = **1.0**

This is solvable in 3-4 steps. A good model should score 0.85+.

---

### Task 2 — Data Leakage (Medium)

**What the environment generates:**
- `train.log` shows val accuracy hitting **99.2% at epoch 1** (impossibly fast)
- `dataset_stats.json` shows train/val split: 80/20 but sample overlap count is non-zero
- `preprocessing.py` contains this code:
```python
# Bug is HERE — normalization fitted on full dataset before split
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_full)  # ← sees val data during fit
X_train, X_val = train_test_split(X_normalized, test_size=0.2)
```

**What the agent needs to do:**
1. `read_logs()` → flags suspiciously high val accuracy at epoch 1
2. `check_dataset_stats()` → sees train/val sample overlap is non-zero
3. `inspect_preprocessing()` → finds `scaler.fit_transform(X_full)` before split
4. `submit_diagnosis(failure_category="data_leakage", root_cause_file="preprocessing.py", root_cause_field="scaler.fit_transform", proposed_fix="Fit scaler only on X_train, then transform X_val separately")`

**Grading:** Same structure. Medium difficulty — requires connecting two signals (log anomaly + preprocessing code).

---

### Task 3 — Silent Label Encoder Mismatch (Hard)

**What the environment generates:**
- `train.log` looks completely normal — loss converges, val accuracy 87%
- `eval_results.json` shows test accuracy **34%** (near-random for 3-class problem)
- `model_card.json` mentions `label_encoder: v2`
- `preprocessing.py` contains:
```python
# Training used this
encoder = LabelEncoder()
encoder.fit(["cat", "dog", "bird"])  # alphabetical: 0=bird, 1=cat, 2=dog

# Evaluation code used this (v1 — different ordering)  
encoder = LabelEncoder()
encoder.fit(["dog", "cat", "bird"])  # 0=bird, 1=dog, 2=cat ← WRONG ORDER
```
- No error is thrown. No warning in logs. Training was fine. Only eval is silent-broken.

**What the agent needs to do:**
1. `read_logs()` → training looks normal, no clues there
2. `read_eval_results()` → test accuracy 34% despite 87% val accuracy — major gap
3. `inspect_preprocessing()` → finds two different encoder initializations
4. `run_sanity_check(sanity_check_type="label_consistency")` → environment returns that train and eval label mappings differ
5. `submit_diagnosis(failure_category="evaluation_bug", root_cause_file="preprocessing.py", root_cause_field="LabelEncoder fit order", proposed_fix="Use same label encoder instance for both training and evaluation")`

**Grading:** Asymmetric — missing this in a deployed model means your model silently predicts wrong classes in production. Penalty for wrong answer is heavier here.

---

## Why This Is Genuinely Hard For Models

- **Task 1:** Solvable by reading 2 files. Easy baseline ~0.8
- **Task 2:** Requires connecting a symptom (log) to a cause (preprocessing code) across different files. Baseline ~0.45
- **Task 3:** No error signal at all. Agent must reason about *what's absent* — the suspicious gap between val (87%) and test (34%) — and trace backwards. Frontier models score ~0.25 here without careful prompting

That variance across tasks (0.8 → 0.45 → 0.25) is exactly what judges want to see.

---

## The Reward Shape

```
Per step:
  +0.02 each time agent reads a new (unread) artifact → encourages exploration
  -0.02 reading the same artifact twice with no new filter → penalizes brute force
  -0.05 submitting a diagnosis before reading ≥3 artifacts → penalizes guessing

At submission:
  +0.15 correct failure_category
  +0.25 correct root_cause_file  
  +0.30 correct root_cause_field
  +0.30 correct proposed_fix (substring match against gold fixes)
  
  Task 3 multiplier: wrong answer gets ×1.5 penalty (silent bugs are worse)
```

Dense signal throughout. Agent is rewarded for systematic investigation, not lucky one-shot guessing.

---

## What Makes This Un-copyable

The key IP of this environment is the **bug generator** — a procedural system that:

1. Picks a bug type from a catalogue (config error, data leak, eval bug, preprocessing bug, version mismatch)
2. Generates all 6 artifacts consistently around that bug
3. Ensures every artifact is internally consistent with the others *except* for the one planted fault

This is the part nobody will replicate in 3 days. Most people will hardcode 3 static scenarios. You'll have a generator that produces different seeds — which means your grader has genuine variance and the baseline script can report `mean ± std` across 10 seeds. That one line in your README signals engineering maturity.

---

Confident in this? I'll start building the full code right now — `bug_generator.py`, `artifact_generator.py`, `environment.py`, `models.py`, `app.py`, `inference.py`, everything.