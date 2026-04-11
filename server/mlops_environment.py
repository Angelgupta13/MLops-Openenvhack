"""
MLOps Pipeline Debugger — Core Environment

Episode flow:
    1. reset(task_id, seed) → generates a broken training run with one planted bug
    2. Agent investigates using 8 actions (reads artifacts, runs sanity checks)
    3. Agent submits a structured diagnosis
    4. Grader compares against planted bug ground truth → score in [0.0, 1.0]

Reward design (dense, not sparse):
    +0.02  per new artifact read (first time — rewards exploration)
    -0.02  per duplicate artifact read (no new filter applied)
    -0.05  submitting diagnosis after reading < 3 distinct artifacts
    
    At submit_diagnosis:
        +0.15  correct failure_category
        +0.25  correct root_cause_file
        +0.30  correct root_cause_field (substring match, case-insensitive)
        +0.30  correct proposed_fix (keyword match against gold fix)
        
    Task 3 (hard) penalty multiplier:
        wrong diagnosis → ×1.5 penalty on the missed components
        (silent bugs that reach production are more costly)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from models import MLOpsAction, MLOpsObservation, MLOpsState, ArtifactMeta
from artifact_generator import (
    ArtifactGenerator, BUG_CATALOGUE, TASK_BUG_POOLS,
    run_sanity_check,
)


TASK_MAX_STEPS = {"easy": 20, "medium": 30, "hard": 40}

TASK_DESCRIPTIONS = {
    "easy": (
        "TASK 1 — CONFIG ERROR DIAGNOSIS (Easy)\n\n"
        "A training run has failed or produced clearly wrong results. The issue is in "
        "the training configuration — a hyperparameter is set to an incorrect value that "
        "causes immediate, visible degradation in training metrics.\n\n"
        "Your job: investigate the training artifacts, identify which configuration "
        "parameter is wrong, and propose the correct fix.\n\n"
        "Strategy: Start by reading the training logs to observe symptom patterns, "
        "then check the config to find the misconfigured parameter. "
        "Run sanity checks (loss_trajectory, gradient_norms) to confirm your hypothesis "
        "before submitting.\n\n"
        "Actions available: read_config | read_logs | check_dataset_stats | "
        "inspect_preprocessing | read_eval_results | run_sanity_check | "
        "query_artifact | submit_diagnosis"
    ),
    "medium": (
        "TASK 2 — DATA LEAKAGE DETECTION (Medium)\n\n"
        "Training metrics look suspiciously good — validation accuracy is anomalously "
        "high from the first epoch, but test performance tells a different story. "
        "The issue is in the data preprocessing pipeline.\n\n"
        "Your job: identify the exact source of data leakage — whether it's a scaler "
        "fitted on the full dataset, overlapping train/val splits from a non-deterministic "
        "split, or an inverted split ratio — and propose the correct fix.\n\n"
        "Strategy: Anomalous val accuracy in the logs is your first signal. "
        "Inspect preprocessing code to find how splits are constructed. "
        "Run the data_leakage and feature_statistics sanity checks to confirm. "
        "The val/test metric gap in eval results is another key clue.\n\n"
        "Actions available: read_config | read_logs | check_dataset_stats | "
        "inspect_preprocessing | read_eval_results | run_sanity_check | "
        "query_artifact | submit_diagnosis"
    ),
    "hard": (
        "TASK 3 — SILENT EVALUATION BUG (Hard)\n\n"
        "Training completed normally. Validation metrics look reasonable. "
        "But test set performance is catastrophically below validation — "
        "and there are NO error logs, NO warnings, NO exceptions thrown.\n\n"
        "Your job: find the silent bug in the evaluation pipeline. It could be "
        "a label encoder mismatch between train and eval (different class orderings), "
        "a metric assignment swap (val/test results mislabeled), or a tokenizer "
        "version drift (training used v2, evaluation uses v1).\n\n"
        "Strategy: The val/test metric gap in eval_results is your only initial signal. "
        "Run metric_gap_analysis first to quantify the anomaly. Then systematically "
        "check label_consistency, encoder_version_match, and inspect the preprocessing "
        "code carefully — the bug produces no error output and will only be visible "
        "by comparing train vs eval pipeline definitions.\n\n"
        "WARNING: Missing this bug in a deployed model means silent wrong predictions "
        "in production. Penalty for wrong diagnosis is weighted 1.5×.\n\n"
        "Actions available: read_config | read_logs | check_dataset_stats | "
        "inspect_preprocessing | read_eval_results | run_sanity_check | "
        "query_artifact | submit_diagnosis"
    ),
}

ARTIFACT_DESCRIPTIONS = {
    "config.yaml":          ("Training configuration — hyperparameters, model, optimizer, scheduler", "~45 lines"),
    "train.log":            ("Epoch-by-epoch training metrics — loss, accuracy, gradient norms", "~30–60 lines"),
    "dataset_stats.json":   ("Dataset split sizes, class distribution, feature statistics", "~35 fields"),
    "preprocessing.py":     ("Data preprocessing pipeline — splits, normalization, encoding", "~40–70 lines"),
    "eval_results.json":    ("Final evaluation metrics — val and test loss/accuracy", "~15 fields"),
    "model_card.json":      ("Model architecture summary, training config, preprocessing versions", "~20 fields"),
}


class MLOpsEnvironment:
    """OpenEnv-compatible MLOps Pipeline Debugging environment."""

    def __init__(self, task_id: str = "easy"):
        assert task_id in TASK_MAX_STEPS, f"task_id must be one of {list(TASK_MAX_STEPS)}"
        self.task_id = task_id
        self._reset_internal(seed=42)

    def _reset_internal(self, seed: int):
        rng = random.Random(seed)

        # Pick bug from this task's pool
        pool = TASK_BUG_POOLS[self.task_id]
        self.bug_type = rng.choice(pool)
        self.bug = BUG_CATALOGUE[self.bug_type]

        # Generate all artifacts
        gen = ArtifactGenerator(self.bug_type, seed)
        self._artifacts: Dict[str, str] = gen.generate_all()
        self._model_cfg = gen.model_cfg
        self._run_id = gen.run_id
        self._rng = rng
        self._seed = seed

        # Cache artifact metadata at reset time (avoids consuming RNG per step)
        self._artifact_meta: List[ArtifactMeta] = [
            ArtifactMeta(
                name=name,
                description=ARTIFACT_DESCRIPTIONS[name][0],
                size_hint=ARTIFACT_DESCRIPTIONS[name][1],
                last_modified=f"2024-03-{rng.randint(1,28):02d}",
            )
            for name in self._artifacts
        ]

        # Episode state
        self._step_count = 0
        self._max_steps = TASK_MAX_STEPS[self.task_id]
        self._done = False
        self._artifacts_read: List[str] = []
        self._last_read_filters: Dict[str, str] = {}
        self._sanity_checks_run: List[str] = []
        self._duplicate_queries = 0
        self._current_score = 0.01
        self._messages: List[str] = []

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> MLOpsObservation:
        import time
        actual_seed = seed if seed is not None else int(time.time() * 1000) % 100000
        self._reset_internal(actual_seed)
        return self._build_obs(
            {"status": "reset", "message": "New training run loaded. Begin investigation."},
        )

    def step(self, action: MLOpsAction) -> Tuple[MLOpsObservation, float, bool, Dict[str, Any]]:
        if self._done:
            return self._build_obs({"status": "done", "message": "Episode over. Call reset()."}), 0.01, True, {"score": max(0.01, min(0.99, self._current_score))}

        self._step_count += 1
        reward = 0.0
        info: Dict[str, Any] = {}
        result: Dict[str, Any] = {}

        if self._step_count >= self._max_steps:
            self._done = True
            score = max(0.01, self._current_score)
            result = {"status": "timeout", "message": f"Max steps ({self._max_steps}) reached.", "score": score}
            return self._build_obs(result), score, True, {"score": score, "reason": "timeout"}

        atype = action.action_type

        # ── read_config ───────────────────────────────────────────────────
        if atype == "read_config":
            reward, result = self._handle_artifact_read("config.yaml", None)

        # ── read_logs ─────────────────────────────────────────────────────
        elif atype == "read_logs":
            reward, result = self._handle_artifact_read("train.log", action.log_filter)

        # ── check_dataset_stats ───────────────────────────────────────────
        elif atype == "check_dataset_stats":
            reward, result = self._handle_artifact_read("dataset_stats.json", None)

        # ── inspect_preprocessing ─────────────────────────────────────────
        elif atype == "inspect_preprocessing":
            reward, result = self._handle_artifact_read("preprocessing.py", None)

        # ── read_eval_results ─────────────────────────────────────────────
        elif atype == "read_eval_results":
            reward, result = self._handle_artifact_read("eval_results.json", None)

        # ── run_sanity_check ──────────────────────────────────────────────
        elif atype == "run_sanity_check":
            check = action.sanity_check_type
            if not check:
                result = {"status": "error", "message": "sanity_check_type is required."}
            else:
                check_result = run_sanity_check(check, self.bug_type, self._artifacts, self._rng)
                if check not in self._sanity_checks_run:
                    self._sanity_checks_run.append(check)
                    reward += 0.01   # small reward for running new checks
                result = {"status": "ok", "sanity_check": check_result}

        # ── query_artifact ────────────────────────────────────────────────
        elif atype == "query_artifact":
            art = action.artifact_name
            field = action.field_path
            if not art or not field:
                result = {"status": "error", "message": "artifact_name and field_path are required."}
            elif art not in self._artifacts:
                result = {"status": "error", "message": f"Artifact '{art}' not found."}
            else:
                val = self._resolve_field(art, field)
                result = {"status": "ok", "artifact": art, "field": field, "value": val}

        # ── submit_diagnosis ──────────────────────────────────────────────
        elif atype == "submit_diagnosis":
            reward, info, result = self._handle_submit(action)
            self._done = True

        obs = self._build_obs(result)
        return obs, reward, self._done, info

    # ── Internal handlers ──────────────────────────────────────────────────────

    def _handle_artifact_read(self, artifact: str, log_filter: Optional[str]) -> Tuple[float, Dict]:
        is_duplicate = (
            artifact in self._artifacts_read
            and self._last_read_filters.get(artifact, "") == (log_filter or "")
        )

        content = self._artifacts[artifact]

        # Apply log filter
        if artifact == "train.log" and log_filter:
            lines = content.split("\n")
            if log_filter.startswith("epoch:"):
                try:
                    parts = log_filter.split(":")[1].split("-")
                    start, end = int(parts[0]), int(parts[1]) if len(parts) > 1 else int(parts[0])
                    filtered = [l for l in lines if any(f"EPOCH {ep:03d}" in l
                                 for ep in range(start, end+1)) or "[INFO  ]" in l or "[ERROR" in l]
                    content = "\n".join(filtered) if filtered else "No log lines match this epoch range."
                except Exception:
                    content = "\n".join(lines)
            else:
                kw = log_filter.lower()
                filtered = [l for l in lines if kw in l.lower()]
                content = "\n".join(filtered) if filtered else f"No log lines contain '{log_filter}'."

        reward = 0.0
        if artifact not in self._artifacts_read:
            self._artifacts_read.append(artifact)
            reward = 0.02   # first read reward
        elif is_duplicate:
            self._duplicate_queries += 1
            reward = -0.02  # duplicate penalty
            self._messages.append(f"⚠️  Duplicate read of {artifact} with same filter. Try a different filter or a new artifact.")

        self._last_read_filters[artifact] = log_filter or ""

        return reward, {
            "status": "ok",
            "artifact": artifact,
            "content": content,
            "note": "Use log_filter='keyword' or 'epoch:N-M' for targeted log queries.",
        }

    def _handle_submit(self, action: MLOpsAction) -> Tuple[float, Dict, Dict]:
        if len(self._artifacts_read) < 3:
            # Penalty for submitting without adequate investigation
            base_penalty = -0.05
            self._messages.append("⚠️  Submitted diagnosis after reading fewer than 3 artifacts.")
        else:
            base_penalty = 0.0

        score = base_penalty
        breakdown: Dict[str, Any] = {}

        # 1. failure_category (+0.15)
        if action.failure_category == self.bug.category:
            score += 0.15
            breakdown["failure_category"] = {"awarded": 0.15, "correct": True}
        else:
            breakdown["failure_category"] = {
                "awarded": 0.0, "correct": False,
                "expected": self.bug.category, "got": action.failure_category,
            }

        # 2. root_cause_file (+0.25)
        if action.root_cause_file and action.root_cause_file.lower() == self.bug.file.lower():
            score += 0.25
            breakdown["root_cause_file"] = {"awarded": 0.25, "correct": True}
        else:
            breakdown["root_cause_file"] = {
                "awarded": 0.0, "correct": False,
                "expected": self.bug.file, "got": action.root_cause_file,
            }

        # 3. root_cause_field (+0.30) — require majority of keywords to match
        field_keywords = [kw.lower() for kw in self.bug.field.replace(".", " ").split() if len(kw) > 1]
        submitted_field = (action.root_cause_field or "").lower()
        field_matches = sum(1 for kw in field_keywords if kw in submitted_field)
        field_threshold = max(1, len(field_keywords) // 2 + 1)  # majority
        field_correct = len(field_keywords) > 0 and field_matches >= field_threshold
        if field_correct:
            score += 0.30
            breakdown["root_cause_field"] = {"awarded": 0.30, "correct": True}
        else:
            breakdown["root_cause_field"] = {
                "awarded": 0.0, "correct": False,
                "expected": self.bug.field, "got": action.root_cause_field,
                "matched_keywords": field_matches, "required": field_threshold,
            }

        # 4. proposed_fix (+0.30) — keyword match against gold fix
        import re as _re
        _stop = {"to", "the", "a", "an", "of", "in", "on", "from", "use", "with", "and", "or", "for", "is", "at", "by"}
        # Strip punctuation from keywords so "(fitted" becomes "fitted"
        fix_keywords = {
            _re.sub(r'[^a-z0-9_.]', '', w)
            for w in self.bug.gold_fix.lower().split()
        } - _stop
        fix_keywords.discard("")  # remove empty strings
        submitted_fix = (action.proposed_fix or "").lower()
        fix_overlap = sum(1 for kw in fix_keywords if kw in submitted_fix)
        fix_score = min(0.30, 0.30 * (fix_overlap / max(1, len(fix_keywords))))
        score += fix_score
        breakdown["proposed_fix"] = {
            "awarded": round(fix_score, 4),
            "correct": fix_score >= 0.20,
            "keyword_overlap": fix_overlap,
            "total_keywords": len(fix_keywords),
        }

        # Hard task penalty multiplier — silent bugs are more costly
        if self.task_id == "hard" and score < 0.70:
            missed = 0.70 - min(score, 0.70)
            score -= missed * 0.5   # 1.5× penalty on missed components
            breakdown["hard_task_penalty_applied"] = True

        score = round(max(0.01, min(0.99, score)), 4)
        self._current_score = score

        info = {
            "score": score,
            "breakdown": breakdown,
            "ground_truth": {
                "bug_type":     self.bug_type,
                "category":     self.bug.category,
                "file":         self.bug.file,
                "field":        self.bug.field,
                "gold_fix":     self.bug.gold_fix,
            },
            "investigation": {
                "artifacts_read":    self._artifacts_read,
                "sanity_checks_run": self._sanity_checks_run,
                "duplicate_queries": self._duplicate_queries,
                "steps_taken":       self._step_count,
            },
        }
        result = {
            "status": "submitted",
            "score": score,
            "breakdown": breakdown,
            "message": f"Diagnosis submitted. Score: {score:.4f}/{1.0:.4f}",
        }
        return score, info, result

    def _resolve_field(self, artifact: str, field_path: str) -> Any:
        """Resolve a dot-notation field path from a JSON artifact."""
        import json as _json
        content = self._artifacts[artifact]
        if artifact.endswith(".json"):
            try:
                data = _json.loads(content)
                parts = field_path.split(".")
                val = data
                for p in parts:
                    if isinstance(val, dict):
                        val = val.get(p, f"Field '{p}' not found")
                    else:
                        return f"Cannot traverse into non-dict at '{p}'"
                return val
            except Exception as e:
                return f"Parse error: {e}"
        elif artifact.endswith(".yaml"):
            # Simple key search for YAML
            for line in content.split("\n"):
                target_key = field_path.split(".")[-1]
                if f"{target_key}:" in line:
                    return line.strip()
            return f"Field '{field_path}' not found in config"
        else:
            # For .py files, return lines containing the field name
            target = field_path.split(".")[-1]
            matches = [l.strip() for l in content.split("\n") if target in l]
            return matches[:5] if matches else f"'{target}' not found in {artifact}"

    def _build_obs(self, last_result: Dict[str, Any]) -> MLOpsObservation:
        return MLOpsObservation(
            task_id=self.task_id,
            task_description=TASK_DESCRIPTIONS[self.task_id],
            run_id=self._run_id,
            run_summary={
                "model":   self._model_cfg["name"],
                "dataset": self._model_cfg["dataset"],
                "task":    self._model_cfg["type"],
                "status":  "FAILED" if self.task_id == "easy" else "COMPLETED_WITH_ANOMALIES",
                "note":    "Investigate artifacts to determine root cause.",
            },
            available_artifacts=list(self._artifact_meta),
            artifacts_read=list(self._artifacts_read),
            last_action_result=last_result,
            step_count=self._step_count,
            max_steps=self._max_steps,
            done=self._done,
            messages=list(self._messages),
        )

    @property
    def state(self) -> MLOpsState:
        return MLOpsState(
            task_id=self.task_id,
            seed=self._seed,
            step_count=self._step_count,
            max_steps=self._max_steps,
            episode_done=self._done,
            bug_type=self.bug_type,
            bug_category=self.bug.category,
            bug_file=self.bug.file,
            bug_field=self.bug.field,
            gold_fix=self.bug.gold_fix,
            artifacts=self._artifacts,
            artifacts_read=list(self._artifacts_read),
            sanity_checks_run=list(self._sanity_checks_run),
            duplicate_queries=self._duplicate_queries,
            current_score=self._current_score,
        )


# ─── Standalone grader ────────────────────────────────────────────────────────

def grade_task(task_id: str, seed: int, diagnosis: Dict[str, Any]) -> float:
    """Deterministic grader callable by OpenEnv validation framework.

    Bypasses the artifact-read penalty since the grader only evaluates
    diagnosis quality, not investigation thoroughness.
    """
    env = MLOpsEnvironment(task_id=task_id)
    env.reset(seed=seed)
    # Pre-populate artifact reads to avoid the < 3 artifacts penalty
    env._artifacts_read = list(env._artifacts.keys())
    action = MLOpsAction(action_type="submit_diagnosis", **diagnosis)
    _, reward, _, info = env.step(action)
    return max(0.01, min(0.99, info.get("score", 0.01)))
