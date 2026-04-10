"""
MLOps Pipeline Debugger — Pydantic Models

The agent acts as an ML engineer investigating a broken training run.
It has access to training artifacts (logs, configs, dataset stats, preprocessing code)
and must diagnose the root cause through systematic investigation.

Action Space:
    read_config           → Get training configuration (hyperparams, model arch, optimizer)
    read_logs             → Get training logs (filterable by keyword/epoch range)
    check_dataset_stats   → Get dataset split sizes, class distribution, feature statistics
    inspect_preprocessing → Read preprocessing pipeline code
    read_eval_results     → Get validation and test set evaluation metrics
    run_sanity_check      → Compute a specific diagnostic check (label overlap, class balance, etc.)
    query_artifact        → Fetch a specific field from any artifact
    submit_diagnosis      → Final answer — triggers grading

Observation Space:
    task_id, task_description
    available_artifacts   — list of artifacts the agent can inspect
    last_action_result    — result of the most recent action
    artifacts_read        — which artifacts have been read so far (exploration tracking)
    step_count, max_steps
    done
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ─── Action ──────────────────────────────────────────────────────────────────

class MLOpsAction(BaseModel):
    """
    One action the agent can take per step.

    action_type determines which fields are used:
        read_config           → (no extra fields)
        read_logs             → log_filter (optional keyword or "epoch:N-M")
        check_dataset_stats   → (no extra fields)
        inspect_preprocessing → (no extra fields)
        read_eval_results     → (no extra fields)
        run_sanity_check      → sanity_check_type (required)
        query_artifact        → artifact_name + field_path (required)
        submit_diagnosis      → all diagnosis fields (required)
    """
    action_type: Literal[
        "read_config",
        "read_logs",
        "check_dataset_stats",
        "inspect_preprocessing",
        "read_eval_results",
        "run_sanity_check",
        "query_artifact",
        "submit_diagnosis",
    ] = Field(..., description="Which action to perform")

    # read_logs
    log_filter: Optional[str] = Field(
        None,
        description="Filter logs by keyword (e.g. 'nan', 'warning', 'error') or epoch range 'epoch:1-5'"
    )

    # run_sanity_check
    sanity_check_type: Optional[Literal[
        "label_consistency",      # Are train/eval label mappings identical?
        "data_leakage",           # Is there train/val sample overlap?
        "gradient_norms",         # Are gradient norms within normal range?
        "class_balance",          # Are classes balanced across splits?
        "feature_statistics",     # Do train/val feature distributions match?
        "encoder_version_match",  # Do all pipeline stages use the same encoder version?
        "loss_trajectory",        # Is the loss curve shape anomalous?
        "metric_gap_analysis",    # Is val vs test metric gap suspiciously large?
    ]] = Field(None, description="Type of sanity check to run")

    # query_artifact
    artifact_name: Optional[Literal[
        "config.yaml",
        "train.log",
        "dataset_stats.json",
        "preprocessing.py",
        "eval_results.json",
        "model_card.json",
    ]] = Field(None, description="Artifact to query a specific field from")
    field_path: Optional[str] = Field(
        None,
        description="Dot-notation field path, e.g. 'optimizer.learning_rate' or 'metrics.val_accuracy'"
    )

    # submit_diagnosis
    failure_category: Optional[Literal[
        "config_error",       # Wrong hyperparameter value
        "data_leakage",       # Train/val contamination
        "evaluation_bug",     # Eval pipeline uses wrong artifacts
        "preprocessing_bug",  # Data transformation applied incorrectly
        "label_mismatch",     # Label encoding inconsistency
        "architecture_bug",   # Model architecture misconfiguration
    ]] = Field(None, description="Category of the failure")
    root_cause_file: Optional[str] = Field(
        None, description="Which artifact file contains the root cause"
    )
    root_cause_field: Optional[str] = Field(
        None, description="Specific parameter, function, or variable that is wrong"
    )
    diagnosis: Optional[str] = Field(
        None, description="Natural language explanation of what went wrong and why"
    )
    proposed_fix: Optional[str] = Field(
        None, description="Concrete change that would fix the issue"
    )


# ─── Observation ─────────────────────────────────────────────────────────────

class ArtifactMeta(BaseModel):
    name: str
    description: str
    size_hint: str   # e.g. "47 lines", "12 fields"
    last_modified: str


class MLOpsObservation(BaseModel):
    """Everything the agent sees after each step / reset."""
    task_id: str
    task_description: str

    # Run summary — always visible
    run_id: str
    run_summary: Dict[str, Any] = Field(
        description="High-level run info: model, dataset, final loss, training status"
    )

    available_artifacts: List[ArtifactMeta]
    artifacts_read: List[str] = Field(
        default_factory=list,
        description="Names of artifacts the agent has already read"
    )

    last_action_result: Dict[str, Any] = Field(default_factory=dict)

    step_count: int = 0
    max_steps: int = 30
    done: bool = False
    messages: List[str] = Field(default_factory=list)


# ─── State ───────────────────────────────────────────────────────────────────

class MLOpsState(BaseModel):
    """Full internal state — for RL harness and debugging."""
    task_id: str
    seed: int
    step_count: int
    max_steps: int
    episode_done: bool

    # Planted bug ground truth
    bug_type: str
    bug_category: str
    bug_file: str
    bug_field: str
    gold_fix: str

    # All generated artifacts (full text)
    artifacts: Dict[str, str]

    # Agent's investigation history
    artifacts_read: List[str]
    sanity_checks_run: List[str]
    duplicate_queries: int

    current_score: float
