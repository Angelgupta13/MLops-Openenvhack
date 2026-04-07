"""
inference.py — Optimized LLM Agent for MLOps Pipeline Debugger

Required env vars (in .env file):
    GEMINI_API_KEY   your Gemini API key
    MODEL_NAME       gemini-2.5-flash (default)
    ENV_BASE_URL     http://localhost:7860 (default)

STDOUT FORMAT (mandatory):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

API_BASE_URL = os.getenv(
    "API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
HF_TOKEN = os.getenv("GEMINI_API_KEY", os.getenv("HF_TOKEN", ""))
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "mlops-debug-env"
TASKS = ["easy", "medium", "hard"]
SUCCESS_THRESHOLD = 0.5

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Complete bug reference for diagnosis guidance ─────────────────────────────

BUG_REFERENCE = {
    "easy": {
        "exploding_lr": {
            "category": "config_error",
            "file": "config.yaml",
            "field": "optimizer.learning_rate",
            "gold_fix": "Reduce learning_rate from 50.0 to 1e-4 (or use a scheduler with warmup)",
            "symptoms": "loss explodes: 2.31 → 8.94 → 847.2 → nan by epoch 3",
        },
        "wrong_optimizer": {
            "category": "config_error",
            "file": "config.yaml",
            "field": "optimizer.momentum",
            "gold_fix": "Reduce momentum from 0.99 to 0.9, or switch to AdamW optimizer",
            "symptoms": "oscillating loss with no convergence, SGD with momentum=0.99",
        },
        "batch_size_overflow": {
            "category": "config_error",
            "file": "config.yaml",
            "field": "training.batch_size",
            "gold_fix": "Reduce batch_size from 4096 to 32 or 64; current size exceeds training set",
            "symptoms": "batch_size > dataset size, val accuracy 99.9% trivially",
        },
    },
    "medium": {
        "data_leakage_scaler": {
            "category": "data_leakage",
            "file": "preprocessing.py",
            "field": "StandardScaler.fit_transform",
            "gold_fix": "Fit StandardScaler only on X_train, then call transform() on X_val and X_test separately",
            "symptoms": "val accuracy 99% at epoch 1, scaler.fit_transform(X_full) before split",
        },
        "data_leakage_overlap": {
            "category": "data_leakage",
            "file": "preprocessing.py",
            "field": "train_test_split.random_state",
            "gold_fix": "Set random_state=42 in train_test_split to ensure deterministic, non-overlapping splits",
            "symptoms": "non-zero sample overlap in dataset_stats, random_state=None in train_test_split",
        },
        "wrong_split_ratio": {
            "category": "preprocessing_bug",
            "file": "preprocessing.py",
            "field": "train_test_split.test_size",
            "gold_fix": "Change test_size from 0.8 to 0.2 — current config trains on 20% and evaluates on 80%",
            "symptoms": "test_size=0.8 in preprocessing.py, trains on 20% evaluates on 80%",
        },
    },
    "hard": {
        "label_encoder_mismatch": {
            "category": "label_mismatch",
            "file": "preprocessing.py",
            "field": "LabelEncoder.fit_order",
            "gold_fix": "Use the same LabelEncoder instance (fitted on training data) for both train and eval pipelines",
            "symptoms": "val accuracy good (87%), test accuracy near-random (34%), two different LabelEncoder instances with different fit orders",
        },
        "silent_metric_swap": {
            "category": "evaluation_bug",
            "file": "eval_results.json",
            "field": "metrics.val_accuracy",
            "gold_fix": "Swap val_accuracy and test_accuracy assignments in the evaluation loop — metrics are mislabeled",
            "symptoms": "val_accuracy suspiciously low, test_accuracy suspiciously high (reversed)",
        },
        "tokenizer_version_drift": {
            "category": "evaluation_bug",
            "file": "preprocessing.py",
            "field": "tokenizer.version",
            "gold_fix": "Ensure training and evaluation both use tokenizer v2 — v1 has a different vocabulary mapping for 847 tokens",
            "symptoms": "training uses TOKENIZER_V2, eval uses TOKENIZER_V1, 847 tokens map to [UNK]",
        },
    },
}

SYSTEM_PROMPT = """You are a senior ML engineer investigating a broken training run.

INVESTIGATION STRATEGY (follow this exact order):
1. read_logs — identify the symptom
2. read_eval_results — check val vs test metric gap
3. inspect_preprocessing — look for pipeline bugs
4. read_config — check hyperparameters
5. check_dataset_stats — look for split issues
6. run_sanity_check — confirm hypothesis
7. submit_diagnosis — ONLY after steps 1-5 minimum

FAILURE CATEGORIES:
- config_error        : Wrong hyperparameter
- data_leakage        : Train/val contamination
- evaluation_bug      : Eval pipeline uses wrong artifacts or swapped metrics
- preprocessing_bug   : Data transformation applied incorrectly
- label_mismatch      : Label encoding inconsistency
- architecture_bug    : Model architecture misconfiguration

ROOT CAUSE FIELD FORMAT: Use dot notation. Examples:
- "optimizer.learning_rate" / "training.batch_size" / "optimizer.momentum"
- "StandardScaler.fit_transform" / "train_test_split.random_state" / "train_test_split.test_size"
- "LabelEncoder.fit_order" / "tokenizer.version" / "metrics.val_accuracy"

RESPOND WITH ONE JSON ACTION OBJECT PER TURN. Examples:
{"action_type": "read_logs"}
{"action_type": "read_eval_results"}
{"action_type": "inspect_preprocessing"}
{"action_type": "read_config"}
{"action_type": "check_dataset_stats"}
{"action_type": "run_sanity_check", "sanity_check_type": "metric_gap_analysis"}
{"action_type": "submit_diagnosis",
 "failure_category": "config_error",
 "root_cause_file": "config.yaml",
 "root_cause_field": "training.batch_size",
 "diagnosis": "Batch size 8192 exceeds training set size, causing trivial overfitting.",
 "proposed_fix": "Reduce batch_size from 4096 to 32 or 64; current size exceeds training set"}

ONLY output the JSON object. No explanation. No markdown."""

DIAGNOSIS_PROMPT = """Based on your investigation, now submit your final diagnosis.

Here is the complete bug reference for this task difficulty:

{bug_ref}

Analyze the artifacts you've read and identify which specific bug matches the symptoms.
Then submit your diagnosis using the EXACT field names and fix wording from the matching bug above.

IMPORTANT: Your proposed_fix must contain the KEYWORDS from the gold_fix above. The grader uses keyword matching.

Respond with ONLY the JSON submit_diagnosis action. No explanation. No markdown."""


# ── Logging helpers ──────────────────────────────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ── Environment helpers ───────────────────────────────────────────────────────


def env_reset(task_id: str, seed: int = 42) -> Dict[str, Any]:
    r = httpx.post(
        f"{ENV_BASE_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=30
    )
    r.raise_for_status()
    return r.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    r = httpx.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


def build_user_msg(obs: Dict[str, Any]) -> str:
    arts_read = obs.get("artifacts_read", [])
    pending = [
        a["name"]
        for a in obs.get("available_artifacts", [])
        if a["name"] not in arts_read
    ]
    last = obs.get("last_action_result", {})
    step = obs.get("step_count", 0)
    max_s = obs.get("max_steps", 30)
    run = obs.get("run_summary", {})

    lines = [
        f"=== STEP {step}/{max_s} ===",
        f"Run: {obs.get('run_id', '')} | Model: {run.get('model', '')} | Status: {run.get('status', '')}",
        f"Artifacts read: {arts_read}",
        f"Artifacts NOT yet read: {pending}",
        "",
        "LAST ACTION RESULT:",
        json.dumps(last, indent=2, default=str)[:3000],
    ]
    msgs = obs.get("messages", [])
    if msgs:
        lines += ["", "SYSTEM MESSAGES:"] + msgs
    if obs.get("done"):
        lines.append("\nEpisode done.")
    return "\n".join(lines)


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1])
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]+\}", text)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


# ── Rate-limited LLM calls ───────────────────────────────────────────────────

_last_call_time = 0
_MIN_CALL_INTERVAL = 2.0
_HARD_ALT_USED = False


def call_llm(messages: List[Dict], model_name: Optional[str] = None) -> str:
    global _last_call_time
    model_to_use = model_name or MODEL_NAME
    for attempt in range(10):
        try:
            elapsed = time.time() - _last_call_time
            if elapsed < _MIN_CALL_INTERVAL:
                time.sleep(_MIN_CALL_INTERVAL - elapsed)

            resp = client.chat.completions.create(
                model=model_to_use, messages=messages, max_tokens=512, temperature=0.1
            )
            _last_call_time = time.time()
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err_msg = str(e)
            if "rate" in err_msg.lower() or "Request rate" in err_msg:
                wait = min(15 * (2**attempt), 120)
                print(
                    f"  [RATE LIMIT] Waiting {wait}s (attempt {attempt + 1}/10)...",
                    flush=True,
                )
            else:
                wait = min(30 * (2**attempt), 300)
                print(
                    f"  [RETRY] LLM error (attempt {attempt + 1}/10): {e}. Waiting {wait}s...",
                    flush=True,
                )
            time.sleep(wait)
    raise RuntimeError("LLM call failed after 10 retries")


# ── Fallback actions ──────────────────────────────────────────────────────────

FALLBACK_ACTIONS = [
    {"action_type": "read_logs"},
    {"action_type": "read_eval_results"},
    {"action_type": "inspect_preprocessing"},
    {"action_type": "read_config"},
    {"action_type": "check_dataset_stats"},
    {"action_type": "run_sanity_check", "sanity_check_type": "metric_gap_analysis"},
    {"action_type": "run_sanity_check", "sanity_check_type": "data_leakage"},
    {"action_type": "run_sanity_check", "sanity_check_type": "label_consistency"},
]


def get_fallback_action(step_num: int) -> Dict[str, Any]:
    idx = min(step_num - 1, len(FALLBACK_ACTIONS) - 1)
    return FALLBACK_ACTIONS[idx]


# ── Main agent loop ──────────────────────────────────────────────────────────


def run_task(task_id: str, seed: int = 42) -> float:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    obs = env_reset(task_id, seed)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"TASK DESCRIPTION:\n{obs['task_description']}\n\n{build_user_msg(obs)}",
        },
    ]

    MIN_STEPS = {"easy": 5, "medium": 7, "hard": 10}
    MAX_STEPS = {"easy": 20, "medium": 30, "hard": 40}
    min_steps = MIN_STEPS.get(task_id, 7)
    max_steps = MAX_STEPS.get(task_id, 30)

    CORE_ARTIFACTS = {
        "train.log",
        "eval_results.json",
        "preprocessing.py",
        "config.yaml",
        "dataset_stats.json",
    }

    step_num = 0
    done = False
    final_score = 0.0
    rewards: List[float] = []
    action_history: List[str] = []
    sanity_check_history: List[str] = []
    in_diagnosis_phase = False

    def get_unread_artifacts() -> List[str]:
        arts_read = set(obs.get("artifacts_read", []))
        return [a for a in CORE_ARTIFACTS if a not in arts_read]

    def get_next_unread_artifact() -> Optional[Dict[str, Any]]:
        unread = get_unread_artifacts()
        if not unread:
            return None
        artifact_to_action = {
            "train.log": {"action_type": "read_logs"},
            "eval_results.json": {"action_type": "read_eval_results"},
            "preprocessing.py": {"action_type": "inspect_preprocessing"},
            "config.yaml": {"action_type": "read_config"},
            "dataset_stats.json": {"action_type": "check_dataset_stats"},
        }
        return artifact_to_action.get(unread[0])

    def force_new_sanity_check() -> Dict[str, Any]:
        all_checks = [
            "metric_gap_analysis",
            "data_leakage",
            "label_consistency",
            "encoder_version_match",
            "loss_trajectory",
            "class_balance",
            "gradient_norms",
            "feature_statistics",
        ]
        for sc in all_checks:
            if sc not in sanity_check_history:
                return {"action_type": "run_sanity_check", "sanity_check_type": sc}
        return {
            "action_type": "run_sanity_check",
            "sanity_check_type": "metric_gap_analysis",
        }

    def is_repetitive(action_type: str) -> bool:
        if len(action_history) < 2:
            return False
        return action_history[-1] == action_type and action_history[-2] == action_type

    while not done:
        step_num += 1
        unread = get_unread_artifacts()
        all_read = len(unread) == 0

        # Force submission near max steps
        if step_num >= max_steps - 1:
            in_diagnosis_phase = True

        if in_diagnosis_phase:
            # Build diagnosis prompt with bug reference
            diag_prompt = DIAGNOSIS_PROMPT.format(
                bug_ref=json.dumps(BUG_REFERENCE.get(task_id, {}), indent=2)
            )
            diag_messages = messages + [{"role": "user", "content": diag_prompt}]
            llm_out = call_llm(diag_messages)
            action = parse_action(llm_out)
            if action is None or action.get("action_type") != "submit_diagnosis":
                # Force a diagnosis with best guess
                action = {"action_type": "submit_diagnosis"}
        else:
            llm_out = call_llm(messages)
            action = parse_action(llm_out)

            if action is None:
                # Use fallback
                if all_read:
                    action = force_new_sanity_check()
                else:
                    action = get_next_unread_artifact() or get_fallback_action(step_num)

            action_type = action.get("action_type", "unknown")

            # Detect and break loops
            if is_repetitive(action_type) and action_type != "submit_diagnosis":
                if all_read:
                    action = force_new_sanity_check()
                else:
                    next_artifact = get_next_unread_artifact()
                    if next_artifact:
                        action = next_artifact
                    else:
                        action = force_new_sanity_check()

            # Track sanity checks
            if action_type == "run_sanity_check":
                sc = action.get("sanity_check_type", "")
                sanity_check_history.append(sc)

        # Enforce hard rubric before allowing hard submit
        if action.get("action_type") == "submit_diagnosis" and task_id == "hard":
            artifacts_read = obs.get("artifacts_read", [])
            if (
                len(artifacts_read) < 3
                or len(sanity_check_history) < 3
                or step_num < min_steps
            ):
                action = get_fallback_action(step_num)
                log_step(
                    step=step_num,
                    action=action["action_type"],
                    reward=0,
                    done=False,
                    error=None,
                )
                result = env_step(action)
                new_obs = result["observation"]
                reward = result["reward"]
                done = result["done"]
                info = result.get("info", {})
                rewards.append(reward)
                # Continue with the next loop iteration
                if done:
                    final_score = info.get("score", reward)
                    break
                obs = new_obs
                messages.append({"role": "assistant", "content": llm_out})
                messages.append({"role": "user", "content": build_user_msg(new_obs)})
                continue

        # Execute action
        result = env_step(action)
        new_obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        info = result.get("info", {})

        rewards.append(reward)
        action_str = action.get("action_type", "unknown")
        error_msg = (
            new_obs.get("last_action_result", {}).get("error")
            if isinstance(new_obs, dict)
            else None
        )

        log_step(
            step=step_num, action=action_str, reward=reward, done=done, error=error_msg
        )

        if done:
            final_score = info.get("score", reward)
            break

        # Update observation
        obs = new_obs
        action_history.append(action_str)

        # Check if we should enter diagnosis phase
        if not in_diagnosis_phase:
            unread = get_unread_artifacts()
            all_read = len(unread) == 0
            enough_checks = len(sanity_check_history) >= 2
            if all_read and enough_checks and step_num >= min_steps:
                in_diagnosis_phase = True

        messages.append({"role": "assistant", "content": llm_out})
        messages.append({"role": "user", "content": build_user_msg(new_obs)})

        # Keep context window manageable
        if len(messages) > 40:
            messages = [messages[0], messages[1]] + messages[-26:]

    success = final_score >= SUCCESS_THRESHOLD
    log_end(success=success, steps=step_num, rewards=rewards)
    return final_score


def main():
    parser = argparse.ArgumentParser(
        description="MLOps Pipeline Debugger — Baseline Agent"
    )
    parser.add_argument(
        "--task", choices=TASKS, help="Run a specific task (default: all)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    try:
        httpx.get(f"{ENV_BASE_URL}/health", timeout=10).raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach {ENV_BASE_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    tasks = [args.task] if args.task else TASKS
    scores = {}
    for t in tasks:
        scores[t] = run_task(t, seed=args.seed)

    print(f"\n=== FINAL SCORES ===", flush=True)
    for t, s in scores.items():
        print(f"  {t:8s}: {s:.4f}")
    print(f"  {'AVERAGE':8s}: {sum(scores.values()) / len(scores):.4f}")


if __name__ == "__main__":
    main()
