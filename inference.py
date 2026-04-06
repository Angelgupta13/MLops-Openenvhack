"""
inference.py — Baseline LLM Agent for MLOps Pipeline Debugger

Required env vars:
    API_BASE_URL   e.g. https://router.huggingface.co/v1
    MODEL_NAME     e.g. Qwen/Qwen2.5-72B-Instruct
    HF_TOKEN       your Hugging Face token
    ENV_BASE_URL   defaults to http://localhost:7860

STDOUT FORMAT (mandatory):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "mlops-debug-env"
TASKS = ["easy", "medium", "hard"]
SUCCESS_THRESHOLD = 0.5

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are a senior ML engineer investigating a broken training run.
You have access to training artifacts and must diagnose the root cause.

INVESTIGATION STRATEGY:
1. Always start by reading training logs to identify the symptom pattern
2. Read eval_results to check val vs test metric gap
3. Inspect preprocessing code for data pipeline issues
4. Read config for hyperparameter problems
5. Run targeted sanity checks to confirm your hypothesis
6. Submit diagnosis ONLY when you are confident

FAILURE CATEGORIES:
- config_error        : Wrong hyperparameter (learning rate, batch size, momentum)
- data_leakage        : Train/val contamination through scaler or overlapping splits
- evaluation_bug      : Eval pipeline uses wrong artifacts or swapped metrics
- preprocessing_bug   : Data transformation applied incorrectly
- label_mismatch      : Label encoding inconsistency between train and eval
- architecture_bug    : Model architecture misconfiguration

RESPOND WITH ONE JSON ACTION OBJECT PER TURN. Examples:
{"action_type": "read_logs", "log_filter": "nan"}
{"action_type": "read_logs", "log_filter": "epoch:1-3"}
{"action_type": "run_sanity_check", "sanity_check_type": "gradient_norms"}
{"action_type": "run_sanity_check", "sanity_check_type": "metric_gap_analysis"}
{"action_type": "inspect_preprocessing"}
{"action_type": "read_config"}
{"action_type": "query_artifact", "artifact_name": "config.yaml", "field_path": "optimizer.learning_rate"}
{"action_type": "submit_diagnosis",
 "failure_category": "config_error",
 "root_cause_file": "config.yaml",
 "root_cause_field": "optimizer.learning_rate",
 "diagnosis": "The learning rate is set to 50.0, causing gradient explosion and NaN loss.",
 "proposed_fix": "Reduce learning_rate to 1e-4 and add gradient clipping"}

ONLY output the JSON object. No explanation. No markdown."""


# ── Logging helpers (mandatory format) ────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# ── Environment helpers ───────────────────────────────────────────────────────

def env_reset(task_id: str, seed: int = 42) -> Dict[str, Any]:
    r = httpx.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    r = httpx.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


def build_user_msg(obs: Dict[str, Any]) -> str:
    arts_read = obs.get("artifacts_read", [])
    pending = [a["name"] for a in obs.get("available_artifacts", []) if a["name"] not in arts_read]
    last = obs.get("last_action_result", {})
    step = obs.get("step_count", 0)
    max_s = obs.get("max_steps", 30)
    run = obs.get("run_summary", {})

    lines = [
        f"=== STEP {step}/{max_s} ===",
        f"Run: {obs.get('run_id','')} | Model: {run.get('model','')} | Status: {run.get('status','')}",
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
        m = re.search(r'\{[\s\S]+\}', text)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


def call_llm(messages: List[Dict]) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, max_tokens=512, temperature=0.1
    )
    return resp.choices[0].message.content.strip()


# ── Safe fallback actions ─────────────────────────────────────────────────────

# Ordered list of safe exploratory actions to try when LLM output is unparseable
FALLBACK_ACTIONS = [
    {"action_type": "read_logs"},
    {"action_type": "read_eval_results"},
    {"action_type": "read_config"},
    {"action_type": "inspect_preprocessing"},
    {"action_type": "check_dataset_stats"},
    {"action_type": "run_sanity_check", "sanity_check_type": "metric_gap_analysis"},
]


def get_fallback_action(step_num: int) -> Dict[str, Any]:
    """Return a safe exploratory action instead of a garbage diagnosis."""
    idx = min(step_num - 1, len(FALLBACK_ACTIONS) - 1)
    return FALLBACK_ACTIONS[idx]


# ── Main agent loop ──────────────────────────────────────────────────────────

def run_task(task_id: str, seed: int = 42) -> float:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    obs = env_reset(task_id, seed)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"TASK DESCRIPTION:\n{obs['task_description']}\n\n{build_user_msg(obs)}"},
    ]

    step_num = 0
    done = False
    final_score = 0.0
    rewards: List[float] = []
    parse_failures = 0

    while not done:
        step_num += 1
        llm_out = call_llm(messages)
        action = parse_action(llm_out)

        if action is None:
            parse_failures += 1
            # Use safe fallback instead of submitting a garbage diagnosis
            action = get_fallback_action(step_num)
        else:
            parse_failures = 0

        result = env_step(action)
        new_obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        info = result.get("info", {})

        rewards.append(reward)
        action_str = action.get("action_type", "unknown")
        error_msg = new_obs.get("last_action_result", {}).get("error") if isinstance(new_obs, dict) else None

        log_step(step=step_num, action=action_str, reward=reward, done=done, error=error_msg)

        if done:
            final_score = info.get("score", reward)
            break

        messages.append({"role": "assistant", "content": llm_out})
        messages.append({"role": "user", "content": build_user_msg(new_obs)})

        # Keep context window manageable: preserve system prompt + recent history
        if len(messages) > 40:
            messages = [messages[0], messages[1]] + messages[-26:]

    success = final_score >= SUCCESS_THRESHOLD
    log_end(success=success, steps=step_num, rewards=rewards)
    return final_score


def main():
    parser = argparse.ArgumentParser(description="MLOps Pipeline Debugger — Baseline Agent")
    parser.add_argument("--task", choices=TASKS, help="Run a specific task (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
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
