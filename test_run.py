import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from client import MLOpsDebugEnv
from models import MLOpsAction

with MLOpsDebugEnv(base_url="http://localhost:7860").sync() as env:
    obs = env.reset(task_id="easy", seed=42)
    print(f"Task: {obs.task_id}")
    print(f"Summary: {obs.run_summary}")
    print()

    r = env.step(MLOpsAction(action_type="read_logs"))
    print(f"Step 1 (read_logs) reward={r.reward}")
    content = str(r.observation.last_action_result.get("content", ""))
    print(content[:500])
    print()

    r = env.step(MLOpsAction(action_type="read_config"))
    print(f"Step 2 (read_config) reward={r.reward}")
    content = str(r.observation.last_action_result.get("content", ""))
    print(content[:500])
    print()

    r = env.step(
        MLOpsAction(
            action_type="submit_diagnosis",
            failure_category="config_error",
            root_cause_file="config.yaml",
            root_cause_field="training.batch_size",
            proposed_fix="Reduce batch_size from 4096 to 32 or 64; current size exceeds training set",
        )
    )
    print(f"Step 3 (submit_diagnosis) reward={r.reward}, done={r.done}")
    print(f"Score: {r.info['score']}")
