"""Tests for MLOpsEnvironment — core episode flow, state management, and step logic."""

import pytest
from mlops_environment import MLOpsEnvironment, TASK_MAX_STEPS
from models import MLOpsAction


class TestReset:
    """reset() should produce a clean, valid initial state."""

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_reset_returns_valid_observation(self, task_id):
        env = MLOpsEnvironment(task_id=task_id)
        obs = env.reset(seed=42)
        assert obs.task_id == task_id
        assert obs.step_count == 0
        assert obs.max_steps == TASK_MAX_STEPS[task_id]
        assert obs.done is False
        assert len(obs.available_artifacts) == 6
        assert obs.artifacts_read == []

    def test_reset_with_same_seed_is_deterministic(self):
        env1 = MLOpsEnvironment(task_id="easy")
        env2 = MLOpsEnvironment(task_id="easy")
        obs1 = env1.reset(seed=123)
        obs2 = env2.reset(seed=123)
        assert obs1.run_id == obs2.run_id
        assert env1.bug_type == env2.bug_type

    def test_reset_with_different_seeds_varies(self):
        env = MLOpsEnvironment(task_id="easy")
        obs1 = env.reset(seed=1)
        run_id_1 = obs1.run_id
        obs2 = env.reset(seed=999)
        assert obs2.run_id != run_id_1

    def test_reset_clears_previous_episode(self):
        env = MLOpsEnvironment(task_id="easy")
        env.reset(seed=42)
        env.step(MLOpsAction(action_type="read_config"))
        assert len(env._artifacts_read) == 1
        env.reset(seed=42)
        assert len(env._artifacts_read) == 0
        assert env._step_count == 0


class TestStepActions:
    """Each action type should return expected structure and reward."""

    @pytest.fixture
    def env(self):
        env = MLOpsEnvironment(task_id="easy")
        env.reset(seed=42)
        return env

    def test_read_config(self, env):
        obs, reward, done, info = env.step(MLOpsAction(action_type="read_config"))
        assert reward == 0.02
        assert done is False
        assert "config.yaml" in obs.artifacts_read
        assert "content" in obs.last_action_result

    def test_read_logs(self, env):
        obs, reward, done, info = env.step(MLOpsAction(action_type="read_logs"))
        assert reward == 0.02
        assert "train.log" in obs.artifacts_read

    def test_read_logs_with_filter(self, env):
        obs, reward, done, info = env.step(
            MLOpsAction(action_type="read_logs", log_filter="epoch:1-3")
        )
        assert reward == 0.02
        content = obs.last_action_result.get("content", "")
        assert "EPOCH" in content or "No log lines" in content

    def test_check_dataset_stats(self, env):
        obs, reward, done, info = env.step(MLOpsAction(action_type="check_dataset_stats"))
        assert reward == 0.02
        assert "dataset_stats.json" in obs.artifacts_read

    def test_inspect_preprocessing(self, env):
        obs, reward, done, info = env.step(MLOpsAction(action_type="inspect_preprocessing"))
        assert reward == 0.02
        assert "preprocessing.py" in obs.artifacts_read

    def test_read_eval_results(self, env):
        obs, reward, done, info = env.step(MLOpsAction(action_type="read_eval_results"))
        assert reward == 0.02
        assert "eval_results.json" in obs.artifacts_read

    def test_run_sanity_check(self, env):
        obs, reward, done, info = env.step(
            MLOpsAction(action_type="run_sanity_check", sanity_check_type="loss_trajectory")
        )
        assert reward == 0.01
        assert obs.last_action_result["status"] == "ok"
        assert "sanity_check" in obs.last_action_result

    def test_query_artifact(self, env):
        env.step(MLOpsAction(action_type="read_config"))
        obs, reward, done, info = env.step(
            MLOpsAction(action_type="query_artifact", artifact_name="config.yaml", field_path="model.architecture")
        )
        assert obs.last_action_result["status"] == "ok"

    def test_duplicate_read_penalty(self, env):
        env.step(MLOpsAction(action_type="read_config"))
        obs, reward, done, info = env.step(MLOpsAction(action_type="read_config"))
        assert reward == -0.02

    def test_step_count_increments(self, env):
        env.step(MLOpsAction(action_type="read_config"))
        env.step(MLOpsAction(action_type="read_logs"))
        assert env._step_count == 2

    def test_done_after_submit(self, env):
        obs, reward, done, info = env.step(MLOpsAction(action_type="submit_diagnosis"))
        assert done is True

    def test_step_after_done_returns_done(self, env):
        env.step(MLOpsAction(action_type="submit_diagnosis"))
        obs, reward, done, info = env.step(MLOpsAction(action_type="read_config"))
        assert done is True
        assert reward == 0.01  # clamped minimum
        assert "score" in info


class TestEpisodeBoundaries:
    """Episode should terminate correctly on submit, timeout, and re-step."""

    def test_timeout_at_max_steps(self):
        env = MLOpsEnvironment(task_id="easy")
        env.reset(seed=42)
        for _ in range(TASK_MAX_STEPS["easy"]):
            obs, reward, done, info = env.step(MLOpsAction(action_type="read_config"))
            if done:
                break
        assert done is True
        assert "score" in info

    def test_submit_ends_episode(self):
        env = MLOpsEnvironment(task_id="medium")
        env.reset(seed=42)
        env.step(MLOpsAction(action_type="read_logs"))
        obs, reward, done, info = env.step(MLOpsAction(action_type="submit_diagnosis"))
        assert done is True
        assert "score" in info
        assert "breakdown" in info
