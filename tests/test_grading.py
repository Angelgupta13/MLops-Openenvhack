"""Tests for the grading system — score ranges, component scoring, and determinism."""

import pytest
from mlops_environment import MLOpsEnvironment, grade_task
from artifact_generator import BUG_CATALOGUE, TASK_BUG_POOLS
from models import MLOpsAction


class TestScoreRange:
    """All scores must be strictly between 0 and 1."""

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_perfect_diagnosis_below_1(self, task_id):
        env = MLOpsEnvironment(task_id=task_id)
        env.reset(seed=42)
        env._artifacts_read = list(env._artifacts.keys())
        bug = env.bug
        obs, reward, done, info = env.step(MLOpsAction(
            action_type="submit_diagnosis",
            failure_category=bug.category,
            root_cause_file=bug.file,
            root_cause_field=bug.field,
            diagnosis="test",
            proposed_fix=bug.gold_fix,
        ))
        score = info["score"]
        assert 0 < score < 1, f"Perfect diagnosis score {score} is not in (0, 1)"
        assert score <= 0.99

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_empty_diagnosis_above_0(self, task_id):
        env = MLOpsEnvironment(task_id=task_id)
        env.reset(seed=42)
        obs, reward, done, info = env.step(MLOpsAction(action_type="submit_diagnosis"))
        score = info["score"]
        assert 0 < score < 1, f"Empty diagnosis score {score} is not in (0, 1)"
        assert score >= 0.01

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_wrong_diagnosis_above_0(self, task_id):
        env = MLOpsEnvironment(task_id=task_id)
        env.reset(seed=42)
        env._artifacts_read = list(env._artifacts.keys())
        obs, reward, done, info = env.step(MLOpsAction(
            action_type="submit_diagnosis",
            failure_category="architecture_bug",
            root_cause_file="nonexistent.py",
            root_cause_field="wrong.field",
            diagnosis="completely wrong",
            proposed_fix="do nothing",
        ))
        score = info["score"]
        assert 0 < score < 1, f"Wrong diagnosis score {score} is not in (0, 1)"

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    @pytest.mark.parametrize("seed", [1, 42, 100, 999, 54321])
    def test_score_range_across_seeds(self, task_id, seed):
        env = MLOpsEnvironment(task_id=task_id)
        env.reset(seed=seed)
        env._artifacts_read = list(env._artifacts.keys())
        bug = env.bug
        obs, reward, done, info = env.step(MLOpsAction(
            action_type="submit_diagnosis",
            failure_category=bug.category,
            root_cause_file=bug.file,
            root_cause_field=bug.field,
            diagnosis="test",
            proposed_fix=bug.gold_fix,
        ))
        score = info["score"]
        assert 0 < score < 1, f"Score {score} out of range for {task_id}/seed={seed}"


class TestComponentScoring:
    """Each scoring component should award correct points."""

    @pytest.fixture
    def env_with_bug(self):
        env = MLOpsEnvironment(task_id="easy")
        env.reset(seed=42)
        env._artifacts_read = list(env._artifacts.keys())
        return env, env.bug

    def test_category_only(self, env_with_bug):
        env, bug = env_with_bug
        obs, reward, done, info = env.step(MLOpsAction(
            action_type="submit_diagnosis",
            failure_category=bug.category,
        ))
        bd = info["breakdown"]
        assert bd["failure_category"]["correct"] is True
        assert bd["failure_category"]["awarded"] == 0.15

    def test_category_plus_file(self, env_with_bug):
        env, bug = env_with_bug
        obs, reward, done, info = env.step(MLOpsAction(
            action_type="submit_diagnosis",
            failure_category=bug.category,
            root_cause_file=bug.file,
        ))
        bd = info["breakdown"]
        assert bd["failure_category"]["correct"] is True
        assert bd["root_cause_file"]["correct"] is True
        assert info["score"] >= 0.35

    def test_file_match_case_insensitive(self, env_with_bug):
        env, bug = env_with_bug
        obs, reward, done, info = env.step(MLOpsAction(
            action_type="submit_diagnosis",
            failure_category=bug.category,
            root_cause_file=bug.file.upper(),
        ))
        assert info["breakdown"]["root_cause_file"]["correct"] is True

    def test_partial_fix_scoring(self, env_with_bug):
        env, bug = env_with_bug
        # Submit just one keyword from the gold fix
        first_word = bug.gold_fix.split()[0]
        obs, reward, done, info = env.step(MLOpsAction(
            action_type="submit_diagnosis",
            failure_category=bug.category,
            proposed_fix=first_word,
        ))
        fix_awarded = info["breakdown"]["proposed_fix"]["awarded"]
        assert fix_awarded > 0  # partial credit


class TestHardTaskPenalty:
    """Hard task should apply 1.5x penalty when score < 0.70."""

    def test_penalty_applied_on_low_score(self):
        env = MLOpsEnvironment(task_id="hard")
        env.reset(seed=42)
        env._artifacts_read = list(env._artifacts.keys())
        # Submit with only category correct → score ~0.15, well below 0.70
        obs, reward, done, info = env.step(MLOpsAction(
            action_type="submit_diagnosis",
            failure_category=env.bug.category,
        ))
        assert info["breakdown"].get("hard_task_penalty_applied") is True
        assert info["score"] < 0.15  # penalty reduces it

    def test_no_penalty_on_high_score(self):
        env = MLOpsEnvironment(task_id="hard")
        env.reset(seed=42)
        env._artifacts_read = list(env._artifacts.keys())
        bug = env.bug
        obs, reward, done, info = env.step(MLOpsAction(
            action_type="submit_diagnosis",
            failure_category=bug.category,
            root_cause_file=bug.file,
            root_cause_field=bug.field,
            diagnosis="test",
            proposed_fix=bug.gold_fix,
        ))
        assert info["breakdown"].get("hard_task_penalty_applied") is not True
        assert info["score"] >= 0.70


class TestGraderDeterminism:
    """Same inputs must always produce identical scores."""

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_same_seed_same_score(self, task_id):
        scores = []
        for _ in range(3):
            env = MLOpsEnvironment(task_id=task_id)
            env.reset(seed=42)
            env._artifacts_read = list(env._artifacts.keys())
            bug = env.bug
            obs, _, _, info = env.step(MLOpsAction(
                action_type="submit_diagnosis",
                failure_category=bug.category,
                root_cause_file=bug.file,
                root_cause_field=bug.field,
                proposed_fix=bug.gold_fix,
            ))
            scores.append(info["score"])
        assert scores[0] == scores[1] == scores[2], f"Non-deterministic: {scores}"


class TestGradeTaskStandalone:
    """grade_task() must match environment grading and respect score range."""

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_grade_task_score_in_range(self, task_id):
        pool = TASK_BUG_POOLS[task_id]
        for bug_name in pool:
            bug = BUG_CATALOGUE[bug_name]
            score = grade_task(task_id, seed=42, diagnosis={
                "failure_category": bug.category,
                "root_cause_file": bug.file,
                "root_cause_field": bug.field,
                "proposed_fix": bug.gold_fix,
            })
            assert 0 < score < 1, f"grade_task score {score} out of range for {bug_name}"

    def test_grade_task_empty_diagnosis(self):
        score = grade_task("easy", seed=42, diagnosis={})
        assert 0 < score < 1
