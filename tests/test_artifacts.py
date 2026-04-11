"""Tests for artifact generation — consistency, determinism, and bug planting."""

import json
import pytest
from artifact_generator import (
    ArtifactGenerator, BUG_CATALOGUE, TASK_BUG_POOLS,
    run_sanity_check,
)
import random


class TestArtifactGeneration:
    """Artifacts should be complete, parseable, and internally consistent."""

    @pytest.mark.parametrize("bug_type", list(BUG_CATALOGUE.keys()))
    def test_generates_all_six_artifacts(self, bug_type):
        gen = ArtifactGenerator(bug_type, seed=42)
        artifacts = gen.generate_all()
        expected = {"config.yaml", "train.log", "dataset_stats.json",
                    "preprocessing.py", "eval_results.json", "model_card.json"}
        assert set(artifacts.keys()) == expected

    @pytest.mark.parametrize("bug_type", list(BUG_CATALOGUE.keys()))
    def test_json_artifacts_are_valid(self, bug_type):
        gen = ArtifactGenerator(bug_type, seed=42)
        artifacts = gen.generate_all()
        for name in ["dataset_stats.json", "eval_results.json", "model_card.json"]:
            data = json.loads(artifacts[name])
            assert isinstance(data, dict), f"{name} is not a dict"

    @pytest.mark.parametrize("bug_type", list(BUG_CATALOGUE.keys()))
    def test_config_yaml_has_required_sections(self, bug_type):
        gen = ArtifactGenerator(bug_type, seed=42)
        artifacts = gen.generate_all()
        config = artifacts["config.yaml"]
        for section in ["model:", "training:", "optimizer:", "scheduler:", "data:"]:
            assert section in config, f"Missing {section} in config.yaml"

    @pytest.mark.parametrize("bug_type", list(BUG_CATALOGUE.keys()))
    def test_train_log_has_epochs(self, bug_type):
        gen = ArtifactGenerator(bug_type, seed=42)
        artifacts = gen.generate_all()
        log = artifacts["train.log"]
        assert "EPOCH" in log or "epoch" in log.lower()

    @pytest.mark.parametrize("bug_type", list(BUG_CATALOGUE.keys()))
    def test_preprocessing_is_valid_python(self, bug_type):
        gen = ArtifactGenerator(bug_type, seed=42)
        artifacts = gen.generate_all()
        code = artifacts["preprocessing.py"]
        compile(code, f"<{bug_type}_preprocessing>", "exec")  # syntax check


class TestDeterminism:
    """Same (bug_type, seed) must produce identical artifacts."""

    @pytest.mark.parametrize("bug_type", ["exploding_lr", "data_leakage_scaler", "label_encoder_mismatch"])
    def test_same_seed_same_artifacts(self, bug_type):
        gen1 = ArtifactGenerator(bug_type, seed=42)
        gen2 = ArtifactGenerator(bug_type, seed=42)
        a1 = gen1.generate_all()
        a2 = gen2.generate_all()
        for name in a1:
            assert a1[name] == a2[name], f"{name} differs between runs"

    def test_different_seeds_differ(self):
        gen1 = ArtifactGenerator("exploding_lr", seed=1)
        gen2 = ArtifactGenerator("exploding_lr", seed=999)
        a1 = gen1.generate_all()
        a2 = gen2.generate_all()
        assert a1["config.yaml"] != a2["config.yaml"]


class TestBugPlanting:
    """Each bug type should plant its specific fault in the artifacts."""

    def test_exploding_lr_has_high_lr(self):
        gen = ArtifactGenerator("exploding_lr", seed=42)
        config = gen.generate_all()["config.yaml"]
        # LR should be absurdly high (10, 25, or 50)
        assert any(f"learning_rate: {lr}" in config for lr in ["50.0", "10.0", "25.0"])

    def test_wrong_optimizer_has_high_momentum(self):
        gen = ArtifactGenerator("wrong_optimizer", seed=42)
        config = gen.generate_all()["config.yaml"]
        assert "momentum: 0.99" in config

    def test_batch_size_overflow_has_large_batch(self):
        gen = ArtifactGenerator("batch_size_overflow", seed=42)
        config = gen.generate_all()["config.yaml"]
        assert any(f"batch_size: {bs}" in config for bs in ["2048", "4096", "8192"])

    def test_data_leakage_scaler_fits_before_split(self):
        gen = ArtifactGenerator("data_leakage_scaler", seed=42)
        code = gen.generate_all()["preprocessing.py"]
        assert "fit_transform" in code
        assert "BUG" in code or "sees val/test" in code

    def test_data_leakage_overlap_has_no_random_state(self):
        gen = ArtifactGenerator("data_leakage_overlap", seed=42)
        code = gen.generate_all()["preprocessing.py"]
        assert "random_state=None" in code

    def test_wrong_split_ratio_has_inverted_split(self):
        gen = ArtifactGenerator("wrong_split_ratio", seed=42)
        code = gen.generate_all()["preprocessing.py"]
        assert "test_size=0.8" in code

    def test_label_encoder_mismatch_has_two_encoders(self):
        gen = ArtifactGenerator("label_encoder_mismatch", seed=42)
        code = gen.generate_all()["preprocessing.py"]
        assert "le_train" in code and "le_eval" in code

    def test_silent_metric_swap_has_swapped_assignments(self):
        gen = ArtifactGenerator("silent_metric_swap", seed=42)
        code = gen.generate_all()["preprocessing.py"]
        assert "test_acc" in code and "val_acc" in code

    def test_tokenizer_drift_has_version_mismatch(self):
        gen = ArtifactGenerator("tokenizer_version_drift", seed=42)
        code = gen.generate_all()["preprocessing.py"]
        assert "TOKENIZER_V1" in code and "TOKENIZER_V2" in code


class TestSanityChecks:
    """Sanity checks should detect the planted bug."""

    def test_gradient_norms_detects_exploding_lr(self):
        gen = ArtifactGenerator("exploding_lr", seed=42)
        artifacts = gen.generate_all()
        rng = random.Random(42)
        result = run_sanity_check("gradient_norms", "exploding_lr", artifacts, rng)
        assert result["result"] == "ANOMALY"

    def test_data_leakage_detects_scaler_leak(self):
        gen = ArtifactGenerator("data_leakage_scaler", seed=42)
        artifacts = gen.generate_all()
        rng = random.Random(42)
        result = run_sanity_check("data_leakage", "data_leakage_scaler", artifacts, rng)
        assert result["result"] == "FAIL"

    def test_label_consistency_detects_mismatch(self):
        gen = ArtifactGenerator("label_encoder_mismatch", seed=42)
        artifacts = gen.generate_all()
        rng = random.Random(42)
        result = run_sanity_check("label_consistency", "label_encoder_mismatch", artifacts, rng)
        assert result["result"] == "FAIL"

    def test_encoder_version_detects_drift(self):
        gen = ArtifactGenerator("tokenizer_version_drift", seed=42)
        artifacts = gen.generate_all()
        rng = random.Random(42)
        result = run_sanity_check("encoder_version_match", "tokenizer_version_drift", artifacts, rng)
        assert result["result"] == "MISMATCH"

    def test_metric_gap_detects_hard_bugs(self):
        for bug_type in TASK_BUG_POOLS["hard"]:
            gen = ArtifactGenerator(bug_type, seed=42)
            artifacts = gen.generate_all()
            rng = random.Random(42)
            result = run_sanity_check("metric_gap_analysis", bug_type, artifacts, rng)
            assert result["result"] == "ANOMALY", f"metric_gap missed {bug_type}"

    def test_unknown_check_returns_unknown(self):
        gen = ArtifactGenerator("exploding_lr", seed=42)
        artifacts = gen.generate_all()
        rng = random.Random(42)
        result = run_sanity_check("nonexistent_check", "exploding_lr", artifacts, rng)
        assert result["result"] == "UNKNOWN"


class TestBugCatalogue:
    """Bug catalogue should be complete and consistent."""

    def test_all_bugs_have_required_fields(self):
        for name, bug in BUG_CATALOGUE.items():
            assert bug.bug_type == name
            assert bug.category in [
                "config_error", "data_leakage", "preprocessing_bug",
                "evaluation_bug", "label_mismatch", "architecture_bug",
            ]
            assert bug.file.endswith((".yaml", ".py", ".json"))
            assert len(bug.field) > 0
            assert len(bug.gold_fix) > 10
            assert bug.task_difficulty in ["easy", "medium", "hard"]

    def test_task_pools_cover_all_bugs(self):
        all_pooled = set()
        for pool in TASK_BUG_POOLS.values():
            all_pooled.update(pool)
        assert all_pooled == set(BUG_CATALOGUE.keys())

    def test_each_pool_has_three_bugs(self):
        for task_id, pool in TASK_BUG_POOLS.items():
            assert len(pool) == 3, f"{task_id} has {len(pool)} bugs, expected 3"
