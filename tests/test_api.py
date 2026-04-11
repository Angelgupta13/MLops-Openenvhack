"""Tests for the FastAPI server — endpoint responses and error handling."""

import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthAndInfo:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "MLOps Pipeline Debugger API" in r.json()["message"]

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_tasks(self, client):
        r = client.get("/tasks")
        assert r.status_code == 200
        tasks = r.json()["tasks"]
        assert len(tasks) == 3
        task_ids = {t["task_id"] for t in tasks}
        assert task_ids == {"easy", "medium", "hard"}


class TestResetEndpoint:
    def test_reset_easy(self, client):
        r = client.post("/reset", json={"task_id": "easy", "seed": 42})
        assert r.status_code == 200
        data = r.json()
        assert data["task_id"] == "easy"
        assert data["step_count"] == 0
        assert data["done"] is False
        assert len(data["available_artifacts"]) == 6

    def test_reset_hard(self, client):
        r = client.post("/reset", json={"task_id": "hard", "seed": 42})
        assert r.status_code == 200
        assert r.json()["task_id"] == "hard"

    def test_reset_default(self, client):
        r = client.post("/reset", json={})
        assert r.status_code == 200
        assert r.json()["task_id"] == "easy"


class TestStepEndpoint:
    def test_step_read_config(self, client):
        client.post("/reset", json={"task_id": "easy", "seed": 42})
        r = client.post("/step", json={"action_type": "read_config"})
        assert r.status_code == 200
        data = r.json()
        assert data["reward"] == 0.02
        assert data["done"] is False

    def test_step_submit_diagnosis(self, client):
        client.post("/reset", json={"task_id": "easy", "seed": 42})
        r = client.post("/step", json={
            "action_type": "submit_diagnosis",
            "failure_category": "config_error",
            "root_cause_file": "config.yaml",
            "root_cause_field": "optimizer.learning_rate",
            "proposed_fix": "Reduce learning_rate",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["done"] is True
        assert 0 < data["info"]["score"] < 1

    def test_step_invalid_action(self, client):
        client.post("/reset", json={"task_id": "easy", "seed": 42})
        r = client.post("/step", json={"action_type": "invalid_action"})
        assert r.status_code == 422

    def test_step_nested_action_format(self, client):
        client.post("/reset", json={"task_id": "easy", "seed": 42})
        r = client.post("/step", json={"action": {"action_type": "read_config"}})
        assert r.status_code == 200


class TestStateEndpoint:
    def test_state_after_reset(self, client):
        client.post("/reset", json={"task_id": "easy", "seed": 42})
        r = client.get("/state")
        assert r.status_code == 200
        data = r.json()
        assert data["task_id"] == "easy"
        assert data["seed"] == 42
        assert "bug_type" in data


class TestOpenEnvState:
    def test_openenv_state(self, client):
        r = client.get("/openenv/state")
        assert r.status_code == 200
        data = r.json()
        assert "scores" in data
        assert "easy" in data["scores"]
