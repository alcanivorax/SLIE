from fastapi.testclient import TestClient

from slie.app import app


client = TestClient(app)


def test_root_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_reset_endpoint() -> None:
    response = client.post("/reset", json={"task_id": "task1", "episode_seed": 0})
    assert response.status_code == 200
    data = response.json()
    assert len(data["observation"]["gesture_embedding"]) == 64
    assert len(data["observation"]["hand_landmarks"]) == 6


def test_reset_endpoint_without_body() -> None:
    response = client.post("/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "task1"
