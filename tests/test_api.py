from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "twitter-roberta" in data["model"]


def test_predict_valid_payload():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "Amazing customer support, very happy!"})
        assert response.status_code == 200
        data = response.json()
        assert data["label"] in {"positive", "neutral", "negative"}
        assert 0.0 <= data["score"] <= 1.0
        assert set(data["probabilities"].keys()) == {"positive", "neutral", "negative"}


def test_predict_empty_text_returns_422():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422


def test_metrics_endpoint_exposed():
    with TestClient(app) as client:
        client.post("/predict", json={"text": "Testing metrics collection"})
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "sentiment_predictions_total" in response.text
