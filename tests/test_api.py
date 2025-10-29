"""Pytest smoke tests for the sentiment and toxicity API."""

import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture(scope="module")
def client():
    """Create a test client with lifespan events."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_analyze_positive_text(client):
    """Test analysis of positive text."""
    response = client.post(
        "/analyze",
        json={"text": "I love this product"}
    )
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "sentiment" in data
    assert "toxicity" in data
    assert "model_backend" in data
    assert "latency_ms" in data
    
    # Check sentiment structure
    assert "label" in data["sentiment"]
    assert "score" in data["sentiment"]
    
    # Validate types
    assert isinstance(data["sentiment"]["label"], str)
    assert isinstance(data["sentiment"]["score"], float)
    assert isinstance(data["toxicity"], float)
    assert isinstance(data["latency_ms"], float)
    
    # Validate ranges
    assert 0.0 <= data["sentiment"]["score"] <= 1.0
    assert 0.0 <= data["toxicity"] <= 1.0
    assert data["latency_ms"] >= 0


def test_analyze_negative_text(client):
    """Test analysis of negative/toxic text."""
    response = client.post(
        "/analyze",
        json={"text": "This is terrible and awful"}
    )
    assert response.status_code == 200
    data = response.json()
    
    # Should detect negative sentiment
    assert data["sentiment"]["label"] in ["POSITIVE", "NEGATIVE"]
    
    # Check all required fields present
    assert "toxicity" in data
    assert "model_backend" in data
    assert "latency_ms" in data


def test_analyze_empty_text_rejected(client):
    """Test that empty text is rejected."""
    response = client.post(
        "/analyze",
        json={"text": ""}
    )
    assert response.status_code == 422  # Validation error


def test_analyze_missing_text_rejected(client):
    """Test that missing text field is rejected."""
    response = client.post(
        "/analyze",
        json={}
    )
    assert response.status_code == 422  # Validation error


def test_backend_identifier(client):
    """Test that backend identifier is valid."""
    response = client.post(
        "/analyze",
        json={"text": "test text"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_backend"] in ["transformer", "tfidf"]

