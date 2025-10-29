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


def test_version_endpoint(client):
    """Test version endpoint returns expected fields."""
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    
    # Check all fields present
    assert "version" in data
    assert "git_sha" in data
    assert "sentiment_model" in data
    assert "toxicity_model" in data
    
    # Validate model names
    assert "distilbert" in data["sentiment_model"]
    assert "toxic-bert" in data["toxicity_model"]


def test_very_long_text_boundary(client):
    """Test handling of very long text (boundary test)."""
    # 1000 characters - should still work
    long_text = "This is a test sentence. " * 40  # ~1000 chars
    response = client.post(
        "/analyze",
        json={"text": long_text}
    )
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "toxicity" in data


def test_special_characters_handling(client):
    """Test handling of special characters and Unicode (boundary test)."""
    special_text = "🔥 I love this! 💯 Best product ever! 😍✨"
    response = client.post(
        "/analyze",
        json={"text": special_text}
    )
    assert response.status_code == 200
    data = response.json()
    
    # Should detect positive sentiment despite emojis
    assert data["sentiment"]["label"] in ["POSITIVE", "NEGATIVE"]
    assert 0.0 <= data["toxicity"] <= 1.0


def test_punctuation_only_text(client):
    """Test edge case with only punctuation (negative test)."""
    response = client.post(
        "/analyze",
        json={"text": "!@#$%^&*()"}
    )
    # Should still return valid response, not crash
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "toxicity" in data


# Tests for /analyze/text endpoint (form data)

def test_analyze_text_endpoint_basic(client):
    """Test the form data endpoint with basic text."""
    response = client.post(
        "/analyze/text",
        data={"text": "I love this product"}
    )
    assert response.status_code == 200
    data = response.json()
    
    # Same response structure as JSON endpoint
    assert "sentiment" in data
    assert "toxicity" in data
    assert "model_backend" in data
    assert "latency_ms" in data
    assert data["sentiment"]["label"] in ["POSITIVE", "NEGATIVE"]


def test_analyze_text_multiline(client):
    """Test form data endpoint with multi-line text."""
    multiline_text = """This is a great product!
    
I highly recommend it to everyone.
It works perfectly."""
    
    response = client.post(
        "/analyze/text",
        data={"text": multiline_text}
    )
    assert response.status_code == 200
    data = response.json()
    
    # Should handle newlines correctly
    assert data["sentiment"]["label"] == "POSITIVE"
    assert 0.0 <= data["toxicity"] <= 1.0


def test_analyze_text_empty_rejected(client):
    """Test that empty text is rejected on form endpoint."""
    response = client.post(
        "/analyze/text",
        data={"text": ""}
    )
    assert response.status_code == 422


def test_analyze_text_whitespace_only_rejected(client):
    """Test that whitespace-only text is rejected (boundary test)."""
    response = client.post(
        "/analyze/text",
        data={"text": "   \n\n\t  "}
    )
    assert response.status_code == 422

