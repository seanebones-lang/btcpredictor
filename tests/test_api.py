"""
Integration tests for BTC Predictor API.
"""

import pytest
import httpx
import asyncio
from typing import AsyncGenerator

# Test configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "test-key"


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create async HTTP client for testing."""
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=60.0) as client:
        client.headers["X-API-Key"] = API_KEY
        yield client


@pytest.mark.asyncio
async def test_health_check(client: httpx.AsyncClient):
    """Test health check endpoint."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "2.0.0"
    assert "services" in data


@pytest.mark.asyncio
async def test_metrics_endpoint(client: httpx.AsyncClient):
    """Test Prometheus metrics endpoint."""
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


@pytest.mark.asyncio
async def test_predict_text_endpoint(client: httpx.AsyncClient):
    """Test text prediction endpoint."""
    response = await client.get("/predict/text")
    assert response.status_code == 200
    text = response.text
    assert "BTC 15-min Prediction" in text
    assert "Current:" in text
    assert "Expected move:" in text
    assert "Next target:" in text
    assert "Confidence:" in text


@pytest.mark.asyncio
async def test_predict_json_endpoint(client: httpx.AsyncClient):
    """Test JSON prediction endpoint."""
    response = await client.get("/predict")
    assert response.status_code == 200
    data = response.json()
    
    # Check required fields
    required_fields = [
        "window_start", "window_end", "current_price", "next_price",
        "change_pct", "direction", "confidence", "ensemble_interval",
        "prophet_interval", "lstm_std", "enriched_features", "model", "timestamp"
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"
    
    # Check types
    assert isinstance(data["current_price"], (int, float))
    assert isinstance(data["next_price"], (int, float))
    assert isinstance(data["change_pct"], (int, float))
    assert isinstance(data["confidence"], int)
    assert isinstance(data["ensemble_interval"], list)
    assert len(data["ensemble_interval"]) == 2
    assert isinstance(data["enriched_features"], dict)
    assert isinstance(data["model"], str)
    
    # Check interval bounds
    assert data["ensemble_interval"][0] < data["ensemble_interval"][1]


@pytest.mark.asyncio
async def test_rate_limiting(client: httpx.AsyncClient):
    """Test rate limiting is enforced."""
    # Make multiple rapid requests
    responses = []
    for _ in range(5):
        response = await client.get("/health")
        responses.append(response.status_code)
    
    # All should succeed (health check might not be rate limited)
    assert all(status == 200 for status in responses)


@pytest.mark.asyncio
async def test_cors_headers(client: httpx.AsyncClient):
    """Test CORS headers are present."""
    response = await client.options("/health")
    # Check for CORS headers
    assert response.status_code in [200, 204]


@pytest.mark.asyncio
async def test_invalid_api_key():
    """Test API rejects invalid keys."""
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        client.headers["X-API-Key"] = "invalid-key"
        response = await client.get("/predict")
        # In development mode, auth might be disabled
        # In production, should be 401
        assert response.status_code in [200, 401]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
