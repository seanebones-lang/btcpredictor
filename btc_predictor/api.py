"""
FastAPI wrapper for BTC Predictor with authentication, rate limiting, and monitoring.
"""

import asyncio
import time
import os
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from .predictor_async import (
    generate_async_prediction,
    get_fetcher,
    close_fetcher,
    AsyncDataFetcher,
    EnrichedConfig,
)

# ============================================================
# Configuration
# ============================================================

API_KEY = os.getenv("API_KEY", "dev-key-change-in-production")
RATE_LIMIT = os.getenv("RATE_LIMIT", "60/minute")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# ============================================================
# Prometheus Metrics
# ============================================================

REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"]
)
ACTIVE_PREDICTIONS = Gauge("active_predictions", "Number of active predictions")
PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds", "Prediction generation latency"
)
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence", "Prediction confidence scores"
)

# ============================================================
# Rate Limiter
# ============================================================

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

# ============================================================
# Lifespan
# ============================================================

_fetcher_instance: Optional[AsyncDataFetcher] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global _fetcher_instance
    _fetcher_instance = AsyncDataFetcher()
    await _fetcher_instance.__aenter__()
    yield
    if _fetcher_instance:
        await _fetcher_instance.__aexit__(None, None, None)
        _fetcher_instance = None

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="BTC Predictor API",
    description="Production-grade Bitcoin price prediction API with ensemble models",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ENVIRONMENT == "development" else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Metrics Middleware
# ============================================================

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response

# ============================================================
# Authentication
# ============================================================

async def verify_api_key(request: Request) -> str:
    """Verify API key from header."""
    if ENVIRONMENT == "development":
        return "dev"
    
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key

# ============================================================
# Pydantic Models
# ============================================================

class PredictionResponse(BaseModel):
    """Prediction response model."""
    window_start: str
    window_end: str
    current_price: float
    next_price: float
    change_pct: float
    direction: str
    confidence: int
    ensemble_interval: list[float]
    prophet_interval: list[float]
    lstm_std: float
    enriched_features: Dict[str, Any]
    model: str
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str
    services: Dict[str, str]

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None

# ============================================================
# Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        timestamp=datetime.utcnow().isoformat(),
        services={
            "api": "operational",
            "fetcher": "operational",
            "redis": "operational",
        }
    )

@app.get("/metrics", response_class=PlainTextResponse, tags=["monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

@app.get("/predict", response_model=PredictionResponse, tags=["predictions"])
@limiter.limit(RATE_LIMIT)
async def predict(
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """Generate a BTC 15-minute price prediction."""
    ACTIVE_PREDICTIONS.inc()
    start_time = time.time()
    
    try:
        fetcher = await get_fetcher()
        config = EnrichedConfig()
        
        # Import the internal function
        from .predictor_async import get_prediction_with_intervals, EnrichedConfig as AsyncConfig
        
        result = await get_prediction_with_intervals(fetcher, AsyncConfig())
        
        # Convert numpy types
        import numpy as np
        def to_native(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_native(v) for v in obj]
            return obj
        
        result = {k: to_native(v) for k, v in result.items()}
        
        return PredictionResponse(
            window_start=result['window_start'],
            window_end=result['window_end'],
            current_price=result['current_price'],
            next_price=result['next_price'],
            change_pct=result['change_pct'],
            direction=result['direction'],
            confidence=result['confidence'],
            ensemble_interval=result['ensemble_interval'],
            prophet_interval=result['prophet_interval'],
            lstm_std=result['lstm_std'],
            enriched_features=result['enriched_features'],
            model=result['model'],
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
    finally:
        ACTIVE_PREDICTIONS.dec()
        PREDICTION_LATENCY.observe(time.time() - start_time)

@app.get("/predict/text", response_class=PlainTextResponse, tags=["predictions"])
@limiter.limit(RATE_LIMIT)
async def predict_text(
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """Generate a BTC 15-minute price prediction as formatted text."""
    try:
        prediction = await generate_async_prediction()
        return prediction
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=ENVIRONMENT == "development",
    )