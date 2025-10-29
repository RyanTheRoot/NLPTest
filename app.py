"""FastAPI application for sentiment and toxicity analysis."""

import time
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from inference import InferenceService


# Request and response models
class AnalyzeRequest(BaseModel):
    """Request model for text analysis."""
    text: str = Field(..., min_length=1, description="Text to analyze")


class SentimentResult(BaseModel):
    """Sentiment analysis result."""
    label: str = Field(..., description="POSITIVE or NEGATIVE")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class AnalyzeResponse(BaseModel):
    """Response model for text analysis."""
    sentiment: SentimentResult
    toxicity: float = Field(..., ge=0.0, le=1.0, description="Toxicity score")
    model_backend: str = Field(..., description="Backend used: transformer or tfidf")
    latency_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str


# Global inference service
inference_service: InferenceService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize inference service on startup."""
    global inference_service
    print("Initializing inference service...")
    inference_service = InferenceService.from_env()
    print("Service ready")
    yield
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Sentiment & Toxicity Analysis API",
    description="Offline API for sentiment and toxicity analysis",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze text for sentiment and toxicity.
    
    Args:
        request: Request containing text to analyze
        
    Returns:
        Analysis results with sentiment, toxicity, backend, and latency
    """
    start_time = time.perf_counter()
    
    # Perform analysis
    result = inference_service.analyze(request.text)
    
    # Calculate latency
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return AnalyzeResponse(
        sentiment=SentimentResult(
            label=result["sentiment"]["label"],
            score=result["sentiment"]["score"]
        ),
        toxicity=result["toxicity"],
        model_backend=inference_service.backend_name(),
        latency_ms=latency_ms
    )

