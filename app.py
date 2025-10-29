"""FastAPI application for sentiment and toxicity analysis."""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional

from inference import InferenceService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Middleware to limit request body size
class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request body size to prevent abuse."""
    
    def __init__(self, app, max_body_size: int = 1_048_576):  # 1 MB default
        super().__init__(app)
        self.max_body_size = max_body_size
    
    async def dispatch(self, request: Request, call_next):
        if request.method in ["POST", "PUT", "PATCH"]:
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_body_size:
                logger.warning(
                    f"Request body too large: {content_length} bytes (max: {self.max_body_size})"
                )
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request body too large. Max size: {self.max_body_size} bytes"}
                )
        return await call_next(request)


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
    logger.info("Initializing inference service...")
    inference_service = InferenceService.from_env()
    logger.info("Service ready")
    yield
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Sentiment & Toxicity Analysis API",
    description="Offline API for sentiment and toxicity analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware to limit request body size (1 MB)
app.add_middleware(BodySizeLimitMiddleware, max_body_size=1_048_576)


class VersionInfo(BaseModel):
    """Version information."""
    version: str
    git_sha: str
    sentiment_model: str
    toxicity_model: str


@app.get("/health", response_model=HealthResponse)
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/version", response_model=VersionInfo)
async def version() -> VersionInfo:
    """Version and model information endpoint."""
    git_sha = os.getenv("GIT_SHA", "dev")
    return VersionInfo(
        version="1.0.0",
        git_sha=git_sha,
        sentiment_model="distilbert-base-uncased-finetuned-sst-2-english",
        toxicity_model="unitary/toxic-bert"
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, http_request: Request) -> AnalyzeResponse:
    """Analyze text for sentiment and toxicity.
    
    Args:
        request: Request containing text to analyze
        http_request: FastAPI request object for logging
        
    Returns:
        Analysis results with sentiment, toxicity, backend, and latency
    """
    start_time = time.perf_counter()
    text_length = len(request.text)
    
    # Perform analysis
    result = inference_service.analyze(request.text)
    
    # Calculate latency
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # Log request metrics
    logger.info(
        "analyze_request",
        extra={
            "text_length": text_length,
            "latency_ms": round(latency_ms, 2),
            "backend": inference_service.backend_name(),
            "sentiment": result["sentiment"]["label"],
            "client_ip": http_request.client.host if http_request.client else "unknown"
        }
    )
    
    return AnalyzeResponse(
        sentiment=SentimentResult(
            label=result["sentiment"]["label"],
            score=result["sentiment"]["score"]
        ),
        toxicity=result["toxicity"],
        model_backend=inference_service.backend_name(),
        latency_ms=latency_ms
    )

