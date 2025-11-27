"""
Exercise 2 Solution: Input Validation & Production Features (BentoML 1.4+)
===========================================================================

Complete solution combining Pydantic validation with production features.

This service demonstrates:
- Pydantic validation for type safety
- Comprehensive error handling
- Structured logging
- Request tracking
- Batch processing
- Production-ready health checks
"""

from __future__ import annotations
import bentoml
from transformers import pipeline
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import time
import logging
import uuid
from datetime import datetime

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class SentimentRequest(BaseModel):
    """Request model with validation"""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to analyze for sentiment"
    )
    request_id: Optional[str] = Field(None, description="Optional request ID for tracking")

    @field_validator('text')
    @classmethod
    def text_must_not_be_empty_or_whitespace(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError('Text cannot be empty or just whitespace')
        return v.strip()


class SentimentResponse(BaseModel):
    """Response model with tracking fields"""
    text: str
    sentiment: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    request_id: str
    timestamp: str


class BatchSentimentRequest(BaseModel):
    """Batch request model"""
    texts: List[str] = Field(..., min_length=1, max_length=100)
    request_id: Optional[str] = None

    @field_validator('texts')
    @classmethod
    def validate_each_text(cls, v: List[str]) -> List[str]:
        for text in v:
            if not text or text.strip() == "":
                raise ValueError('All texts must be non-empty')
        return [text.strip() for text in v]


class BatchSentimentResponse(BaseModel):
    """Batch response with metadata"""
    results: List[SentimentResponse]
    metadata: dict
    request_id: str


class ErrorResponse(BaseModel):
    """Standardized error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    request_id: str = Field(..., description="Request ID for debugging")
    timestamp: str = Field(..., description="Error timestamp")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_request_id(provided_id: Optional[str] = None) -> str:
    """Generate or use provided request ID"""
    if provided_id:
        return provided_id
    else:
        return str(uuid.uuid4())[:8]


def get_timestamp() -> str:
    """Get ISO formatted timestamp"""
    return datetime.utcnow().isoformat()


# =============================================================================
# SERVICE DEFINITION
# =============================================================================

@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 30},
)
class SentimentService:
    """
    Production-ready sentiment analysis service

    Features:
    - Pydantic validation
    - Error handling
    - Request tracking
    - Batch processing
    - Structured logging
    """

    def __init__(self) -> None:
        """Initialize the sentiment analysis pipeline"""
        self.pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        logger.info("Model loaded and ready")

    @bentoml.api
    def predict(self, request: SentimentRequest) -> SentimentResponse:
        """
        Predict sentiment with validation and error handling

        Args:
            request: SentimentRequest with validated text

        Returns:
            SentimentResponse with prediction and tracking info
        """
        request_id = generate_request_id(request.request_id)

        logger.info(
            f"[{request_id}] Single prediction request",
            extra={"text_length": len(request.text)}
        )

        try:
            start_time = time.time()

            # Run prediction
            result = self.pipeline(request.text)

            # Calculate latency
            latency = (time.time() - start_time) * 1000

            logger.info(
                f"[{request_id}] Prediction successful",
                extra={
                    "latency_ms": round(latency, 2),
                    "sentiment": result[0]['label'],
                    "confidence": result[0]['score']
                }
            )

            return SentimentResponse(
                text=request.text,
                sentiment=result[0]['label'],
                confidence=round(result[0]['score'], 4),
                request_id=request_id,
                timestamp=get_timestamp()
            )

        except Exception as e:
            logger.error(
                f"[{request_id}] Prediction failed: {str(e)}",
                exc_info=True
            )

            return SentimentResponse(
                text=request.text,
                sentiment="ERROR",
                confidence=0.0,
                request_id=request_id,
                timestamp=get_timestamp()
            )

    @bentoml.api
    def batch_predict(self, request: BatchSentimentRequest) -> BatchSentimentResponse:
        """
        Batch predict with error handling

        Args:
            request: BatchSentimentRequest with list of texts

        Returns:
            BatchSentimentResponse with all predictions and metadata
        """
        request_id = generate_request_id(request.request_id)

        logger.info(
            f"[{request_id}] Batch prediction request",
            extra={"batch_size": len(request.texts)}
        )

        try:
            start_time = time.time()
            texts = request.texts
            num_texts = len(texts)

            # Run batch prediction
            results = self.pipeline(texts)

            # Calculate metrics
            latency = (time.time() - start_time) * 1000
            throughput = num_texts / (latency / 1000) if latency > 0 else 0

            logger.info(
                f"[{request_id}] Batch successful",
                extra={
                    "count": num_texts,
                    "latency_ms": round(latency, 2),
                    "throughput_per_sec": round(throughput, 1)
                }
            )

            # Format results
            timestamp = get_timestamp()
            predictions = [
                SentimentResponse(
                    text=text,
                    sentiment=result['label'],
                    confidence=round(result['score'], 4),
                    request_id=f"{request_id}-{i}",
                    timestamp=timestamp
                )
                for i, (text, result) in enumerate(zip(texts, results))
            ]

            return BatchSentimentResponse(
                results=predictions,
                metadata={
                    "count": num_texts,
                    "latency_ms": round(latency, 2),
                    "throughput_per_sec": round(throughput, 1),
                    "avg_latency_per_text_ms": round(latency / num_texts, 2)
                },
                request_id=request_id
            )

        except Exception as e:
            logger.error(
                f"[{request_id}] Batch failed: {str(e)}",
                exc_info=True
            )

            return BatchSentimentResponse(
                results=[],
                metadata={
                    "error": "BatchModelError",
                    "message": f"Batch prediction failed: {str(e)}",
                    "count": 0
                },
                request_id=request_id
            )

    @bentoml.api
    def health(self) -> dict:
        """Production health check with timestamp"""
        logger.debug("Health check requested")

        return {
            "status": "healthy",
            "service": "sentiment_analysis",
            "timestamp": get_timestamp()
        }
