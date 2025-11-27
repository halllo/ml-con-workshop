"""
Exercise 2: Input Validation & Production Features (BentoML 1.4+)
==================================================================
"""

from __future__ import annotations
import bentoml
from transformers import pipeline

# TODO 1: Import Pydantic for validation
# FILL IN: from pydantic import BaseModel, Field, field_validator
# YOUR CODE HERE

# TODO 2: Import typing, time, logging, uuid, datetime for production features
# FILL IN: from typing import List, Optional
# FILL IN: import time, logging, uuid
# FILL IN: from datetime import datetime
# YOUR CODE HERE

print("=" * 70)
print("EXERCISE 2: Input Validation & Production Features")
print("=" * 70)

# =============================================================================
# PART 1: PYDANTIC VALIDATION (TODOs 3-12)
# =============================================================================

print("\n[Part 1: Pydantic Validation]")
print("\n[1/6] Defining Pydantic models...")

# TODO 3: Define the SentimentRequest model
# FILL IN: Create a Pydantic BaseModel with:
#   - text: str field with Field(..., min_length=1, max_length=5000, description="...")
#   - request_id: Optional[str] = Field(None, description="...")
class SentimentRequest(BaseModel):
    """
    Request model for sentiment prediction

    Defines expected input format with validation rules
    """
    # YOUR CODE HERE
    # text: str = Field(...)
    # request_id: Optional[str] = Field(None, description="Optional request ID")
    pass  # Remove this line after adding your code


# TODO 4: Add custom validator for text field (Pydantic v2 style)
# FILL IN: Use @field_validator('text') decorator with @classmethod
# Hint: Check if v.strip() is empty, raise ValueError if so, return v.strip()
    # YOUR DECORATOR AND FUNCTION HERE
    # @field_validator('text')
    # @classmethod
    # def text_must_not_be_empty_or_whitespace(cls, v: str) -> str:
    #     if not v or v.strip() == "":
    #         raise ValueError('Text cannot be empty or just whitespace')
    #     return v.strip()


# TODO 5: Define the SentimentResponse model
# FILL IN: Create a Pydantic BaseModel with:
#   - text: str
#   - sentiment: str
#   - confidence: float (with Field(..., ge=0.0, le=1.0))
#   - request_id: str
#   - timestamp: str
class SentimentResponse(BaseModel):
    """
    Response model for sentiment prediction

    Includes tracking fields for production use
    """
    # YOUR CODE HERE
    # text: str
    # sentiment: str
    # confidence: float = Field(..., ge=0.0, le=1.0)
    # request_id: str
    # timestamp: str
    pass  # Remove this line after adding your code


# TODO 6: Define BatchSentimentRequest model
# FILL IN: Create model with:
#   - texts: List[str] = Field(..., min_length=1, max_length=100)
#   - request_id: Optional[str] = None
class BatchSentimentRequest(BaseModel):
    """Batch request model for multiple texts"""
    # YOUR CODE HERE
    # texts: List[str] = Field(...)
    # request_id: Optional[str] = None
    pass  # Remove this line after adding your code


# TODO 7: Define BatchSentimentResponse model
# FILL IN: Create model with:
#   - results: List[SentimentResponse]
#   - metadata: dict
#   - request_id: str
class BatchSentimentResponse(BaseModel):
    """Batch response with metadata"""
    # YOUR CODE HERE
    # results: List[SentimentResponse]
    # metadata: dict
    # request_id: str
    pass  # Remove this line after adding your code


# TODO 8: Define ErrorResponse model
# FILL IN: Create model with error, message, request_id, timestamp fields (all str)
class ErrorResponse(BaseModel):
    """Standardized error response"""
    # YOUR CODE HERE
    # error: str = Field(...)
    # message: str = Field(...)
    # request_id: str = Field(...)
    # timestamp: str = Field(...)
    pass  # Remove this line after adding your code


print("  ✓ Pydantic models defined (fill in TODOs 3-8)")

# =============================================================================
# PART 2: PRODUCTION FEATURES (TODOs 9-25)
# =============================================================================

print("\n[Part 2: Production Features]")
print("\n[2/6] Configuring logging...")

# TODO 9: Configure logging
# FILL IN: Use logging.basicConfig() with:
#   - level=logging.INFO
#   - format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#   - datefmt='%Y-%m-%d %H:%M:%S'
# YOUR CODE HERE

# TODO 10: Create logger instance
# FILL IN: logger = logging.getLogger(__name__)
logger = None  # YOUR CODE HERE

print("  ✓ Logging configured (fill in TODOs 9-10)")

# =============================================================================
# PART 3: HELPER FUNCTIONS
# =============================================================================

print("\n[3/6] Defining helper functions...")

# TODO 11: Implement generate_request_id function
# FILL IN: Return provided_id if exists, else generate with str(uuid.uuid4())[:8]
def generate_request_id(provided_id: Optional[str] = None) -> str:
    """Generate or use provided request ID"""
    # YOUR CODE HERE
    # Hint: if provided_id: return provided_id
    #       else: return str(uuid.uuid4())[:8]
    pass


# TODO 12: Implement get_timestamp function
# FILL IN: Return datetime.utcnow().isoformat()
def get_timestamp() -> str:
    """Get ISO formatted timestamp"""
    # YOUR CODE HERE
    pass


print("  ✓ Helper functions defined (fill in TODOs 11-12)")

# =============================================================================
# PART 4: SERVICE DEFINITION
# =============================================================================

print("\n[4/6] Creating production service...")

# TODO 13: Add @bentoml.service decorator
# FILL IN: @bentoml.service(resources={"cpu": "2"}, traffic={"timeout": 30})
# YOUR DECORATOR HERE
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
        # TODO 14: Load the pipeline
        # FILL IN: self.pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.pipeline = None  # YOUR CODE HERE

        # TODO 15: Log that model is ready
        # FILL IN: logger.info("Model loaded and ready")
        # YOUR CODE HERE

    # TODO 16: Add @bentoml.api decorator for predict endpoint
    # FILL IN: @bentoml.api
    # YOUR DECORATOR HERE
    def predict(self, request: SentimentRequest) -> SentimentResponse:
        """
        Predict sentiment with validation and error handling

        Args:
            request: SentimentRequest with validated text

        Returns:
            SentimentResponse with prediction and tracking info
        """
        # Generate request ID
        request_id = generate_request_id(request.request_id)

        # TODO 17: Log incoming request
        # FILL IN: logger.info(f"[{request_id}] Single prediction request")
        # YOUR CODE HERE

        # TODO 18: Add try/except block around prediction
        # FILL IN: Wrap the prediction logic in try/except Exception as e
        try:
            # Start timing
            start_time = time.time()

            # TODO 19: Run prediction
            # FILL IN: result = self.pipeline(request.text)
            result = None  # YOUR CODE HERE

            # Calculate latency
            latency = (time.time() - start_time) * 1000

            # TODO 20: Log successful prediction
            # FILL IN: logger.info(f"[{request_id}] Prediction successful", extra={"latency_ms": round(latency, 2)})
            # YOUR CODE HERE

            # TODO 21: Return SentimentResponse
            # FILL IN: Create and return SentimentResponse with all required fields
            # Hint: text=request.text, sentiment=result[0]['label'],
            #       confidence=round(result[0]['score'], 4), request_id=request_id, timestamp=get_timestamp()
            return None  # YOUR CODE HERE

        except Exception as e:
            # TODO 22: Log error with stack trace
            # FILL IN: logger.error(f"[{request_id}] Prediction failed: {str(e)}", exc_info=True)
            # YOUR CODE HERE

            # TODO 23: Return error response (as SentimentResponse with ERROR sentiment)
            # FILL IN: Return SentimentResponse with sentiment="ERROR", confidence=0.0
            # Hint: SentimentResponse(text=request.text, sentiment="ERROR", confidence=0.0, request_id=request_id, timestamp=get_timestamp())
            return None  # YOUR CODE HERE

    # TODO 24: Add @bentoml.api decorator for batch_predict endpoint
    # FILL IN: @bentoml.api
    # YOUR DECORATOR HERE
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

            # Return batch response
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

            # Return batch response with error info
            return BatchSentimentResponse(
                results=[],
                metadata={
                    "error": "BatchModelError",
                    "message": f"Batch prediction failed: {str(e)}",
                    "count": 0
                },
                request_id=request_id
            )

    # TODO 25: Add @bentoml.api decorator and implement health check
    # FILL IN: Add @bentoml.api decorator
    # FILL IN: Return dict with status, service, and timestamp
    # YOUR DECORATOR HERE
    def health(self) -> dict:
        """Production health check with timestamp"""
        # YOUR CODE HERE
        # Hint: return {"status": "healthy", "service": "sentiment_analysis", "timestamp": get_timestamp()}
        pass


print("  ✓ Production service defined (fill in TODOs 13-25)")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2 SETUP COMPLETE!")
print("=" * 70)

print("\n📋 How to run this service:")
print("  bentoml serve service_with_validation:SentimentService")
print("\n  Service will start at: http://localhost:3000")

print("\n🧪 How to test:")

print("\n  1. Valid request with tracking:")
print('     curl -X POST http://localhost:3000/predict \\')
print('          -H "Content-Type: application/json" \\')
print('          -d \'{"text": "This workshop is amazing!", "request_id": "test-123"}\'')

print("\n  2. Invalid request (empty text):")
print('     curl -X POST http://localhost:3000/predict \\')
print('          -H "Content-Type: application/json" \\')
print('          -d \'{"text": ""}\'')
print("     Expected: ValidationError with clear message")

print("\n  3. Batch prediction:")
print('     curl -X POST http://localhost:3000/batch_predict \\')
print('          -H "Content-Type: application/json" \\')
print('          -d \'{"texts": ["Amazing!", "Terrible!", "Okay"]}\'')

print("\n  4. Health check:")
print('     curl http://localhost:3000/health')

print("\n💡 Features Implemented:")
print("  Part 1: Pydantic Validation")
print("    ✓ Request/response models with type safety")
print("    ✓ Input validation (length, whitespace, required fields)")
print("    ✓ Custom validators")
print("    ✓ Structured error responses")
print("\n  Part 2: Production Features")
print("    ✓ Comprehensive error handling (try/except)")
print("    ✓ Structured logging with timestamps")
print("    ✓ Request ID tracking for debugging")
print("    ✓ Performance metrics (latency, throughput)")
print("    ✓ Batch processing endpoint")
print("    ✓ Production-ready health check")

print("\n✅ Validation:")
print("  pytest tests/test_implementation.py::TestExercise2 -v")

print("\n🎯 Production Ready:")
print("  ✓ Can be deployed to production")
print("  ✓ Ready for load testing")
print("  ✓ Monitoring integration ready")
print("  ✓ Easy to debug with request IDs")
print("  ✓ Scalable with batch processing")

print("\nNext: Build and containerize with 'bentoml build'")
print("=" * 70)
