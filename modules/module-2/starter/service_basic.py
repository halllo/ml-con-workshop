"""
Exercise 1: Basic BentoML Service (BentoML 1.4+ API)
=====================================================
"""

from __future__ import annotations
import bentoml
from transformers import pipeline

print("=" * 70)
print("EXERCISE 1: Basic BentoML Service (BentoML 1.4+)")
print("=" * 70)

# =============================================================================
# PART 1: Define the Service Class
# =============================================================================

print("\n[1/3] Defining BentoML service class...")

# TODO 1: Add @bentoml.service decorator to the class
# FILL IN: Use @bentoml.service(resources={"cpu": "2"}, traffic={"timeout": 30})
# Hint: The decorator configures resource requirements and traffic settings
@bentoml.service(
    resources={"cpu": "2"},      # Resource requirements
    traffic={"timeout": 30},     # Request timeout in seconds
)

# YOUR DECORATOR HERE
class SentimentService:
    """
    Basic sentiment analysis service

    This service loads a sentiment model and provides a prediction endpoint
    """

    # TODO 2: Define __init__ method to initialize the model
    # FILL IN: Create __init__(self) -> None method
    # Hint: Load the sentiment analysis pipeline in __init__
    # YOUR CODE HERE
    def __init__(self) -> None:
        """
        Initialize the service

        TODO 3: Load the sentiment analysis pipeline
        FILL IN: Use pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        Hint: Store it as self.pipeline
        """
        print("\n  Loading sentiment analysis model...")

        # YOUR CODE HERE
        self.pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

        print("  ✓ Model loaded and ready")

    # TODO 4: Add @bentoml.api decorator to the predict method
    # FILL IN: Use @bentoml.api decorator (no parameters needed)
    # Hint: This makes the method an HTTP endpoint

    # YOUR DECORATOR HERE
    @bentoml.api
    def predict(self, text: str) -> dict:
        """
        Predict sentiment of input text

        Args:
            input_data: Dict with 'text' key
                Example: {"text": "This movie is great!"}

        Returns:
            Dict with sentiment prediction
                Example: {"label": "POSITIVE", "score": 0.9998}
        """
        # TODO 5: Extract text and run prediction
        # FILL IN: Get text from input_data and use self.pipeline to predict
        # Hint: text = input_data["text"], then result = self.pipeline(text)

        # YOUR CODE HERE
        result = self.pipeline(text)

        # TODO 6: Return the first result
        # FILL IN: The pipeline returns a list, extract the first element
        # Hint: Use result[0]
        return result[0]  # YOUR CODE HERE


print("  ✓ Service class structure defined (fill in TODOs 1-6)")

# =============================================================================
# HOW TO RUN
# =============================================================================

print("\n" + "=" * 70)
print("EXERCISE 1 SETUP COMPLETE!")
print("=" * 70)

print("\n📋 How to run this service:")
print("  bentoml serve service_basic:SentimentService")
print("\n  Service will start at: http://localhost:3000")

print("\n🧪 How to test:")
print('  curl -X POST http://localhost:3000/predict \\')
print('       -H "Content-Type: application/json" \\')
print('       -d \'{"text": "This workshop is amazing!"}\'')

print("\n💡 Expected response:")
print('  {"label": "POSITIVE", "score": 0.9998}')

print("\n✅ Validation:")
print("  pytest tests/test_implementation.py::TestExercise1 -v")

print("\n⚠️  What's missing:")
print("  - No input validation")
print("  - No error handling")
print("  - No health check endpoint")

print("\nNext: service_with_validation.py")
print("=" * 70)
