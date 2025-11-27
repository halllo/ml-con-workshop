"""
Step 1: Basic BentoML Service (BentoML 1.4+ API)
================================================
"""

from __future__ import annotations
import bentoml
from transformers import pipeline

print("=" * 60)
print("STEP 1: Basic BentoML Service (BentoML 1.4+ API)")
print("=" * 60)

# -----------------------------------------------------------------------------
# PART 1: Define the Service Class
# -----------------------------------------------------------------------------
# In BentoML 1.4+, services are defined as classes with the @bentoml.service decorator

print("\n[1/3] Defining BentoML service class...")

@bentoml.service(
    resources={"cpu": "2"},      # Resource requirements
    traffic={"timeout": 30},     # Request timeout in seconds
)
class SentimentService:
    """
    Basic sentiment analysis service

    This service:
    - Loads a pre-trained sentiment model from Hugging Face
    - Exposes a /predict endpoint for sentiment analysis
    - Returns sentiment label and confidence score
    """

    def __init__(self) -> None:
        """
        Initialize the service

        This runs once when the service starts.
        Load your models here to avoid loading them on every request.
        """
        print("\n  Loading sentiment analysis model...")
        print("  (Using distilbert-base-uncased-finetuned-sst-2-english)")

        # Create a sentiment analysis pipeline
        # This automatically downloads and caches the model
        self.pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

        print("  ✓ Model loaded and ready")

    @bentoml.api
    def predict(self, text: str) -> dict:
        """
        Predict sentiment of input text

        This is the simplest possible endpoint:
        - Takes a string input
        - Runs prediction
        - Returns a dictionary result

        Args:
            text: Text to analyze for sentiment
                Example: "This movie is great!"

        Returns:
            Dict with sentiment prediction
                Example: {"label": "POSITIVE", "score": 0.9998}
        """
        # Run prediction using the pipeline
        # The pipeline returns a list with one result
        result = self.pipeline(text)

        # Return the first result (unwrap from list)
        return result[0]

print("  ✓ SentimentService class defined")
print("    - Resource requirements: 2 CPUs, 30s timeout")
print("    - Model: distilbert-base-uncased-finetuned-sst-2-english")
print("    - Endpoint: /predict")

# -----------------------------------------------------------------------------
# HOW TO USE THIS SERVICE
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 1 COMPLETE!")
print("=" * 60)

print("\n📋 How to run this service:")
print("  1. Save this file as 'service_basic_solution.py'")
print("  2. Run: bentoml serve service_basic_solution:SentimentService")
print("  3. Service starts at http://localhost:3000")

print("\n🧪 How to test:")
print("  # Test with curl:")
print('  curl -X POST http://localhost:3000/predict \\')
print('       -H "Content-Type: application/json" \\')
print('       -d \'"This movie is amazing!"\'')

print("\n  # Or use Python:")
print("  import requests")
print('  response = requests.post(')
print('      "http://localhost:3000/predict",')
print('      json="This movie is terrible"')
print('  )')
print('  print(response.json())')

print("\n💡 Tips:")
print("  - Visit http://localhost:3000 to see auto-generated Swagger UI")
print("  - Press Ctrl+C to stop the service")
print("  - Use --reload flag for auto-reload during development")

print("\n⚠️  What's MISSING (we'll add in next steps):")
print("  ✗ No input validation (what if text is empty?)")
print("  ✗ No error handling (what if prediction fails?)")
print("  ✗ No health check endpoint")
print("  ✗ No batch processing (process multiple texts at once)")
print("  ✗ No logging or monitoring")

print("\n→ Next: service_with_validation_solution.py (adds Pydantic validation)")
print("=" * 60)
