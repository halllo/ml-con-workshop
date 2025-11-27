"""
Exercise 2: MLflow Experiment Tracking & Model Registry
========================================================

This exercise teaches you MLflow in two progressive parts:

**PART 1: Basic MLflow Tracking** (20 minutes, REQUIRED)
- Set up MLflow experiments
- Log training parameters
- Log evaluation metrics
- Save models to MLflow registry
- View results in MLflow UI

**PART 2: Model Registry Workflow** (20 minutes, ADVANCED/OPTIONAL)
- Register multiple model versions
- Transition models through stages (using MLflow 2.9+ aliases)
- Load models by stage
- Implement automated promotion logic

Learning objectives:
- Master MLflow experiment tracking
- Understand model lifecycle management
- Learn production model deployment patterns
- Implement automated model promotion

Before running:
1. Install MLflow: pip install mlflow
2. Start MLflow UI: mlflow ui
3. Open browser: http://localhost:5000

Usage:
    python train_with_mlflow.py               # Run Part 1 (basic tracking)
    python train_with_mlflow.py --advanced    # Run Part 2 (registry workflow)
"""

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# TODO 1: Import MLflow and MLflow's transformers integration
# Hint: import mlflow
# Hint: import mlflow.transformers
# TODO: Add imports here

# For Part 2 (Advanced)
from mlflow.tracking import MlflowClient
from transformers import pipeline
import time

# =============================================================================
# PART 1: BASIC MLFLOW TRACKING
# =============================================================================

# -----------------------------------------------------------------------------
# Metrics Computation
# -----------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """Compute comprehensive metrics for evaluation"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------

def load_model(model_name: str):
    """Load pre-trained model and tokenizer from Hugging Face"""
    print(f"\n[1/6] Loading model: {model_name}...")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"✓ Model loaded: {model_name}")
    print(f"  - Parameters: ~{model.num_parameters() / 1e6:.1f}M")

    return model, tokenizer

# -----------------------------------------------------------------------------
# Dataset Loading
# -----------------------------------------------------------------------------

def load_dataset_imdb(train_samples: int = 1000, test_samples: int = 100):
    """Load IMDB dataset"""
    print(f"\n[2/6] Loading dataset...")

    dataset = load_dataset("imdb")
    train_dataset = dataset["train"].select(range(train_samples))
    test_dataset = dataset["test"].select(range(test_samples))

    print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")

    return train_dataset, test_dataset

# -----------------------------------------------------------------------------
# Tokenization
# -----------------------------------------------------------------------------

def tokenize_datasets(train_dataset, test_dataset, tokenizer, max_length: int = 128):
    """Tokenize datasets"""
    print(f"\n[3/6] Tokenizing dataset...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    print("✓ Tokenization complete")

    return train_dataset, test_dataset

# -----------------------------------------------------------------------------
# Training with MLflow
# -----------------------------------------------------------------------------

def train_model_with_mlflow(
    model,
    train_dataset,
    test_dataset,
    model_name: str,
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-5
):
    """Train model with MLflow tracking"""
    print(f"\n[4/6] Training with MLflow tracking...")

    # TODO 2: Log the model architecture parameter
    # Hint: mlflow.log_param("model_name", model_name)
    # TODO: Add parameter logging here

    # TODO 3: Log training hyperparameters
    # Hint: Log epochs, batch_size, and learning_rate
    # TODO: Add hyperparameter logging here

    # TODO 4: Log dataset sizes
    # Hint: Log train_samples and test_samples using len(train_dataset)
    # TODO: Add dataset size logging here

    # Configure training
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=50,
        eval_strategy="epoch",  # Note: Using eval_strategy (new API)
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"Training started...")
    print("=" * 70)

    # Train
    train_result = trainer.train()

    print("=" * 70)
    print("✓ Training complete!")

    # TODO 5: Log training loss to MLflow
    # Hint: mlflow.log_metric("train_loss", train_result.training_loss)
    # TODO: Add training loss logging here

    return trainer, train_result

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate_and_log_metrics(trainer):
    """Evaluate model and log all metrics to MLflow"""
    print(f"\n[5/6] Evaluating and logging metrics...")

    # Evaluate
    eval_results = trainer.evaluate()

    # TODO 6: Log evaluation metrics to MLflow
    # Hint: Log eval_loss, accuracy, precision, and f1_score
    # Hint: Access metrics like eval_results["eval_loss"], eval_results["eval_accuracy"]
    # TODO: Add metric logging here

    print(f"✓ Metrics logged to MLflow:")
    print(f"  - Accuracy:  {eval_results['eval_accuracy']:.4f}")
    print(f"  - Precision: {eval_results['eval_precision']:.4f}")
    print(f"  - Recall:    {eval_results['eval_recall']:.4f}")
    print(f"  - F1 Score:  {eval_results['eval_f1']:.4f}")

    return eval_results

# -----------------------------------------------------------------------------
# Model Logging
# -----------------------------------------------------------------------------

def log_model_to_mlflow(model, tokenizer):
    """Log trained model to MLflow for versioning"""
    print(f"\n[6/6] Logging model to MLflow...")

    # TODO 7: Log the model and tokenizer to MLflow
    # Hint: Use mlflow.transformers.log_model()
    # Hint: Pass transformers_model={"model": model, "tokenizer": tokenizer}
    # Hint: Set artifact_path="model" and task="text-classification"
    # TODO: Add model logging here

    print(f"✓ Model logged to MLflow")
    print(f"  💡 View in MLflow UI: http://localhost:5000")

# =============================================================================
# PART 2: MODEL REGISTRY WORKFLOW (ADVANCED/OPTIONAL)
# =============================================================================

# -----------------------------------------------------------------------------
# Train and Register Model
# -----------------------------------------------------------------------------

def train_and_register_model(version_name: str, epochs: int, accuracy: float):
    """
    Train a model and register it in MLflow Model Registry

    Args:
        version_name: Descriptive name for this version (e.g., "baseline")
        epochs: Number of epochs (for demonstration)
        accuracy: Simulated accuracy (for demonstration)

    This function should:
    1. Create a Hugging Face pipeline
    2. Start an MLflow run
    3. Log parameters (version_name, epochs)
    4. Log metrics (accuracy)
    5. Log the model with registered_model_name=MODEL_NAME
    """

    print(f"  Training {version_name} model (epochs={epochs})...")

    # YOUR CODE HERE
    # Hint: Use mlflow.start_run() context manager
    # Hint: Use mlflow.log_param() for parameters
    # Hint: Use mlflow.log_metric() for metrics
    # Hint: Use mlflow.transformers.log_model() with registered_model_name

    # Example structure:
    # with mlflow.start_run(run_name=f"model-{version_name}") as run:
    #     # Load model
    #     model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    #
    #     # Log parameters
    #     mlflow.log_param(???, ???)
    #
    #     # Log metrics
    #     mlflow.log_metric(???, ???)
    #
    #     # Register model
    #     mlflow.transformers.log_model(
    #         transformers_model=???,
    #         artifact_path="model",
    #         registered_model_name=???
    #     )

    pass  # Remove this and add your code


# -----------------------------------------------------------------------------
# Set Model Alias (Replaces deprecated stages)
# -----------------------------------------------------------------------------

def transition_to_stage(model_name: str, version: int, stage: str):
    """
    Set an alias for a model version (replaces deprecated stage transitions)

    MLflow 2.9+ uses aliases instead of stages.
    Aliases are tags like "production", "staging", "champion", "challenger"

    Args:
        model_name: Name of the registered model
        version: Version number to set alias for
        stage: Alias name (e.g., "production", "staging")

    This function should:
    1. Convert stage name to lowercase for alias
    2. Use MlflowClient to set the alias for this version
    """

    print(f"  Setting {model_name} v{version} → alias '{stage}'...")

    # YOUR CODE HERE
    # Hint: Convert stage to lowercase: alias = stage.lower()
    # Hint: Use client.set_registered_model_alias()
    # Hint: Parameters: name, alias, version

    # Example:
    # alias = ???
    # client.set_registered_model_alias(
    #     name=???,
    #     alias=???,
    #     version=???
    # )

    pass  # Remove this and add your code


# -----------------------------------------------------------------------------
# Load Model by Alias
# -----------------------------------------------------------------------------

def get_model_by_stage(model_name: str, stage: str):
    """
    Load a model using an alias (replaces deprecated stage loading)

    Args:
        model_name: Name of the registered model
        stage: Alias name (e.g., "production", "staging")

    Returns:
        Loaded model

    This function should:
    1. Convert stage to lowercase for alias
    2. Construct the model URI: models:/<model_name>@<alias>
    3. Load the model using mlflow.transformers.load_model()
    """

    print(f"  Loading {model_name} with alias '{stage}'...")

    # YOUR CODE HERE
    # Hint: Convert stage to lowercase: alias = stage.lower()
    # Hint: New URI format is "models:/model-name@alias"
    # Hint: Use mlflow.transformers.load_model()

    # Example:
    # alias = ???
    # model_uri = f"models:/???@???"
    # model = mlflow.transformers.load_model(???)
    # return model

    pass  # Remove this and add your code


# -----------------------------------------------------------------------------
# Compare and Promote Models
# -----------------------------------------------------------------------------

def compare_and_promote(model_name: str):
    """
    Compare staging vs production models and promote if staging is better

    Args:
        model_name: Name of the registered model

    This function should:
    1. Get models by 'staging' and 'production' aliases
    2. Retrieve their metrics (accuracy)
    3. If staging accuracy > production accuracy:
       - Remove 'production' alias from old version
       - Set 'production' alias to staging version
    4. Otherwise, keep current production

    Note: Uses MLflow 2.9+ aliases instead of deprecated stages
    """

    print(f"\n  Comparing models for potential promotion...")

    # YOUR CODE HERE
    # Hint: Use client.get_model_version_by_alias(model_name, "staging")
    # Hint: Use client.get_model_version_by_alias(model_name, "production")
    # Hint: Wrap in try/except to handle missing aliases
    # Hint: Use client.get_run(version.run_id) to get metrics
    # Hint: Access metrics with run.data.metrics.get("accuracy")
    # Hint: Use client.delete_registered_model_alias() to remove old alias
    # Hint: Use client.set_registered_model_alias() to set new alias

    # Example structure:
    # # Get models by alias (with error handling)
    # try:
    #     staging_version = client.get_model_version_by_alias(???, ???)
    # except Exception:
    #     staging_version = None
    #
    # try:
    #     prod_version = client.get_model_version_by_alias(???, ???)
    # except Exception:
    #     prod_version = None
    #
    # if not staging_version or not prod_version:
    #     print("  ⚠️  Need both aliases set")
    #     return
    #
    # # Get metrics
    # staging_run = client.get_run(staging_version.run_id)
    # prod_run = client.get_run(prod_version.run_id)
    #
    # staging_acc = staging_run.data.metrics.get("accuracy")
    # prod_acc = prod_run.data.metrics.get("accuracy")
    #
    # # Promote if better
    # if staging_acc > prod_acc:
    #     # Remove production alias from old version
    #     client.delete_registered_model_alias(name=???, alias=???)
    #     # Set production alias to new version
    #     client.set_registered_model_alias(name=???, alias=???, version=???)
    # else:
    #     print("  ✗ Staging not better. Keeping current production.")

    pass  # Remove this and add your code


# =============================================================================
# MAIN WORKFLOWS
# =============================================================================

def main_basic_training():
    """Main workflow for Part 1: Basic MLflow Tracking"""

    print("=" * 70)
    print("PART 1: BASIC MLFLOW TRACKING")
    print("=" * 70)
    print("\nThis script demonstrates:")
    print("  ✓ Experiment tracking")
    print("  ✓ Parameter logging")
    print("  ✓ Metric logging")
    print("  ✓ Model versioning")

    # Configuration
    EXPERIMENT_NAME = "sentiment-analysis-workshop"
    MODEL_NAME = "distilbert-base-uncased"
    TRAIN_SAMPLES = 1000
    TEST_SAMPLES = 100
    MAX_LENGTH = 128
    EPOCHS = 2
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5

    # TODO 8: Set up MLflow experiment
    # Hint: mlflow.set_experiment(EXPERIMENT_NAME)
    # TODO: Add experiment setup here

    print(f"\n✓ MLflow experiment: {EXPERIMENT_NAME}")
    print(f"  💡 Start MLflow UI: mlflow ui")
    print(f"  💡 View at: http://localhost:5000")

    # Start MLflow run
    with mlflow.start_run(run_name="train-basic") as run:
        print(f"✓ MLflow run started: {run.info.run_id[:8]}...")

        # Set tags
        mlflow.set_tag("model_type", "sentiment-analysis")
        mlflow.set_tag("framework", "huggingface-transformers")

        # Step 1: Load model
        model, tokenizer = load_model(MODEL_NAME)

        # Step 2: Load dataset
        train_dataset, test_dataset = load_dataset_imdb(TRAIN_SAMPLES, TEST_SAMPLES)

        # Step 3: Tokenize
        train_dataset, test_dataset = tokenize_datasets(
            train_dataset, test_dataset, tokenizer, MAX_LENGTH
        )

        # Step 4: Train with MLflow tracking
        trainer, train_result = train_model_with_mlflow(
            model, train_dataset, test_dataset,
            MODEL_NAME, EPOCHS, BATCH_SIZE, LEARNING_RATE
        )

        # Step 5: Evaluate and log metrics
        eval_results = evaluate_and_log_metrics(trainer)

        # Step 6: Log model
        log_model_to_mlflow(model, tokenizer)

        # Summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE WITH MLFLOW!")
        print("=" * 70)
        print(f"\nRun ID: {run.info.run_id}")
        print(f"Experiment: {EXPERIMENT_NAME}")
        print(f"\nMetrics:")
        print(f"  - Accuracy: {eval_results['eval_accuracy']:.4f}")
        print(f"  - F1 Score: {eval_results['eval_f1']:.4f}")
        print(f"\nView results: http://localhost:5000")
        print(f"\nLoad this model:")
        print(f"  model = mlflow.transformers.load_model('runs:/{run.info.run_id}/model')")
        print("\nNext: Try 'python train_with_mlflow.py --advanced' for registry workflow!")
        print("=" * 70)


def main_registry_workflow():
    """Main workflow for Part 2: Model Registry (ADVANCED)"""

    print("=" * 70)
    print("PART 2: MODEL REGISTRY WORKFLOW (ADVANCED)")
    print("=" * 70)

    # Initialize MLflow
    mlflow.set_experiment("model-registry-demo")
    client = MlflowClient()

    MODEL_NAME = "sentiment-classifier"

    # -------------------------------------------------------------------------
    # Step 1: Register Multiple Model Versions
    # -------------------------------------------------------------------------
    print("\n[1/4] Registering model versions...")

    # Register version 1 (baseline)
    train_and_register_model("baseline", epochs=1, accuracy=0.85)
    time.sleep(1)  # Small delay to ensure versions are sequential

    # Register version 2 (improved)
    train_and_register_model("improved", epochs=2, accuracy=0.88)
    time.sleep(1)

    # Register version 3 (experimental)
    train_and_register_model("experimental", epochs=3, accuracy=0.86)

    print(f"\n  ✓ Registered 3 versions of '{MODEL_NAME}'")

    # -------------------------------------------------------------------------
    # Step 2: Transition Models to Stages
    # -------------------------------------------------------------------------
    print("\n[2/4] Setting up model stages...")

    # Set version 1 to Production
    transition_to_stage(MODEL_NAME, version=1, stage="Production")

    # Set version 2 to Staging
    transition_to_stage(MODEL_NAME, version=2, stage="Staging")

    # Version 3 stays in "None" stage (just registered)

    print(f"\n  ✓ Stages configured:")
    print(f"    - Version 1: Production")
    print(f"    - Version 2: Staging")
    print(f"    - Version 3: None")

    # -------------------------------------------------------------------------
    # Step 3: Load and Test Models by Stage
    # -------------------------------------------------------------------------
    print("\n[3/4] Loading models by stage...")

    # Load Production model
    prod_model = get_model_by_stage(MODEL_NAME, "Production")

    # Load Staging model
    staging_model = get_model_by_stage(MODEL_NAME, "Staging")

    # Test both models
    test_text = "This workshop is amazing!"
    print(f"\n  Testing with: '{test_text}'")

    if prod_model:
        prod_result = prod_model(test_text)
        print(f"  Production: {prod_result[0]['label']} ({prod_result[0]['score']:.4f})")

    if staging_model:
        staging_result = staging_model(test_text)
        print(f"  Staging: {staging_result[0]['label']} ({staging_result[0]['score']:.4f})")

    # -------------------------------------------------------------------------
    # Step 4: Automated Promotion
    # -------------------------------------------------------------------------
    print("\n[4/4] Checking for automated promotion...")

    compare_and_promote(MODEL_NAME)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL REGISTRY WORKFLOW COMPLETE!")
    print("=" * 70)

    print("\nView results in MLflow UI:")
    print("  1. Open http://localhost:5000")
    print("  2. Click 'Models' tab")
    print(f"  3. Click on '{MODEL_NAME}'")
    print("  4. See all versions and their aliases")

    print("\nLoad production model in code:")
    print(f"  # New MLflow 2.9+ format using aliases")
    print(f"  model_uri = 'models:/{MODEL_NAME}@production'")
    print(f"  model = mlflow.transformers.load_model(model_uri)")

    print("\n💡 Key Takeaways:")
    print("  ✓ Model Registry manages model lifecycle")
    print("  ✓ MLflow 2.9+ uses aliases (not deprecated stages)")
    print("  ✓ Aliases are flexible tags: 'production', 'staging', 'champion', etc.")
    print("  ✓ Load by alias (@alias), not by version number")
    print("  ✓ Automated promotion based on metrics")
    print("  ✓ Audit trail for all transitions")

    print("\n" + "=" * 70)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train sentiment analysis model with MLflow"
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Run advanced model registry workflow (Part 2)"
    )

    args = parser.parse_args()

    if args.advanced:
        main_registry_workflow()
    else:
        main_basic_training()
