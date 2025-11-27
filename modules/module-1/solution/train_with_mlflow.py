"""
Exercise 2: MLflow Experiment Tracking & Model Registry - SOLUTION
===================================================================

This is the complete solution combining:
- Part 1: Basic MLflow Tracking
- Part 2: Model Registry Workflow

Compare with your implementation in starter/train_with_mlflow.py
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

# SOLUTION TODO 1: Import MLflow
import mlflow
import mlflow.transformers

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

    # SOLUTION TODO 2: Log model architecture parameter
    mlflow.log_param("model_name", model_name)

    # SOLUTION TODO 3: Log training hyperparameters
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)

    # SOLUTION TODO 4: Log dataset sizes
    mlflow.log_param("train_samples", len(train_dataset))
    mlflow.log_param("test_samples", len(test_dataset))

    # Configure training
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=50,
        eval_strategy="epoch",
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

    # SOLUTION TODO 5: Log training loss
    mlflow.log_metric("train_loss", train_result.training_loss)

    return trainer, train_result

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate_and_log_metrics(trainer):
    """Evaluate model and log all metrics to MLflow"""
    print(f"\n[5/6] Evaluating and logging metrics...")

    # Evaluate
    eval_results = trainer.evaluate()

    # SOLUTION TODO 6: Log evaluation metrics to MLflow
    mlflow.log_metric("eval_loss", eval_results["eval_loss"])
    mlflow.log_metric("accuracy", eval_results["eval_accuracy"])
    mlflow.log_metric("precision", eval_results["eval_precision"])
    mlflow.log_metric("f1_score", eval_results["eval_f1"])

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

    # SOLUTION TODO 7: Log the model and tokenizer to MLflow
    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path="model",
        task="text-classification"
    )

    print(f"✓ Model logged to MLflow")
    print(f"  💡 View in MLflow UI: http://localhost:5000")

# =============================================================================
# PART 2: MODEL REGISTRY WORKFLOW (ADVANCED/OPTIONAL)
# =============================================================================

# Initialize client (used in Part 2 functions)
client = MlflowClient()

# -----------------------------------------------------------------------------
# Train and Register Model
# -----------------------------------------------------------------------------

def train_and_register_model(version_name: str, epochs: int, accuracy: float, model_name: str = "sentiment-classifier"):
    """
    Train a model and register it in MLflow Model Registry
    
    SOLUTION: Complete implementation
    """
    
    print(f"  Training {version_name} model (epochs={epochs})...")

    with mlflow.start_run(run_name=f"model-{version_name}") as run:
        # Load model (using Hugging Face pipeline for simplicity)
        model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

        # Log parameters
        mlflow.log_param("version_name", version_name)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("model_type", "distilbert-base-uncased")

        # Log metrics (simulated for demo)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("training_time", epochs * 10)  # Simulated

        # Register model in Model Registry
        mlflow.transformers.log_model(
            transformers_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )

        print(f"    ✓ Registered as {model_name} (run_id: {run.info.run_id[:8]}...)")


# -----------------------------------------------------------------------------
# Set Model Alias (Replaces deprecated stages)
# -----------------------------------------------------------------------------

def transition_to_stage(model_name: str, version: int, stage: str):
    """
    Set an alias for a model version (replaces deprecated stage transitions)
    
    SOLUTION: Complete implementation using MLflow 2.9+ aliases
    """

    print(f"  Setting {model_name} v{version} → alias '{stage}'...")

    # Convert stage name to lowercase for alias
    alias = stage.lower()

    # Set alias for this version
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=version
    )

    print(f"    ✓ Version {version} now has alias '{alias}'")


# -----------------------------------------------------------------------------
# Load Model by Alias
# -----------------------------------------------------------------------------

def get_model_by_stage(model_name: str, stage: str):
    """
    Load a model using an alias (replaces deprecated stage loading)
    
    SOLUTION: Complete implementation
    """

    alias = stage.lower()
    print(f"  Loading {model_name} with alias '{alias}'...")

    try:
        # Construct model URI using alias
        # New format: models:/<model_name>@<alias>
        model_uri = f"models:/{model_name}@{alias}"

        # Load model
        model = mlflow.transformers.load_model(model_uri)

        # Get version info by alias
        try:
            model_version = client.get_model_version_by_alias(model_name, alias)
            print(f"    ✓ Loaded version {model_version.version} (alias: '{alias}')")
        except:
            print(f"    ✓ Loaded model with alias '{alias}'")

        return model

    except Exception as e:
        print(f"    ✗ Error loading model: {e}")
        return None


# -----------------------------------------------------------------------------
# Compare and Promote Models
# -----------------------------------------------------------------------------

def compare_and_promote(model_name: str):
    """
    Compare Staging vs Production models and promote if Staging is better
    
    SOLUTION: Complete implementation with automated promotion logic
    """

    print(f"\n  Comparing models for potential promotion...")

    # Get models by alias (replaces deprecated get_latest_versions with stages)
    try:
        staging_version = client.get_model_version_by_alias(model_name, "staging")
    except Exception:
        staging_version = None

    try:
        prod_version = client.get_model_version_by_alias(model_name, "production")
    except Exception:
        prod_version = None

    # Check if both aliases have models
    if not staging_version:
        print("  ⚠️  No model with 'staging' alias. Nothing to promote.")
        return

    if not prod_version:
        print("  ⚠️  No model with 'production' alias. Promoting staging directly.")
        transition_to_stage(model_name, staging_version.version, "Production")
        return

    # Get metrics from runs
    staging_run = client.get_run(staging_version.run_id)
    prod_run = client.get_run(prod_version.run_id)

    staging_acc = staging_run.data.metrics.get("accuracy", 0)
    prod_acc = prod_run.data.metrics.get("accuracy", 0)

    print(f"\n  Metrics Comparison:")
    print(f"    Staging (v{staging_version.version}): accuracy = {staging_acc:.4f}")
    print(f"    Production (v{prod_version.version}): accuracy = {prod_acc:.4f}")

    # Automated promotion logic
    if staging_acc > prod_acc:
        improvement = ((staging_acc - prod_acc) / prod_acc) * 100
        print(f"\n  ✓ Staging model is {improvement:.2f}% better!")
        print(f"  Promoting to Production...")

        # Remove production alias from old version
        client.delete_registered_model_alias(
            name=model_name,
            alias="production"
        )
        print(f"    ✓ Removed 'production' alias from version {prod_version.version}")

        # Set production alias to new version
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=staging_version.version
        )
        print(f"    ✓ Version {staging_version.version} promoted to Production!")

    else:
        print(f"\n  ✗ Staging model not better than Production")
        print(f"  Keeping current Production (v{prod_version.version})")


# =============================================================================
# MAIN WORKFLOWS
# =============================================================================

def main_basic_training():
    """Main workflow for Part 1: Basic MLflow Tracking"""

    print("=" * 70)
    print("PART 1: BASIC MLFLOW TRACKING - SOLUTION")
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

    # SOLUTION TODO 8: Set up MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

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
    print("PART 2: MODEL REGISTRY WORKFLOW (ADVANCED) - SOLUTION")
    print("=" * 70)

    # Initialize MLflow
    mlflow.set_experiment("model-registry-demo")

    MODEL_NAME = "sentiment-classifier"

    # -------------------------------------------------------------------------
    # Step 1: Register Multiple Model Versions
    # -------------------------------------------------------------------------
    print("\n[1/4] Registering model versions...")

    # Register version 1 (baseline)
    train_and_register_model("baseline", epochs=1, accuracy=0.85, model_name=MODEL_NAME)
    time.sleep(1)  # Small delay to ensure versions are sequential

    # Register version 2 (improved)
    train_and_register_model("improved", epochs=2, accuracy=0.88, model_name=MODEL_NAME)
    time.sleep(1)

    # Register version 3 (experimental)
    train_and_register_model("experimental", epochs=3, accuracy=0.86, model_name=MODEL_NAME)

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
    print(f"    - Version 1: Production (baseline, acc=0.85)")
    print(f"    - Version 2: Staging (improved, acc=0.88)")
    print(f"    - Version 3: None (experimental, acc=0.86)")

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

    print("\nWhat happened:")
    print("  1. Registered 3 model versions with different accuracies")
    print("  2. Set up initial aliases (v1→'production', v2→'staging')")
    print("  3. Loaded and tested models by alias")
    print("  4. Compared staging vs production (0.88 > 0.85)")
    print("  5. Automatically promoted v2 to 'production' alias")
    print("  6. Removed 'production' alias from v1")

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
        description="Train sentiment analysis model with MLflow - SOLUTION"
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
