"""
Exercise 1: Basic Model Training (No MLflow) - SOLUTION
========================================================

This is the complete solution for Exercise 1.
Compare with your implementation in starter/train_basic.py
"""

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# =============================================================================
# PART 1: Load Pre-trained Model
# =============================================================================

def load_model(model_name: str):
    """Load pre-trained model and tokenizer from Hugging Face Hub"""
    print(f"\n[1/5] Loading model: {model_name}...")

    # SOLUTION TODO 1: Load model with 2 labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    # SOLUTION TODO 2: Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"✓ Model loaded: {model_name}")
    print(f"  - Parameters: ~{model.num_parameters() / 1e6:.1f}M")
    print(f"  - Task: Binary Sentiment Classification")

    return model, tokenizer

# =============================================================================
# PART 2: Load Dataset
# =============================================================================

def load_dataset_imdb(train_samples: int = 1000, test_samples: int = 100):
    """Load IMDB movie review dataset"""
    print(f"\n[2/5] Loading dataset...")

    # SOLUTION TODO 3: Load the IMDB dataset
    dataset = load_dataset("imdb")

    # Select subset for quick training
    train_dataset = dataset["train"].select(range(train_samples))
    test_dataset = dataset["test"].select(range(test_samples))

    print(f"✓ Dataset loaded: IMDB")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Test samples: {len(test_dataset)}")
    print(f"\nExample review:")
    print(f"  Text: {train_dataset[0]['text'][:100]}...")
    print(f"  Label: {'Positive (1)' if train_dataset[0]['label'] == 1 else 'Negative (0)'}")

    return train_dataset, test_dataset

# =============================================================================
# PART 3: Tokenize Data
# =============================================================================

def tokenize_datasets(train_dataset, test_dataset, tokenizer, max_length: int = 128):
    """Convert text to tokens (numbers) that the model can process"""
    print(f"\n[3/5] Tokenizing dataset...")

    def tokenize_function(examples):
        # SOLUTION TODO 4: Tokenize with padding and truncation
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    # Apply tokenization to datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    print("✓ Tokenization complete")
    print(f"  - Max length: {max_length} tokens")

    return train_dataset, test_dataset

# =============================================================================
# PART 4: Train Model
# =============================================================================

def train_model(model, train_dataset, test_dataset, epochs: int = 2, batch_size: int = 8):
    """Train the sentiment analysis model"""
    print(f"\n[4/5] Setting up training...")

    # SOLUTION TODO 5: Create TrainingArguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=50,
        eval_strategy="epoch"
    )

    # SOLUTION TODO 6: Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # Start training
    print(f"✓ Training configuration set")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"\nTraining started... (this may take 3-5 minutes)")
    print("=" * 70)

    # SOLUTION TODO 7: Train the model
    trainer.train()

    print("=" * 70)
    print("✓ Training complete!")

    return trainer

# =============================================================================
# PART 5: Evaluate Model
# =============================================================================

def evaluate_model(trainer):
    """Evaluate the trained model on test set"""
    print(f"\n[5/5] Evaluating model...")

    # SOLUTION TODO 8: Evaluate the model
    eval_results = trainer.evaluate()

    # Display results
    print(f"✓ Evaluation complete!")
    print(f"\nResults:")
    print(f"  - Loss: {eval_results['eval_loss']:.4f}")

    return eval_results

# =============================================================================
# PART 6: Save Model
# =============================================================================

def save_model(trainer, tokenizer, output_dir: str = "./sentiment_model"):
    """Save trained model and tokenizer to disk"""
    print(f"\nSaving model...")

    # SOLUTION TODO 9: Save the model
    trainer.save_model(output_dir)

    # SOLUTION TODO 10: Save the tokenizer
    tokenizer.save_pretrained(output_dir)

    print(f"✓ Model saved to: {output_dir}")
    print(f"\nTo load this model later:")
    print(f'  model = AutoModelForSequenceClassification.from_pretrained("{output_dir}")')
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{output_dir}")')

# =============================================================================
# Main Training Workflow
# =============================================================================

def main():
    """Main training workflow - orchestrates all steps"""

    print("=" * 70)
    print("EXERCISE 1: Basic Model Training - SOLUTION")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Load a pre-trained DistilBERT model")
    print("  2. Load the IMDB sentiment dataset")
    print("  3. Tokenize the text data")
    print("  4. Train the model on 1000 examples")
    print("  5. Evaluate on 100 test examples")
    print("  6. Save the trained model")

    # Configuration
    MODEL_NAME = "distilbert-base-uncased"
    TRAIN_SAMPLES = 1000
    TEST_SAMPLES = 100
    MAX_LENGTH = 128
    EPOCHS = 2
    BATCH_SIZE = 8

    # Step 1: Load model and tokenizer
    model, tokenizer = load_model(MODEL_NAME)

    # Step 2: Load dataset
    train_dataset, test_dataset = load_dataset_imdb(TRAIN_SAMPLES, TEST_SAMPLES)

    # Step 3: Tokenize
    train_dataset, test_dataset = tokenize_datasets(
        train_dataset, test_dataset, tokenizer, MAX_LENGTH
    )

    # Step 4: Train
    trainer = train_model(model, train_dataset, test_dataset, EPOCHS, BATCH_SIZE)

    # Step 5: Evaluate
    eval_results = evaluate_model(trainer)

    # Step 6: Save
    save_model(trainer, tokenizer)

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Loss: {eval_results['eval_loss']:.4f}")
    print(f"\nWhat's missing? No experiment tracking!")
    print("Next: Try 'train_with_mlflow.py' to add MLflow tracking")
    print("=" * 70)

if __name__ == "__main__":
    main()
