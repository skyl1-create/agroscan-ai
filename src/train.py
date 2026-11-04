import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import logging
import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import (
    MODEL_PATH,
    CLASS_INDICES_PATH,
    TRAINING_CURVES_PATH,
    EPOCHS,
    LEARNING_RATE,
    DATASET_DIR
)
from src.dataset import load_datasets
from src.model import build_crop_disease_model, save_class_indices
from src.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_training_history(history: tf.keras.callbacks.History, save_path: Path = TRAINING_CURVES_PATH):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#2e7d32", lw=2)
    if val_acc:
        plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#1565c0", lw=2)
    plt.title("Model Accuracy over Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#c62828", lw=2)
    if val_loss:
        plt.plot(epochs_range, val_loss, label="Validation Loss", color="#f57c00", lw=2)
    plt.title("Model Loss over Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved training curves to %s", save_path)

def train(
    dataset_path: Path = None,
    epochs: int = 5,
    learning_rate: float = 0.001
):
    if dataset_path is None:
        balanced = Path(__file__).resolve().parent.parent / "data" / "balanced_dataset"
        dataset_path = balanced if balanced.exists() else DATASET_DIR

    logger.info("Starting AI Crop Disease Model Training Pipeline on: %s", dataset_path)
    
    train_ds, val_ds, test_ds, class_names = load_datasets(dataset_path)

    if train_ds is None or len(class_names) == 0:
        logger.error("Dataset folder is empty: %s", dataset_path)
        return

    logger.info("Detected %d classes. Saving class index mapping...", len(class_names))
    save_class_indices(class_names, CLASS_INDICES_PATH)

    # Build model architecture with pre-trained ImageNet weights
    model = build_crop_disease_model(
        num_classes=len(class_names),
        learning_rate=learning_rate
    )
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy" if val_ds is not None else "accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if val_ds is not None else "loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if val_ds is not None else "loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Save final curves
    plot_training_history(history)

    # Evaluation
    eval_ds = val_ds if val_ds is not None else train_ds
    if eval_ds is not None:
        logger.info("Running post-training evaluation & confusion matrix generation...")
        try:
            evaluate_model(model, eval_ds, class_names)
        except Exception as e:
            logger.warning("Post-training evaluation encountered a warning: %s", e)

    # Save final model
    model.save(str(MODEL_PATH))
    logger.info("Training complete! High-accuracy model saved to %s", MODEL_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Crop Disease Model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()
    
    train(epochs=args.epochs, learning_rate=args.lr)
