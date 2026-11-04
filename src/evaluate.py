import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from src.config import REPORTS_DIR, CONFUSION_MATRIX_PATH, EVALUATION_REPORT_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: List[str],
    save_dir: Path = REPORTS_DIR
) -> dict:
    """
    Evaluates the model on the given dataset, prints metrics,
    and generates a seaborn confusion matrix and classification report.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    y_true_list = []
    y_pred_list = []

    logger.info("Computing predictions over evaluation dataset...")
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_pred_classes = np.argmax(preds, axis=1)
        
        y_true_list.extend(labels.numpy())
        y_pred_list.extend(y_pred_classes)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    # 1. Classification Report
    short_class_names = [c.split("___")[-1].replace("_", " ") for c in class_names]
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=short_class_names,
        output_dict=True,
        zero_division=0
    )
    
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(EVALUATION_REPORT_PATH)
    logger.info("Classification report saved to %s", EVALUATION_REPORT_PATH)

    # 2. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=short_class_names,
        yticklabels=short_class_names
    )
    plt.title("Crop Disease Diagnosis Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("Ground Truth Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()
    logger.info("Confusion matrix plot saved to %s", CONFUSION_MATRIX_PATH)

    accuracy = float(np.mean(y_true == y_pred))
    logger.info("Overall Evaluation Accuracy: %.2f%%", accuracy * 100)

    return {
        "accuracy": accuracy,
        "classification_report": report_dict,
        "confusion_matrix_path": str(CONFUSION_MATRIX_PATH),
        "report_csv_path": str(EVALUATION_REPORT_PATH)
    }
