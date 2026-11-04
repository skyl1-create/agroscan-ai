import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Base Paths
BASE_DIR = PROJECT_ROOT
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
for directory in [DATA_DIR, DATASET_DIR, SAMPLES_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Training & Model Configuration
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 15
RANDOM_SEED = 42

# Default Best Model Path
MODEL_PATH = MODELS_DIR / "crop_disease_mobilenetv2.keras"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"
EVALUATION_REPORT_PATH = REPORTS_DIR / "classification_report.csv"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
TRAINING_CURVES_PATH = REPORTS_DIR / "training_curves.png"

# Standard Agricultural Classes (Crop & Disease Taxonomy)
DEFAULT_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn___Common_rust",
    "Corn___Gray_leaf_spot",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___healthy"
]
