# Continual Learning & Active Feedback Engine for AgroScan AI
import json
import uuid
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback_store"
FEEDBACK_IMG_DIR = FEEDBACK_DIR / "images"
METADATA_FILE = FEEDBACK_DIR / "feedback_registry.json"

FEEDBACK_IMG_DIR.mkdir(parents=True, exist_ok=True)

def initialize_registry():
    if not METADATA_FILE.exists():
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

def save_sample_feedback(
    image: Image.Image,
    predicted_class: str,
    confidence: float,
    user_confirmed: bool,
    corrected_class: Optional[str] = None,
    city: str = "Unknown",
    temperature: Optional[float] = None,
    humidity: Optional[float] = None
) -> Dict[str, Any]:
    """
    Saves an uploaded image and its ground truth metadata into the Continual Learning store.
    """
    initialize_registry()
    
    sample_id = str(uuid.uuid4())[:10]
    timestamp = datetime.datetime.now().isoformat()
    filename = f"{sample_id}.jpg"
    img_path = FEEDBACK_IMG_DIR / filename
    
    # Save image (convert to RGB if RGBA)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(img_path, format="JPEG", quality=90)
    
    actual_label = predicted_class if user_confirmed else (corrected_class or predicted_class)
    
    record = {
        "id": sample_id,
        "timestamp": timestamp,
        "image_file": filename,
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "user_confirmed": user_confirmed,
        "ground_truth_label": actual_label,
        "city": city,
        "temperature": temperature,
        "humidity": humidity,
        "used_in_retraining": False
    }
    
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = []
        
    data.append(record)
    
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    return record

def get_continual_learning_stats() -> Dict[str, Any]:
    """
    Returns statistics on accumulated field samples awaiting retraining.
    """
    initialize_registry()
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception:
        records = []
        
    total = len(records)
    unretrained = [r for r in records if not r.get("used_in_retraining", False)]
    confirmed = [r for r in records if r.get("user_confirmed", False)]
    corrected = [r for r in records if not r.get("user_confirmed", False)]
    
    return {
        "total_samples": total,
        "pending_retrain": len(unretrained),
        "confirmed_count": len(confirmed),
        "corrected_count": len(corrected),
        "records": records[-10:] # last 10 records for display
    }

def trigger_incremental_fine_tuning(epochs: int = 2) -> Dict[str, Any]:
    """
    Performs real-time fine-tuning on newly acquired field images.
    """
    import tensorflow as tf
    from src.config import MODEL_PATH, CLASS_INDICES_PATH, IMAGE_SIZE
    
    initialize_registry()
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    unretrained = [r for r in records if not r.get("used_in_retraining", False)]
    
    if len(unretrained) < 1:
        return {"success": False, "message": "No new samples available for fine-tuning."}
        
    # Load class mapping
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_dict = json.load(f)
    label_to_idx = {v: int(k) for k, v in class_dict.items()}
    
    X_train = []
    y_train = []
    
    for rec in unretrained:
        img_p = FEEDBACK_IMG_DIR / rec["image_file"]
        if not img_p.exists():
            continue
        label = rec["ground_truth_label"]
        if label not in label_to_idx:
            continue
            
        img = Image.open(img_p).resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        X_train.append(arr)
        y_train.append(label_to_idx[label])
        
    if len(X_train) == 0:
        return {"success": False, "message": "Valid samples count is 0."}
        
    X_arr = np.array(X_train)
    y_arr = tf.keras.utils.to_categorical(y_train, num_classes=len(label_to_idx))
    
    # Load model
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    
    # Unfreeze top layers and compile with low learning rate to prevent forgetting
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    history = model.fit(
        X_arr, y_arr,
        batch_size=min(4, len(X_arr)),
        epochs=epochs,
        verbose=0
    )
    
    # Save improved weights
    model.save(str(MODEL_PATH))
    
    # Mark as trained
    for rec in records:
        if not rec.get("used_in_retraining", False):
            rec["used_in_retraining"] = True
            
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        
    final_loss = float(history.history["loss"][-1])
    final_acc = float(history.history["accuracy"][-1])
    
    return {
        "success": True,
        "samples_learned": len(X_arr),
        "loss": round(final_loss, 4),
        "accuracy": round(final_acc * 100, 2),
        "message": f"Model successfully fine-tuned on {len(X_arr)} new field samples!"
    }

