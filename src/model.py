import json
import logging
from pathlib import Path
from typing import Tuple, Optional, List
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from src.config import INPUT_SHAPE, LEARNING_RATE, DEFAULT_CLASSES, MODEL_PATH, CLASS_INDICES_PATH, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_augmentation_layer() -> tf.keras.Sequential:
    """
    Builds data augmentation layers for image robustness against variable field lighting,
    angles, and camera noise.
    """
    return models.Sequential([
        layers.RandomFlip("horizontal_and_vertical", name="random_flip"),
        layers.RandomRotation(0.15, name="random_rotation"),
        layers.RandomZoom(0.15, name="random_zoom"),
        layers.RandomTranslation(0.05, 0.05, name="random_translation"),
    ], name="data_augmentation")

def build_crop_disease_model(
    num_classes: int = len(DEFAULT_CLASSES),
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    fine_tune: bool = False,
    learning_rate: float = LEARNING_RATE
) -> tf.keras.Model:
    """
    Constructs a high-accuracy, lightweight crop disease classification network
    powered by MobileNetV2 transfer learning.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # 1. Image preprocessing normalized for MobileNetV2 ([-1, 1])
    x = layers.Rescaling(scale=1./127.5, offset=-1.0, name="mobilenetv2_rescaling")(inputs)
    
    # 2. Base feature extractor (MobileNetV2 with pre-trained ImageNet weights)
    local_weights = MODELS_DIR / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
    if local_weights.exists():
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=None
        )
        base_model.load_weights(str(local_weights))
        logger.info("Loaded pre-trained ImageNet weights from local workspace file: %s", local_weights)
    else:
        try:
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet"
            )
            logger.info("Loaded MobileNetV2 pre-trained ImageNet weights from Keras cache.")
        except Exception:
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights=None
            )
            logger.warning("Initializing MobileNetV2 with random weights.")

    # Freeze base model for rapid feature extraction and optimal transfer learning
    base_model.trainable = fine_tune

    x = base_model(x, training=False)
    
    # 3. Custom Classification Head with Regularization
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_features"
    )(x)
    x = layers.Dropout(0.35, name="dropout")(x)
    
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="disease_predictions"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CropDisease_MobileNetV2")

    # Compile model with Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy")
        ]
    )
    
    return model

def save_class_indices(class_names: List[str], path: Path = CLASS_INDICES_PATH) -> None:
    mapping = {idx: name for idx, name in enumerate(class_names)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    logger.info("Class indices successfully saved to %s", path)

def load_class_indices(path: Path = CLASS_INDICES_PATH) -> List[str]:
    if not path.exists():
        return DEFAULT_CLASSES
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Sort by integer index
        return [data[str(i)] for i in range(len(data))]
