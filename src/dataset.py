import logging
from pathlib import Path
from typing import Tuple, Optional, List
import tensorflow as tf

from src.config import DATASET_DIR, IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED
from src.model import build_augmentation_layer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_datasets(
    dataset_dir: Path = DATASET_DIR,
    val_split: float = 0.2,
    test_split: float = 0.1,
    batch_size: int = BATCH_SIZE,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    seed: int = RANDOM_SEED
) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Optional[tf.data.Dataset], List[str]]:
    """
    Loads training, validation, and test datasets from directory.
    Supports either pre-split folders (train/val/test) or a unified class directory.
    """
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    test_dir = dataset_dir / "test"

    # Case 1: Pre-existing train/val splits
    if train_dir.exists() and any(train_dir.iterdir()):
        logger.info("Found pre-split directory structure: train, val, test")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            seed=seed
        )
        class_names = train_ds.class_names
        
        val_ds = None
        if val_dir.exists() and any(val_dir.iterdir()):
            val_ds = tf.keras.utils.image_dataset_from_directory(
                val_dir,
                image_size=image_size,
                batch_size=batch_size,
                shuffle=False
            )
            
        test_ds = None
        if test_dir.exists() and any(test_dir.iterdir()):
            test_ds = tf.keras.utils.image_dataset_from_directory(
                test_dir,
                image_size=image_size,
                batch_size=batch_size,
                shuffle=False
            )
            
    # Case 2: Unified directory (auto-split)
    elif dataset_dir.exists() and any(dataset_dir.iterdir()):
        logger.info("Using automatic train/val splitting from %s", dataset_dir)
        train_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            validation_split=val_split,
            subset="training",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            validation_split=val_split,
            subset="validation",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size
        )
        class_names = train_ds.class_names
        test_ds = None
    else:
        logger.warning("No image data found in %s", dataset_dir)
        return None, None, None, []

    # Apply data augmentation only on training set
    aug_layer = build_augmentation_layer()
    train_ds = train_ds.map(
        lambda x, y: (aug_layer(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Prefetch for optimal throughput
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    if val_ds is not None:
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
