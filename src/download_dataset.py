import os
import sys
import shutil
import zipfile
import urllib.request
import logging
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_DIR, DEFAULT_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Curated High-Impact Classes for AgroScan AI
TARGET_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy"
]

def verify_and_clean_images(folder: Path):
    """Scan and verify that all downloaded images are valid RGB readable files."""
    valid_count = 0
    removed_count = 0
    for root, _, files in os.walk(folder):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    valid_count += 1
                except Exception:
                    logger.warning("Corrupt image encountered and removed: %s", file_path)
                    file_path.unlink(missing_ok=True)
                    removed_count += 1
    logger.info("Dataset Verification: %d valid images verified (%d corrupt removed).", valid_count, removed_count)
    return valid_count

def download_via_git():
    """
    Downloads curated PlantVillage color images using Git Sparse-Checkout.
    This selectively downloads ONLY the necessary crop leaf classes (saving ~1GB bandwidth).
    """
    temp_git_dir = PROJECT_ROOT / "data" / "plantvillage_git_temp"
    try:
        if temp_git_dir.exists():
            shutil.rmtree(temp_git_dir)
            
        logger.info("Initializing sparse clone from official PlantVillage repository...")
        clone_cmd = f'git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git "{temp_git_dir}"'
        res = os.system(clone_cmd)
        if res != 0:
            raise RuntimeError("Git clone failed or network unreachable.")

        # Set sparse checkout targets
        cwd = os.getcwd()
        os.chdir(temp_git_dir)
        try:
            os.system('git sparse-checkout set raw/color')
        finally:
            os.chdir(cwd)

        source_color_dir = temp_git_dir / "raw" / "color"
        if not source_color_dir.exists():
            raise RuntimeError("raw/color directory not found in clone.")

        # Move target classes to data/dataset
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        for class_dir in source_color_dir.iterdir():
            if class_dir.is_dir():
                # Map git names to our normalized naming
                normalized_name = class_dir.name.replace("Corn_(maize)___Common_rust_", "Corn___Common_rust") \
                                                .replace("Corn_(maize)___Northern_Leaf_Blight", "Corn___Northern_Leaf_Blight") \
                                                .replace("Corn_(maize)___healthy", "Corn___healthy")
                
                dest_dir = DATASET_DIR / normalized_name
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(class_dir, dest_dir)
                logger.info("Synchronized class: %s (%d images)", normalized_name, len(list(dest_dir.glob("*.*"))))

        # Cleanup temp git clone
        shutil.rmtree(temp_git_dir, ignore_errors=True)
        logger.info("Dataset downloaded and placed in %s", DATASET_DIR)
        verify_and_clean_images(DATASET_DIR)
        return True

    except Exception as e:
        logger.warning("Sparse git download could not complete: %s", e)
        if temp_git_dir.exists():
            shutil.rmtree(temp_git_dir, ignore_errors=True)
        return False

def download_via_direct_zip():
    """
    Fallback method: downloads curated archive if git is not available or blocked.
    """
    archive_url = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
    zip_path = PROJECT_ROOT / "data" / "plantvillage.zip"
    logger.info("Downloading PlantVillage archive from %s...", archive_url)
    
    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0 and count % 500 == 0:
                percent = int(count * block_size * 100 / total_size)
                print(f"\rDownloading dataset: {percent}% complete", end="", flush=True)

        urllib.request.urlretrieve(archive_url, zip_path, reporthook=reporthook)
        print("\nDownload complete. Extracting selected classes...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if "raw/color/" in member and not member.endswith("/"):
                    # Extract file
                    zip_ref.extract(member, PROJECT_ROOT / "data")
                    
        zip_path.unlink(missing_ok=True)
        return True
    except Exception as e:
        logger.error("Direct zip download failed: %s", e)
        return False

if __name__ == "__main__":
    logger.info("Starting automated dataset downloader for AgroScan AI...")
    success = download_via_git()
    if not success:
        logger.info("Attempting alternative direct download...")
        success = download_via_direct_zip()
    
    if success:
        logger.info("Data download complete! You can now run `python src/train.py`.")
    else:
        logger.error("Download encountered an error. If network access is restricted, grant permission or place your PlantVillage images into data/dataset/.")

