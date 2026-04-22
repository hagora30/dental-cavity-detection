"""
data_ingest.py
--------------
Downloads the dental cavity dataset from Roboflow into data/raw/.
Modified to bypass the Roboflow 'location' bug safely.
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

# Edit these three values to match your Roboflow project
WORKSPACE   = "bhoomikashetty"   
PROJECT     = "dental-cavity-qfnzu"     
VERSION_NUM = 2                        

# Where the raw download lands
RAW_DATA_DIR = Path("data/raw")


def download_dataset(workspace: str, project: str, version: int, dest: Path) -> Path:
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise EnvironmentError("ROBOFLOW_API_KEY not found in .env")

    print(f"[data_ingest] Connecting to Roboflow workspace: '{workspace}'")
    rf = Roboflow(api_key=api_key)

    project_ref = rf.workspace(workspace).project(project)
    dataset_version = project_ref.version(version)

    print(f"[data_ingest] Downloading version {version} of '{project}'...")
    
    # FIX: Download without the buggy 'location' parameter first
    dataset = dataset_version.download("yolov8")
    downloaded_path = Path(dataset.location)

    print(f"[data_ingest] Moving files to {dest}/...")
    dest.mkdir(parents=True, exist_ok=True)

    # Move files from the default download folder to data/raw/
    for item in downloaded_path.iterdir():
        target_item = dest / item.name
        if target_item.exists():
            if target_item.is_dir():
                shutil.rmtree(target_item)
            else:
                target_item.unlink()
        shutil.move(str(item), str(dest))

    # Clean up the empty directory Roboflow left behind
    downloaded_path.rmdir()

    print(f"[data_ingest] Download complete. Dataset root: {dest}")
    return dest


def verify_structure(dataset_path: Path) -> None:
    expected_splits = ["train", "valid"]
    all_ok = True

    for split in expected_splits:
        img_dir   = dataset_path / split / "images"
        label_dir = dataset_path / split / "labels"

        img_count   = len(list(img_dir.glob("*")))   if img_dir.exists()   else 0
        label_count = len(list(label_dir.glob("*"))) if label_dir.exists() else 0

        status = "✓" if img_count > 0 and label_count > 0 else "✗ MISSING"
        print(f"  [{status}] {split:6s} → {img_count} images, {label_count} labels")

        if img_count == 0 or label_count == 0:
            all_ok = False

    yaml_path = dataset_path / "data.yaml"
    yaml_status = "✓" if yaml_path.exists() else "✗ MISSING"
    print(f"  [{yaml_status}] data.yaml")

    if not all_ok:
        raise FileNotFoundError(
            "Dataset structure verification failed. Check the Roboflow export."
        )

    print("[data_ingest] Structure verified — ready for EDA.")


if __name__ == "__main__":
    dataset_path = download_dataset(
        workspace=WORKSPACE,
        project=PROJECT,
        version=VERSION_NUM,
        dest=RAW_DATA_DIR,
    )

    print("\n[data_ingest] Verifying downloaded structure...")
    verify_structure(dataset_path)