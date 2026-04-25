
import cv2
import random
import shutil
import numpy as np
import albumentations as A
from pathlib import Path
from collections import defaultdict



PROCESSED_DIR = Path("data/processed")
TRAIN_IMG_DIR = PROCESSED_DIR / "train" / "images"
TRAIN_LBL_DIR = PROCESSED_DIR / "train" / "labels"

CLASS_NAMES = ["Cavity", "Fillings", "Impacted Tooth", "Implant"]

# How many total augmented copies to generate per minority image
# These are tuned to bring minority classes closer to Fillings count
AUGMENT_TARGETS = {
    "Cavity":         6,   # was 3 — double it
    "Impacted Tooth": 4,   # was 3 — slight increase
    "Implant":        1,   # unchanged
    "Fillings":       0,   # never augment
}

RANDOM_SEED = 42
MIN_VISIBLE_FRACTION = 0.50   # discard box if < 50% remains after transform



def build_augmentation_pipeline() -> A.Compose:
    """
    Builds the Albumentations transform pipeline.
    BboxParams ensures bounding boxes are transformed in sync with the image
    and automatically discards boxes that become too small after cropping.
    """
    return A.Compose(
        [
            # ── Geometry (mild — preserve small boxes) 
            A.HorizontalFlip(p=0.5),

            A.ShiftScaleRotate(
                shift_limit=0.05,    # max 5% translation
                scale_limit=0.10,    # max ±10% zoom
                rotate_limit=10,     # max ±10° rotation
                border_mode=cv2.BORDER_CONSTANT,
                value=0,             # black fill for X-rays
                p=0.7
            ),

            # ── Intensity (mimics X-ray exposure variation)
            A.RandomBrightnessContrast(
                brightness_limit=0.20,
                contrast_limit=0.20,
                p=0.8
            ),

            A.CLAHE(
                clip_limit=3.0,
                tile_grid_size=(8, 8),
                p=0.5
            ),

            # ── Noise (mimics X-ray sensor noise) 
            A.GaussNoise(
                var_limit=(5.0, 20.0),
                p=0.3
            ),

            # ── Blur (mimics slight motion / focus variation) 
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.2),

            # ── Elastic distortion (subtle anatomical variation)
            A.ElasticTransform(
                alpha=30,
                sigma=5,
                p=0.2
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=MIN_VISIBLE_FRACTION,
        ),
    )



def read_yolo_labels(label_path: Path) -> tuple[list[int], list[list[float]]]:
    """
    Returns (class_ids, bboxes) where bboxes are [cx, cy, w, h] normalised.
    """
    class_ids, bboxes = [], []
    if not label_path.exists():
        return class_ids, bboxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_ids.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:]])
    return class_ids, bboxes


def write_yolo_labels(
    label_path: Path,
    class_ids: list[int],
    bboxes: list[list[float]]
) -> None:
    """Writes YOLO label file from class ids and bbox list."""
    with open(label_path, "w") as f:
        for cls_id, bbox in zip(class_ids, bboxes):
            cx, cy, w, h = bbox
            # Cast cls_id to int — Albumentations may return floats
            f.write(f"{int(float(cls_id))} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")



def get_dominant_class(class_ids: list[int]) -> str:
    """
    Returns the class name that appears most in a label file.
    Used to decide which augmentation multiplier to apply to an image.
    An image is augmented based on its rarest/most-valuable class.
    """
    if not class_ids:
        return "Fillings"   # default — won't be augmented

    # Prioritise minority classes: if an image has ANY Cavity or Impacted Tooth,
    # treat it as that class for augmentation targeting
    priority_order = ["Cavity", "Impacted Tooth", "Implant", "Fillings"]
    present = {CLASS_NAMES[i] for i in class_ids if i < len(CLASS_NAMES)}

    for cls in priority_order:
        if cls in present:
            return cls

    return "Fillings"


def augment_training_set() -> None:
    """
    Main augmentation loop. For each training image whose dominant class
    has AUGMENT_TARGETS > 0, generates N augmented copies with new filenames.
    """
    transform  = build_augmentation_pipeline()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    image_files = sorted(
        list(TRAIN_IMG_DIR.glob("*.jpg")) +
        list(TRAIN_IMG_DIR.glob("*.jpeg")) +
        list(TRAIN_IMG_DIR.glob("*.png"))
    )

    # Track how many images we augment per class
    aug_counts = defaultdict(int)
    new_image_count = 0

    print(f"[augmentation] Processing {len(image_files)} training images...")

    for img_path in image_files:
        stem       = img_path.stem
        label_path = TRAIN_LBL_DIR / (stem + ".txt")

        class_ids, bboxes = read_yolo_labels(label_path)
        if not class_ids:
            continue   # skip unannotated images

        dominant_cls = get_dominant_class(class_ids)
        n_copies     = AUGMENT_TARGETS.get(dominant_cls, 0)

        if n_copies == 0:
            continue   # Fillings — skip

        # Read image (keep as-is — X-rays may already be grayscale or RGB)
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[augmentation] WARNING: could not read {img_path.name} — skipping")
            continue

        # Convert BGR → RGB for Albumentations
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for copy_idx in range(n_copies):
            try:
                result = transform(
                    image=image_rgb,
                    bboxes=bboxes,
                    class_labels=class_ids,
                )
            except Exception as e:
                print(f"[augmentation] WARNING: transform failed on {stem}: {e}")
                continue

            aug_image  = result["image"]
            aug_bboxes = result["bboxes"]
            aug_labels = result["class_labels"]

            # Skip if all boxes were lost during transformation
            if not aug_bboxes:
                continue

            # New filename: <original_stem>_aug<N>.<ext>
            new_stem = f"{stem}_aug{copy_idx + 1}"
            new_img_path = TRAIN_IMG_DIR / (new_stem + img_path.suffix)
            new_lbl_path = TRAIN_LBL_DIR / (new_stem + ".txt")

            # Write augmented image (convert back to BGR for OpenCV)
            aug_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(new_img_path), aug_bgr)

            # Write augmented labels
            write_yolo_labels(new_lbl_path, list(aug_labels), list(aug_bboxes))

            new_image_count += 1
            aug_counts[dominant_cls] += 1

    print("\n[augmentation] ── Augmentation complete ──")
    print(f"  New images generated : {new_image_count}")
    for cls in CLASS_NAMES:
        print(f"  {cls:<16s}: +{aug_counts[cls]} augmented images")

    total_train = len(list(TRAIN_IMG_DIR.glob("*.*")))
    print(f"\n  Total train images now: {total_train}")
    print(f"  Output directory      : {TRAIN_IMG_DIR}")



if __name__ == "__main__":
    print("[augmentation] Starting offline augmentation pipeline...")
    print(f"[augmentation] Target multipliers: {AUGMENT_TARGETS}")
    augment_training_set()
    
    