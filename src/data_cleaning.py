"""
data_cleaning.py
----------------
Cleans the raw dataset and writes sanitised output to data/cleaned/.

Actions performed:
  1. Remove annotations with near-zero bounding box area (noise)
  2. Detect Roboflow-duplicated images (same source, multiple rf. hashes)
  3. Verify no image appears in both train and valid splits (leakage check)
  4. Copy clean images + labels to data/cleaned/ preserving split structure
  5. Write a cleaning report to notebooks/eda_outputs/cleaning_report.txt

Usage (from project root):
    python src/data_cleaning.py
"""

import shutil
import cv2
from pathlib import Path
from collections import defaultdict
import pandas as pd


# ── Config ────────────────────────────────────────────────────────────────────

RAW_DIR     = Path("data/raw")
CLEAN_DIR   = Path("data/cleaned")
REPORT_PATH = Path("notebooks/eda_outputs/cleaning_report.txt")
SPLITS      = ["train", "valid", "test"]
CLASS_NAMES = ["Cavity", "Fillings", "Impacted Tooth", "Implant"]

# Boxes with normalised area below this threshold are treated as noise
MIN_BOX_AREA = 0.0005

# Images whose annotation count exceeds mean + N*std are flagged (not removed)
OUTLIER_STD_MULTIPLIER = 3.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_source_stem(filename: str) -> str:
    """
    Extracts the original image ID from a Roboflow-augmented filename.

    Roboflow renames files as:  <original_stem>_jpg.rf.<hash>
    We want just:               <original_stem>

    Example:
        '0546_jpg.rf.1d24c3b34f81049db8632b4c1607bba8'  →  '0546'
    """
    name = Path(filename).stem
    if "_jpg.rf." in name:
        return name.split("_jpg.rf.")[0]
    if "_png.rf." in name:
        return name.split("_png.rf.")[0]
    return name   # not a Roboflow-augmented name — return as-is


def read_labels(label_path: Path) -> list[list[float]]:
    """
    Reads a YOLO label file and returns a list of
    [class_id, cx, cy, w, h] rows. Returns empty list if file is missing.
    """
    if not label_path.exists():
        return []
    rows = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                rows.append([float(p) for p in parts])
    return rows


def write_labels(label_path: Path, rows: list[list[float]]) -> None:
    """Writes cleaned label rows back to a YOLO .txt file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for row in rows:
            cls = int(row[0])
            f.write(f"{cls} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}\n")


# ── Cleaning passes ───────────────────────────────────────────────────────────

def pass1_remove_tiny_boxes(
    rows: list[list[float]],
    filename: str,
    report_lines: list[str]
) -> tuple[list[list[float]], int]:
    """
    Pass 1: Drop any box whose normalised area (w*h) is below MIN_BOX_AREA.
    Returns the cleaned rows and the count of removed boxes.
    """
    cleaned = []
    removed = 0
    for row in rows:
        _, cx, cy, w, h = row
        area = w * h
        if area < MIN_BOX_AREA:
            report_lines.append(
                f"  [REMOVED tiny box] {filename}  area={area:.6f}  "
                f"cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f}"
            )
            removed += 1
        else:
            cleaned.append(row)
    return cleaned, removed


def pass2_clamp_boxes(
    rows: list[list[float]],
    filename: str,
    report_lines: list[str]
) -> tuple[list[list[float]], int]:
    """
    Pass 2: Clamp any box coordinates that fall outside [0, 1].
    Roboflow augmentations can occasionally produce boxes that slightly
    exceed image boundaries.
    """
    clamped_count = 0
    fixed_rows = []
    for row in rows:
        cls, cx, cy, w, h = row

        # Clamp centre and size so the box stays within [0,1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w  = max(0.001, min(1.0, w))
        h  = max(0.001, min(1.0, h))

        # Ensure the box doesn't extend past image edges
        x1 = cx - w / 2
        x2 = cx + w / 2
        y1 = cy - h / 2
        y2 = cy + h / 2

        needs_clamp = x1 < 0 or x2 > 1 or y1 < 0 or y2 > 1
        if needs_clamp:
            x1, x2 = max(0.0, x1), min(1.0, x2)
            y1, y2 = max(0.0, y1), min(1.0, y2)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w  = x2 - x1
            h  = y2 - y1
            clamped_count += 1
            report_lines.append(
                f"  [CLAMPED box] {filename}  class={int(cls)} "
                f"new: cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f}"
            )

        fixed_rows.append([cls, cx, cy, w, h])

    return fixed_rows, clamped_count


def pass3_detect_duplicates(all_stems: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Pass 3: Groups filenames by their Roboflow source stem.
    Returns only groups that have more than one image (true duplicates).

    all_stems: { split -> [filename_stem, ...] }
    """
    source_map = defaultdict(list)   # source_stem -> [(split, filename)]

    for split, stems in all_stems.items():
        for stem in stems:
            source = get_source_stem(stem)
            source_map[source].append((split, stem))

    duplicates = {
        src: entries
        for src, entries in source_map.items()
        if len(entries) > 1
    }
    return duplicates


def pass4_check_leakage(all_stems: dict[str, list[str]]) -> list[str]:
    """
    Pass 4: Checks whether the same source image appears in both
    train and valid/test splits — which would constitute data leakage.
    """
    split_sources = {}
    for split, stems in all_stems.items():
        split_sources[split] = {get_source_stem(s) for s in stems}

    leaking = []
    train_sources = split_sources.get("train", set())
    for split in ["valid", "test"]:
        overlap = train_sources & split_sources.get(split, set())
        if overlap:
            for src in sorted(overlap):
                leaking.append(f"  [LEAKAGE] source '{src}' appears in both train and {split}")

    return leaking


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_cleaning() -> None:
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("  DATA CLEANING REPORT")
    report_lines.append("=" * 60)

    total_removed_boxes   = 0
    total_clamped_boxes   = 0
    total_images_copied   = 0
    all_stems: dict[str, list[str]] = {}
    class_counts_after: dict[str, dict[str, int]] = {}

    # ── Process each split ────────────────────────────────────────
    for split in SPLITS:
        img_dir_raw   = RAW_DIR   / split / "images"
        lbl_dir_raw   = RAW_DIR   / split / "labels"
        img_dir_clean = CLEAN_DIR / split / "images"
        lbl_dir_clean = CLEAN_DIR / split / "labels"

        img_dir_clean.mkdir(parents=True, exist_ok=True)
        lbl_dir_clean.mkdir(parents=True, exist_ok=True)

        if not img_dir_raw.exists():
            print(f"[cleaning] Skipping {split} — directory not found")
            continue

        image_files = sorted(
            list(img_dir_raw.glob("*.jpg")) +
            list(img_dir_raw.glob("*.jpeg")) +
            list(img_dir_raw.glob("*.png"))
        )

        split_removed = 0
        split_clamped = 0
        split_stems   = []
        class_counts_after[split] = defaultdict(int)

        report_lines.append(f"\n── {split.upper()} split ({len(image_files)} images) ──")

        for img_path in image_files:
            stem       = img_path.stem
            label_path = lbl_dir_raw / (stem + ".txt")
            split_stems.append(stem)

            rows = read_labels(label_path)

            # Apply cleaning passes
            rows, n_removed = pass1_remove_tiny_boxes(rows, stem, report_lines)
            rows, n_clamped = pass2_clamp_boxes(rows, stem, report_lines)

            split_removed += n_removed
            split_clamped += n_clamped

            # Count class distribution after cleaning
            for row in rows:
                cls_name = CLASS_NAMES[int(row[0])] if int(row[0]) < len(CLASS_NAMES) else "unknown"
                class_counts_after[split][cls_name] += 1

            # Copy image to cleaned directory
            shutil.copy2(img_path, img_dir_clean / img_path.name)

            # Write cleaned labels (even if unchanged — ensures consistency)
            clean_label_path = lbl_dir_clean / (stem + ".txt")
            write_labels(clean_label_path, rows)
            total_images_copied += 1

        all_stems[split] = split_stems
        total_removed_boxes += split_removed
        total_clamped_boxes += split_clamped
        print(f"[cleaning] {split:6s} → {len(image_files)} images processed  "
              f"| {split_removed} boxes removed  | {split_clamped} boxes clamped")

    # ── Duplicate and leakage checks ──────────────────────────────
    report_lines.append("\n── Duplicate image detection ──")
    duplicates = pass3_detect_duplicates(all_stems)
    if duplicates:
        report_lines.append(f"  Found {len(duplicates)} source images with Roboflow duplicates:")
        for src, entries in sorted(duplicates.items()):
            splits_present = [e[0] for e in entries]
            report_lines.append(f"  {src}: {len(entries)} copies → splits: {splits_present}")
    else:
        report_lines.append("  ✓ No duplicates found")

    report_lines.append("\n── Data leakage check ──")
    leakage = pass4_check_leakage(all_stems)
    if leakage:
        for line in leakage:
            report_lines.append(line)
    else:
        report_lines.append("  ✓ No leakage detected — train/valid/test are clean")

    # ── Class distribution after cleaning ─────────────────────────
    report_lines.append("\n── Class distribution after cleaning (train) ──")
    train_counts = class_counts_after.get("train", {})
    total_train  = sum(train_counts.values())
    for cls in CLASS_NAMES:
        count = train_counts.get(cls, 0)
        pct   = count / total_train * 100 if total_train > 0 else 0
        report_lines.append(f"  {cls:<16s} {count:5d}  ({pct:5.1f}%)")

    # ── Summary ───────────────────────────────────────────────────
    report_lines.append("\n── Summary ──")
    report_lines.append(f"  Total images copied : {total_images_copied}")
    report_lines.append(f"  Tiny boxes removed  : {total_removed_boxes}")
    report_lines.append(f"  Boxes clamped       : {total_clamped_boxes}")
    report_lines.append(f"  Duplicate groups    : {len(duplicates)}")
    report_lines.append(f"  Leakage issues      : {len(leakage)}")
    report_lines.append("\n  data/cleaned/ is ready for augmentation.")
    report_lines.append("=" * 60)

    # ── Write report ──────────────────────────────────────────────
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))

    # Print full report to terminal
    print("\n" + "\n".join(report_lines))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[cleaning] Starting data cleaning pipeline...")
    print(f"[cleaning] Source : {RAW_DIR}")
    print(f"[cleaning] Output : {CLEAN_DIR}")
    run_cleaning()