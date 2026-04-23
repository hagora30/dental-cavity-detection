
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from ultralytics import YOLO



CLASS_NAMES  = ["Cavity", "Fillings", "Impacted Tooth", "Implant"]
CLASS_COLORS = {
    "Cavity":         "#E24B4A",
    "Fillings":       "#378ADD",
    "Impacted Tooth": "#1D9E75",
    "Implant":        "#EF9F27",
}
DATA_YAML    = "configs/dataset.yaml"
OUTPUT_DIR   = Path("notebooks/eda_outputs/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_validation(model: YOLO) -> dict:
    """
    Runs YOLOv8 validation on the test split and returns metrics dict.
    """
    print("[evaluate] Running validation on test split...")
    metrics = model.val(
        data=str(Path(DATA_YAML).resolve()),
        split="test",
        imgsz=640,
        batch=8,
        device="cpu",
        verbose=True,
        plots=True,
        save_json=True,
        project=str(OUTPUT_DIR),
        name="val_results",
        exist_ok=True,
    )
    return metrics


def print_metrics_report(metrics) -> None:
    """Prints a clean per-class metrics table and saves to txt."""
    lines = []
    lines.append("=" * 65)
    lines.append("  EVALUATION REPORT — YOLOv8m Dental Cavity Detection")
    lines.append("=" * 65)

    lines.append("\n── Per-class metrics (test split) ───────────────────────")
    lines.append(f"  {'Class':<16} {'Precision':>10} {'Recall':>10} "
                 f"{'mAP50':>10} {'mAP50-95':>10}")
    lines.append("  " + "─" * 55)

    # Overall metrics
    mp  = metrics.box.mp    # mean precision
    mr  = metrics.box.mr    # mean recall
    map50    = metrics.box.map50
    map5095  = metrics.box.map

    lines.append(f"  {'ALL':<16} {mp:>10.3f} {mr:>10.3f} "
                 f"{map50:>10.3f} {map5095:>10.3f}")
    lines.append("  " + "─" * 55)

    # Per-class metrics
    for i, cls_name in enumerate(CLASS_NAMES):
        try:
            p    = metrics.box.p[i]
            r    = metrics.box.r[i]
            ap50 = metrics.box.ap50[i]
            ap   = metrics.box.ap[i]
            lines.append(f"  {cls_name:<16} {p:>10.3f} {r:>10.3f} "
                         f"{ap50:>10.3f} {ap:>10.3f}")
        except IndexError:
            lines.append(f"  {cls_name:<16} {'N/A':>10}")

    # Clinical interpretation
    lines.append("\n── Clinical interpretation ───────────────────────────────")
    try:
        cavity_recall = metrics.box.r[0]
        if cavity_recall < 0.5:
            lines.append(f"  ⚠ Cavity recall {cavity_recall:.3f} — model misses "
                         f"{(1-cavity_recall)*100:.1f}% of cavities")
            lines.append("    → Recommend: more Cavity augmentation + yolov8l")
        elif cavity_recall < 0.7:
            lines.append(f"  △ Cavity recall {cavity_recall:.3f} — acceptable but improvable")
            lines.append("    → Recommend: fine-tune with higher cls loss weight for Cavity")
        else:
            lines.append(f"  ✓ Cavity recall {cavity_recall:.3f} — clinically acceptable")
    except IndexError:
        pass

    lines.append("\n── Recommendations for next iteration ────────────────────")
    lines.append("  1. Increase Cavity augmentation multiplier from 3x to 5x")
    lines.append("  2. Try yolov8l architecture for better small object detection")
    lines.append("  3. Increase imgsz from 640 to 1280 — cavities are tiny")
    lines.append("  4. Set patience=50 to allow longer convergence")
    lines.append("=" * 65)

    report = "\n".join(lines)
    print("\n" + report)

    report_path = OUTPUT_DIR / "evaluation_report.txt"
    report_path.write_text(report)
    print(f"\n[evaluate] Report saved → {report_path}")


def visualise_test_predictions(model: YOLO, n_samples: int = 8) -> None:
    """
    Runs inference on random test images and visualises predictions
    vs ground truth side by side.
    """
    print("\n[evaluate] Generating prediction visualisations...")

    test_img_dir = Path("data/processed/test/images")
    test_lbl_dir = Path("data/processed/test/labels")

    image_files = sorted(list(test_img_dir.glob("*.jpg")) +
                         list(test_img_dir.glob("*.png")))

    if not image_files:
        print("[evaluate] No test images found — skipping visualisation")
        return

    # Sample randomly
    import random
    random.seed(42)
    samples = random.sample(image_files, min(n_samples, len(image_files)))

    fig, axes = plt.subplots(
        len(samples), 2,
        figsize=(14, 4 * len(samples))
    )
    if len(samples) == 1:
        axes = [axes]

    fig.suptitle(
        "Ground Truth (left) vs Predictions (right)",
        fontsize=14, fontweight="bold"
    )

    for row_idx, img_path in enumerate(samples):
        stem       = img_path.stem
        label_path = test_lbl_dir / (stem + ".txt")

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w    = img_rgb.shape[:2]

        # ── Left: ground truth ────────────────────────────────────
        ax_gt = axes[row_idx][0]
        ax_gt.imshow(img_rgb)
        ax_gt.set_title(f"GT: {stem[:25]}", fontsize=8)
        ax_gt.axis("off")

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(float(parts[0]))
                    cx, cy, bw, bh = map(float, parts[1:])
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    cls_name = CLASS_NAMES[cls_id]
                    color    = CLASS_COLORS[cls_name]
                    rect = patches.Rectangle(
                        (x1, y1), bw * w, bh * h,
                        linewidth=1.5, edgecolor=color, facecolor="none"
                    )
                    ax_gt.add_patch(rect)
                    ax_gt.text(
                        x1, y1 - 3, cls_name, fontsize=6,
                        color=color, fontweight="bold",
                        bbox=dict(facecolor="black", alpha=0.4,
                                  pad=1, edgecolor="none")
                    )

        # ── Right: predictions ────────────────────────────────────
        ax_pred = axes[row_idx][1]
        ax_pred.imshow(img_rgb)
        ax_pred.set_title("Predictions", fontsize=8)
        ax_pred.axis("off")

        results = model.predict(
            source=str(img_path),
            imgsz=640,
            conf=0.25,
            device="cpu",
            verbose=False,
        )

        for result in results:
            for box in result.boxes:
                cls_id   = int(box.cls.item())
                conf     = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
                color    = CLASS_COLORS.get(cls_name, "#ffffff")

                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=1.5, edgecolor=color, facecolor="none"
                )
                ax_pred.add_patch(rect)
                ax_pred.text(
                    x1, y1 - 3,
                    f"{cls_name} {conf:.2f}",
                    fontsize=6, color=color, fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.4,
                              pad=1, edgecolor="none")
                )

    plt.tight_layout()
    out = OUTPUT_DIR / "predictions_vs_gt.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved → {out}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/yolov8m-baseline/weights/best.pt",
        help="Path to trained model weights"
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at: {weights_path}")

    print(f"[evaluate] Loading model from: {weights_path}")
    model = YOLO(str(weights_path))

    metrics = run_validation(model)
    print_metrics_report(metrics)
    visualise_test_predictions(model)

    print("\n[evaluate] Complete. Check notebooks/eda_outputs/evaluation/")