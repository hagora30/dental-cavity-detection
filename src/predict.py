import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO



DEFAULT_WEIGHTS = "runs/yolov8m-run2-imgsz1280/weights/best.pt"
DEFAULT_CONF    = 0.30
DEFAULT_IOU     = 0.45
OUTPUT_DIR      = Path("runs/predictions")

CLASS_NAMES = ["Cavity", "Fillings", "Impacted Tooth", "Implant"]

# BGR colors for each class (OpenCV uses BGR not RGB)
CLASS_COLORS_BGR = {
    "Cavity":         (74,  75,  226),   # red
    "Fillings":       (221, 138, 55),    # blue
    "Impacted Tooth": (117, 158, 29),    # green
    "Implant":        (39,  159, 239),   # amber
}



def draw_predictions(image: np.ndarray, results) -> np.ndarray:
    """
    Draws bounding boxes and labels on the image for all detections.
    Returns the annotated image.
    """
    annotated = image.copy()

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf   = box.conf.item()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
            color    = CLASS_COLORS_BGR.get(cls_name, (255, 255, 255))
            label    = f"{cls_name} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - th - 6), (x1 + tw + 4, y1),
                color, -1
            )

            # Draw label text
            cv2.putText(
                annotated, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA
            )

    return annotated



def run_inference(
    source: str,
    weights: str,
    conf: float,
    iou: float,
) -> None:
    """
    Loads the model and runs inference on all images in source.
    Saves annotated results to OUTPUT_DIR.
    """
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    # Collect image files
    if source_path.is_dir():
        image_files = sorted(
            list(source_path.glob("*.jpg")) +
            list(source_path.glob("*.jpeg")) +
            list(source_path.glob("*.png"))
        )
    else:
        image_files = [source_path]

    if not image_files:
        raise ValueError(f"No images found in: {source_path}")

    print(f"[predict] Loading model: {weights}")
    model = YOLO(weights)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[predict] Running inference on {len(image_files)} image(s)...")
    print(f"[predict] Confidence threshold : {conf}")
    print(f"[predict] Output directory     : {OUTPUT_DIR}/\n")

    total_detections = {cls: 0 for cls in CLASS_NAMES}

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[predict] WARNING: could not read {img_path.name} — skipping")
            continue

        # Run inference
        results = model.predict(
            source=str(img_path),
            imgsz=1280,
            conf=conf,
            iou=iou,
            device="cpu",
            verbose=False,
        )

        # Count detections per class
        detection_summary = {cls: 0 for cls in CLASS_NAMES}
        for result in results:
            for box in result.boxes:
                cls_id   = int(box.cls.item())
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
                detection_summary[cls_name]   += 1
                total_detections[cls_name]    += 1

        # Draw and save
        annotated = draw_predictions(image, results)
        out_path  = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), annotated)

        # Print per-image summary
        found = [f"{cls}:{n}" for cls, n in detection_summary.items() if n > 0]
        summary = ", ".join(found) if found else "no detections"
        print(f"  {img_path.name:<40s} → {summary}")

    # Print overall summary
    print(f"\n[predict] ── Summary ──────────────────────────")
    print(f"  Images processed : {len(image_files)}")
    for cls, count in total_detections.items():
        print(f"  {cls:<16s} : {count} detections")
    print(f"  Saved to         : {OUTPUT_DIR}/")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run dental cavity detection inference on X-ray images"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Path to image file or folder of images"
    )
    parser.add_argument(
        "--weights", type=str, default=DEFAULT_WEIGHTS,
        help="Path to model weights (.pt file)"
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF,
        help="Confidence threshold (default: 0.30)"
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_IOU,
        help="IoU threshold for NMS (default: 0.45)"
    )
    args = parser.parse_args()

    run_inference(
        source  = args.source,
        weights = args.weights,
        conf    = args.conf,
        iou     = args.iou,
    )