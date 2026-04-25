# Results

## Training — Run 2 (YOLOv8m, imgsz=1280, L40S GPU)

| File | Description |
|------|-------------|
| `training/results.png` | Loss and mAP curves across 118 epochs |
| `training/confusion_matrix_normalized.png` | Per-class confusion matrix |
| `training/F1_curve.png` | F1 score vs confidence threshold |
| `training/PR_curve.png` | Precision-Recall curve per class |
| `training/results.csv` | Raw metrics per epoch |

## EDA

| File | Description |
|------|-------------|
| `eda/01_class_distribution.png` | Annotation counts per class per split |
| `eda/02_bbox_statistics.png` | Bounding box size distribution |
| `eda/03_bbox_heatmap.png` | Spatial heatmap of bbox centres |
| `eda/04_sample_annotations.png` | Sample annotated X-rays |

## Evaluation — Test Split (114 images, never seen during training)

| File | Description |
|------|-------------|
| `evaluation/predictions_vs_gt.png` | Ground truth vs model predictions |
| `evaluation/evaluation_report.txt` | Full per-class metrics report |

### Final Test Metrics

| Class | mAP50 | Recall |
|-------|-------|--------|
| All | 0.758 | 0.639 |
| Cavity | 0.433 | 0.281 |
| Fillings | 0.820 | 0.721 |
| Impacted Tooth | 0.816 | 0.659 |
| Implant | 0.961 | 0.896 |
