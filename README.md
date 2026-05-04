<h1 style="color:#2c5d85;">Dental Cavity Detection</h1>

<p style="font-size:18px;">YOLOv8 Object Detection Pipeline</p>

<div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:10px;">
  <div style="background:#2c6fb7; color:white; padding:6px 12px; border-radius:4px;">
    Model: YOLOv8m
  </div>
  <div style="background:#2e8b57; color:white; padding:6px 12px; border-radius:4px;">
    GPU: NVIDIA L40S
  </div>
  <div style="background:#7a3db8; color:white; padding:6px 12px; border-radius:4px;">
    mAP50: 0.758
  </div>
  <div style="background:#c96a00; color:white; padding:6px 12px; border-radius:4px;">
    Export: ONNX 99MB
  </div>
</div>

<hr>


## Executive Summary
A complete, end-to-end machine learning pipeline for automated detection of dental conditions in panoramic X-ray images.

The system identifies and localizes four key dental conditions using a fine-tuned YOLOv8m model trained on 1,148 annotated X-rays.

Challenges addressed:
- 236 data leakage cases
- 11.6x class imbalance
- Small object detection (<1% image area)

Intended use:
Clinical decision support. The model highlights regions for review, not a replacement for clinical judgment.

---

## Detection Classes

| ID | Class            | Description |
|----|------------------|------------|
| 0  | Cavity           | Active dental caries |
| 1  | Fillings         | Restored teeth |
| 2  | Impacted Tooth   | Non-erupted tooth |
| 3  | Implant          | Artificial tooth root |

---

## Dataset

| Property | Value |
|----------|------|
| Source | Roboflow — dental-cavity-qfnzu v3 |
| License | CC BY 4.0 |
| Images | 1,148 |
| Annotations | 8,500 |
| Split | 804 train / 230 valid / 114 test |
| Avg annotations per image | 7.4 |

processed dataset on kaggle : https://www.kaggle.com/datasets/hagargalall/x-ray-dental-data
---

## Key Challenges

Data Leakage:
236 duplicated images across splits. Fixed by rebuilding splits.

Class Imbalance:
Fillings = 65.4%  
Reduced imbalance from 11.6x to 6.5x.

Small Objects:
Median size = 0.78%  
Resolution increased from 640 to 1280.

---

## Training Pipeline

    Raw Dataset
       ↓
    Data Cleaning
       ↓
    Split Rebuild
       ↓
    Augmentation
       ↓
    Local Test (CPU)
       ↓
    Cloud Training (L40S)
       ↓
    Evaluation
       ↓
    Export ONNX

---

## Training Settings

- Model: YOLOv8m  
- Image Size: 1280  
- Batch: 24  
- Epochs: 118  
- Optimizer: AdamW  
- Dropout: 0.1  
- Tracking: Weights & Biases  

---

## Results

### Run Comparison

| Setting | Run 1 | Run 2 |
|--------|------|------|
| imgsz | 640 | 1280 |
| Cavity augmentation | x3 | x6 |
| Dropout | 0 | 0.1 |
| Epochs | 73 | 118 |
| Val to Test mAP50 gap | 0.574 | 0.033 |

---

## Final Metrics

| Class | Precision | Recall | mAP50 | mAP50-95 |
|------|----------|--------|------|----------|
| All | 0.815 | 0.639 | 0.758 | 0.462 |
| Cavity | 0.728 | 0.281 | 0.433 | 0.252 |
| Fillings | 0.847 | 0.721 | 0.820 | 0.484 |
| Impacted Tooth | 0.732 | 0.659 | 0.816 | 0.520 |
| Implant | 0.954 | 0.896 | 0.961 | 0.592 |

---

## Deployment

### Local Setup

    git clone https://github.com/hagora30/dental-cavity-detection.git
    cd dental-cavity-detection
    conda env create -f environment.yml
    conda activate dental-cv

### Cloud Setup

    git clone https://github.com/hagora30/dental-cavity-detection.git
    cd dental-cavity-detection
    pip install -r requirements-cloud.txt

---

## Configuration

Create a .env file:

    ROBOFLOW_API_KEY=your_key
    WANDB_API_KEY=your_key

---

## Run Pipeline

    python src/data_ingest.py
    python src/eda.py
    python src/data_cleaning.py
    python src/rebuild_splits.py
    python src/augmentation.py
    python src/train.py --smoke
    python src/train.py
    python src/evaluate.py --weights path/to/best.pt
    python src/predict.py --source path/to/xray.jpg
    python tests/test_pipeline_smoke.py

---

## Tech Stack

- Model: YOLOv8m (Ultralytics)
- Augmentation: Albumentations
- Tracking: Weights & Biases
- Cloud: Lightning AI (L40S)
- Local: WSL + Conda
- Version Control: GitHub
- Export: ONNX

---

## Conclusion

Dental cavity detection using computer vision for clinical decision support.

https://github.com/hagora30/dental-cavity-detection
