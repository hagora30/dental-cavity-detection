
import argparse
import os
import sys
from pathlib import Path

import wandb
import yaml
from dotenv import load_dotenv
from ultralytics import YOLO



def load_config(config_path: str) -> dict:
    """Loads and returns the YAML training config as a dict."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    print(f"[train] Config loaded from: {config_path}")
    return cfg


def resolve_dataset_yaml(cfg: dict) -> str:
    """
    Returns the absolute path to dataset.yaml.
    YOLOv8 requires an absolute path when called from arbitrary working dirs.
    """
    rel_path = cfg["data"]["config"]
    abs_path = str(Path(rel_path).resolve())
    if not Path(abs_path).exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at: {abs_path}\n"
            f"Make sure you run train.py from the project root."
        )
    return abs_path


def init_wandb(cfg: dict, smoke_test: bool) -> None:
    """
    Initialises a WandB run if enabled in config.
    Skipped automatically if WANDB_API_KEY is not set.
    """
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")

    if not cfg["logging"].get("wandb", False):
        print("[train] WandB logging disabled in config — skipping")
        return

    if not api_key:
        print("[train] WANDB_API_KEY not found in .env — skipping WandB init")
        print("[train] Add WANDB_API_KEY=your_key to .env to enable tracking")
        return

    run_name = cfg["logging"]["run_name"]
    if smoke_test:
        run_name = f"{run_name}-smoke"

    wandb.init(
        project=cfg["logging"]["project"],
        name=run_name,
        config=cfg,
        tags=["yolov8", "dental", "smoke-test" if smoke_test else "full-run"],
    )
    print(f"[train] WandB run initialised: {run_name}")



def run_training(config_path: str, smoke_test: bool = False) -> None:
    """
    Loads config, initialises WandB, and launches the YOLOv8 training loop.
    """
    cfg        = load_config(config_path)
    data_yaml  = resolve_dataset_yaml(cfg)

    # ── Override settings for smoke test ──────────────────────────
    epochs = 3    if smoke_test else cfg["training"]["epochs"]
    batch  = 2    if smoke_test else cfg["training"]["batch"]
    imgsz  = 320  if smoke_test else cfg["model"]["imgsz"]
    device = "cpu" if smoke_test else "cuda"

    print(f"\n[train] ── Run configuration ──────────────────────")
    print(f"  Mode      : {'SMOKE TEST' if smoke_test else 'FULL TRAINING'}")
    print(f"  Model     : {cfg['model']['architecture']}")
    print(f"  Epochs    : {epochs}")
    print(f"  Batch     : {batch}")
    print(f"  Imgsz     : {imgsz}")
    print(f"  Device    : {device}")
    print(f"  Data      : {data_yaml}")
    print(f"──────────────────────────────────────────────────\n")

    init_wandb(cfg, smoke_test)

    model_name = cfg["model"]["architecture"]
    model = YOLO(f"{model_name}.pt")   # downloads pretrained weights if needed
    print(f"[train] Model loaded: {model_name}")

    aug = cfg["augmentation"]

    results = model.train(
        data       = data_yaml,
        epochs     = epochs,
        batch      = batch,
        imgsz      = imgsz,
        device     = device,
        workers    = 0 if smoke_test else cfg["training"]["workers"],
        patience   = cfg["training"]["patience"],
        optimizer  = cfg["training"]["optimizer"],
        lr0        = cfg["training"]["lr0"],
        lrf        = cfg["training"]["lrf"],
        warmup_epochs = cfg["training"]["warmup_epochs"],
        weight_decay  = cfg["training"]["weight_decay"],
        momentum      = cfg["training"]["momentum"],
        # Augmentation
        hsv_h      = aug["hsv_h"],
        hsv_s      = aug["hsv_s"],
        hsv_v      = aug["hsv_v"],
        degrees    = aug["degrees"],
        translate  = aug["translate"],
        scale      = aug["scale"],
        fliplr     = aug["fliplr"],
        flipud     = aug["flipud"],
        mosaic       = aug["mosaic"],
        mixup        = aug["mixup"],
        copy_paste   = aug["copy_paste"],
        dropout      = cfg["training"].get("dropout", 0.0),
        close_mosaic = cfg["training"].get("close_mosaic", 10),
        # Output
        project    = cfg["logging"]["save_dir"],
        name       = cfg["logging"]["run_name"],
        exist_ok   = True,
    )

    print("\n[train] ── Training complete ──")
    print(f"  Results saved to: runs/{cfg['logging']['run_name']}/")

    if wandb.run:
        wandb.finish()
        print("[train] WandB run closed.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 dental cavity detector")
    parser.add_argument(
        "--config", type=str,
        default="configs/train_config.yaml",
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run a 3-epoch CPU smoke test to verify pipeline integrity"
    )
    args = parser.parse_args()

    run_training(config_path=args.config, smoke_test=args.smoke)