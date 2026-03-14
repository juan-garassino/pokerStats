"""
Train YOLOv8s on the PokerStars synthetic dataset.
Run: python train_yolo.py

Requirements:
  pip install ultralytics torch torchvision
  python dataset_builder.py  (generate dataset first)
"""

from ultralytics import YOLO
from pathlib import Path
import yaml

DATASET_YAML = "data/pokerstars_cards/dataset.yaml"
OUTPUT_DIR   = "models"
RUN_NAME     = "pokerstars_v1"

def train():
    # Use YOLOv8s — small model is fast enough for 52 card classes
    # at near-perfect accuracy (cards have high visual distinctiveness)
    model = YOLO("yolov8s.pt")

    results = model.train(
        data=DATASET_YAML,
        epochs=100,
        imgsz=640,
        batch=32,
        patience=20,           # early stopping: stop if no improvement for 20 epochs
        lr0=0.01,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Augmentations — keep moderate, cards are already distinctive
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=15.0,          # rotation up to ±15°
        translate=0.1,
        scale=0.3,
        flipud=0.0,            # cards are orientation-sensitive, no vertical flip
        fliplr=0.0,            # same: Ah ≠ mirror(Ah)
        mosaic=0.5,
        mixup=0.1,
        # Output
        project=OUTPUT_DIR,
        name=RUN_NAME,
        save=True,
        save_period=10,        # save checkpoint every 10 epochs
        device=0,              # GPU 0 — change to "cpu" if no GPU
        workers=8,
        verbose=True,
    )

    # Copy best weights to a clean path
    best = Path(OUTPUT_DIR) / RUN_NAME / "weights" / "best.pt"
    dest = Path(OUTPUT_DIR) / "pokerstars_yolov8s.pt"
    if best.exists():
        import shutil
        shutil.copy(best, dest)
        print(f"\nBest model saved to {dest}")

    return results


def validate(model_path: str = f"{OUTPUT_DIR}/pokerstars_yolov8s.pt"):
    """Quick validation pass — prints mAP@50 and per-class AP."""
    model = YOLO(model_path)
    metrics = model.val(data=DATASET_YAML, imgsz=640, batch=16)
    print(f"\nmAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print("\nPer-class AP (top 10 lowest — focus training effort here):")
    class_ap = list(zip(metrics.box.ap_class_index, metrics.box.ap50))
    class_ap.sort(key=lambda x: x[1])
    from dataset_builder import ALL_CLASSES
    for cls_id, ap in class_ap[:10]:
        print(f"  {ALL_CLASSES[cls_id]:6s}  AP@50={ap:.3f}")


def export_to_onnx(model_path: str = f"{OUTPUT_DIR}/pokerstars_yolov8s.pt"):
    """
    Export to ONNX for faster CPU inference (no PyTorch dependency at runtime).
    Useful if running the bot on a separate low-power machine.
    """
    model = YOLO(model_path)
    model.export(format="onnx", imgsz=640, optimize=True)
    print(f"Exported ONNX to {model_path.replace('.pt','.onnx')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","validate","export"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        print("Step 1: Generating dataset...")
        from dataset_builder import synthetic_dataset
        synthetic_dataset(n_images=4000)
        print("\nStep 2: Training YOLOv8s...")
        train()

    elif args.mode == "validate":
        validate()

    elif args.mode == "export":
        export_to_onnx()

    # ── Expected results ──────────────────────────────────────────────────────
    # With 4000 synthetic + fine-tuned on ~200 real PokerStars screenshots:
    #   mAP@50 > 0.97 is achievable in ~2 hours on an RTX 3080
    #   Inference: ~8ms per ROI on GPU, ~25ms on CPU (ONNX)
    #   False positive rate on empty slots: <1%
