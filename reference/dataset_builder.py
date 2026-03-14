"""
PokerStars YOLO dataset builder
────────────────────────────────
Two modes:
  1. synthetic_dataset()   — render card images onto random PokerStars table backgrounds
  2. label_from_screenshots() — semi-auto label from your own PokerStars screenshots

Outputs a YOLOv8-compatible dataset in /data/pokerstars_cards/
"""

import os
import cv2
import numpy as np
import random
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Card class definitions ───────────────────────────────────────────────────
RANKS = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]
SUITS = ["c","d","h","s"]  # clubs, diamonds, hearts, spades

# 52 classes, e.g. "2c", "Ah", "Ks"
CARD_CLASSES = [f"{r}{s}" for r in RANKS for s in SUITS]
CLASS_MAP = {c: i for i, c in enumerate(CARD_CLASSES)}

# Add non-card classes
EXTRA_CLASSES = ["card_back", "empty_slot"]
ALL_CLASSES = CARD_CLASSES + EXTRA_CLASSES
NUM_CLASSES = len(ALL_CLASSES)

# PokerStars 4-color deck defaults: d=blue, c=green, h=red, s=black
SUIT_COLORS_4COLOR = {
    "c": (0, 128, 0),    # green
    "d": (0, 0, 200),    # blue
    "h": (200, 0, 0),    # red
    "s": (20, 20, 20),   # near-black
}
SUIT_SYMBOLS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}

DATASET_ROOT = Path("data/pokerstars_cards")


def render_card(rank: str, suit: str, width=72, height=96, four_color=True) -> np.ndarray:
    """
    Render a single card as a numpy BGR image.
    Mimics PokerStars card style closely enough for training.
    """
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Card border
    draw.rounded_rectangle([(0, 0), (width-1, height-1)], radius=6,
                            outline=(180, 180, 180), width=2)

    color = SUIT_COLORS_4COLOR[suit] if four_color else (
        (200, 0, 0) if suit in ("h", "d") else (20, 20, 20)
    )
    symbol = SUIT_SYMBOLS[suit]

    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = font_large

    # Top-left rank + suit
    draw.text((4, 2),  rank,   fill=color, font=font_small)
    draw.text((4, 14), symbol, fill=color, font=font_small)

    # Center rank + suit (large)
    draw.text((width//2 - 8, height//2 - 18), rank,   fill=color, font=font_large)
    draw.text((width//2 - 8, height//2 + 2),  symbol, fill=color, font=font_large)

    # Bottom-right (inverted)
    draw.text((width-16, height-26), rank,   fill=color, font=font_small)
    draw.text((width-16, height-14), symbol, fill=color, font=font_small)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def make_background(width=960, height=760) -> np.ndarray:
    """Generate a PokerStars-like green felt background with slight texture."""
    # Base felt green
    bg = np.full((height, width, 3), (34, 85, 34), dtype=np.uint8)
    # Add noise for felt texture
    noise = np.random.randint(-15, 15, bg.shape, dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Table oval border
    cv2.ellipse(bg, (width//2, height//2), (width//2 - 40, height//2 - 40),
                0, 0, 360, (20, 60, 20), 8)
    return bg


def augment_card(card_img: np.ndarray) -> np.ndarray:
    """Random augmentation: rotation, brightness, blur, perspective."""
    h, w = card_img.shape[:2]

    # Random rotation ±15°
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    card_img = cv2.warpAffine(card_img, M, (w, h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255,255,255))

    # Random brightness
    factor = random.uniform(0.7, 1.3)
    card_img = np.clip(card_img.astype(float) * factor, 0, 255).astype(np.uint8)

    # Random blur (simulate motion/focus)
    if random.random() < 0.3:
        k = random.choice([3, 5])
        card_img = cv2.GaussianBlur(card_img, (k, k), 0)

    return card_img


def overlay_card(background: np.ndarray, card: np.ndarray, x: int, y: int):
    """Place card onto background at (x, y). Returns updated bg + bbox."""
    h, w = card.shape[:2]
    bh, bw = background.shape[:2]
    x2, y2 = min(x + w, bw), min(y + h, bh)
    background[y:y2, x:x2] = card[:y2-y, :x2-x]
    # YOLO format: cx cy w h (normalized)
    cx = (x + w/2) / bw
    cy = (y + h/2) / bh
    nw = w / bw
    nh = h / bh
    return background, (cx, cy, nw, nh)


def synthetic_dataset(n_images: int = 3000, split: tuple = (0.8, 0.1, 0.1)):
    """
    Generate n_images synthetic labeled images.
    Each image has 2-7 cards placed randomly on a felt background.
    """
    splits = ["train", "val", "test"]
    counts = [int(n_images * s) for s in split]
    counts[-1] = n_images - sum(counts[:-1])  # fix rounding

    for sp in splits:
        (DATASET_ROOT / "images" / sp).mkdir(parents=True, exist_ok=True)
        (DATASET_ROOT / "labels" / sp).mkdir(parents=True, exist_ok=True)

    img_idx = 0
    for split_name, count in zip(splits, counts):
        for i in range(count):
            bg = make_background()
            h, w = bg.shape[:2]
            annotations = []

            n_cards = random.randint(2, 7)
            placed = []
            for _ in range(n_cards):
                rank = random.choice(RANKS)
                suit = random.choice(SUITS)
                label = CLASS_MAP[f"{rank}{suit}"]
                card = render_card(rank, suit, four_color=random.random() > 0.3)
                card = augment_card(card)
                ch, cw = card.shape[:2]

                # Random non-overlapping position
                for _ in range(20):
                    x = random.randint(20, w - cw - 20)
                    y = random.randint(20, h - ch - 20)
                    overlap = any(
                        abs(x - px) < cw and abs(y - py) < ch
                        for px, py in placed
                    )
                    if not overlap:
                        break
                placed.append((x, y))
                bg, bbox = overlay_card(bg, card, x, y)
                annotations.append(f"{label} {' '.join(f'{v:.6f}' for v in bbox)}")

            fname = f"{img_idx:06d}"
            cv2.imwrite(str(DATASET_ROOT / "images" / split_name / f"{fname}.jpg"), bg)
            with open(DATASET_ROOT / "labels" / split_name / f"{fname}.txt", "w") as f:
                f.write("\n".join(annotations))
            img_idx += 1

        print(f"  {split_name}: {count} images")

    # Write dataset.yaml
    yaml_content = f"""path: {DATASET_ROOT.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: {NUM_CLASSES}
names: {ALL_CLASSES}
"""
    with open(DATASET_ROOT / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\nDataset ready at {DATASET_ROOT}/")
    print(f"Total images: {n_images} | Classes: {NUM_CLASSES}")
    return str(DATASET_ROOT / "dataset.yaml")


if __name__ == "__main__":
    print("Generating PokerStars YOLO dataset...")
    yaml_path = synthetic_dataset(n_images=4000)
    print(f"\nNext step:\n  yolo train model=yolov8s.pt data={yaml_path} epochs=100 imgsz=640")
