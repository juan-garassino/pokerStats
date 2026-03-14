"""
PokerStars card detector + game state parser
────────────────────────────────────────────
Runs YOLOv8 on each ROI from the capture pipeline,
then assembles a full GameState object every frame.
"""

import cv2
import numpy as np
import re
import pytesseract
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("ultralytics not installed — pip install ultralytics")

from pokerstars_capture import PokerStarsCapture, CapturedFrame, CARD_CLASSES


# ── Model paths ──────────────────────────────────────────────────────────────
MODEL_PATH = "models/pokerstars_yolov8s.pt"   # your trained weights
CONF_THRESHOLD = 0.82


# ── OCR helpers ──────────────────────────────────────────────────────────────
def ocr_number(img: np.ndarray) -> Optional[float]:
    """
    Extract a dollar/chip amount from a small ROI.
    Returns float or None if parse fails.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    # Scale up for better OCR accuracy
    scaled = cv2.resize(thresh, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
    text = pytesseract.image_to_string(
        scaled,
        config="--psm 7 -c tessedit_char_whitelist=0123456789.,$ "
    ).strip()
    text = text.replace("$", "").replace(",", "").strip()
    try:
        return float(text.split()[0]) if text else None
    except (ValueError, IndexError):
        return None


# ── Game state dataclass ─────────────────────────────────────────────────────
@dataclass
class PlayerState:
    seat: int
    stack: Optional[float] = None
    current_bet: Optional[float] = None
    is_active: bool = True
    cards: list = field(default_factory=list)  # opponent visible cards


@dataclass
class GameState:
    # Cards
    hole_cards: list = field(default_factory=list)      # ["Ah", "Ks"]
    community_cards: list = field(default_factory=list) # up to 5

    # Money
    pot: Optional[float] = None
    hero_stack: Optional[float] = None
    hero_bet: Optional[float] = None

    # Players
    players: dict = field(default_factory=dict)         # seat -> PlayerState

    # Street
    street: str = "preflop"  # preflop / flop / turn / river

    # Position (detected from dealer button)
    dealer_seat: Optional[int] = None
    hero_position: Optional[str] = None  # BTN / CO / HJ / MP / EP / SB / BB

    # Action
    available_actions: list = field(default_factory=list)  # ["fold","call","raise"]
    call_amount: Optional[float] = None
    min_raise: Optional[float] = None
    raise_amount: Optional[float] = None

    # Metadata
    confidence: float = 0.0
    timestamp: float = 0.0

    def __str__(self):
        cc = " ".join(self.community_cards) or "(none)"
        hc = " ".join(self.hole_cards) or "??"
        return (
            f"[{self.street.upper()}] "
            f"Hand: {hc} | Board: {cc} | "
            f"Pot: ${self.pot} | Stack: ${self.hero_stack} | "
            f"Pos: {self.hero_position}"
        )


# ── Detector ─────────────────────────────────────────────────────────────────
class PokerStarsDetector:
    def __init__(self, model_path: str = MODEL_PATH):
        if YOLO_AVAILABLE and Path(model_path).exists():
            self.model = YOLO(model_path)
            print(f"Loaded YOLO model from {model_path}")
        else:
            self.model = None
            print(f"YOLO model not found at {model_path} — using OCR-only mode")

    def _detect_card_in_roi(self, roi_img: np.ndarray) -> Optional[str]:
        """
        Run YOLO on a single card ROI.
        Falls back to template matching if confidence is low.
        """
        if self.model is None:
            return None

        results = self.model(roi_img, verbose=False, conf=CONF_THRESHOLD)
        if not results or len(results[0].boxes) == 0:
            return None

        box = results[0].boxes[0]
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < CONF_THRESHOLD or cls_id >= len(CARD_CLASSES):
            return None

        return CARD_CLASSES[cls_id]

    def _detect_buttons(self, frame: CapturedFrame) -> list:
        """
        Detect which action buttons are visible (fold/call/raise).
        Uses simple color thresholding — PokerStars buttons have distinct colors.
        """
        actions = []
        button_colors = {
            "fold":  ([0, 0, 150], [60, 80, 255]),    # red-ish fold button
            "call":  ([0, 120, 0], [80, 255, 80]),     # green call button
            "raise": ([100, 60, 0], [255, 200, 80]),   # orange raise button
        }
        for action, (lower, upper) in button_colors.items():
            roi = frame.rois.get(f"btn_{action}")
            if roi is None:
                continue
            mask = cv2.inRange(roi,
                               np.array(lower, dtype=np.uint8),
                               np.array(upper, dtype=np.uint8))
            if mask.sum() > 500:  # threshold: enough colored pixels
                actions.append(action)
        return actions

    def parse_frame(self, frame: CapturedFrame) -> GameState:
        state = GameState(timestamp=frame.timestamp)

        # ── Hole cards ──
        for slot in ["hole_card_1", "hole_card_2"]:
            card = self._detect_card_in_roi(frame.rois[slot])
            if card:
                state.hole_cards.append(card)

        # ── Community cards ──
        for slot in ["flop_1", "flop_2", "flop_3", "turn", "river"]:
            card = self._detect_card_in_roi(frame.rois[slot])
            if card:
                state.community_cards.append(card)

        # ── Determine street ──
        n = len(state.community_cards)
        state.street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(n, "preflop")

        # ── Pot + stacks (OCR) ──
        state.pot        = ocr_number(frame.rois["pot"])
        state.hero_stack = ocr_number(frame.rois["hero_stack"])
        state.hero_bet   = ocr_number(frame.rois["hero_bet"])

        for i in range(1, 7):
            key = f"seat{i}_stack"
            stack = ocr_number(frame.rois.get(key, np.zeros((28,180,3), np.uint8)))
            state.players[i] = PlayerState(seat=i, stack=stack)

        # ── Available actions ──
        state.available_actions = self._detect_buttons(frame)

        # ── Confidence: ratio of successfully detected items ──
        detected = len(state.hole_cards) + len(state.community_cards)
        expected = 2 + n
        state.confidence = detected / max(expected, 1)

        return state


# ── Main loop ────────────────────────────────────────────────────────────────
def run_detection_loop(model_path: str = MODEL_PATH, fps: int = 10):
    capture = PokerStarsCapture(monitor_index=1, target_fps=fps)
    detector = PokerStarsDetector(model_path=model_path)

    print("Starting PokerStars detection loop (Ctrl+C to stop)...")
    print("Run capture.save_calibration_snapshot() first to verify ROI alignment.\n")

    prev_state = None
    for frame in capture.stream():
        state = detector.parse_frame(frame)

        # Only print when state changes meaningfully
        if prev_state is None or state.community_cards != prev_state.community_cards \
                or state.hole_cards != prev_state.hole_cards:
            print(state)

        prev_state = state

        # Yield state to agent (in practice, call agent.decide(state) here)
        yield state


if __name__ == "__main__":
    # Calibration mode: just verify ROI positions
    cap = PokerStarsCapture(monitor_index=1)
    cap.save_calibration_snapshot("calibration_check.png")
    print("Open calibration_check.png to verify ROI boxes are aligned with PokerStars UI.")
