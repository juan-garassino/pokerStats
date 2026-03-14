"""
PokerStars screen capture + ROI extractor
Works for the default 6-max table layout at 1920x1080.
Adjust ROI coords if using a different resolution or table skin.
"""

import mss
import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional

# ── PokerStars 6-max 1920x1080 ROI map ──────────────────────────────────────
# These are (left, top, width, height) in screen pixels.
# Capture these once, then run YOLO on each region independently.

POKERSTARS_ROIS = {
    # Your hole cards (bottom center)
    "hole_card_1":     (845,  870, 72, 96),
    "hole_card_2":     (925,  870, 72, 96),

    # Community cards (center table)
    "flop_1":          (668,  490, 72, 96),
    "flop_2":          (752,  490, 72, 96),
    "flop_3":          (836,  490, 72, 96),
    "turn":            (920,  490, 72, 96),
    "river":           (1004, 490, 72, 96),

    # Pot size text region
    "pot":             (840,  440, 240, 32),

    # Your stack + bet
    "hero_stack":      (820,  820, 200, 28),
    "hero_bet":        (820,  770, 200, 28),

    # Opponent stacks (6-max seat positions)
    "seat1_stack":     (200,  600, 180, 28),
    "seat2_stack":     (200,  280, 180, 28),
    "seat3_stack":     (600,  140, 180, 28),
    "seat4_stack":     (1140, 140, 180, 28),
    "seat5_stack":     (1540, 280, 180, 28),
    "seat6_stack":     (1540, 600, 180, 28),

    # Action buttons
    "btn_fold":        (730,  960, 140, 44),
    "btn_call":        (890,  960, 140, 44),
    "btn_raise":       (1050, 960, 140, 44),

    # Raise slider + amount box
    "raise_amount":    (890,  910, 180, 36),

    # Dealer button area (for position detection)
    "dealer_zone":     (620,  400, 680, 280),

    # Turn indicator (whose action it is)
    "action_indicator":(400,  50,  400, 40),
}

# Whole table snapshot (for full-frame YOLO pass as fallback)
TABLE_REGION = {"left": 500, "top": 100, "width": 960, "height": 760}


@dataclass
class CapturedFrame:
    timestamp: float
    full_table: np.ndarray
    rois: dict = field(default_factory=dict)


class PokerStarsCapture:
    def __init__(self, monitor_index: int = 1, target_fps: int = 10):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_index]
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self._last_frame_time = 0.0

    def _grab_region(self, roi: tuple) -> np.ndarray:
        """Grab a single (left, top, w, h) region and return as BGR numpy array."""
        l, t, w, h = roi
        region = {
            "left":   self.monitor["left"] + l,
            "top":    self.monitor["top"]  + t,
            "width":  w,
            "height": h,
        }
        raw = self.sct.grab(region)
        frame = np.array(raw)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def grab_full_table(self) -> np.ndarray:
        region = {
            "left":   self.monitor["left"] + TABLE_REGION["left"],
            "top":    self.monitor["top"]  + TABLE_REGION["top"],
            "width":  TABLE_REGION["width"],
            "height": TABLE_REGION["height"],
        }
        raw = self.sct.grab(region)
        return cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)

    def grab_all_rois(self) -> dict:
        return {name: self._grab_region(roi) for name, roi in POKERSTARS_ROIS.items()}

    def capture_frame(self) -> Optional[CapturedFrame]:
        now = time.monotonic()
        elapsed = now - self._last_frame_time
        if elapsed < self.frame_interval:
            time.sleep(self.frame_interval - elapsed)
        self._last_frame_time = time.monotonic()

        full = self.grab_full_table()
        rois = self.grab_all_rois()
        return CapturedFrame(timestamp=self._last_frame_time, full_table=full, rois=rois)

    def stream(self):
        """Generator — yields CapturedFrame at target_fps indefinitely."""
        while True:
            yield self.capture_frame()

    def save_calibration_snapshot(self, path: str = "calibration.png"):
        """
        Saves the full monitor with ROI rectangles drawn.
        Use this to verify your ROI coordinates are correct.
        """
        raw = self.sct.grab(self.monitor)
        img = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)
        colors = {
            "hole": (0, 255, 0),
            "flop": (0, 200, 255),
            "turn": (0, 200, 255),
            "river":(0, 200, 255),
            "pot":  (255, 200, 0),
            "hero": (255, 100, 0),
            "seat": (200, 0, 255),
            "btn":  (0, 0, 255),
        }
        for name, (l, t, w, h) in POKERSTARS_ROIS.items():
            key = next((k for k in colors if name.startswith(k)), "hole")
            color = colors[key]
            cv2.rectangle(img, (l, t), (l + w, t + h), color, 2)
            cv2.putText(img, name, (l, t - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.imwrite(path, img)
        print(f"Calibration snapshot saved to {path}")
        return img
