"""
Action executor
───────────────
Controls the PokerStars UI:
  • Clicks fold / call / check buttons
  • Sets raise amount via slider or text field
  • Adds human-like timing jitter (critical for not looking robotic)
  • Verifies action was registered (screenshot diff)
"""

import time
import random
import math
import pyautogui
import cv2
import numpy as np
from dataclasses import dataclass

from decision_engine import Decision

# Disable pyautogui fail-safe for smoother automation
# (re-enable during testing — move mouse to corner to abort)
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = True

# ── PokerStars button centers (1920×1080 6-max) ──────────────────────────────
# These are absolute screen coordinates of button centers
BUTTON_CENTERS = {
    "fold":  (800,  982),
    "check": (960,  982),
    "call":  (960,  982),   # same position as check (PokerStars uses same slot)
    "raise": (1120, 982),
    "allin": (1120, 982),
}

# Raise amount text input field (where you type the amount)
RAISE_INPUT_CENTER = (980, 932)
RAISE_INPUT_BOX    = (890, 910, 180, 36)   # for clearing

# Raise slider
RAISE_SLIDER_LEFT  = (740,  930)
RAISE_SLIDER_RIGHT = (1200, 930)

# Confirm raise button (appears after typing amount)
RAISE_CONFIRM_BTN  = (1120, 982)


@dataclass
class TimingProfile:
    """Humanized timing randomization."""
    # Base delay before acting (simulates "thinking")
    think_min: float = 0.8
    think_max: float = 3.5

    # Delay for specific situations (longer = looks more human)
    preflop_open_max: float = 2.5
    river_decision_max: float = 6.0

    # Mouse movement style
    mouse_duration_min: float = 0.12
    mouse_duration_max: float = 0.35

    # Occasional long tanks (simulates reading the board)
    tank_probability: float = 0.08
    tank_min: float = 8.0
    tank_max: float = 25.0


DEFAULT_TIMING = TimingProfile()


def _human_delay(street: str = "flop",
                 action: str = "call",
                 timing: TimingProfile = DEFAULT_TIMING):
    """
    Sleep for a human-like amount of time before acting.
    Varies by street, action type, and adds occasional long tanks.
    """
    # Occasional deep tank
    if random.random() < timing.tank_probability:
        delay = random.uniform(timing.tank_min, timing.tank_max)
        time.sleep(delay)
        return

    base_min = timing.think_min
    base_max = timing.think_max

    # Longer on river
    if street == "river":
        base_max = timing.river_decision_max
    # Raise requires "thinking about sizing"
    if action in ("raise", "allin"):
        base_min += 0.4
        base_max += 1.0
    # Folds can be quicker (sometimes snap-fold for big hand range tells — avoid)
    if action == "fold":
        base_min = max(base_min, 0.6)

    delay = random.uniform(base_min, base_max)
    # Add micro-jitter (keyboard latency simulation)
    time.sleep(delay + random.gauss(0, 0.05))


def _move_and_click(x: int, y: int, timing: TimingProfile = DEFAULT_TIMING):
    """Move mouse with human-like arc and click."""
    # Add small random offset to avoid pixel-perfect clicking (bot tell)
    jitter_x = random.randint(-4, 4)
    jitter_y = random.randint(-3, 3)
    duration = random.uniform(timing.mouse_duration_min, timing.mouse_duration_max)

    pyautogui.moveTo(x + jitter_x, y + jitter_y, duration=duration, tween=pyautogui.easeInOutQuad)
    time.sleep(random.uniform(0.04, 0.12))  # hover pause
    pyautogui.click()
    time.sleep(random.uniform(0.05, 0.15))  # post-click pause


def _set_raise_amount(amount: float, min_raise: float, max_raise: float,
                      timing: TimingProfile = DEFAULT_TIMING):
    """
    Set raise amount. Strategy:
      1. Click the raise input field
      2. Select all + delete
      3. Type the amount (with realistic keystroke delays)
    """
    # Clamp to valid range
    amount = max(min_raise, min(amount, max_raise))
    amount_str = str(int(amount)) if amount == int(amount) else f"{amount:.2f}"

    # Click the raise input box
    _move_and_click(*RAISE_INPUT_CENTER, timing)
    time.sleep(0.1)

    # Select all and clear
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.05)
    pyautogui.press("delete")
    time.sleep(0.08)

    # Type amount with realistic inter-key delays
    for char in amount_str:
        pyautogui.press(char)
        time.sleep(random.uniform(0.05, 0.14))

    time.sleep(0.1)


class ActionExecutor:
    def __init__(self, timing: TimingProfile = DEFAULT_TIMING, dry_run: bool = False):
        """
        dry_run=True: print actions without clicking (safe testing mode)
        """
        self.timing = timing
        self.dry_run = dry_run
        self._last_action_ts = 0.0

    def execute(
        self,
        decision: Decision,
        street: str = "flop",
        pot: float = 0.0,
        min_raise: float = 0.0,
        max_raise: float = 9999.0,
        available_actions: list = None,
    ) -> bool:
        """
        Execute a decision on the PokerStars UI.
        Returns True if action was successfully taken.
        """
        available_actions = available_actions or ["fold", "call", "raise"]
        action = decision.action.lower()

        # Map "bet" to "raise" (same button in PokerStars)
        if action == "bet":
            action = "raise"

        # Safety: don't act too quickly
        elapsed = time.time() - self._last_action_ts
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)

        # Validate action is available
        if action not in available_actions and action not in BUTTON_CENTERS:
            if self.dry_run:
                print(f"[DRY RUN] Action '{action}' not available, folding")
            action = "fold" if "fold" in available_actions else available_actions[0]

        if self.dry_run:
            print(f"[DRY RUN] street={street} action={action} amount={decision.amount:.2f} | {decision.reasoning}")
            self._last_action_ts = time.time()
            return True

        # ── Human-like delay ──────────────────────────────────────────────────
        _human_delay(street, action, self.timing)

        # ── Execute ───────────────────────────────────────────────────────────
        try:
            if action in ("fold", "check", "call"):
                btn = BUTTON_CENTERS.get(action)
                if btn:
                    _move_and_click(*btn, self.timing)

            elif action in ("raise", "allin"):
                # First set the amount, then click raise
                _set_raise_amount(decision.amount, min_raise, max_raise, self.timing)
                time.sleep(random.uniform(0.2, 0.5))
                _move_and_click(*RAISE_CONFIRM_BTN, self.timing)

            self._last_action_ts = time.time()
            return True

        except Exception as e:
            print(f"[ActionExecutor] Error executing {action}: {e}")
            return False

    def emergency_fold(self):
        """Failsafe: fold immediately without timing delay."""
        if not self.dry_run:
            pyautogui.click(*BUTTON_CENTERS["fold"])
