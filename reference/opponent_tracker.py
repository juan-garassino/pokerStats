"""
Opponent tracker
────────────────
• OCR player names from PokerStars seat labels
• Track every action per player per hand
• Detect bluffs at showdown (their cards visible + they bet/raised)
• Build real-time stats fed to the decision engine
"""

import cv2
import numpy as np
import re
import time
from dataclasses import dataclass, field
from typing import Optional
import pytesseract

from hand_history_db import HandHistoryDB, OpponentProfile


# ── PokerStars seat name ROIs (1920×1080, 6-max) ────────────────────────────
# Name labels sit just above the avatar/chip stack area for each seat
SEAT_NAME_ROIS = {
    1: (155,  638, 200, 26),   # bottom-left
    2: (155,  310, 200, 26),   # mid-left
    3: (560,  165, 200, 26),   # top-left
    4: (1155, 165, 200, 26),   # top-right
    5: (1545, 310, 200, 26),   # mid-right
    6: (1545, 638, 200, 26),   # bottom-right
}

# Action label ROIs — PokerStars shows "folds", "calls $X", "raises to $X" above each seat
SEAT_ACTION_ROIS = {
    1: (155,  610, 220, 22),
    2: (155,  282, 220, 22),
    3: (560,  137, 220, 22),
    4: (1155, 137, 220, 22),
    5: (1545, 282, 220, 22),
    6: (1545, 610, 220, 22),
}

# Opponent hole card ROIs (only visible at showdown)
SEAT_CARD_ROIS = {
    1: [(90,  655, 60, 80), (155, 655, 60, 80)],
    2: [(90,  330, 60, 80), (155, 330, 60, 80)],
    3: [(490, 180, 60, 80), (555, 180, 60, 80)],
    4: [(1155,180, 60, 80), (1220,180, 60, 80)],
    5: [(1480,330, 60, 80), (1545,330, 60, 80)],
    6: [(1480,655, 60, 80), (1545,655, 60, 80)],
}


def ocr_text(img: np.ndarray, whitelist: str = "") -> str:
    """OCR a region, return cleaned text."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Increase contrast for PokerStars white-on-dark name labels
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    scaled = cv2.resize(thresh, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
    cfg = "--psm 7"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist='{whitelist}'"
    text = pytesseract.image_to_string(scaled, config=cfg).strip()
    # Remove common OCR artifacts
    text = re.sub(r"[^\w\s\-_\.]", "", text).strip()
    return text[:24]  # PokerStars max name length is 24


def parse_action_text(text: str) -> tuple:
    """
    Parse PokerStars action overlay text.
    Returns (action_type, amount) e.g. ("raises", 45.0)
    """
    text = text.lower().strip()
    if "folds"  in text: return ("fold",  0.0)
    if "checks" in text: return ("check", 0.0)

    m = re.search(r"calls?\s*\$?([\d,.]+)", text)
    if m: return ("call", float(m.group(1).replace(",", "")))

    m = re.search(r"raises?\s+to\s*\$?([\d,.]+)", text)
    if m: return ("raise", float(m.group(1).replace(",", "")))

    m = re.search(r"bets?\s*\$?([\d,.]+)", text)
    if m: return ("bet", float(m.group(1).replace(",", "")))

    m = re.search(r"all.?in\s*\$?([\d,.]+)?", text)
    if m:
        amt = float(m.group(1).replace(",", "")) if m.group(1) else 0.0
        return ("allin", amt)

    return ("unknown", 0.0)


@dataclass
class SeatState:
    seat: int
    name: str = ""
    stack: float = 0.0
    current_action: str = ""
    current_bet: float = 0.0
    cards: list = field(default_factory=list)
    is_active: bool = True
    last_action_ts: float = 0.0


class OpponentTracker:
    def __init__(self, db: HandHistoryDB, capture):
        self.db = db
        self.capture = capture
        self.seats: dict = {}             # seat -> SeatState
        self.current_hand_id: Optional[int] = None
        self._prev_action_texts: dict = {}

        # In-hand action sequence per opponent (reset each hand)
        self._hand_actions: dict = field(default_factory=dict)
        self._hand_actions = {}

    # ── Name detection ────────────────────────────────────────────────────────
    def refresh_seat_names(self):
        """Re-OCR all seat name regions. Call at start of each hand."""
        for seat, roi in SEAT_NAME_ROIS.items():
            img = self.capture._grab_region(roi)
            name = ocr_text(img)
            if name and len(name) > 1:
                if seat not in self.seats:
                    self.seats[seat] = SeatState(seat=seat)
                self.seats[seat].name = name
                self.db.upsert_opponent(name)

    # ── Live action detection ─────────────────────────────────────────────────
    def poll_actions(self, street: str, pot: float, board: list) -> list:
        """
        Check all seats for new action overlays.
        Returns list of (seat, name, action, amount) tuples for any new actions.
        """
        new_actions = []
        for seat, roi in SEAT_ACTION_ROIS.items():
            img = self.capture._grab_region(roi)
            text = ocr_text(img, whitelist="abcdefghijklmnopqrstuvwxyz$0123456789.,- ")
            if not text or text == self._prev_action_texts.get(seat, ""):
                continue

            self._prev_action_texts[seat] = text
            action_type, amount = parse_action_text(text)
            if action_type == "unknown":
                continue

            seat_state = self.seats.get(seat)
            name = seat_state.name if seat_state else f"seat{seat}"
            if not name:
                continue

            # Log to DB
            if self.current_hand_id:
                self.db.log_action(
                    self.current_hand_id, street, name,
                    action_type, amount, pot, board
                )
                # Update live stats
                self._update_live_stats(name, street, action_type, amount, pot)

            # Track in-hand sequence
            if name not in self._hand_actions:
                self._hand_actions[name] = []
            self._hand_actions[name].append({
                "street": street, "action": action_type,
                "amount": amount, "pot": pot
            })

            new_actions.append((seat, name, action_type, amount))

        return new_actions

    def _update_live_stats(self, name: str, street: str, action: str, amount: float, pot: float):
        """Increment running stats counters in DB."""
        if street == "preflop":
            if action in ("call", "raise", "bet", "allin"):
                self.db.increment_stat(name, "vpip")
            if action in ("raise", "bet", "allin"):
                self.db.increment_stat(name, "pfr")
        if street == "flop" and action in ("bet", "raise"):
            self.db.increment_stat(name, "cbet_flop")
        if action == "fold":
            # Check if this was a fold to a 3-bet preflop
            prior = self._hand_actions.get(name, [])
            if street == "preflop" and len(prior) >= 1:
                self.db.increment_stat(name, "fold_to_3bet")
            # Fold to cbet on flop
            if street == "flop":
                self.db.increment_stat(name, "fold_to_cbet")

    # ── Showdown detection ────────────────────────────────────────────────────
    def detect_showdown(self, street: str, pot: float) -> list:
        """
        At showdown, OCR opponent hole cards and classify as bluff or value.
        Returns list of showdown events.
        """
        from pokerstars_detector import PokerStarsDetector
        detector = PokerStarsDetector.__new__(PokerStarsDetector)

        events = []
        for seat, card_rois in SEAT_CARD_ROIS.items():
            seat_state = self.seats.get(seat)
            if not seat_state or not seat_state.name:
                continue

            # Try to detect cards in both slots
            detected = []
            for roi in card_rois:
                img = self.capture._grab_region(roi)
                # Use YOLO if available, else skip
                if hasattr(detector, 'model') and detector.model:
                    card = detector._detect_card_in_roi(img)
                    if card:
                        detected.append(card)

            if len(detected) < 2:
                continue

            # Classify bluff: did they bet/raise this street with a weak hand?
            prior = self._hand_actions.get(seat_state.name, [])
            last_aggressive = next(
                (a for a in reversed(prior) if a["action"] in ("bet", "raise", "allin")),
                None
            )
            was_bluff = False
            bet_size = 0.0
            if last_aggressive:
                bet_size = last_aggressive["amount"]
                # Simple bluff heuristic: bet on river with no pair/draw
                # (full equity calculation would go here with treys)
                was_bluff = self._classify_bluff(detected, last_aggressive)

            name = seat_state.name
            self.db.increment_stat(name, "total_showdowns")
            if was_bluff:
                self.db.increment_stat(name, "bluffs_seen")

            if self.current_hand_id:
                self.db.log_showdown(
                    self.current_hand_id, name, detected,
                    was_bluff, last_aggressive["action"] if last_aggressive else "check",
                    pot, bet_size
                )

            events.append({
                "seat": seat, "name": name,
                "cards": detected, "was_bluff": was_bluff
            })

        return events

    def _classify_bluff(self, cards: list, last_action: dict) -> bool:
        """
        Heuristic bluff classifier. In production, replace with full equity eval.
        Bluff = bet/raise on river + hand has no pair (high-card / missed draw).
        """
        try:
            from treys import Card, Evaluator
            evaluator = Evaluator()
            hand = [Card.new(c[0] + c[1].lower()) for c in cards if len(c) == 2]
            # Without the board here we can't evaluate exactly — mark unknown
            # Full version would pass board cards too
            return False  # placeholder: requires board context
        except Exception:
            return False

    # ── Hand boundary management ──────────────────────────────────────────────
    def on_new_hand(self, hand_id: int):
        self.current_hand_id = hand_id
        self._hand_actions.clear()
        self._prev_action_texts.clear()
        self.refresh_seat_names()
        for seat_state in self.seats.values():
            if seat_state.name:
                self.db.increment_stat(seat_state.name, "hands_played")

    def get_table_profiles(self) -> dict:
        """Return OpponentProfile for all seated players, keyed by seat number."""
        profiles = {}
        for seat, state in self.seats.items():
            if state.name:
                profile = self.db.get_opponent(state.name)
                if profile:
                    profiles[seat] = profile
        return profiles
