"""
Decision engine
───────────────
Street-aware (preflop → flop → turn → river) decision making:
  1. Equity calculation  (Monte Carlo with treys)
  2. GTO baseline        (pot odds + position + equity threshold)
  3. Exploit layer       (opponent profile adjustments)
  4. Bet sizing          (mixed strategy sampling)
  5. ICM stub            (placeholder for tournament mode)
"""

import random
import math
from dataclasses import dataclass
from typing import Optional

try:
    from treys import Card, Deck, Evaluator
    TREYS_OK = True
except ImportError:
    TREYS_OK = False
    print("pip install treys")

from hand_history_db import OpponentProfile


# ── Decision output ───────────────────────────────────────────────────────────
@dataclass
class Decision:
    action: str          # "fold" | "call" | "raise" | "check" | "allin"
    amount: float = 0.0  # raise amount (absolute), 0 for fold/call/check
    reasoning: str = ""  # human-readable explanation
    confidence: float = 0.0


# ── Monte Carlo equity engine ─────────────────────────────────────────────────
def calculate_equity(
    hole_cards: list,       # ["Ah", "Ks"]
    community_cards: list,  # ["2c", "7d", "Jh"] etc.
    num_opponents: int = 1,
    simulations: int = 5000,
) -> float:
    """
    Run Monte Carlo equity simulation.
    Returns win probability [0.0 – 1.0].
    """
    if not TREYS_OK or not hole_cards or len(hole_cards) < 2:
        return 0.5  # fallback

    try:
        evaluator = Evaluator()

        def to_card(s: str) -> int:
            # treys format: "Ah" -> Card.new("Ah")
            rank = s[0].upper()
            suit = s[1].lower()
            return Card.new(f"{rank}{suit}")

        hero = [to_card(c) for c in hole_cards]
        board = [to_card(c) for c in community_cards]
        used   = set(hole_cards + community_cards)

        # Full deck minus used cards
        all_cards = []
        for r in "23456789TJQKA":
            for s in "cdhs":
                c = f"{r}{s}"
                if c not in used:
                    all_cards.append(c)

        wins = 0
        for _ in range(simulations):
            random.shuffle(all_cards)
            remaining_board = board[:]
            drawn = list(all_cards)

            # Complete board
            idx = 0
            while len(remaining_board) < 5:
                remaining_board.append(to_card(drawn[idx]))
                idx += 1

            # Deal opponent hands
            opp_hands = []
            for o in range(num_opponents):
                opp_hands.append([to_card(drawn[idx]), to_card(drawn[idx+1])])
                idx += 2

            # Evaluate
            hero_score = evaluator.evaluate(remaining_board, hero)
            opp_scores = [evaluator.evaluate(remaining_board, opp) for opp in opp_hands]

            # Lower score = better hand in treys
            if hero_score < min(opp_scores):
                wins += 1
            elif hero_score == min(opp_scores):
                wins += 0.5  # split

        return wins / simulations

    except Exception as e:
        return 0.5


# ── Street-specific GTO thresholds ────────────────────────────────────────────
# (equity required to continue, indexed by position quality and street)
# These are simplified — a real implementation uses CFR lookup trees

GTO_THRESHOLDS = {
    # (street, position_quality) -> min_equity_to_continue
    ("preflop", "ip"):   0.35,   # in position: play wider
    ("preflop", "oop"):  0.42,
    ("flop",    "ip"):   0.32,
    ("flop",    "oop"):  0.38,
    ("turn",    "ip"):   0.36,
    ("turn",    "oop"):  0.42,
    ("river",   "ip"):   0.40,   # river: need stronger equity to call/raise
    ("river",   "oop"):  0.46,
}

# Bet sizing ranges by street (fraction of pot)
BET_SIZES = {
    "preflop":  [2.2, 2.5, 3.0],   # open-raise multipliers (BB)
    "flop":     [0.33, 0.50, 0.75],
    "turn":     [0.50, 0.66, 0.80],
    "river":    [0.50, 0.75, 1.00],
}

POSITIONS_IP = {"btn", "co"}  # in-position vs most of the table


class DecisionEngine:
    def __init__(self):
        self.equity_cache: dict = {}

    def _is_ip(self, position: str) -> bool:
        return str(position).lower() in POSITIONS_IP

    def _pot_odds(self, call_amount: float, pot: float) -> float:
        """Required equity to break even on a call."""
        if call_amount <= 0:
            return 0.0
        return call_amount / (pot + call_amount)

    def _sample_bet_size(self, street: str, pot: float, equity: float,
                         opponent_profile: Optional[OpponentProfile] = None) -> float:
        """
        Sample a bet size from the mixed strategy distribution.
        Adjusts sizing based on opponent tendencies.
        """
        sizes = BET_SIZES.get(street, [0.5])

        # If opponent folds to large bets, go bigger as a bluff
        if opponent_profile and opponent_profile.fold_to_cbet > 0.65:
            sizes = [max(sizes)]  # polarize: bet big

        # If opponent is a calling station, use smaller value bets
        if opponent_profile and opponent_profile.player_type == "fish":
            sizes = [min(sizes)]

        frac = random.choice(sizes)
        return round(pot * frac, 2)

    def _preflop_hand_strength(self, hole_cards: list) -> float:
        """
        Fast preflop hand strength without simulation.
        Uses Chen formula approximation.
        """
        if len(hole_cards) < 2:
            return 0.0

        rank_map = {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,
                    "9":9,"T":10,"J":11,"Q":12,"K":13,"A":14}

        r1 = rank_map.get(hole_cards[0][0], 0)
        r2 = rank_map.get(hole_cards[1][0], 0)
        s1 = hole_cards[0][1]
        s2 = hole_cards[1][1]

        suited = s1 == s2
        pair   = r1 == r2
        gap    = abs(r1 - r2)
        hi     = max(r1, r2)

        score = hi * 0.5
        if pair:   score = max(score * 2, 5.0)
        if suited: score += 2
        if gap == 0 and not pair: score += 1  # connected
        if gap == 1: score += 0.5

        # Normalize to [0,1]
        return min(score / 20.0, 1.0)

    def decide(
        self,
        hole_cards: list,
        community_cards: list,
        street: str,
        pot: float,
        hero_stack: float,
        call_amount: float,
        position: str,
        num_opponents: int,
        opponent_profiles: dict,   # seat -> OpponentProfile
        available_actions: list,
    ) -> Decision:

        if not hole_cards:
            return Decision("fold", reasoning="no cards detected")

        # ── 1. Equity ─────────────────────────────────────────────────────────
        cache_key = (tuple(hole_cards), tuple(community_cards), num_opponents)
        if cache_key not in self.equity_cache:
            if street == "preflop":
                # Faster: use Chen formula for preflop, MC for postflop
                equity = self._preflop_hand_strength(hole_cards)
            else:
                equity = calculate_equity(
                    hole_cards, community_cards,
                    num_opponents=num_opponents,
                    simulations=3000 if street in ("turn","river") else 2000
                )
            self.equity_cache[cache_key] = equity
        equity = self.equity_cache[cache_key]

        # ── 2. Pot odds ───────────────────────────────────────────────────────
        pot_odds = self._pot_odds(call_amount, pot)
        position_quality = "ip" if self._is_ip(position) else "oop"
        gto_threshold = GTO_THRESHOLDS.get((street, position_quality), 0.40)

        # ── 3. Opponent exploit overlay ───────────────────────────────────────
        # Aggregate table-level tendencies
        profiles = list(opponent_profiles.values())
        avg_bluff_freq = sum(p.bluff_frequency for p in profiles) / max(len(profiles), 1)
        avg_fold_cbet  = sum(p.fold_to_cbet   for p in profiles) / max(len(profiles), 1)
        has_maniacs    = any(p.player_type == "maniac" for p in profiles)
        has_fish       = any(p.player_type == "fish"   for p in profiles)

        # Tighten vs maniacs (they have higher natural equity, but we trap more)
        if has_maniacs:
            gto_threshold += 0.03

        # Loosen vs fish (they call too wide, so we value-bet more)
        if has_fish:
            gto_threshold -= 0.04

        # Adjust call threshold based on observed bluff frequency
        # If opponents bluff >40% of the time, call down more
        bluff_adjusted_threshold = pot_odds * (1 - avg_bluff_freq * 0.5)

        # ── 4. Decision logic ────────────────────────────────────────────────
        # Primary opponent (for sizing, use the one who's most active)
        primary_opp = max(profiles, key=lambda p: p.hands_played) if profiles else None

        # No bet to call → check or bet
        if "raise" not in available_actions and call_amount == 0:
            if "check" in available_actions:
                if equity > 0.55:
                    bet_amt = self._sample_bet_size(street, pot, equity, primary_opp)
                    return Decision(
                        "raise", bet_amt,
                        f"value bet {equity:.0%} equity vs check",
                        confidence=equity
                    )
                return Decision("check", 0.0, f"check {equity:.0%} equity", confidence=equity)

        # Facing a bet
        if call_amount > 0:
            spr = hero_stack / max(pot, 1)   # stack-to-pot ratio

            # Fold: equity below pot odds even after bluff adjustment
            if equity < bluff_adjusted_threshold and equity < gto_threshold - 0.05:
                return Decision(
                    "fold", 0.0,
                    f"fold: eq={equity:.0%} < pot_odds={pot_odds:.0%}, opp_bluff={avg_bluff_freq:.0%}",
                    confidence=1 - equity
                )

            # Raise / re-raise: strong equity + opponent folds to pressure
            if (equity > 0.65 and "raise" in available_actions) or \
               (equity > 0.55 and avg_fold_cbet > 0.60):
                bet_amt = self._sample_bet_size(street, pot, equity, primary_opp)
                if street == "river" and equity > 0.75:
                    bet_amt = pot * 1.0   # pot-sized value bet on river
                return Decision(
                    "raise", round(call_amount + bet_amt, 2),
                    f"raise: eq={equity:.0%}, opp_fold_cbet={avg_fold_cbet:.0%}",
                    confidence=equity
                )

            # All-in: near-nut hand + short stack
            if equity > 0.78 and spr < 3 and "allin" in available_actions:
                return Decision(
                    "allin", hero_stack,
                    f"shove: eq={equity:.0%}, spr={spr:.1f}",
                    confidence=equity
                )

            # Call
            return Decision(
                "call", call_amount,
                f"call: eq={equity:.0%} > pot_odds={pot_odds:.0%}",
                confidence=equity
            )

        # Default
        return Decision("check", 0.0, "no action detected", confidence=0.5)
