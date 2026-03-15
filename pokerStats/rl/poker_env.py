"""
No-Limit Texas Hold'em environment
────────────────────────────────────
OpenAI Gym-compatible multi-agent poker environment.

Design choices:
  - Observation:  145-dim vector (hand + board + betting + position + hand strength)
  - Action space: Discrete(10) — fold, check, call, raise 33/50/75/100/125/150/200%, all_in
  - Reward:       BB-normalized (chips won / big_blind)
  - Info asymmetry: each agent only sees its own hole cards
  - Supports 2-6 players
"""

import copy
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_OK = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_OK = True
    except ImportError:
        GYM_OK = False

try:
    from treys import Card, Deck, Evaluator
    TREYS_OK = True
    _evaluator = Evaluator()
except ImportError:
    TREYS_OK = False

from .renderer import render_table, render_hand


# ── Action encoding ───────────────────────────────────────────────────────────
# Hybrid discrete-continuous: 5 action types + continuous raise sizing
class Action(IntEnum):
    FOLD   = 0
    CHECK  = 1   # only valid when no bet to call
    CALL   = 2
    RAISE  = 3   # continuous sizing via raise_frac in [0, 1]
    ALL_IN = 4

NUM_ACTIONS = len(Action)

# raise_frac mapping: 0.0 = min raise, 1.0 = all-in
# Exponential curve: smooth near 0 (fine control on small bets),
# steep near 1 (coarse on overbets). Formula:
#   actual = (e^(k * frac) - 1) / (e^k - 1),  k = 3.0
# Landmarks (k=3):
#   frac=0.0 → min raise     frac=0.3 → ~12% of range
#   frac=0.5 → ~26%          frac=0.7 → ~51%
#   frac=0.9 → ~82%          frac=1.0 → all-in
import math
RAISE_CURVE_K = 3.0
_EXP_K = math.exp(RAISE_CURVE_K) - 1.0

def raise_frac_to_amount(frac: float, min_raise: float, max_raise: float) -> float:
    """Map [0,1] to [min_raise, max_raise] with exponential curve."""
    frac = max(0.0, min(1.0, frac))
    curved = (math.exp(RAISE_CURVE_K * frac) - 1.0) / _EXP_K
    return min_raise + curved * (max_raise - min_raise)

# ── Observation space layout ──────────────────────────────────────────────────
# Padded to MAX_PLAYERS=9 so one network works for 2, 3, 6, or 8 players.
# Empty seats get zeros — the agent learns to ignore them.
#
# [0:52]    hole cards (one-hot, 52 bits)
# [52:104]  community cards (one-hot, 52 bits)
# [104:113] per-player stack normalized (9 slots, padded)
# [113:122] per-player current bet normalized
# [122:131] per-player active flags
# [131:140] per-player position (dealer=0,SB=1,BB=2... normalized)
# [140:144] pot, call_amount, min_raise, num_active (normalized)
# [144:148] street one-hot [preflop, flop, turn, river]
# [148:155] last 7 actions normalized
# [155]     hand strength (0=trash, 1=nuts)
# [156]     draw potential (0=none, 1=strong draw)
# [157]     raises_this_street / max_raises (aggression level)
# [158]     facing_raise (1.0 if must call a raise)
# [159]     pot_odds (call / (pot + call), 0 if no bet)
MAX_PLAYERS = 9
OBS_DIM = 160

STREETS = ["preflop", "flop", "turn", "river"]


# ── Card encoding ─────────────────────────────────────────────────────────────
RANKS = "23456789TJQKA"
SUITS = "cdhs"
ALL_CARDS = [f"{r}{s}" for r in RANKS for s in SUITS]
CARD_IDX  = {c: i for i, c in enumerate(ALL_CARDS)}


def cards_to_onehot(cards: list) -> np.ndarray:
    v = np.zeros(52, dtype=np.float32)
    for c in cards:
        if c in CARD_IDX:
            v[CARD_IDX[c]] = 1.0
    return v


def hand_strength(hole_cards: list, community: list) -> float:
    """
    Estimate hand strength 0-1 (1 = nuts).
    Uses treys evaluator postflop, simple heuristic preflop.
    This lets the agent explicitly know when it's bluffing.
    """
    if not hole_cards:
        return 0.0

    if not community:
        # Preflop heuristic based on card ranks
        r0 = RANKS.index(hole_cards[0][0])
        r1 = RANKS.index(hole_cards[1][0])
        suited = hole_cards[0][1] == hole_cards[1][1]
        pair = r0 == r1
        high = max(r0, r1) / 12.0
        if pair:
            return 0.5 + high * 0.5  # pairs: 0.5 (22) to 1.0 (AA)
        gap = abs(r0 - r1)
        connected = gap <= 1
        score = high * 0.4 + (0.1 if suited else 0) + (0.1 if connected else 0) - gap * 0.02
        return max(0.0, min(1.0, score))

    if TREYS_OK:
        try:
            board = [Card.new(c[0] + c[1]) for c in community]
            hand = [Card.new(c[0] + c[1]) for c in hole_cards]
            rank = _evaluator.evaluate(board, hand)
            # treys rank: 1 (royal flush) to 7462 (worst)
            return 1.0 - (rank / 7462.0)
        except Exception:
            return 0.5

    return 0.5


def draw_potential(hole_cards: list, community: list) -> float:
    """
    Estimate drawing potential 0-1.
    Detects flush draws (4 to a flush) and straight draws (open-ended).
    """
    if not community or not hole_cards:
        return 0.0

    all_cards = hole_cards + community
    suits = [c[1] for c in all_cards]
    ranks = sorted(set(RANKS.index(c[0]) for c in all_cards))

    score = 0.0

    # Flush draw: 4+ of same suit
    max_suited = max(suits.count(s) for s in "cdhs")
    if max_suited >= 4:
        score += 0.5
    if max_suited >= 5:
        score += 0.3  # made flush, still has draw value for better flush

    # Straight draw: 4 cards within a window of 5
    for i in range(len(ranks) - 3):
        if ranks[i + 3] - ranks[i] <= 4:
            score += 0.5
            break

    return min(1.0, score)


# ── Player state ──────────────────────────────────────────────────────────────
@dataclass
class Player:
    idx:         int
    stack:       float
    hole_cards:  list  = field(default_factory=list)
    current_bet: float = 0.0
    total_bet:   float = 0.0   # total committed this hand
    folded:      bool  = False
    all_in:      bool  = False
    is_human:    bool  = False

    @property
    def active(self) -> bool:
        return not self.folded and not self.all_in

    @property
    def can_act(self) -> bool:
        return not self.folded and not self.all_in and self.stack > 0


# ── Side pot helper ───────────────────────────────────────────────────────────
def compute_pots(players: list) -> list:
    """
    Compute side pots for all-in scenarios.
    Returns list of (pot_amount, eligible_player_indices).
    """
    commitments = sorted(set(p.total_bet for p in players if p.total_bet > 0))
    pots  = []
    prev  = 0.0
    for level in commitments:
        eligible = [p.idx for p in players if p.total_bet >= level and not p.folded]
        pot_amt  = sum(min(p.total_bet, level) - min(p.total_bet, prev) for p in players)
        if pot_amt > 0:
            pots.append((pot_amt, eligible))
        prev = level
    return pots


class _LazyObs:
    """Lazy observation dict — only builds obs vector on first access per player."""
    __slots__ = ('_env', '_cache')

    def __init__(self, env):
        self._env = env
        self._cache = {}

    def __getitem__(self, idx):
        if idx not in self._cache:
            self._cache[idx] = self._env._build_obs(idx)
        return self._cache[idx]

    def __contains__(self, idx):
        return 0 <= idx < self._env.num_players

    def get(self, idx, default=None):
        if idx in self:
            return self[idx]
        return default


# ── Main environment ──────────────────────────────────────────────────────────
class PokerEnv:
    """
    Multi-agent No-Limit Texas Hold'em environment.

    Usage (self-play):
        env = PokerEnv(num_players=6, starting_stack=200, big_blind=2)
        obs = env.reset()
        while True:
            player_idx = env.current_player
            legal      = env.legal_actions()
            action     = agent.act(obs[player_idx], legal)
            obs, rewards, done, info = env.step(action)
            if done:
                obs = env.reset()
    """

    def __init__(
        self,
        num_players:     int   = 6,
        starting_stack:  float = 200.0,
        small_blind:     float = 1.0,
        big_blind:       float = 2.0,
        render_mode:     str   = "none",   # "unicode" | "none"
        max_raises:      int   = 6,        # max raises per street (allows 4-bet+)
    ):
        self.num_players    = num_players
        self.starting_stack = starting_stack
        self.small_blind    = small_blind
        self.big_blind      = big_blind
        self.base_small_blind = small_blind
        self.base_big_blind   = big_blind
        self.render_mode    = render_mode
        self.max_raises     = max_raises
        # Stack bounds: rebuy below min, trim above max (prevents accumulation)
        self.min_stack      = big_blind * 10      # 10 BB — short-stack floor
        self.max_stack      = starting_stack * 3   # 300 BB — deep-stack ceiling
        self.tournament_len = 100                  # hard reset every N hands (new session)
        # Optional blind escalation (tournament-style)
        # List of (hand_threshold, sb, bb) — blinds increase at each threshold
        # Thresholds are relative to tournament start (0 = first hand of session)
        self.blind_schedule = None  # set externally to enable

        # Gym spaces
        if GYM_OK:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(NUM_ACTIONS)

        # State
        self.players:      list  = []
        self.community:    list  = []
        self.deck:         list  = []
        self.pot:          float = 0.0
        self.street_idx:   int   = 0
        self.current_player: int = 0
        self.dealer_idx:   int   = 0
        self.hand_num:     int   = 0
        self.episode:      int   = 0
        self.last_bet:     float = 0.0
        self.raises_this_street: int = 0
        self._action_log:  list  = []
        self._action_ids:  list  = []   # numeric action history for obs vector
        self._done:        bool  = False
        self._rewards:     dict  = {}

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self) -> dict:
        self.hand_num  += 1
        self.community  = []
        self.pot        = 0.0
        self.street_idx = 0
        self.last_bet   = self.big_blind
        self.raises_this_street = 0
        self._action_log = []
        self._action_ids = []
        self._done      = False
        self._rewards   = {i: 0.0 for i in range(self.num_players)}
        self._hs_cache  = {}  # (player_idx, street_idx) -> (hand_strength, draw_potential)
        self._showdown_info = {}  # filled by _showdown()

        # Rebuild deck
        self.deck = ALL_CARDS[:]
        random.shuffle(self.deck)

        # Reset players (or create them)
        new_tournament = (self.hand_num % self.tournament_len == 0)
        if not self.players or new_tournament:
            # New session — reset blinds to base level
            self.small_blind = self.base_small_blind
            self.big_blind   = self.base_big_blind
            self.players = [
                Player(idx=i, stack=self.starting_stack)
                for i in range(self.num_players)
            ]
        else:
            for p in self.players:
                p.hole_cards  = []
                p.current_bet = 0.0
                p.total_bet   = 0.0
                p.folded      = False
                p.all_in      = False
                # Rebuy if short-stacked, trim if deep-stacked.
                # Keeps stacks in a realistic range so the agent learns
                # both deep-stack and short-stack play without accumulation.
                if p.stack < self.min_stack:
                    p.stack = self.starting_stack
                elif p.stack > self.max_stack:
                    p.stack = self.max_stack

        # Blind escalation (tournament-style, optional)
        if self.blind_schedule is not None:
            hands_into_session = self.hand_num % self.tournament_len
            for threshold, sb, bb in self.blind_schedule:
                if hands_into_session >= threshold:
                    self.small_blind = sb
                    self.big_blind   = bb
            self.last_bet = self.big_blind

        # Deal hole cards
        for p in self.players:
            p.hole_cards = [self.deck.pop(), self.deck.pop()]

        # Rotate dealer
        self.dealer_idx = (self.dealer_idx + 1) % self.num_players

        # Post blinds
        sb_idx = (self.dealer_idx + 1) % self.num_players
        bb_idx = (self.dealer_idx + 2) % self.num_players
        self._post_blind(sb_idx, self.small_blind)
        self._post_blind(bb_idx, self.big_blind)

        # First to act preflop: UTG (dealer+3)
        self.current_player = (self.dealer_idx + 3) % self.num_players

        if self.render_mode == "unicode":
            print(self._render_unicode())

        return self._get_obs()

    def _post_blind(self, idx: int, amount: float):
        p      = self.players[idx]
        actual = min(amount, p.stack)
        p.stack      -= actual
        p.current_bet = actual
        p.total_bet   = actual
        self.pot      += actual
        if p.stack == 0:
            p.all_in = True

    # ── Step ─────────────────────────────────────────────────────────────────
    def step(self, action: int, raise_frac: float = 0.5) -> tuple:
        """
        Take one action for the current player.

        Args:
            action:     Action type (0-4: fold/check/call/raise/all-in)
            raise_frac: Continuous raise sizing in [0, 1] when action == RAISE.
                        0.0 = minimum raise, 1.0 = all-in.

        Returns (obs_dict, rewards_dict, done, info).
        """
        if self._done:
            raise RuntimeError("Call reset() before stepping after done=True")

        p = self.players[self.current_player]
        legal = self.legal_actions()

        # Map illegal actions to the best legal fallback
        if action not in legal:
            for fallback in [Action.CHECK, Action.CALL, Action.FOLD]:
                if fallback in legal:
                    action = fallback
                    break

        call_amt = self._call_amount(self.current_player)
        action   = Action(action)
        amt      = 0.0

        if action == Action.FOLD:
            p.folded = True
            self._log(f"{self._pname(p)}: fold")

        elif action == Action.CHECK:
            self._log(f"{self._pname(p)}: check")

        elif action == Action.CALL:
            amt = min(call_amt, p.stack)
            p.stack      -= amt
            p.current_bet += amt
            p.total_bet   += amt
            self.pot      += amt
            if p.stack == 0:
                p.all_in = True
            self._log(f"{self._pname(p)}: call ${amt:.1f}")

        elif action == Action.RAISE:
            # Continuous sizing with exponential curve
            min_raise  = max(self.last_bet * 2, call_amt + self.big_blind)
            max_raise  = p.stack + p.current_bet
            raise_to   = raise_frac_to_amount(raise_frac, min_raise, max_raise)
            actual     = raise_to - p.current_bet
            actual     = min(actual, p.stack)
            p.stack      -= actual
            p.current_bet += actual
            p.total_bet   += actual
            self.pot      += actual
            self.last_bet  = raise_to
            self.raises_this_street += 1
            if p.stack == 0:
                p.all_in = True
            self._log(f"{self._pname(p)}: raise ${actual:.0f} to ${raise_to:.0f}")

        elif action == Action.ALL_IN:
            amt = p.stack
            p.stack       = 0
            p.current_bet += amt
            p.total_bet   += amt
            self.pot      += amt
            p.all_in      = True
            self.last_bet  = max(self.last_bet, p.current_bet)
            self._log(f"{self._pname(p)}: all-in ${p.total_bet:.0f}")

        # Record action id for obs vector (fixes _action_ids bug)
        self._action_ids.append(int(action))

        # Track action metadata for opponent modeling
        action_player = self.current_player
        action_taken = int(action)
        action_raise_frac = raise_frac if action == Action.RAISE else 0.0

        # Advance to next player
        self._advance()

        obs  = self._get_obs()
        done = self._done
        info = {
            "street": STREETS[self.street_idx],
            "pot": self.pot,
            "hand_num": self.hand_num,
            "showdown": self._showdown_info.get("occurred", False) if done else False,
            "showdown_players": self._showdown_info.get("players", []) if done else [],
            "revealed_cards": self._showdown_info.get("cards", {}) if done else {},
            "action_player": action_player,
            "action_taken": action_taken,
            "raise_frac": action_raise_frac,
        }

        if self.render_mode == "unicode":
            print(self._render_unicode())

        return obs, dict(self._rewards), done, info

    # ── Action advancement ────────────────────────────────────────────────────
    def _advance(self):
        """Move to the next player or next street."""
        active = [p for p in self.players if not p.folded]

        # Everyone but one folded -> end hand
        if len(active) == 1:
            self._end_hand(active)
            return

        # Check if betting round is complete
        can_act = [p for p in active if p.can_act]
        max_bet = max((p.current_bet for p in active), default=0)
        all_matched = all(p.current_bet == max_bet or not p.can_act for p in active)

        if all_matched and not can_act:
            # All in — run out the board
            while self.street_idx < 3:
                self._deal_next_street()
            self._showdown()
            return

        if all_matched and can_act:
            # Street complete
            if self.street_idx < 3:
                self._deal_next_street()
                return
            else:
                self._showdown()
                return

        # Find next active player
        nxt = (self.current_player + 1) % self.num_players
        while self.players[nxt].folded or not self.players[nxt].can_act:
            nxt = (nxt + 1) % self.num_players
            if nxt == self.current_player:
                break
        self.current_player = nxt

    def _deal_next_street(self):
        """Deal community cards and reset betting for new street."""
        self.street_idx += 1
        self.raises_this_street = 0
        for p in self.players:
            p.current_bet = 0.0
        self.last_bet = self.big_blind

        if self.street_idx == 1:    # flop
            self.community = [self.deck.pop() for _ in range(3)]
        elif self.street_idx in (2, 3):  # turn, river
            self.community.append(self.deck.pop())

        # First to act postflop: first active left of dealer
        nxt = (self.dealer_idx + 1) % self.num_players
        start = nxt
        while self.players[nxt].folded or self.players[nxt].all_in:
            nxt = (nxt + 1) % self.num_players
            if nxt == start:
                # Everyone is all-in or folded — no one to act
                break
        self.current_player = nxt

        if self.render_mode == "unicode":
            line = "\u2500" * 40
            print(f"\n  {line}")
            print(f"  {STREETS[self.street_idx].upper()}: {' '.join(self.community)}")

    # ── Showdown + hand resolution ────────────────────────────────────────────
    def _showdown(self):
        active  = [p for p in self.players if not p.folded]
        pots    = compute_pots(self.players)

        # Record showdown info for opponent modeling
        self._showdown_info = {
            "occurred": True,
            "players": [p.idx for p in active],
            "cards": {p.idx: list(p.hole_cards) for p in active},
        }

        if self.render_mode == "unicode":
            print("\n  SHOWDOWN")
            for p in active:
                print(f"  {self._pname(p)}: {' '.join(p.hole_cards)}")

        total_awarded = 0.0
        for pot_amt, eligible_idx in pots:
            eligible = [p for p in active if p.idx in eligible_idx]
            if not eligible:
                continue

            if len(eligible) == 1:
                winner = eligible[0]
                winner.stack   += pot_amt
                self._rewards[winner.idx] += pot_amt
                total_awarded  += pot_amt
            else:
                # Evaluate hands
                if TREYS_OK and self.community:
                    board_treys = [Card.new(c[0]+c[1]) for c in self.community]
                    scores = {}
                    for p in eligible:
                        hand_treys = [Card.new(c[0]+c[1]) for c in p.hole_cards]
                        try:
                            scores[p.idx] = _evaluator.evaluate(board_treys, hand_treys)
                        except Exception:
                            scores[p.idx] = 9999
                    best = min(scores.values())
                    winners = [p for p in eligible if scores[p.idx] == best]
                else:
                    winners = eligible  # fallback: chop

                share = pot_amt / len(winners)
                for w in winners:
                    w.stack           += share
                    self._rewards[w.idx] += share
                    total_awarded     += share

        # Normalize rewards by big blind (standard poker metric)
        for i in range(self.num_players):
            committed = self.players[i].total_bet
            self._rewards[i] = (self._rewards[i] - committed) / self.big_blind

        self._done = True

    def _end_hand(self, active: list):
        """One player wins uncontested."""
        winner = active[0]
        winner.stack           += self.pot
        self._rewards[winner.idx] += self.pot
        for i in range(self.num_players):
            self._rewards[i] = (self._rewards[i] - self.players[i].total_bet) / self.big_blind
        self._done = True

    # ── Legal actions ─────────────────────────────────────────────────────────
    def legal_actions(self) -> list:
        p        = self.players[self.current_player]
        call_amt = self._call_amount(self.current_player)
        legal    = []

        if p.stack == 0 or p.folded:
            return []

        # Fold always available (unless check is free)
        if call_amt > 0:
            legal.append(Action.FOLD)

        # Check: free if no bet to call
        if call_amt == 0:
            legal.append(Action.CHECK)
        else:
            legal.append(Action.CALL)

        # Raise: available if under raise cap and have chips for min raise
        if self.raises_this_street < self.max_raises:
            min_raise_amt = self.last_bet * 2
            if p.stack + p.current_bet > min_raise_amt:
                legal.append(Action.RAISE)

        # All-in always available if have chips
        if p.stack > 0:
            legal.append(Action.ALL_IN)

        return [int(a) for a in legal]

    def legal_mask(self) -> np.ndarray:
        """Returns a boolean mask over NUM_ACTIONS for the current player."""
        mask  = np.zeros(NUM_ACTIONS, dtype=bool)
        legal = self.legal_actions()
        for a in legal:
            mask[a] = True
        return mask

    # ── Observation ───────────────────────────────────────────────────────────
    def _get_obs(self) -> '_LazyObs':
        """Returns a lazy observation dict: obs[i] builds on first access."""
        return _LazyObs(self)

    def _build_obs(self, player_idx: int) -> np.ndarray:
        p = self.players[player_idx]
        v = np.zeros(OBS_DIM, dtype=np.float32)

        # [0:52] own hole cards
        v[0:52] = cards_to_onehot(p.hole_cards)

        # [52:104] community cards
        v[52:104] = cards_to_onehot(self.community)

        # Player slots padded to MAX_PLAYERS=9 (empty seats stay zero)
        max_stack = self.max_stack

        # [104:113] normalized stacks
        for i, pl in enumerate(self.players):
            v[104+i] = pl.stack / max_stack

        # [113:122] normalized current bets
        for i, pl in enumerate(self.players):
            v[113+i] = pl.current_bet / max_stack

        # [122:131] active flags
        for i, pl in enumerate(self.players):
            v[122+i] = float(not pl.folded and not pl.all_in)

        # [131:140] positions (dealer=0, normalized)
        for i in range(self.num_players):
            pos = (i - self.dealer_idx) % self.num_players
            v[131+i] = pos / self.num_players

        # [140:144] game scalars
        v[140] = self.pot / max_stack
        v[141] = self._call_amount(player_idx) / max_stack
        v[142] = self.last_bet / max_stack
        v[143] = sum(1 for pl in self.players if not pl.folded) / MAX_PLAYERS

        # [144:148] street one-hot
        v[144 + self.street_idx] = 1.0

        # [148:155] last 7 actions (action type normalized within 7 slots)
        recent = self._action_ids[-7:]
        for j, a in enumerate(recent):
            v[148 + j] = a / NUM_ACTIONS

        # [155-156] hand strength + draw potential (cached per player per street)
        cache_key = (player_idx, self.street_idx)
        if cache_key not in self._hs_cache:
            self._hs_cache[cache_key] = (
                hand_strength(p.hole_cards, self.community),
                draw_potential(p.hole_cards, self.community),
            )
        v[155], v[156] = self._hs_cache[cache_key]

        # [157] raises this street (normalized by max_raises)
        v[157] = self.raises_this_street / max(self.max_raises, 1)

        # [158] facing a raise (1.0 if call_amount > 0 and there's been a raise)
        call_amt = self._call_amount(player_idx)
        v[158] = float(call_amt > 0 and self.raises_this_street > 0)

        # [159] pot odds: call / (pot + call) — key for raise/call/fold decisions
        if call_amt > 0:
            v[159] = call_amt / (self.pot + call_amt)
        else:
            v[159] = 0.0

        # Safety clamp — prevents numerical overflow in the network
        np.clip(v, -10.0, 10.0, out=v)

        return v

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _call_amount(self, player_idx: int) -> float:
        p       = self.players[player_idx]
        max_bet = max((pl.current_bet for pl in self.players), default=0)
        return max(0.0, max_bet - p.current_bet)

    def _pname(self, p: Player) -> str:
        return f"P{p.idx}"

    def _log(self, msg: str):
        self._action_log.append(msg)

    # ── Unicode render ────────────────────────────────────────────────────────
    def _render_unicode(self) -> str:
        hero = self.players[0]
        hero_eq = hand_strength(hero.hole_cards, self.community) if hero.hole_cards else -1
        return render_table({
            "episode":          self.episode,
            "hand_num":         self.hand_num,
            "street":           STREETS[self.street_idx],
            "community_cards":  self.community,
            "pot":              self.pot,
            "current_player":   self.current_player,
            "dealer_idx":       self.dealer_idx,
            "hero_equity":      hero_eq,
            "showdown":         self._done,
            "players": [
                {
                    "name":        f"P{p.idx}",
                    "stack":       p.stack,
                    "current_bet": p.current_bet,
                    "folded":      p.folded,
                    "hole_cards":  p.hole_cards,
                    "is_hero":     (p.idx == 0),
                }
                for p in self.players
            ],
            "last_actions": self._action_log,
        })

    def clone(self) -> 'PokerEnv':
        """Create a lightweight copy for MCTS simulation."""
        c = PokerEnv.__new__(PokerEnv)
        c.num_players    = self.num_players
        c.starting_stack = self.starting_stack
        c.small_blind    = self.small_blind
        c.big_blind      = self.big_blind
        c.base_small_blind = self.base_small_blind
        c.base_big_blind   = self.base_big_blind
        c.render_mode    = "none"
        c.max_raises     = self.max_raises
        c.min_stack      = self.min_stack
        c.max_stack      = self.max_stack
        c.tournament_len = self.tournament_len
        c.blind_schedule = self.blind_schedule
        c.players        = [copy.copy(p) for p in self.players]
        for p in c.players:
            p.hole_cards = list(p.hole_cards)
        c.community      = list(self.community)
        c.deck           = list(self.deck)
        c.pot            = self.pot
        c.street_idx     = self.street_idx
        c.current_player = self.current_player
        c.dealer_idx     = self.dealer_idx
        c.hand_num       = self.hand_num
        c.episode        = self.episode
        c.last_bet       = self.last_bet
        c.raises_this_street = self.raises_this_street
        c._action_log    = list(self._action_log)
        c._action_ids    = list(self._action_ids)
        c._done          = self._done
        c._rewards       = dict(self._rewards)
        c._hs_cache      = dict(self._hs_cache)
        c._showdown_info = dict(self._showdown_info)
        return c

    def render(self):
        print(self._render_unicode())
