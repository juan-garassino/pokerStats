"""
No-Limit Texas Hold'em environment
────────────────────────────────────
OpenAI Gym-compatible multi-agent poker environment.

Design choices:
  - Observation:  143-dim vector (hand + board + betting history + position)
  - Action space: Discrete(7) — fold, check, call, raise_25%, raise_50%,
                               raise_100% (pot), all_in
  - Reward:       BB-normalized (chips won / big_blind)
  - Info asymmetry: each agent only sees its own hole cards
  - Supports 2-6 players
"""

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
class Action(IntEnum):
    FOLD       = 0
    CHECK      = 1   # only valid when no bet to call
    CALL       = 2
    RAISE_25   = 3   # 25% of pot
    RAISE_50   = 4   # 50% of pot
    RAISE_100  = 5   # 100% of pot (pot-sized bet)
    ALL_IN     = 6

NUM_ACTIONS = len(Action)

# Raise multipliers (fraction of pot)
RAISE_FRACS = {
    Action.RAISE_25:  0.25,
    Action.RAISE_50:  0.50,
    Action.RAISE_100: 1.00,
}

# ── Observation space layout ──────────────────────────────────────────────────
# [0:52]   hole cards (one-hot, 52 bits)
# [52:104] community cards (one-hot, 52 bits)
# [104:110] per-player stack normalized (6 players)
# [110:116] per-player current bet normalized
# [116:122] per-player active flags
# [122:128] per-player position (dealer=0,SB=1,BB=2... normalized)
# [128:132] pot, call_amount, min_raise, num_active (normalized)
# [132:136] street one-hot [preflop, flop, turn, river]
# [136:143] last 7 actions one-hot encoded
OBS_DIM = 143

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
        max_raises:      int   = 4,        # max raises per street
    ):
        self.num_players    = num_players
        self.starting_stack = starting_stack
        self.small_blind    = small_blind
        self.big_blind      = big_blind
        self.render_mode    = render_mode
        self.max_raises     = max_raises

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

        # Rebuild deck
        self.deck = ALL_CARDS[:]
        random.shuffle(self.deck)

        # Reset players (or create them)
        if not self.players:
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
                # Rebuy if busted
                if p.stack < self.big_blind:
                    p.stack = self.starting_stack

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
    def step(self, action: int) -> tuple:
        """
        Take one action for the current player.
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

        elif action in RAISE_FRACS:
            frac     = RAISE_FRACS[action]
            raise_to = call_amt + self.pot * frac
            raise_to = max(raise_to, self.last_bet * 2)
            raise_to = min(raise_to, p.stack + p.current_bet)
            actual   = raise_to - p.current_bet
            actual   = min(actual, p.stack)
            p.stack      -= actual
            p.current_bet += actual
            p.total_bet   += actual
            self.pot      += actual
            self.last_bet  = raise_to
            self.raises_this_street += 1
            if p.stack == 0:
                p.all_in = True
            self._log(f"{self._pname(p)}: raise to ${raise_to:.1f}")

        elif action == Action.ALL_IN:
            amt = p.stack
            p.stack       = 0
            p.current_bet += amt
            p.total_bet   += amt
            self.pot      += amt
            p.all_in      = True
            self.last_bet  = max(self.last_bet, p.current_bet)
            self._log(f"{self._pname(p)}: all-in ${p.total_bet:.1f}")

        # Record action id for obs vector (fixes _action_ids bug)
        self._action_ids.append(int(action))

        # Advance to next player
        self._advance()

        obs  = self._get_obs()
        done = self._done
        info = {"street": STREETS[self.street_idx], "pot": self.pot, "hand_num": self.hand_num}

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
        while self.players[nxt].folded or self.players[nxt].all_in:
            nxt = (nxt + 1) % self.num_players
        self.current_player = nxt

        if self.render_mode == "unicode":
            line = "\u2500" * 40
            print(f"\n  {line}")
            print(f"  {STREETS[self.street_idx].upper()}: {' '.join(self.community)}")

    # ── Showdown + hand resolution ────────────────────────────────────────────
    def _showdown(self):
        active  = [p for p in self.players if not p.folded]
        pots    = compute_pots(self.players)

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

        # Raises: available if under raise cap and have chips
        if self.raises_this_street < self.max_raises:
            for act in [Action.RAISE_25, Action.RAISE_50, Action.RAISE_100]:
                min_raise_amt = self.last_bet * 2
                if p.stack + p.current_bet > min_raise_amt:
                    legal.append(act)

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
    def _get_obs(self) -> dict:
        """Returns an observation dict: {player_idx: obs_vector}."""
        obs = {}
        for i, p in enumerate(self.players):
            obs[i] = self._build_obs(i)
        return obs

    def _build_obs(self, player_idx: int) -> np.ndarray:
        p = self.players[player_idx]
        v = np.zeros(OBS_DIM, dtype=np.float32)

        # [0:52] own hole cards
        v[0:52] = cards_to_onehot(p.hole_cards)

        # [52:104] community cards
        v[52:104] = cards_to_onehot(self.community)

        # [104:110] normalized stacks
        max_stack = self.starting_stack * 2
        for i, pl in enumerate(self.players):
            v[104+i] = pl.stack / max_stack

        # [110:116] normalized current bets
        for i, pl in enumerate(self.players):
            v[110+i] = pl.current_bet / max_stack

        # [116:122] active flags
        for i, pl in enumerate(self.players):
            v[116+i] = float(not pl.folded and not pl.all_in)

        # [122:128] positions (dealer=0, normalized)
        for i in range(self.num_players):
            pos = (i - self.dealer_idx) % self.num_players
            v[122+i] = pos / self.num_players

        # [128:132] game scalars
        v[128] = self.pot / max_stack
        v[129] = self._call_amount(player_idx) / max_stack
        v[130] = self.last_bet / max_stack
        v[131] = sum(1 for pl in self.players if not pl.folded) / self.num_players

        # [132:136] street one-hot
        v[132 + self.street_idx] = 1.0

        # [136:143] last 7 actions (action type normalized within 7 slots)
        recent = self._action_ids[-7:]
        for j, a in enumerate(recent):
            v[136 + j] = a / NUM_ACTIONS

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
        return render_table({
            "episode":          self.episode,
            "hand_num":         self.hand_num,
            "street":           STREETS[self.street_idx],
            "community_cards":  self.community,
            "pot":              self.pot,
            "current_player":   self.current_player,
            "showdown":         self._done,
            "players": [
                {
                    "name":        f"Agent-{p.idx}",
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

    def render(self):
        print(self._render_unicode())
