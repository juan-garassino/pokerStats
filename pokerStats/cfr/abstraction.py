"""
Hand abstraction for CFR
────────────────────────
Reduces the game tree by bucketing similar hands together.

Preflop:  1326 combos → 169 canonical groups (AA, AKs, AKo, ..., 22)
Postflop: Equity histogram clustering via k-means (~200 buckets per street)
"""

import random
import numpy as np
from itertools import combinations
from pathlib import Path

from pokerStats.rl.poker_env import RANKS, SUITS, ALL_CARDS, CARD_IDX, hand_strength

try:
    from treys import Card, Evaluator
    TREYS_OK = True
    _evaluator = Evaluator()
except ImportError:
    TREYS_OK = False


# ── Preflop abstraction ─────────────────────────────────────────────────────

# 169 canonical preflop hands: pairs (13), suited (78), offsuit (78)
# Ordered: AA=0, KK=1, ..., 22=12, AKs=13, AQs=14, ..., 32s=90, AKo=91, ...

def _canonical_key(card1: str, card2: str) -> tuple:
    """Return (high_rank_idx, low_rank_idx, suited_flag) for a hole card pair."""
    r0 = RANKS.index(card1[0])
    r1 = RANKS.index(card2[0])
    suited = card1[1] == card2[1]
    high, low = max(r0, r1), min(r0, r1)
    return (high, low, suited)


def _build_preflop_lut() -> dict:
    """Build canonical key → bucket_id lookup table for all 169 groups."""
    lut = {}
    bucket_id = 0

    # Pairs: AA(12,12) down to 22(0,0)
    for r in range(12, -1, -1):
        lut[(r, r, False)] = bucket_id
        lut[(r, r, True)] = bucket_id  # pairs: suited flag irrelevant
        bucket_id += 1

    # Suited non-pairs: AKs, AQs, ..., 32s
    for high in range(12, 0, -1):
        for low in range(high - 1, -1, -1):
            lut[(high, low, True)] = bucket_id
            bucket_id += 1

    # Offsuit non-pairs: AKo, AQo, ..., 32o
    for high in range(12, 0, -1):
        for low in range(high - 1, -1, -1):
            lut[(high, low, False)] = bucket_id
            bucket_id += 1

    return lut


class PreflopAbstraction:
    """Maps 1326 hole card combos → 169 canonical preflop groups."""

    NUM_BUCKETS = 169

    def __init__(self):
        self._lut = _build_preflop_lut()

    def get_bucket(self, hole_cards: list) -> int:
        """
        Get preflop bucket for a pair of hole cards.

        Args:
            hole_cards: ["Ah", "Ks"] — order doesn't matter.

        Returns:
            Bucket ID in [0, 168].
        """
        key = _canonical_key(hole_cards[0], hole_cards[1])
        return self._lut[key]

    def all_combos_for_bucket(self, bucket_id: int) -> list:
        """Return all (card1, card2) combos that map to a given bucket."""
        combos = []
        for c1, c2 in combinations(ALL_CARDS, 2):
            key = _canonical_key(c1, c2)
            if self._lut.get(key) == bucket_id:
                combos.append((c1, c2))
        return combos


# ── Postflop abstraction ────────────────────────────────────────────────────

def compute_equity_histogram(
    hole_cards: list,
    community: list,
    n_bins: int = 50,
    n_samples: int = 300,
) -> np.ndarray:
    """
    Build an equity distribution histogram by sampling opponent hands.

    For each sampled opponent hand, compute hero's equity against it.
    The histogram captures the *shape* of equity — not just the mean.
    Hands with similar histograms play similarly.

    Returns:
        np.ndarray of shape (n_bins,), normalized to sum to 1.
    """
    if not TREYS_OK:
        # Fallback: single-bin based on hand_strength
        hist = np.zeros(n_bins, dtype=np.float32)
        hs = hand_strength(hole_cards, community)
        bin_idx = min(int(hs * n_bins), n_bins - 1)
        hist[bin_idx] = 1.0
        return hist

    used = set(hole_cards + community)
    available = [c for c in ALL_CARDS if c not in used]

    hero_treys = [Card.new(c[0] + c[1]) for c in hole_cards]
    board_treys = [Card.new(c[0] + c[1]) for c in community]

    equities = []
    for _ in range(n_samples):
        random.shuffle(available)

        # Complete the board if needed
        remaining_board = list(board_treys)
        draw_idx = 0
        while len(remaining_board) < 5:
            remaining_board.append(Card.new(available[draw_idx][0] + available[draw_idx][1]))
            draw_idx += 1

        # Deal opponent hand
        if draw_idx + 1 >= len(available):
            continue
        opp_cards = [
            Card.new(available[draw_idx][0] + available[draw_idx][1]),
            Card.new(available[draw_idx + 1][0] + available[draw_idx + 1][1]),
        ]

        try:
            hero_rank = _evaluator.evaluate(remaining_board, hero_treys)
            opp_rank = _evaluator.evaluate(remaining_board, opp_cards)
            # Lower rank = better in treys
            if hero_rank < opp_rank:
                equities.append(1.0)
            elif hero_rank == opp_rank:
                equities.append(0.5)
            else:
                equities.append(0.0)
        except Exception:
            continue

    if not equities:
        hist = np.zeros(n_bins, dtype=np.float32)
        hist[n_bins // 2] = 1.0
        return hist

    hist, _ = np.histogram(equities, bins=n_bins, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


class PostflopAbstraction:
    """
    Cluster postflop hands into buckets using equity histogram k-means.

    Each street gets its own set of centroids. At inference, a hand is
    assigned to the nearest centroid (L2 distance on histograms).
    """

    def __init__(self, n_buckets: int = 200, n_bins: int = 50):
        self.n_buckets = n_buckets
        self.n_bins = n_bins
        # street → (n_buckets, n_bins) centroid array
        self.centroids: dict = {}

    def train_buckets(
        self,
        street: str,
        n_samples: int = 10000,
        n_kmeans_iters: int = 30,
    ):
        """
        Train equity-histogram k-means for a given street.

        Generates random (hole_cards, community) combos for the street,
        computes equity histograms, then runs k-means.
        """
        community_sizes = {"flop": 3, "turn": 4, "river": 5}
        n_community = community_sizes.get(street, 3)

        histograms = []
        for _ in range(n_samples):
            deck = ALL_CARDS[:]
            random.shuffle(deck)
            hole = [deck.pop(), deck.pop()]
            community = [deck.pop() for _ in range(n_community)]
            hist = compute_equity_histogram(hole, community, self.n_bins, n_samples=100)
            histograms.append(hist)

        data = np.array(histograms, dtype=np.float32)  # (n_samples, n_bins)
        centroids = self._kmeans(data, self.n_buckets, n_kmeans_iters)
        self.centroids[street] = centroids

    def _kmeans(self, data: np.ndarray, k: int, n_iters: int) -> np.ndarray:
        """Simple k-means clustering on equity histograms."""
        n = data.shape[0]
        # Initialize centroids with k-means++
        centroids = np.zeros((k, data.shape[1]), dtype=np.float32)
        centroids[0] = data[random.randint(0, n - 1)]

        for i in range(1, k):
            dists = np.min(
                np.sum((data[:, None, :] - centroids[None, :i, :]) ** 2, axis=2),
                axis=1,
            )
            probs = dists / (dists.sum() + 1e-10)
            centroids[i] = data[np.random.choice(n, p=probs)]

        # Iterate
        for _ in range(n_iters):
            # Assign
            dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)

            # Update
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centroids[j] = data[mask].mean(axis=0)

        return centroids

    def get_bucket(self, hole_cards: list, community: list) -> int:
        """Assign a postflop hand to the nearest centroid bucket."""
        if len(community) <= 2:
            street = "flop"
        elif len(community) == 4:
            street = "turn"
        else:
            street = "river"

        if street not in self.centroids:
            # Untrained: fall back to raw equity bin
            hs = hand_strength(hole_cards, community)
            return min(int(hs * self.n_buckets), self.n_buckets - 1)

        hist = compute_equity_histogram(hole_cards, community, self.n_bins, n_samples=100)
        dists = np.sum((self.centroids[street] - hist[None, :]) ** 2, axis=1)
        return int(np.argmin(dists))


# ── Facade ───────────────────────────────────────────────────────────────────

class HandAbstraction:
    """
    Combined preflop + postflop hand abstraction.

    Single entry point: get_bucket(hole_cards, community) → int
    """

    def __init__(self, n_postflop_buckets: int = 200, n_bins: int = 50):
        self.preflop = PreflopAbstraction()
        self.postflop = PostflopAbstraction(n_buckets=n_postflop_buckets, n_bins=n_bins)

    def get_bucket(self, hole_cards: list, community: list) -> int:
        """
        Get the abstraction bucket for a hand.

        Preflop: returns bucket in [0, 168]
        Postflop: returns 169 + postflop_bucket (offset so IDs don't collide)
        """
        if not community:
            return self.preflop.get_bucket(hole_cards)
        return PreflopAbstraction.NUM_BUCKETS + self.postflop.get_bucket(hole_cards, community)

    def save(self, path: str):
        """Save abstraction to compressed numpy archive."""
        save_dict = {
            "preflop_lut_keys": np.array(list(self.preflop._lut.keys())),
            "preflop_lut_vals": np.array(list(self.preflop._lut.values())),
            "n_postflop_buckets": np.array(self.postflop.n_buckets),
            "n_bins": np.array(self.postflop.n_bins),
        }
        for street, centroids in self.postflop.centroids.items():
            save_dict[f"centroids_{street}"] = centroids
        np.savez_compressed(path, **save_dict)

    def load(self, path: str):
        """Load abstraction from compressed numpy archive."""
        data = np.load(path, allow_pickle=True)
        self.postflop.n_buckets = int(data["n_postflop_buckets"])
        self.postflop.n_bins = int(data["n_bins"])
        for key in data.files:
            if key.startswith("centroids_"):
                street = key[len("centroids_"):]
                self.postflop.centroids[street] = data[key]
