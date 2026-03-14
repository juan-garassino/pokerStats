"""Tests for pokerStats.cfr.abstraction"""
import numpy as np
from itertools import combinations
from pokerStats.cfr.abstraction import (
    PreflopAbstraction, PostflopAbstraction, HandAbstraction,
    compute_equity_histogram, _canonical_key,
)
from pokerStats.rl.poker_env import ALL_CARDS, RANKS, SUITS


def test_preflop_169_groups():
    """All 1326 hole card combos map to exactly 169 canonical groups."""
    pa = PreflopAbstraction()
    buckets = set()
    for c1, c2 in combinations(ALL_CARDS, 2):
        bucket = pa.get_bucket([c1, c2])
        buckets.add(bucket)
        assert 0 <= bucket < 169, f"Bucket {bucket} out of range for {c1}, {c2}"
    assert len(buckets) == 169, f"Expected 169 groups, got {len(buckets)}"


def test_preflop_symmetry():
    """AhKs and KsAh map to the same bucket."""
    pa = PreflopAbstraction()
    assert pa.get_bucket(["Ah", "Ks"]) == pa.get_bucket(["Ks", "Ah"])
    assert pa.get_bucket(["2c", "7d"]) == pa.get_bucket(["7d", "2c"])
    assert pa.get_bucket(["Td", "Td"]) == pa.get_bucket(["Td", "Td"])


def test_preflop_pairs_same_bucket():
    """All suited combos of the same pair map to the same bucket (e.g., AhAs == AcAd)."""
    pa = PreflopAbstraction()
    # All AA combos should be the same bucket
    aa_buckets = set()
    for s1 in SUITS:
        for s2 in SUITS:
            if s1 >= s2:
                continue
            bucket = pa.get_bucket([f"A{s1}", f"A{s2}"])
            aa_buckets.add(bucket)
    assert len(aa_buckets) == 1, f"AA should be one bucket, got {aa_buckets}"


def test_preflop_suited_vs_offsuit():
    """AKs and AKo are different buckets."""
    pa = PreflopAbstraction()
    aks = pa.get_bucket(["Ah", "Kh"])  # suited
    ako = pa.get_bucket(["Ah", "Kc"])  # offsuit
    assert aks != ako, "Suited and offsuit should be different buckets"


def test_canonical_key_consistency():
    """_canonical_key produces consistent (high, low, suited) tuples."""
    key1 = _canonical_key("Ah", "Ks")
    key2 = _canonical_key("Ks", "Ah")
    assert key1 == key2


def test_equity_histogram_shape():
    """Equity histogram has correct shape and sums to ~1."""
    hist = compute_equity_histogram(
        ["Ah", "Kh"], ["Qh", "Jh", "2c"],
        n_bins=50, n_samples=100,
    )
    assert hist.shape == (50,)
    assert abs(hist.sum() - 1.0) < 0.01, f"Histogram should sum to 1, got {hist.sum()}"


def test_equity_histogram_strong_hand():
    """Strong hands should have equity concentrated in upper bins."""
    hist = compute_equity_histogram(
        ["Ah", "Kh"], ["Qh", "Jh", "Th"],  # Royal flush
        n_bins=10, n_samples=200,
    )
    # Most equity should be in the top bins
    top_half = hist[5:].sum()
    assert top_half > 0.5, f"Royal flush should have high equity, top_half={top_half}"


def test_postflop_bucket_valid():
    """Postflop bucketing returns valid bucket IDs."""
    pa = PostflopAbstraction(n_buckets=50, n_bins=10)
    # Without training, falls back to hand_strength binning
    bucket = pa.get_bucket(["Ah", "Kh"], ["Qh", "Jh", "2c"])
    assert 0 <= bucket < 50


def test_hand_abstraction_preflop_range():
    """HandAbstraction preflop returns [0, 168]."""
    ha = HandAbstraction()
    bucket = ha.get_bucket(["Ah", "Ks"], [])
    assert 0 <= bucket < 169


def test_hand_abstraction_postflop_offset():
    """HandAbstraction postflop returns ≥ 169 (offset from preflop)."""
    ha = HandAbstraction(n_postflop_buckets=50)
    bucket = ha.get_bucket(["Ah", "Ks"], ["2c", "7d", "Jh"])
    assert bucket >= 169, f"Postflop bucket should be >= 169, got {bucket}"


def test_hand_abstraction_save_load(tmp_path):
    """Save and load roundtrip preserves abstraction."""
    ha = HandAbstraction(n_postflop_buckets=20, n_bins=10)

    # Get a bucket before saving
    pre_bucket = ha.get_bucket(["Ah", "Ks"], [])

    path = str(tmp_path / "abstraction.npz")
    ha.save(path)

    ha2 = HandAbstraction()
    ha2.load(path)

    pre_bucket2 = ha2.get_bucket(["Ah", "Ks"], [])
    assert pre_bucket == pre_bucket2


def test_all_combos_for_bucket():
    """all_combos_for_bucket returns the right set of combos."""
    pa = PreflopAbstraction()
    # AA bucket
    aa_bucket = pa.get_bucket(["Ah", "As"])
    combos = pa.all_combos_for_bucket(aa_bucket)
    # C(4,2) = 6 combos of AA
    assert len(combos) == 6, f"AA should have 6 combos, got {len(combos)}"
