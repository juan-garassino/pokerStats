"""Tests for pokerStats.rl.renderer"""
from pokerStats.rl.renderer import card_unicode, render_hand, render_table, render_training_stats


def test_card_unicode_mapping():
    """All 52 cards map to a valid Unicode character."""
    ranks = "23456789TJQKA"
    suits = "cdhs"
    for r in ranks:
        for s in suits:
            card = f"{r}{s}"
            glyph = card_unicode(card)
            assert glyph != "?", f"Card {card} has no Unicode mapping"
            assert len(glyph) == 1, f"Card {card} should map to single char, got {glyph!r}"


def test_card_unicode_specific():
    """Spot-check specific card mappings."""
    assert card_unicode("As") == "\U0001F0A1"  # Ace of spades
    assert card_unicode("Kh") == "\U0001F0BE"  # King of hearts
    assert card_unicode("2c") == "\U0001F0D2"  # 2 of clubs


def test_card_unicode_unknown():
    """Unknown card strings return '?'."""
    assert card_unicode("Xx") == "?"


def test_render_hand_basic():
    """render_hand returns a non-empty string."""
    result = render_hand(["Ah", "Ks"], label="Hero")
    assert "Hero" in result
    assert len(result) > 0


def test_render_hand_empty():
    """render_hand with empty list returns descriptive string."""
    result = render_hand([], label="Hero")
    assert "none" in result


def test_render_hand_hidden():
    """render_hand with hidden=True uses card backs."""
    result = render_hand(["Ah", "Ks"], hidden=True)
    assert "\U0001F0A0" in result


def test_render_table_output():
    """render_table produces complete table output."""
    state = {
        "episode": 1,
        "hand_num": 1,
        "street": "flop",
        "community_cards": ["Ah", "7c", "2d"],
        "pot": 10.0,
        "current_player": 0,
        "showdown": False,
        "players": [
            {"name": "Hero", "stack": 190.0, "current_bet": 5.0,
             "folded": False, "hole_cards": ["Kh", "Qh"], "is_hero": True},
            {"name": "Villain", "stack": 195.0, "current_bet": 5.0,
             "folded": False, "hole_cards": ["Tc", "9c"], "is_hero": False},
        ],
        "last_actions": ["Hero raise $5", "Villain call $5"],
    }
    result = render_table(state)
    assert "FLOP" in result
    assert "Pot" in result
    assert "Hero" in result


def test_render_training_stats():
    """render_training_stats produces a two-line string with key metrics."""
    stats = {
        "episode": 100, "step": 5000, "mean_reward": 0.5,
        "win_pct": 0.5, "bb_per_100": 2.0, "elo": 1500,
        "policy_loss": 0.01, "value_loss": 0.05, "entropy": 1.5,
        "hands_per_sec": 12.3, "phase_pct": 45.0, "ppo_updates": 3,
    }
    result = render_training_stats(stats)
    assert "hand" in result
    assert "elo" in result
    assert "ploss" in result
    assert "hands/s" in result
