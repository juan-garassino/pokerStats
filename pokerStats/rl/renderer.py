"""
Unicode playing card renderer
──────────────────────────────
Renders cards using Unicode playing card codepoints (U+1F0A0 block)
for compact, visual feedback during RL training loops.
"""

# ANSI color codes
RED    = "\033[91m"
BLUE   = "\033[94m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
GRAY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ── Unicode playing card mapping ─────────────────────────────────────────────
# Each suit block starts at a base codepoint. Within each block:
#   offset 1=A, 2=2, 3=3, ..., 10=T, 11=J, 13=Q, 14=K
# (offset 12 is the Knight, which we skip)

_SUIT_BASE = {
    "s": 0x1F0A0,  # spades
    "h": 0x1F0B0,  # hearts
    "d": 0x1F0C0,  # diamonds
    "c": 0x1F0D0,  # clubs
}

_RANK_OFFSET = {
    "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "T": 10, "J": 11, "Q": 13, "K": 14,
}

CARD_BACK = "\U0001F0A0"

# Build lookup dict: "Ah" -> 🂱, "Ks" -> 🂮, etc.
_CARD_MAP = {}
for _suit, _base in _SUIT_BASE.items():
    for _rank, _off in _RANK_OFFSET.items():
        _CARD_MAP[f"{_rank}{_suit}"] = chr(_base + _off)

SUIT_COLOR = {"s": BOLD, "c": GREEN, "h": RED, "d": BLUE}


def card_unicode(card_str: str) -> str:
    """Convert a card string like 'Ah' to its Unicode playing card character."""
    return _CARD_MAP.get(card_str, "?")


def render_hand(cards: list, hidden: bool = False, label: str = "") -> str:
    """Render a row of Unicode cards on one line."""
    if not cards:
        return f"{label}: (none)" if label else "(none)"

    prefix = f"{BOLD}{label}{RESET}  " if label else ""

    if hidden:
        glyphs = f"  ".join(CARD_BACK for _ in cards)
    else:
        glyphs = "  ".join(
            f"{SUIT_COLOR.get(c[1], '')}{card_unicode(c)}{RESET}" for c in cards
        )
    return f"{prefix}{glyphs}"


def equity_bar(equity: float, width: int = 10) -> str:
    """Render an equity gauge: ▰▰▰▰▰▰▰▱▱▱ 72%"""
    filled = int(equity * width)
    bar = "\u25b0" * filled + "\u25b1" * (width - filled)
    if equity >= 0.65:
        col = GREEN
    elif equity >= 0.40:
        col = YELLOW
    else:
        col = RED
    return f"{col}{bar}{RESET} {equity * 100:.0f}%"


# Position labels by seat offset from dealer
_POS_LABELS_6 = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
_POS_LABELS_3 = ["BTN", "SB", "BB"]
_POS_LABELS_2 = ["BTN", "BB"]


def _pos_label(seat_idx: int, dealer_idx: int, num_players: int) -> str:
    """Get position label for a seat."""
    if num_players <= 2:
        labels = _POS_LABELS_2
    elif num_players <= 3:
        labels = _POS_LABELS_3
    else:
        labels = _POS_LABELS_6
    offset = (seat_idx - dealer_idx) % num_players
    if offset < len(labels):
        return labels[offset]
    return f"S{seat_idx}"


def render_table(state: dict) -> str:
    """
    Compact table view — two main lines + optional action line.

    Line 1: hand/street + board cards + pot + equity bar
    Line 2: all players inline with position tags
    Line 3: (optional) last actions

    state keys: community_cards, pot, street, current_player, players,
                hand_num, episode, last_actions, showdown,
                dealer_idx (optional), hero_equity (optional)
    """
    lines = []

    # ── Line 1: board state ──────────────────────────────────────────────
    hand   = state.get("hand_num", 0)
    street = state.get("street", "preflop").upper()

    board = state.get("community_cards", [])
    board_glyphs = []
    for c in board:
        col = SUIT_COLOR.get(c[1], "")
        board_glyphs.append(f"{col}{card_unicode(c)}{RESET}")
    while len(board_glyphs) < 5:
        board_glyphs.append(f"{GRAY}__{RESET}")

    pot = state.get("pot", 0)
    hero_eq = state.get("hero_equity", -1)

    eq_str = ""
    if hero_eq >= 0:
        eq_str = f"  {equity_bar(hero_eq)}"

    lines.append(
        f"  {BOLD}#{hand} [{street}]{RESET}  "
        f"{'  '.join(board_glyphs)}  "
        f"pot ${pot:.0f}{eq_str}"
    )

    # ── Line 2: players ──────────────────────────────────────────────────
    players    = state.get("players", [])
    cur        = state.get("current_player", -1)
    dealer_idx = state.get("dealer_idx", 0)
    n_players  = len(players)
    showdown   = state.get("showdown", False)

    parts = []
    for i, p in enumerate(players):
        name   = p.get("name", f"P{i}")
        stack  = p.get("stack", 0)
        bet    = p.get("current_bet", 0)
        folded = p.get("folded", False)
        is_hero = p.get("is_hero", False)
        cards  = p.get("hole_cards", [])
        show   = is_hero or showdown

        # Position tag
        pos = _pos_label(i, dealer_idx, n_players)

        # Marker
        marker = f"{GREEN}\u25ba{RESET}" if i == cur else " "

        # Name + position
        if is_hero:
            tag = f"{BOLD}{name}{RESET}[{YELLOW}{pos}{RESET}]"
        else:
            tag = f"{name}[{GRAY}{pos}{RESET}]"

        # Cards
        if folded:
            card_str = f"{GRAY}--{RESET}"
        elif cards and show:
            card_str = "".join(
                f"{SUIT_COLOR.get(c[1], '')}{card_unicode(c)}{RESET}" for c in cards
            )
        elif cards:
            card_str = "".join(CARD_BACK for _ in cards)
        else:
            card_str = ""

        # Stack + bet
        if folded:
            info = f"{GRAY}fold{RESET}"
        elif bet > 0:
            info = f"${stack:.0f} bet${bet:.0f}"
        else:
            info = f"${stack:.0f}"

        parts.append(f"{marker}{tag} {card_str} {info}")

    lines.append("  " + "  " + "  ".join(parts))

    # ── Line 3: recent actions (compact) ─────────────────────────────────
    last = state.get("last_actions", [])
    if last:
        recent = " \u2192 ".join(last[-4:])
        lines.append(f"  {GRAY}{recent}{RESET}")

    return "\n".join(lines)


def render_training_stats(stats: dict) -> str:
    """Two-line training log: key metrics + detail line."""
    ep    = stats.get("episode", 0)
    step  = stats.get("step", 0)
    wpct  = stats.get("win_pct", 0)
    bb    = stats.get("bb_per_100", 0)
    elo   = stats.get("elo", 1500)
    ploss = stats.get("policy_loss", 0)
    vloss = stats.get("value_loss", 0)
    ent   = stats.get("entropy", 0)
    hps   = stats.get("hands_per_sec", 0)
    pct   = stats.get("phase_pct", 0)
    upd   = stats.get("ppo_updates", 0)

    bar_len = int(wpct * 20)
    filled = "\u2588" * bar_len
    empty  = "\u2591" * (20 - bar_len)
    bar = f"{GREEN}{filled}{GRAY}{empty}{RESET}"

    line1 = (
        f"  hand {ep:>6,}  step {step:>8,}  "
        f"win {bar} {wpct * 100:4.1f}%  "
        f"bb/100 {bb:+7.2f}  elo {elo:>5.0f}"
    )
    # Auxiliary losses (AlphaPoker)
    aux_card  = stats.get("aux_card_loss", 0)
    aux_range = stats.get("aux_range_loss", 0)
    aux_str = ""
    if aux_card > 0 or aux_range > 0:
        aux_str = f"  aux_c={aux_card:.4f}  aux_r={aux_range:.4f}"

    line2 = (
        f"    {GRAY}ploss={ploss:+.4f}  vloss={vloss:.4f}  "
        f"ent={ent:.3f}{aux_str}  updates={upd}  "
        f"{hps:.1f} hands/s  [{pct:.0f}%]{RESET}"
    )
    return f"{line1}\n{line2}"


if __name__ == "__main__":
    # Quick visual demo
    print(render_hand(["Ah", "Ks"], label="Hero"))
    print(render_hand(["2c", "7d", "Jh"], label="Flop"))
    print()
    print("Equity examples:")
    for eq in [0.15, 0.42, 0.72, 0.95]:
        print(f"  {equity_bar(eq)}")
    print()

    demo_state = {
        "episode": 1024,
        "hand_num": 47,
        "street": "flop",
        "community_cards": ["Ah", "7c", "2d"],
        "pot": 45.50,
        "current_player": 0,
        "dealer_idx": 0,
        "hero_equity": 0.72,
        "showdown": False,
        "players": [
            {"name": "Hero", "stack": 182.0, "current_bet": 20.0,
             "folded": False, "hole_cards": ["Kh", "Qh"], "is_hero": True},
            {"name": "V1", "stack": 210.0, "current_bet": 20.0,
             "folded": False, "hole_cards": ["Tc", "9c"], "is_hero": False},
            {"name": "V2", "stack": 0.0, "current_bet": 0.0,
             "folded": True, "hole_cards": ["5s", "3d"], "is_hero": False},
        ],
        "last_actions": [
            "V1 raise $20",
            "V2 fold",
            "Hero call $20",
        ],
    }
    print(render_table(demo_state))

    # Showdown
    print()
    demo_state["showdown"] = True
    demo_state["street"] = "river"
    demo_state["community_cards"] = ["Ah", "7c", "2d", "Kd", "3s"]
    demo_state["hero_equity"] = 0.91
    demo_state["pot"] = 90.0
    print(render_table(demo_state))

    # Training stats demo
    print()
    print(render_training_stats({
        "episode": 5000, "step": 42000, "mean_reward": 0.5,
        "win_pct": 0.55, "bb_per_100": 3.2, "elo": 1620,
        "policy_loss": 0.023, "entropy": 1.42,
    }))
