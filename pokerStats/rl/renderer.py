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


def render_table(state: dict) -> str:
    """
    Full table view with Unicode cards during training.

    state keys: hole_cards, community_cards, pot, stacks, street,
                current_player, last_actions, hand_num, episode, players
    """
    sep = "\u2500" * 58
    lines = []

    # Header
    ep   = state.get("episode", 0)
    hand = state.get("hand_num", 0)
    street = state.get("street", "preflop").upper()
    lines.append(f"\n{sep}")
    lines.append(f"  {BOLD}Episode {ep}  Hand {hand}  [{street}]{RESET}")
    lines.append(sep)

    # Board
    board = state.get("community_cards", [])
    board_glyphs = []
    for c in board:
        col = SUIT_COLOR.get(c[1], "")
        board_glyphs.append(f"{col}{card_unicode(c)}{RESET}")
    # Pad to 5 slots
    while len(board_glyphs) < 5:
        board_glyphs.append("__")

    pot = state.get("pot", 0)
    lines.append(f"  Board: {'  '.join(board_glyphs)}          Pot: ${pot:.2f}")
    lines.append("")

    # Players
    players = state.get("players", [])
    cur = state.get("current_player", -1)
    for i, p in enumerate(players):
        marker  = f"{GREEN}\u25ba{RESET}" if i == cur else " "
        name    = p.get("name", f"Agent-{i}")
        stack   = p.get("stack", 0)
        bet     = p.get("current_bet", 0)
        folded  = p.get("folded", False)
        is_hero = p.get("is_hero", False)
        cards   = p.get("hole_cards", [])
        show    = is_hero or state.get("showdown", False)

        status = f"{GRAY}FOLDED{RESET}" if folded else f"bet=${bet:.2f}"
        label  = f"{BOLD}{name}{RESET}" if is_hero else name

        if cards and show:
            card_str = " ".join(
                f"{SUIT_COLOR.get(c[1], '')}{card_unicode(c)}{RESET}" for c in cards
            )
        elif cards:
            card_str = " ".join(CARD_BACK for _ in cards)
        else:
            card_str = ""

        lines.append(f"  {marker} {label:20s} {card_str}   stack=${stack:8.2f}  {status}")

    # Recent actions
    last = state.get("last_actions", [])
    if last:
        lines.append("")
        recent = " \u2192 ".join(last[-4:])
        lines.append(f"  {GRAY}Recent: {recent}{RESET}")

    lines.append(sep)
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

    demo_state = {
        "episode": 1024,
        "hand_num": 47,
        "street": "flop",
        "community_cards": ["Ah", "7c", "2d"],
        "pot": 45.50,
        "current_player": 0,
        "showdown": False,
        "players": [
            {"name": "Hero", "stack": 182.0, "current_bet": 20.0,
             "folded": False, "hole_cards": ["Kh", "Qh"], "is_hero": True},
            {"name": "PPO-Agent-2", "stack": 210.0, "current_bet": 20.0,
             "folded": False, "hole_cards": ["Tc", "9c"], "is_hero": False},
            {"name": "PPO-Agent-3", "stack": 0.0, "current_bet": 0.0,
             "folded": True, "hole_cards": ["5s", "3d"], "is_hero": False},
        ],
        "last_actions": [
            "Agent-2 raise $20",
            "Agent-3 fold",
            "Hero call $20",
        ],
    }
    print(render_table(demo_state))

    # Training stats demo
    print()
    print(render_training_stats({
        "episode": 5000, "step": 42000, "mean_reward": 0.5,
        "win_pct": 0.55, "bb_per_100": 3.2, "elo": 1620,
        "policy_loss": 0.023, "entropy": 1.42,
    }))
