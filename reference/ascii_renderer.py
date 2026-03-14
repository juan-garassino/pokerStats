"""
ASCII poker card renderer
─────────────────────────
Renders cards and tables in terminal during RL training loops.
Gives you visual feedback without a GUI.
"""

SUIT_SYMBOLS = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
SUIT_NAMES   = {"s": "spades", "h": "hearts", "d": "diamonds", "c": "clubs"}
RANK_DISPLAY = {"T": "10", "J": "J", "Q": "Q", "K": "K", "A": "A"}

# ANSI color codes
RED    = "\033[91m"
BLUE   = "\033[94m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
GRAY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

SUIT_COLOR = {"s": BOLD, "c": GREEN, "h": RED, "d": BLUE}


def _rank_str(card: str) -> str:
    r = card[0]
    return RANK_DISPLAY.get(r, r)


def card_ascii(card: str, hidden: bool = False) -> list:
    """
    Return a 5-line ASCII card as a list of strings.

    ┌───┐
    │A  │
    │ ♠ │
    │  A│
    └───┘
    """
    if hidden:
        return [
            "┌───┐",
            "│░░░│",
            "│░░░│",
            "│░░░│",
            "└───┘",
        ]

    rank = _rank_str(card)
    suit = card[1]
    sym  = SUIT_SYMBOLS[suit]
    col  = SUIT_COLOR[suit]
    r2   = rank.rjust(2)  # right-align for bottom

    return [
        "┌───┐",
        f"│{col}{rank:<2}{RESET} │",
        f"│ {col}{sym}{RESET} │",
        f"│ {col}{r2}{RESET}│",
        "└───┘",
    ]


def empty_slot_ascii() -> list:
    return [
        "┌───┐",
        "│   │",
        "│   │",
        "│   │",
        "└───┘",
    ]


def render_hand(cards: list, hidden: bool = False, label: str = "") -> str:
    """Render a row of cards side by side."""
    if not cards:
        return f"{label}: (none)"

    rendered = [card_ascii(c, hidden) for c in cards]
    lines = []
    if label:
        lines.append(f"{BOLD}{label}{RESET}")
    for row in range(5):
        lines.append("  ".join(r[row] for r in rendered))
    return "\n".join(lines)


def render_table(state: dict) -> str:
    """
    Full ASCII table view during training.
    state keys: hole_cards, community_cards, pot, stacks, street,
                current_player, last_actions, hand_num, episode
    """
    sep  = "─" * 64
    lines = []

    # Header
    ep   = state.get("episode", 0)
    hand = state.get("hand_num", 0)
    lines.append(f"\n{sep}")
    lines.append(f"  {BOLD}Episode {ep}  Hand {hand}  [{state.get('street','preflop').upper()}]{RESET}")
    lines.append(sep)

    # Community cards
    board = state.get("community_cards", [])
    board_display = board + [""] * (5 - len(board))
    rendered_board = []
    for c in board_display:
        rendered_board.append(card_ascii(c) if c else empty_slot_ascii())

    lines.append(f"\n  {BOLD}Board:{RESET}")
    for row in range(5):
        lines.append("  " + "  ".join(r[row] for r in rendered_board))

    # Pot
    pot = state.get("pot", 0)
    lines.append(f"\n  {YELLOW}Pot: ${pot:.2f}{RESET}")
    lines.append("")

    # Players
    players = state.get("players", [])
    cur     = state.get("current_player", -1)
    for i, p in enumerate(players):
        marker  = f"{GREEN}►{RESET}" if i == cur else " "
        name    = p.get("name", f"Agent {i}")
        stack   = p.get("stack", 0)
        bet     = p.get("current_bet", 0)
        folded  = p.get("folded", False)
        is_hero = p.get("is_hero", False)

        cards = p.get("hole_cards", [])
        show  = is_hero or state.get("showdown", False)

        status = f"{GRAY}FOLDED{RESET}" if folded else f"bet=${bet:.2f}"
        label  = f"{BOLD}{name}{RESET}" if is_hero else name

        lines.append(f"  {marker} {label:20s} stack=${stack:8.2f}  {status}")
        if cards and show:
            card_rows = [card_ascii(c, hidden=not show) for c in cards]
            for row in range(5):
                lines.append("    " + "  ".join(r[row] for r in card_rows))
        elif cards:
            card_rows = [card_ascii(c, hidden=True) for c in cards]
            for row in range(5):
                lines.append("    " + "  ".join(r[row] for r in card_rows))

    # Last actions
    last = state.get("last_actions", [])
    if last:
        lines.append(f"\n  {GRAY}Recent actions:{RESET}")
        for a in last[-4:]:
            lines.append(f"    {GRAY}{a}{RESET}")

    lines.append(sep)
    return "\n".join(lines)


def render_training_stats(stats: dict) -> str:
    """Compact one-liner for training loop output."""
    ep   = stats.get("episode", 0)
    step = stats.get("step", 0)
    rew  = stats.get("mean_reward", 0)
    wpct = stats.get("win_pct", 0)
    bb   = stats.get("bb_per_100", 0)
    elo  = stats.get("elo", 1500)
    loss = stats.get("policy_loss", 0)
    ent  = stats.get("entropy", 0)

    bar_len = int(wpct * 20)
    bar = f"{GREEN}{'█' * bar_len}{GRAY}{'░' * (20-bar_len)}{RESET}"

    return (
        f"  ep={ep:>7,}  step={step:>9,}  "
        f"win={bar} {wpct*100:4.1f}%  "
        f"bb/100={bb:+6.2f}  elo={elo:>5.0f}  "
        f"loss={loss:.4f}  ent={ent:.3f}"
    )


if __name__ == "__main__":
    # Quick visual test
    print(render_hand(["Ah", "Ks"], label="Hero"))
    print()
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
            {"name": "PPO-Agent-1", "stack": 182.0, "current_bet": 20.0,
             "folded": False, "hole_cards": ["Kh", "Qh"], "is_hero": True},
            {"name": "PPO-Agent-2", "stack": 210.0, "current_bet": 20.0,
             "folded": False, "hole_cards": ["Tc", "9c"], "is_hero": False},
            {"name": "PPO-Agent-3", "stack": 0.0,   "current_bet": 0.0,
             "folded": True,  "hole_cards": ["5s", "3d"], "is_hero": False},
        ],
        "last_actions": [
            "PPO-Agent-2: raise $20",
            "PPO-Agent-3: fold",
            "PPO-Agent-1: call $20",
        ],
    }
    print(render_table(demo_state))
