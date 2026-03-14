"""
Autonomous PokerStars agent
────────────────────────────
Full self-playing loop:
  1. Screen capture (mss)
  2. YOLO card detection  (hole cards + board, street-ordered)
  3. Game state parsing   (pot, stacks, actions, position)
  4. Opponent OCR + stat tracking (names, bluff history, VPIP/PFR)
  5. Decision engine      (equity + GTO + opponent exploit)
  6. UI action execution  (pyautogui with human timing)

Run:
  python autonomous_agent.py              # live (dry_run=False)
  python autonomous_agent.py --dry-run    # print decisions, no clicks
  python autonomous_agent.py --stats      # print opponent database
"""

import time
import argparse
import signal
import sys
from dataclasses import dataclass, field
from typing import Optional

from pokerstars_capture import PokerStarsCapture
from pokerstars_detector import PokerStarsDetector, GameState
from opponent_tracker import OpponentTracker
from decision_engine import DecisionEngine, Decision
from action_executor import ActionExecutor, TimingProfile
from hand_history_db import HandHistoryDB


# ─────────────────────────────────────────────────────────────────────────────
#  Hand lifecycle tracker
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class HandContext:
    hand_id: int = 0
    hole_cards: list = field(default_factory=list)
    community_cards: list = field(default_factory=list)
    street: str = "preflop"
    acted_streets: set = field(default_factory=set)   # streets where we already acted
    pot: float = 0.0
    hero_stack: float = 0.0
    position: str = "unknown"
    num_opponents: int = 1
    is_active: bool = True   # False once we fold or hand ends


class AutonomousAgent:
    def __init__(
        self,
        model_path: str  = "models/pokerstars_yolov8s.pt",
        dry_run: bool    = False,
        fps: int         = 10,
    ):
        self.dry_run = dry_run
        self.fps     = fps

        # Core components
        self.db        = HandHistoryDB()
        self.capture   = PokerStarsCapture(monitor_index=1, target_fps=fps)
        self.detector  = PokerStarsDetector(model_path=model_path)
        self.executor  = ActionExecutor(timing=TimingProfile(), dry_run=dry_run)
        self.engine    = DecisionEngine()

        # Opponent tracker (needs capture reference for ROI grabs)
        self.tracker   = OpponentTracker(db=self.db, capture=self.capture)

        # Current hand state
        self.hand_ctx: Optional[HandContext] = None
        self._running = True

        # Register Ctrl+C handler
        signal.signal(signal.SIGINT, self._shutdown)

        print(f"{'[DRY RUN] ' if dry_run else ''}Autonomous agent initialized.")
        print(f"  YOLO model: {model_path}")
        print(f"  Capture:    {fps} FPS")
        print(f"  DB:         data/hand_history.db\n")

    # ── Shutdown ──────────────────────────────────────────────────────────────
    def _shutdown(self, *_):
        print("\nShutting down agent...")
        self._running = False
        sys.exit(0)

    # ── Hand boundary detection ───────────────────────────────────────────────
    def _is_new_hand(self, state: GameState) -> bool:
        """
        New hand when:
          - Hole cards appear after being empty
          - Street resets to preflop with new hole cards
        """
        if self.hand_ctx is None:
            return len(state.hole_cards) == 2

        prev_hole = self.hand_ctx.hole_cards
        new_hole  = state.hole_cards

        # Different hole cards = new hand
        if new_hole and sorted(new_hole) != sorted(prev_hole):
            return True

        # We folded last hand, now new cards appeared
        if not self.hand_ctx.is_active and len(new_hole) == 2:
            return True

        return False

    def _detect_street_transition(self, state: GameState) -> Optional[str]:
        """Return new street name if we just transitioned, else None."""
        if self.hand_ctx is None:
            return None

        prev_n = len(self.hand_ctx.community_cards)
        curr_n = len(state.community_cards)

        if curr_n == prev_n:
            return None

        transitions = {
            (0, 3): "flop",
            (3, 4): "turn",
            (4, 5): "river",
        }
        return transitions.get((prev_n, curr_n))

    def _on_new_hand(self, state: GameState):
        """Initialize a new hand context and log to DB."""
        hand_id = self.db.start_hand(
            state.hole_cards,
            num_players=len([p for p in state.players.values() if p.is_active]),
            position=state.hero_position or "unknown"
        )
        self.hand_ctx = HandContext(
            hand_id     = hand_id,
            hole_cards  = list(state.hole_cards),
            community_cards = [],
            street      = "preflop",
            pot         = state.pot or 0.0,
            hero_stack  = state.hero_stack or 0.0,
            position    = state.hero_position or "unknown",
            num_opponents = max(len(state.players), 1),
        )
        self.tracker.on_new_hand(hand_id)

        print(f"\n{'═'*64}")
        print(f"  NEW HAND #{hand_id}  |  Hole: {' '.join(state.hole_cards)}  |  Pos: {state.hero_position}")
        print(f"  Stack: ${state.hero_stack}  |  Players at table: {self.hand_ctx.num_opponents}")
        self._print_table_reads()

    def _print_table_reads(self):
        """Print opponent profiles for the current table."""
        profiles = self.tracker.get_table_profiles()
        if not profiles:
            print("  Opponents: no data yet")
            return
        for seat, p in profiles.items():
            print(f"  Seat {seat}: {p.name:20s} | {p.player_type:8s} | "
                  f"VPIP={p.vpip:.0%} PFR={p.pfr:.0%} "
                  f"bluff={p.bluff_frequency:.0%} hands={p.hands_played} | "
                  f"→ {p.exploit_note()[:50]}")

    # ── Main decision logic ───────────────────────────────────────────────────
    def _should_act(self, state: GameState) -> bool:
        """Return True if it's our turn to act."""
        if self.hand_ctx is None or not self.hand_ctx.is_active:
            return False
        if not state.available_actions:
            return False
        if not state.hole_cards:
            return False
        # Don't re-act on same street we already acted
        if state.street in self.hand_ctx.acted_streets:
            return False
        return True

    def _make_and_execute_decision(self, state: GameState):
        """Run the decision engine and execute the chosen action."""
        profiles = self.tracker.get_table_profiles()

        decision = self.engine.decide(
            hole_cards        = state.hole_cards,
            community_cards   = state.community_cards,
            street            = state.street,
            pot               = state.pot or 0.0,
            hero_stack        = state.hero_stack or 0.0,
            call_amount       = state.call_amount or 0.0,
            position          = state.hero_position or "oop",
            num_opponents     = self.hand_ctx.num_opponents,
            opponent_profiles = profiles,
            available_actions = state.available_actions,
        )

        # Log decision to DB
        self.db.log_action(
            self.hand_ctx.hand_id,
            state.street, "hero",
            decision.action, decision.amount,
            state.pot or 0.0,
            state.community_cards
        )

        # Print decision
        board_str = " ".join(state.community_cards) or "(preflop)"
        print(f"\n  [{state.street.upper()}] Board: {board_str}")
        print(f"  Hand: {' '.join(state.hole_cards)}")
        print(f"  Pot: ${state.pot}  Call: ${state.call_amount}  Stack: ${state.hero_stack}")
        print(f"  Decision: {decision.action.upper()} "
              f"{'$'+str(decision.amount) if decision.amount else ''}"
              f"  (conf={decision.confidence:.0%})")
        print(f"  Reason: {decision.reasoning}")

        # Execute on UI
        success = self.executor.execute(
            decision,
            street          = state.street,
            pot             = state.pot or 0.0,
            min_raise       = state.min_raise or 0.0,
            max_raise       = state.hero_stack or 9999,
            available_actions = state.available_actions,
        )

        # Mark street as acted
        if success:
            self.hand_ctx.acted_streets.add(state.street)
            if decision.action == "fold":
                self.hand_ctx.is_active = False

        return decision

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        print("Starting autonomous agent. Watching PokerStars screen...\n")

        prev_board = []
        prev_hole  = []

        for frame in self.capture.stream():
            if not self._running:
                break

            state = self.detector.parse_frame(frame)

            # Skip low-confidence frames
            if state.confidence < 0.4 and not state.hole_cards:
                continue

            # ── New hand? ──
            if self._is_new_hand(state):
                # Finish previous hand if any
                if self.hand_ctx:
                    self.db.finish_hand(
                        self.hand_ctx.hand_id,
                        self.hand_ctx.community_cards,
                        self.hand_ctx.pot,
                        0.0   # P&L filled in post-hand (requires result detection)
                    )
                self._on_new_hand(state)

            if self.hand_ctx is None:
                continue

            # ── Street transition? ──
            new_street = self._detect_street_transition(state)
            if new_street:
                print(f"\n  ── {new_street.upper()} ── board: {' '.join(state.community_cards)}")
                self.hand_ctx.community_cards = list(state.community_cards)
                self.hand_ctx.street = new_street

            # ── Poll opponent actions ──
            new_opp_actions = self.tracker.poll_actions(
                state.street, state.pot or 0.0, state.community_cards
            )
            for seat, name, action, amount in new_opp_actions:
                pot_frac = amount / max(state.pot or 1, 1)
                print(f"  Seat {seat} ({name}): {action.upper()} "
                      f"{'$'+str(amount)+f' ({pot_frac:.0%} pot)' if amount else ''}")

            # ── Showdown detection ──
            if state.street == "river" and state.community_cards:
                showdowns = self.tracker.detect_showdown(state.street, state.pot or 0.0)
                for s in showdowns:
                    bluff_tag = "BLUFF" if s["was_bluff"] else "value"
                    print(f"  SHOWDOWN: {s['name']} showed {s['cards']} [{bluff_tag}]")

            # ── Our turn to act? ──
            if self._should_act(state):
                self._make_and_execute_decision(state)

            # Track state changes
            prev_board = list(state.community_cards)
            prev_hole  = list(state.hole_cards)

    # ── Stats mode ────────────────────────────────────────────────────────────
    def print_stats(self):
        """Print all tracked opponents sorted by hands played."""
        opponents = self.db.get_all_opponents()
        if not opponents:
            print("No opponent data yet.")
            return

        print(f"\n{'─'*90}")
        print(f"{'Name':<22} {'Type':<10} {'Hands':>6} {'VPIP':>6} {'PFR':>6} "
              f"{'3bet':>6} {'Fcbet':>6} {'BluffF':>7} {'SD':>4}  Exploit")
        print(f"{'─'*90}")
        for p in opponents:
            print(
                f"{p.name:<22} {p.player_type:<10} {p.hands_played:>6} "
                f"{p.vpip:>6.0%} {p.pfr:>6.0%} {p.three_bet:>6.0%} "
                f"{p.fold_to_cbet:>6.0%} {p.bluff_frequency:>7.0%} "
                f"{p.total_showdowns:>4}  {p.exploit_note()[:35]}"
            )
        print(f"{'─'*90}")
        print(f"Total opponents tracked: {len(opponents)}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous PokerStars agent")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print decisions without clicking")
    parser.add_argument("--stats",   action="store_true",
                        help="Print opponent database and exit")
    parser.add_argument("--fps",     type=int, default=10)
    parser.add_argument("--model",   default="models/pokerstars_yolov8s.pt")
    args = parser.parse_args()

    agent = AutonomousAgent(
        model_path = args.model,
        dry_run    = args.dry_run,
        fps        = args.fps,
    )

    if args.stats:
        agent.print_stats()
    else:
        agent.run()
