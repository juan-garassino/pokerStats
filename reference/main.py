"""
PokerStars AI — main entry point
──────────────────────────────────
Ties together: screen capture → YOLO detection → game state → agent decision

Run:
  python main.py              # live mode (reads screen)
  python main.py --calibrate  # verify ROI alignment first
  python main.py --test-state # print game state from current screen
"""

import argparse
import time
from pokerstars_capture import PokerStarsCapture
from pokerstars_detector import PokerStarsDetector, GameState


# ── Stub agent (replace with your GTO/RL agent) ──────────────────────────────
class PokerAgent:
    """
    Placeholder agent. Replace decide() with your GTO/CFR/RL logic.
    See architecture overview for full agent implementation.
    """
    def decide(self, state: GameState) -> dict:
        if not state.hole_cards:
            return {"action": "wait", "reason": "cards not detected yet"}

        # Stub: just log what we see
        return {
            "action":  "observe",
            "hole":    state.hole_cards,
            "board":   state.community_cards,
            "street":  state.street,
            "pot":     state.pot,
            "stack":   state.hero_stack,
            "actions": state.available_actions,
        }


# ── Pipeline ─────────────────────────────────────────────────────────────────
def run(fps: int = 10, model_path: str = "models/pokerstars_yolov8s.pt"):
    capture  = PokerStarsCapture(monitor_index=1, target_fps=fps)
    detector = PokerStarsDetector(model_path=model_path)
    agent    = PokerAgent()

    print(f"Running at {fps} FPS. Ctrl+C to stop.\n")

    prev_street = None
    prev_community = []

    for frame in capture.stream():
        state = detector.parse_frame(frame)

        # Only trigger agent on meaningful state changes
        state_changed = (
            state.community_cards != prev_community
            or state.street != prev_street
            or ("raise" in state.available_actions and state.hole_cards)
        )

        if state_changed:
            decision = agent.decide(state)
            print(f"\n{'─'*60}")
            print(f"[{state.street.upper()}]  conf={state.confidence:.0%}")
            print(f"  Hole:  {' '.join(state.hole_cards) or '??'}")
            print(f"  Board: {' '.join(state.community_cards) or '(preflop)'}")
            print(f"  Pot:   ${state.pot}   Stack: ${state.hero_stack}")
            print(f"  Actions available: {state.available_actions}")
            print(f"  → Agent: {decision}")

            prev_street    = state.street
            prev_community = list(state.community_cards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PokerStars AI")
    parser.add_argument("--calibrate",  action="store_true", help="Save calibration snapshot")
    parser.add_argument("--test-state", action="store_true", help="Print one game state and exit")
    parser.add_argument("--fps",        type=int, default=10, help="Capture FPS (default 10)")
    parser.add_argument("--model",      default="models/pokerstars_yolov8s.pt")
    args = parser.parse_args()

    if args.calibrate:
        cap = PokerStarsCapture(monitor_index=1)
        cap.save_calibration_snapshot("calibration_check.png")
        print("Saved calibration_check.png — verify all ROI boxes align with PokerStars UI")

    elif args.test_state:
        cap = PokerStarsCapture(monitor_index=1, target_fps=1)
        det = PokerStarsDetector(model_path=args.model)
        frame = cap.capture_frame()
        state = det.parse_frame(frame)
        print(state)

    else:
        run(fps=args.fps, model_path=args.model)
