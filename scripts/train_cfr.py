#!/usr/bin/env python3
"""
CFR training pipeline
─────────────────────
Three phases:
  1. abstraction — Build preflop LUT + postflop equity clusters (~30 min)
  2. solve       — External Sampling MCCFR (~4-12 hours)
  3. blueprint   — Extract average strategies → compressed .npz

Usage:
  python scripts/train_cfr.py --phase abstraction --output-dir checkpoints/cfr
  python scripts/train_cfr.py --phase solve --iterations 10000000 --output-dir checkpoints/cfr
  python scripts/train_cfr.py --phase blueprint --output-dir checkpoints/cfr
  python scripts/train_cfr.py --phase solve --resume checkpoints/cfr/solver_5M.npz
  python scripts/train_cfr.py --phase solve --iterations 5000 --render-every 500
"""

import argparse
import os
import time
import random
import numpy as np
from pathlib import Path
from collections import deque

from pokerStats.cfr.abstraction import HandAbstraction
from pokerStats.cfr.cfr_solver import (
    CFRSolver, NUM_CFR_ACTIONS, cfr_to_env_action, env_legal_to_cfr_mask,
)
from pokerStats.cfr.blueprint import BlueprintStrategy
from pokerStats.rl.poker_env import PokerEnv, Action, STREETS
from pokerStats.rl.renderer import render_training_stats


# ── ANSI colors (same as renderer.py) ───────────────────────────────────────
GREEN = "\033[32m"
GRAY  = "\033[90m"
CYAN  = "\033[36m"
YELLOW = "\033[33m"
RESET = "\033[0m"
BOLD  = "\033[1m"


def _render_cfr_stats(stats: dict) -> str:
    """Two-line CFR training log matching PPO trainer style."""
    it     = stats.get("iteration", 0)
    n_info = stats.get("info_sets", 0)
    rate   = stats.get("iter_per_sec", 0)
    pct    = stats.get("phase_pct", 0)
    exploit = stats.get("exploitability", None)

    bar_len = int(pct / 100 * 20)
    filled = "█" * bar_len
    empty  = "░" * (20 - bar_len)
    bar = f"{GREEN}{filled}{GRAY}{empty}{RESET}"

    line1 = (
        f"  iter {it:>10,}  "
        f"progress {bar} {pct:4.1f}%  "
        f"info_sets {n_info:>8,}"
    )

    exploit_str = ""
    if exploit is not None:
        exploit_str = f"  exploit={exploit:+.4f} BB/hand"

    line2 = (
        f"    {GRAY}{rate:.0f} iter/s{exploit_str}  "
        f"[{pct:.0f}%]{RESET}"
    )
    return f"{line1}\n{line2}"


def _play_demo_hand(solver: CFRSolver, abstraction: HandAbstraction):
    """Play one rendered hand using the current CFR strategy (visual feedback)."""
    env = PokerEnv(num_players=2, render_mode="unicode")
    env.reset()

    action_history = ()
    steps = 0
    while not env._done and steps < 50:
        player = env.players[env.current_player]
        hand_bucket = abstraction.get_bucket(player.hole_cards, env.community)
        street = STREETS[env.street_idx]

        key = solver._info_set_key(hand_bucket, street, action_history)
        info_set = solver._get_info_set(key)
        legal_mask = env_legal_to_cfr_mask(env)

        strategy = info_set.average_strategy(legal_mask)

        # Sample action from average strategy
        legal_indices = np.where(legal_mask)[0]
        if len(legal_indices) == 0:
            break
        legal_probs = strategy[legal_indices]
        legal_probs = legal_probs / (legal_probs.sum() + 1e-10)
        cfr_action = np.random.choice(legal_indices, p=legal_probs)

        env_action, raise_frac = cfr_to_env_action(cfr_action)

        # Show strategy distribution
        action_names = ["fold", "check", "call", "r33%", "r75%", "r150%", "allin"]
        strat_str = "  ".join(
            f"{action_names[i]}={strategy[i]:.0%}"
            for i in range(NUM_CFR_ACTIONS)
            if legal_mask[i]
        )
        print(f"    {CYAN}CFR strategy: {strat_str}{RESET}")

        obs, rewards, done, info = env.step(int(env_action), raise_frac)
        action_history = action_history + (cfr_action,)
        steps += 1

    print(f"    {YELLOW}Rewards: P0={rewards[0]:+.1f} BB  P1={rewards[1]:+.1f} BB{RESET}\n")


def phase_abstraction(output_dir: str, n_postflop_buckets: int, n_samples: int):
    """Build hand abstractions."""
    print(f"\n{'═' * 60}")
    print(f"  {BOLD}Phase 1: Building hand abstractions{RESET}")
    print(f"{'═' * 60}\n")

    abstraction = HandAbstraction(n_postflop_buckets=n_postflop_buckets)

    # Preflop is instant
    print(f"  Preflop: {abstraction.preflop.NUM_BUCKETS} canonical groups (instant)")

    # Spot-check some preflop buckets
    examples = [
        (["Ah", "As"], "AA"), (["Ah", "Kh"], "AKs"),
        (["Ah", "Kc"], "AKo"), (["7d", "2c"], "72o"),
    ]
    for cards, name in examples:
        b = abstraction.get_bucket(cards, [])
        print(f"    {name:5s} → bucket {b}")

    # Postflop clustering
    for street in ["flop", "turn", "river"]:
        print(f"\n  Training {street} buckets ({n_postflop_buckets} clusters, "
              f"{n_samples} samples)...")
        t0 = time.time()
        abstraction.postflop.train_buckets(
            street, n_samples=n_samples, n_kmeans_iters=30
        )
        elapsed = time.time() - t0
        print(f"    ✓ Done in {elapsed:.1f}s")

    path = os.path.join(output_dir, "abstraction.npz")
    abstraction.save(path)
    print(f"\n  Saved abstraction → {path}")


def phase_solve(
    output_dir: str,
    n_iterations: int,
    abstraction_path: str,
    resume_path: str = None,
    checkpoint_every: int = 500000,
    log_every: int = 1000,
    render_every: int = 0,
    exploit_check_every: int = 0,
):
    """Run MCCFR training with rendered hands and training stats."""
    print(f"\n{'═' * 60}")
    print(f"  {BOLD}Phase 2: MCCFR training ({n_iterations:,} iterations){RESET}")
    print(f"{'═' * 60}\n")

    # Load abstraction
    abstraction = HandAbstraction()
    abstraction.load(abstraction_path)
    print(f"  Loaded abstraction from {abstraction_path}")

    # Create or resume solver
    solver = CFRSolver(abstraction=abstraction)
    if resume_path:
        solver.load(resume_path)
        print(f"  Resumed from {resume_path} (iteration {solver.iterations})")

    start_iter = solver.iterations
    target_iter = start_iter + n_iterations
    remaining = n_iterations

    if remaining <= 0:
        print(f"  Already completed {solver.iterations} iterations, nothing to do")
        return

    # Auto-scale log frequency: at least 10 log lines during training
    if log_every >= remaining:
        log_every = max(1, remaining // 10)

    # Auto-scale render frequency too
    if render_every > 0 and render_every >= remaining:
        render_every = max(1, remaining // 4)

    print(f"  Running {remaining:,} iterations (log every {log_every:,})...")
    if render_every > 0:
        print(f"  Rendering demo hand every {render_every:,} iterations")
    print(flush=True)

    t0 = time.time()
    env_iterations = 0
    last_exploit = None

    while env_iterations < remaining:
        batch = min(log_every, remaining - env_iterations)
        solver.train(batch, log_every=0)
        env_iterations += batch

        elapsed = time.time() - t0
        rate = env_iterations / max(elapsed, 1)
        pct = env_iterations / remaining * 100

        # Training stats
        stats = {
            "iteration": solver.iterations,
            "info_sets": len(solver.info_sets),
            "iter_per_sec": rate,
            "phase_pct": pct,
            "exploitability": last_exploit,
        }
        print(_render_cfr_stats(stats), flush=True)

        # Exploitability check
        if exploit_check_every > 0 and solver.iterations % exploit_check_every == 0:
            print(f"\n    {CYAN}Computing exploitability estimate...{RESET}", end="", flush=True)
            last_exploit = solver.exploitability_estimate(n_samples=200)
            print(f" {last_exploit:+.4f} BB/hand\n")

        # Render a demo hand using current strategy
        if render_every > 0 and solver.iterations % render_every == 0:
            print(f"\n    {BOLD}── Demo hand (iter {solver.iterations:,}) ──{RESET}")
            _play_demo_hand(solver, abstraction)

        # Checkpoint
        if checkpoint_every > 0 and solver.iterations % checkpoint_every == 0:
            ckpt_path = os.path.join(
                output_dir,
                f"solver_{solver.iterations // 1000000}M.npz"
            )
            solver.save(ckpt_path)
            print(f"    Checkpoint → {ckpt_path}")

    # Final save
    final_path = os.path.join(output_dir, "solver_final.npz")
    solver.save(final_path)

    total_time = time.time() - t0
    dline = "═" * 60
    print(f"\n{dline}")
    print(f"  {BOLD}MCCFR training complete!{RESET}")
    print(f"  Iterations:    {solver.iterations:,}")
    print(f"  Info sets:     {len(solver.info_sets):,}")
    print(f"  Total time:    {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Avg rate:      {env_iterations/max(total_time,1):.0f} iter/s")
    print(f"  Saved to:      {final_path}")
    print(f"{dline}\n")

    # Final demo hand
    if render_every > 0:
        print(f"  {BOLD}── Final demo hand ──{RESET}")
        _play_demo_hand(solver, abstraction)


def phase_blueprint(output_dir: str, solver_path: str, min_visits: int):
    """Extract blueprint from trained solver."""
    print(f"\n{'═' * 60}")
    print(f"  {BOLD}Phase 3: Extracting blueprint strategy{RESET}")
    print(f"{'═' * 60}\n")

    # Load solver
    abstraction = HandAbstraction()
    abs_path = os.path.join(output_dir, "abstraction.npz")
    if os.path.exists(abs_path):
        abstraction.load(abs_path)

    solver = CFRSolver(abstraction=abstraction)
    solver.load(solver_path)
    print(f"  Loaded solver: {len(solver.info_sets):,} info sets, "
          f"{solver.iterations:,} iterations")

    # Extract
    blueprint = BlueprintStrategy.from_solver(solver, min_visits=min_visits)
    print(f"  Extracted {len(blueprint)} strategies (min_visits={min_visits})")
    print(f"  Estimated size: {blueprint.memory_usage_mb():.1f} MB")

    # Show a few sample strategies
    print(f"\n  {BOLD}Sample blueprint strategies:{RESET}")
    shown = 0
    action_names = ["fold", "check", "call", "r33%", "r75%", "r150%", "allin"]
    for key, strat in list(blueprint.strategies.items())[:5]:
        strat_f64 = strat.astype(np.float64)
        parts = [f"{action_names[i]}={strat_f64[i]:.0%}" for i in range(len(strat_f64)) if strat_f64[i] > 0.01]
        print(f"    {key:40s} → {', '.join(parts)}")
        shown += 1

    path = os.path.join(output_dir, "blueprint.npz")
    blueprint.save(path)
    print(f"\n  Saved blueprint → {path}")


def main():
    parser = argparse.ArgumentParser(description="CFR training pipeline")
    parser.add_argument("--phase", required=True,
                        choices=["abstraction", "solve", "blueprint"],
                        help="Training phase to run")
    parser.add_argument("--output-dir", default="checkpoints/cfr",
                        help="Output directory for checkpoints")
    parser.add_argument("--iterations", type=int, default=10_000_000,
                        help="Number of MCCFR iterations (solve phase)")
    parser.add_argument("--resume", default=None,
                        help="Resume solver from checkpoint path")
    parser.add_argument("--n-buckets", type=int, default=200,
                        help="Number of postflop buckets")
    parser.add_argument("--n-samples", type=int, default=10000,
                        help="Samples per street for abstraction training")
    parser.add_argument("--checkpoint-every", type=int, default=500000,
                        help="Checkpoint interval (iterations)")
    parser.add_argument("--log-every", type=int, default=1000,
                        help="Log stats every N iterations")
    parser.add_argument("--render-every", type=int, default=0,
                        help="Render a demo hand every N iterations (0=off)")
    parser.add_argument("--exploit-every", type=int, default=0,
                        help="Check exploitability every N iterations (0=off)")
    parser.add_argument("--min-visits", type=int, default=10,
                        help="Minimum visits to include in blueprint")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.phase == "abstraction":
        phase_abstraction(args.output_dir, args.n_buckets, args.n_samples)

    elif args.phase == "solve":
        phase_solve(
            args.output_dir,
            args.iterations,
            abstraction_path=os.path.join(args.output_dir, "abstraction.npz"),
            resume_path=args.resume,
            checkpoint_every=args.checkpoint_every,
            log_every=args.log_every,
            render_every=args.render_every,
            exploit_check_every=args.exploit_every,
        )

    elif args.phase == "blueprint":
        solver_path = os.path.join(args.output_dir, "solver_final.npz")
        phase_blueprint(args.output_dir, solver_path, args.min_visits)


if __name__ == "__main__":
    main()
