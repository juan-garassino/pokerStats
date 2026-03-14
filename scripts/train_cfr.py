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
"""

import argparse
import os
import time
from pathlib import Path

from pokerStats.cfr.abstraction import HandAbstraction
from pokerStats.cfr.cfr_solver import CFRSolver
from pokerStats.cfr.blueprint import BlueprintStrategy


def phase_abstraction(output_dir: str, n_postflop_buckets: int, n_samples: int):
    """Build hand abstractions."""
    print("=" * 60)
    print("Phase 1: Building hand abstractions")
    print("=" * 60)

    abstraction = HandAbstraction(n_postflop_buckets=n_postflop_buckets)

    # Preflop is instant (pure lookup table)
    print(f"  Preflop: {abstraction.preflop.NUM_BUCKETS} canonical groups (instant)")

    # Postflop clustering
    for street in ["flop", "turn", "river"]:
        print(f"  Training {street} buckets ({n_postflop_buckets} clusters, "
              f"{n_samples} samples)...")
        t0 = time.time()
        abstraction.postflop.train_buckets(
            street, n_samples=n_samples, n_kmeans_iters=30
        )
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")

    path = os.path.join(output_dir, "abstraction.npz")
    abstraction.save(path)
    print(f"\n  Saved abstraction → {path}")


def phase_solve(
    output_dir: str,
    n_iterations: int,
    abstraction_path: str,
    resume_path: str = None,
    checkpoint_every: int = 500000,
    log_every: int = 10000,
):
    """Run MCCFR training."""
    print("=" * 60)
    print(f"Phase 2: MCCFR training ({n_iterations:,} iterations)")
    print("=" * 60)

    # Load abstraction
    abstraction = HandAbstraction()
    abstraction.load(abstraction_path)
    print(f"  Loaded abstraction from {abstraction_path}")

    # Create or resume solver
    solver = CFRSolver(abstraction=abstraction)
    if resume_path:
        solver.load(resume_path)
        print(f"  Resumed from {resume_path} (iteration {solver.iterations})")

    remaining = n_iterations - solver.iterations
    if remaining <= 0:
        print(f"  Already completed {solver.iterations} iterations, nothing to do")
        return

    print(f"  Running {remaining:,} iterations...")
    t0 = time.time()

    env_iterations = 0
    while env_iterations < remaining:
        batch = min(log_every, remaining - env_iterations)
        solver.train(batch, log_every=0)
        env_iterations += batch

        elapsed = time.time() - t0
        rate = env_iterations / max(elapsed, 1)
        print(f"  iter {solver.iterations:>10,} | "
              f"info_sets {len(solver.info_sets):>8,} | "
              f"{rate:.0f} iter/s | "
              f"elapsed {elapsed:.0f}s")

        # Checkpoint
        if solver.iterations % checkpoint_every == 0:
            ckpt_path = os.path.join(
                output_dir,
                f"solver_{solver.iterations // 1000000}M.npz"
            )
            solver.save(ckpt_path)
            print(f"  Checkpoint → {ckpt_path}")

    # Final save
    final_path = os.path.join(output_dir, "solver_final.npz")
    solver.save(final_path)
    print(f"\n  Saved solver → {final_path}")
    print(f"  Total info sets: {len(solver.info_sets):,}")
    print(f"  Total time: {time.time() - t0:.0f}s")


def phase_blueprint(output_dir: str, solver_path: str, min_visits: int):
    """Extract blueprint from trained solver."""
    print("=" * 60)
    print("Phase 3: Extracting blueprint strategy")
    print("=" * 60)

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

    path = os.path.join(output_dir, "blueprint.npz")
    blueprint.save(path)
    print(f"  Saved blueprint → {path}")


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
    parser.add_argument("--min-visits", type=int, default=10,
                        help="Minimum visits to include in blueprint")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.phase == "abstraction":
        phase_abstraction(args.output_dir, args.n_buckets, args.n_samples)

    elif args.phase == "solve":
        abs_path = args.resume or os.path.join(args.output_dir, "abstraction.npz")
        if not args.resume:
            abs_path = os.path.join(args.output_dir, "abstraction.npz")
        phase_solve(
            args.output_dir,
            args.iterations,
            abstraction_path=os.path.join(args.output_dir, "abstraction.npz"),
            resume_path=args.resume,
            checkpoint_every=args.checkpoint_every,
        )

    elif args.phase == "blueprint":
        solver_path = os.path.join(args.output_dir, "solver_final.npz")
        phase_blueprint(args.output_dir, solver_path, args.min_visits)


if __name__ == "__main__":
    main()
