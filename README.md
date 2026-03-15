# PokerStats AI

Autonomous poker agent that learns No-Limit Hold'em through self-play reinforcement learning with GTO (Game Theory Optimal) knowledge distillation.

## What it does

1. **Trains a CFR blueprint** — solves a simplified poker game tree via Monte Carlo Counterfactual Regret Minimization to produce near-Nash equilibrium strategies
2. **Trains a neural network via PPO self-play** — an AlphaPokerNet learns by playing against itself across 6 seats with shared weights
3. **Distills CFR into the network** — the blueprint's GTO strategies are used as a teaching signal so the network internalizes game-theory-optimal play as its baseline
4. **Exploits opponents** — an opponent encoder tracks player tendencies (VPIP, PFR, bluff frequency) and learns to deviate from GTO against exploitable players
5. **Plays live on PokerStars** — reads the screen via YOLO card detection + OCR, makes decisions, and controls the UI with human-like timing

## Architecture

```
AlphaPokerNet
├── Opponent Encoder (Transformer) — per-seat history → latent vector
├── Shared Trunk (3 ResBlocks, 256 hidden)
├── Policy Head — 5 discrete actions (fold/check/call/raise/all-in)
├── Raise Head — Beta distribution for continuous raise sizing [0,1]
├── Value Head — scalar state value estimate
├── CFR Head — 7-dim blueprint strategy prediction (distillation)
└── Auxiliary Heads (training only)
    ├── Card Predictor — predict opponent hole cards from history
    ├── Action Predictor — predict opponent's next action
    └── Range Head — estimate opponent hand ranges
```

## Training modes

| Mode | Command | What it produces | Use case |
|------|---------|-----------------|----------|
| **Unified** | `make train-unified-colab` | Single network with internalized GTO | Best overall agent |
| **Hybrid** | `make train-hybrid` | Separate CFR table + PPO network | Two-system fallback |
| **Pure PPO** | `make train-colab` | PPO network only | Fast iteration |

### Unified training pipeline

```
CFR (CPU)                          PPO + Distillation (GPU)
┌─────────────────────┐           ┌──────────────────────────────┐
│ Hand Abstraction    │           │ Phase 1: Self-play           │
│ 169 preflop buckets │           │   cfr_coef = 1.0 (learn GTO)│
│ k-means postflop    │           │                              │
│         ↓           │           │ Phase 2: League play         │
│ MCCFR Solving       │──blueprint──▶ cfr_coef = 0.3 (blend)    │
│ External sampling   │           │                              │
│         ↓           │           │ Phase 3: Archetypes          │
│ Blueprint Extraction│           │   cfr_coef = 0.1 (exploit)  │
└─────────────────────┘           └──────────────────────────────┘
                                            ↓
                                    checkpoints/best.pt
                                   (single file, deploys anywhere)
```

## Quick start

### Install

```bash
pip install -r requirements.txt
```

### Run tests

```bash
make test
```

### Watch a demo (no training needed)

```bash
make demo                     # random agent, 3 players
make demo-6max                # random agent, 6 players
make demo-vs-maniac           # random agent vs maniac bot
```

### Train on Google Colab

CFR runs on CPU and PPO runs on GPU. Train them separately to avoid paying for GPU during CFR.

```bash
# 0. Package the repo (run locally)
make zip-repo
```

**Step 1 — CFR on CPU instance (no GPU needed)**

```python
# Upload pokerStats-repo.zip, then:
!unzip pokerStats-repo.zip
!pip install -r requirements.txt
!make cfr-cpu NUM_PLAYERS=6         # ~30 min on CPU

# Download the blueprint
!make zip-output
files.download('pokerStats-output.zip')
```

**Step 2 — PPO on GPU instance (needs blueprint from step 1)**

```python
# Upload pokerStats-repo.zip AND pokerStats-output.zip, then:
!unzip pokerStats-repo.zip
!unzip pokerStats-output.zip        # restores checkpoints/cfr/
!pip install -r requirements.txt
!make ppo-gpu NUM_PLAYERS=6         # ~1-2 hours on GPU

# Download trained network
!make zip-output
files.download('pokerStats-output.zip')
```

**Or all-in-one (single instance with GPU)**

```bash
!make train-unified-colab NUM_PLAYERS=6
```

| Size | CFR | PPO | Time (Colab) | Human equivalent |
|------|-----|-----|-------------|-----------------|
| **XS** | `cfr-xs` 1k iters | `ppo-xs` 20k hands | ~1h | **Drunk tourist** — knows the rules, folds or calls randomly, occasionally stumbles into a win |
| **S** | `cfr-s` 3k iters | `ppo-s` 50k hands | ~4h | **Home game regular** — understands position and pot odds, won't throw away strong hands, but bluffs poorly and overplays medium holdings |
| **M** | `cfr-m` 10k iters | `ppo-m` 90k hands | ~12h | **Casino 1/2 NL grinder** — solid TAG style, knows when to fold, sizes bets reasonably, exploits obvious fish. Profitable at low stakes |
| **L** | `cfr-l` 50k iters | `ppo-l` 300k hands | ~4 days | **Online 200NL regular** — near-GTO preflop, balanced bluff/value ranges, adjusts to opponents. Beats most live players convincingly |
| **XL** | `cfr-xl` 500k iters | `ppo-xl` 700k hands | Weeks | **High-stakes pro** — GTO baseline with precise exploits, balanced across all streets, almost impossible to read. Crushes anything below nosebleeds |

> Benchmark: 0.5 iter/s on Colab CPU. All-in-one: `make train-s NUM_PLAYERS=6`
>
> CFR runs on CPU, PPO runs on GPU. Run them separately to save costs (see workflow below).

### Watch a trained agent play

```bash
make demo-trained                  # vs fish
make demo-trained-search           # with MCTS lookahead
```

## Project structure

```
pokerStats/
├── rl/                        # Reinforcement learning
│   ├── poker_env.py           # NLHE Gym environment (2-9 players)
│   ├── ppo_agent.py           # AlphaPokerNet + PPO trainer + CFR distillation
│   ├── self_play_trainer.py   # Multi-agent training loop (3-phase curriculum)
│   ├── opponent_encoder.py    # Transformer encoder for opponent histories
│   ├── mcts.py                # Monte Carlo Tree Search for lookahead
│   ├── renderer.py            # Unicode terminal rendering
│   └── rl_live_bridge.py      # Load checkpoint → live play
│
├── cfr/                       # Counterfactual Regret Minimization
│   ├── abstraction.py         # Hand bucketing (preflop 169 + postflop k-means)
│   ├── cfr_solver.py          # External sampling MCCFR solver
│   ├── blueprint.py           # Compressed strategy table extraction
│   ├── distillation.py        # BlueprintTeacher for training-time targets
│   ├── hybrid_agent.py        # Blended CFR+PPO agent (inference)
│   ├── subgame_solver.py      # Real-time subgame refinement
│   └── cfr_live_bridge.py     # CFR decision engine for live play
│
├── vision/                    # Screen capture + YOLO detection (WIP)
│
├── reference/                 # Target architecture (read-only)
│
├── scripts/                   # Training & evaluation scripts
│   ├── train_cfr.py           # CFR training CLI
│   ├── demo_play.py           # Interactive demo viewer
│   └── eval_rl.py             # Agent evaluation
│
└── tests/                     # Test suite (95 tests)
```

## Make targets reference

### Training

| Size | CPU step | GPU step | All-in-one | Human equivalent |
|------|----------|----------|------------|-----------------|
| XS | `cfr-xs` (~35 min) | `ppo-xs` | `train-xs` | Drunk tourist |
| S | `cfr-s` (~2.5h) | `ppo-s` | `train-s` | Home game regular |
| M | `cfr-m` (~8h) | `ppo-m` | `train-m` | Casino 1/2 grinder |
| L | `cfr-l` (~3 days) | `ppo-l` | `train-l` | Online 200NL reg |
| XL | `cfr-xl` (weeks) | `ppo-xl` | `train-xl` | High-stakes pro |

All accept `NUM_PLAYERS=6`. Pure PPO without distillation: `make train-colab`.

### Demos

| Target | Description |
|--------|-------------|
| `demo` | Random agent, 3 players |
| `demo-6max` | Random agent, 6 players |
| `demo-vs-maniac` | Random agent vs maniac bot |
| `demo-vs-nit` | Random agent vs nit bot |
| `demo-trained` | Trained agent vs fish |
| `demo-trained-search` | Trained agent with MCTS |

### Packaging

| Target | Description |
|--------|-------------|
| `zip-repo` | Zip source code for Colab upload |
| `zip-output` | Zip checkpoints + data for download |

### Configurable variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_PLAYERS` | 6 | Table size (2-9) |
| `PHASE1_HANDS` | 20000 | Self-play hands |
| `PHASE2_HANDS` | 50000 | League play hands |
| `PHASE3_HANDS` | 20000 | Archetype fine-tune hands |
| `RENDER_EVERY` | 5000 | Render table every N hands |
| `CFR_ITERS` | 5000 | MCCFR iterations |
| `CFR_BUCKETS` | 50 | Postflop abstraction buckets |
| `USE_CFR` | 0 | Enable CFR distillation (0/1) |

Example: `make train-colab NUM_PLAYERS=3 PHASE1_HANDS=10000 RENDER_EVERY=1000`

## Key data types

### Action space

```python
class Action(IntEnum):
    FOLD   = 0
    CHECK  = 1
    CALL   = 2
    RAISE  = 3   # continuous sizing via raise_frac in [0, 1]
    ALL_IN = 4
```

Raise sizing is continuous: the network outputs a Beta distribution over `[0, 1]` which maps to `[min_raise, max_raise]` via an exponential curve.

### Observation vector (145-dim)

| Slice | Content |
|-------|---------|
| `[0:52]` | Hero hole cards (one-hot) |
| `[52:104]` | Community cards (one-hot) |
| `[104:128]` | Per-player stacks, bets, active flags, positions |
| `[128:136]` | Pot, call amount, min raise, street one-hot |
| `[136:143]` | Recent action history |
| `[143:145]` | Hand strength, draw potential |

### Card notation

- Ranks: `2 3 4 5 6 7 8 9 T J Q K A` (T = ten)
- Suits: `c d h s` (clubs, diamonds, hearts, spades)
- Examples: `Ah` (ace of hearts), `Tc` (ten of clubs), `2s` (two of spades)

## Training curriculum

| Phase | Hands | Opponents | CFR coef | Goal |
|-------|-------|-----------|----------|------|
| 1 | 20k-50k | Self-play (all 6 seats) | 1.0 | Learn GTO baseline |
| 2 | 50k-200k | League (frozen checkpoints) | 0.3 | Robustness + diversity |
| 3 | 20k-50k | Scripted archetypes (nit/fish/maniac/TAG) | 0.1 | Exploit real humans |

## Outputs

After training, `checkpoints/` contains:

| File | Description |
|------|-------------|
| `best.pt` | Best PPO checkpoint (by ELO) |
| `final.pt` | Final checkpoint after all phases |
| `ckpt_hand_XXXXXXXX.pt` | Periodic checkpoints |
| `cfr/blueprint.npz` | CFR blueprint strategies |
| `cfr/abstraction.npz` | Hand abstraction centroids |
| `cfr/solver_final.npz` | Raw CFR solver state |

## License

Private project.
