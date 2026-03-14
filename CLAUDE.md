# PokerStars AI — CLAUDE.md

This is an autonomous poker agent that:
1. Reads the PokerStars screen via YOLO card detection + OCR
2. Tracks opponents by name, builds a bluff/stat profile per player
3. Makes decisions using a GTO rule engine OR a trained PPO policy
4. Controls the PokerStars UI with pyautogui (human-like timing)
5. Trains a self-play RL agent via a custom Gym environment

---

## Project structure

```
POKERSTATS/
├── pokerStats/               # main package — legacy code lives here
├── reference/                # modern reference implementations (read-only source of truth)
│   ├── pokerstars_capture.py     Screen capture + ROI pixel map (1920x1080 6-max)
│   ├── pokerstars_detector.py    YOLO inference -> GameState dataclass
│   ├── dataset_builder.py        Synthetic card dataset generator (52 classes)
│   ├── train_yolo.py             YOLOv8s training + validation + ONNX export
│   ├── opponent_tracker.py       OCR seat names, poll actions, detect bluffs
│   ├── hand_history_db.py        SQLite: hands / actions / opponents / showdowns
│   ├── decision_engine.py        GTO rule engine (equity + pot odds + exploit)
│   ├── action_executor.py        pyautogui UI controller with jitter timing
│   ├── autonomous_agent.py       Main self-playing loop
│   ├── poker_env.py              NLHE Gym environment (core RL training env)
│   ├── ppo_agent.py              PPO actor-critic with action masking
│   ├── self_play_trainer.py      3-phase curriculum: self-play -> league -> archetypes
│   ├── ascii_renderer.py         Terminal card/table renderer
│   ├── rl_live_bridge.py         Loads PPO checkpoint, hot-swaps into live agent
│   └── main.py                   Entry point for detection-only / calibration mode
├── notebooks/                # exploratory work and experiments
├── scripts/                  # one-off utility scripts
├── tests/                    # test suite
├── CLAUDE.md
├── Makefile
├── requirements.txt
└── setup.py
```

### Reference folder policy
`reference/` is the target architecture — read-only source of truth. When modifying code in `pokerStats/`, use reference files for data types, interfaces, and patterns. Do not copy them directly — adapt their logic to the existing `pokerStats/` structure. Example: `"refactor pokerStats/detector.py to use the GameState interface from reference/pokerstars_detector.py"`

---

## Key data types

### `GameState` (pokerstars_detector.py)
The central data object passed between all modules.
```python
@dataclass
class GameState:
    hole_cards:        list   # ["Ah", "Ks"]
    community_cards:   list   # up to 5, in deal order [flop1, flop2, flop3, turn, river]
    pot:               float
    hero_stack:        float
    hero_bet:          float
    players:           dict   # seat_num -> PlayerState
    street:            str    # "preflop" | "flop" | "turn" | "river"
    dealer_seat:       int
    hero_position:     str    # "btn" | "co" | "hj" | "mp" | "ep" | "sb" | "bb"
    available_actions: list   # ["fold", "call", "raise"]
    call_amount:       float
    min_raise:         float
    confidence:        float  # 0–1, ratio of cards successfully detected
    timestamp:         float
```

### `Decision` (decision_engine.py)
Output of both the rule engine and RL bridge.
```python
@dataclass
class Decision:
    action:     str    # "fold" | "call" | "check" | "raise" | "allin"
    amount:     float  # absolute raise size; 0 for fold/call/check
    reasoning:  str    # human-readable explanation
    confidence: float
```

### `OpponentProfile` (hand_history_db.py)
Built from SQLite hand history, queried per hand.
```python
@dataclass
class OpponentProfile:
    name:             str
    hands_played:     int
    vpip:             float   # voluntarily put $ in pot %
    pfr:              float   # preflop raise %
    three_bet:        float
    fold_to_3bet:     float
    cbet_flop:        float
    fold_to_cbet:     float
    bluffs_seen:      int
    bluffs_caught:    int
    total_showdowns:  int
    avg_bluff_size:   float   # as fraction of pot
    avg_value_size:   float
    # Computed properties: bluff_frequency, wtsd, player_type, exploit_note()
```

### `Action` enum (poker_env.py)
Used throughout the RL environment and PPO agent.
```python
class Action(IntEnum):
    FOLD       = 0
    CHECK      = 1
    CALL       = 2
    RAISE_25   = 3   # 25% of pot
    RAISE_50   = 4   # 50% of pot
    RAISE_100  = 5   # pot-sized
    ALL_IN     = 6
```

---

## ROI coordinates (pokerstars_capture.py)

All screen regions are `(left, top, width, height)` in pixels, relative to the primary monitor, for the **default PokerStars 6-max table at 1920×1080**.

If you change resolution or table skin, update `POKERSTARS_ROIS` and `SEAT_NAME_ROIS` / `SEAT_ACTION_ROIS` / `SEAT_CARD_ROIS` in `opponent_tracker.py`.

Run `python main.py --calibrate` to get a screenshot with all ROIs drawn — use this to verify alignment before any live run.

---

## Observation vector layout (poker_env.py)

The 143-dim obs vector fed to the PPO network:

| Slice     | Content                                 |
|-----------|-----------------------------------------|
| `[0:52]`  | Hero hole cards (one-hot, 52 cards)     |
| `[52:104]`| Community cards (one-hot)               |
| `[104:110]`| Per-player stacks, normalized           |
| `[110:116]`| Per-player current bets, normalized     |
| `[116:122]`| Per-player active flags                 |
| `[122:128]`| Per-player seat positions, normalized   |
| `[128:132]`| Pot, call amount, min raise, n_active   |
| `[132:136]`| Street one-hot [pre, flop, turn, river] |
| `[136:143]`| Last 7 actions (action_id / NUM_ACTIONS)|

**This layout must stay in sync between `poker_env.py` and `rl_live_bridge.py`.**
If you change obs dimensions, update `OBS_DIM` in both files.

---

## Database schema (hand_history_db.py)

SQLite at `data/hand_history.db`. Four tables:

- `hands` — one row per hand: hole cards, final board, pot, hero P&L, position
- `actions` — one row per action: hand_id, street, actor name, action type, amount, pot before
- `opponents` — one row per player name, running stats (VPIP/PFR/bluff freq etc.)
- `showdowns` — one row per showdown: was_bluff flag, bet sizing, opponent name

All stat columns in `opponents` are raw counts, **not percentages** — they get normalized in `OpponentProfile` computed properties. When updating stats, use `db.increment_stat(name, stat)`.

---

## Training curriculum (self_play_trainer.py)

Three phases configured in the `CFG` dict at the top of the file:

| Phase | Hands   | Opponents                        | Goal                              |
|-------|---------|----------------------------------|-----------------------------------|
| 1     | 50,000  | Copies of current agent          | Learn basic strategy              |
| 2     | 200,000 | League pool (frozen checkpoints) | Robustness, prevent collapse      |
| 3     | 50,000  | Scripted archetypes (nit/fish/maniac/tag) | Generalize to real humans |

League members are added every 2,000 hands if ELO improves by ≥10 points. Pool capped at 8 members (weakest evicted). Checkpoints saved to `checkpoints/` every 5,000 hands.

---

## How to run

```bash
# Verify ROI alignment (always do this first)
python main.py --calibrate

# Generate dataset + train YOLO
python train_yolo.py --mode train
python train_yolo.py --mode validate

# Run RL training
python self_play_trainer.py
python self_play_trainer.py --render      # with ASCII visualization
python self_play_trainer.py --resume checkpoints/ckpt_hand_00050000.pt

# Live agent — rule-based engine
python autonomous_agent.py --dry-run     # no clicks, safe test
python autonomous_agent.py               # live

# Live agent — trained RL policy
python rl_live_bridge.py --model checkpoints/best.pt --dry-run
python rl_live_bridge.py --model checkpoints/best.pt

# Opponent stats dashboard
python autonomous_agent.py --stats
```

---

## Common changes

### Swap to a different screen resolution
Edit `POKERSTARS_ROIS` in `pokerstars_capture.py` and the three `SEAT_*_ROIS` dicts in `opponent_tracker.py`. Run `--calibrate` to verify.

### Change number of table seats (2-max, 9-max)
Change `num_players` in `PokerEnv(...)` in `self_play_trainer.py`. Update seat ROI maps for the new layout. The obs vector seat slots `[104:128]` support up to 6 players — extend those slices if going to 9-max and update `OBS_DIM`.

### Tune the decision engine without retraining
Edit `GTO_THRESHOLDS` and `BET_SIZES` dicts in `decision_engine.py`. These are the equity thresholds and bet sizing fractions per street and position.

### Change bet sizing granularity
Edit the `Action` enum and `RAISE_FRACS` dict in `poker_env.py`, then update `NUM_ACTIONS` (currently 7). The PPO network output layer size in `ppo_agent.py` (`PokerNet.__init__`) must match `NUM_ACTIONS`.

### Add a new opponent stat
1. Add the column to `SCHEMA` in `hand_history_db.py`
2. Add the field to `OpponentProfile`
3. Increment it in `OpponentTracker._update_live_stats()`
4. Use it in `DecisionEngine.decide()` exploit overlay

### Adjust training speed / phases
Edit the `CFG` dict at the top of `self_play_trainer.py`. Key knobs: `phase1_hands`, `phase2_hands`, `phase3_hands`, `rollout_steps`, `lr`.

### Replace PPO with a different RL algorithm
Implement a class with `.get_action(obs, legal_mask)` and `.update(buffer)` matching the interface in `ppo_agent.py:PPOAgent`. Swap it in `self_play_trainer.py:SelfPlayTrainer.__init__`.

### Use the RL agent in dry-run (no clicks)
```bash
python rl_live_bridge.py --model checkpoints/best.pt --dry-run
```

---

## Conventions

- Card strings are always `"{rank}{suit}"` lowercase suit: `"Ah"`, `"Tc"`, `"2s"`
- Ranks: `2 3 4 5 6 7 8 9 T J Q K A` — `T` for ten, never `10`
- Suits: `c d h s` (clubs, diamonds, hearts, spades)
- Money is always in float dollars; pot/stack/bet are never normalized outside the obs vector
- All DB stats are **raw counts**, normalized at read time in `OpponentProfile`
- `dry_run=True` on `ActionExecutor` prints decisions without touching the mouse
- `render_mode="ascii"` on `PokerEnv` prints the full table to stdout each step