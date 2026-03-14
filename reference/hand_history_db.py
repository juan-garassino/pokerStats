"""
Hand history database
─────────────────────
SQLite backend storing:
  • Every hand played (cards, board, pot, result)
  • Every action per street per hand
  • Opponent profiles (stats built from history)
  • Bluff events (showdown reveals)
"""

import sqlite3
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

DB_PATH = Path("data/hand_history.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS hands (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL,
    hole_cards   TEXT,           -- JSON list ["Ah","Ks"]
    final_board  TEXT,           -- JSON list up to 5 cards
    pot_final    REAL,
    hero_result  REAL,           -- positive = won, negative = lost
    num_players  INTEGER,
    hero_position TEXT
);

CREATE TABLE IF NOT EXISTS actions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id      INTEGER REFERENCES hands(id),
    street       TEXT,           -- preflop/flop/turn/river
    actor        TEXT,           -- "hero" or opponent name
    action       TEXT,           -- fold/call/raise/check
    amount       REAL,
    pot_before   REAL,
    board_state  TEXT            -- JSON board at time of action
);

CREATE TABLE IF NOT EXISTS opponents (
    name         TEXT PRIMARY KEY,
    first_seen   REAL,
    last_seen    REAL,
    hands_played INTEGER DEFAULT 0,
    -- Preflop stats
    vpip         REAL DEFAULT 0,    -- voluntarily put $ in pot %
    pfr          REAL DEFAULT 0,    -- preflop raise %
    three_bet    REAL DEFAULT 0,
    fold_to_3bet REAL DEFAULT 0,
    -- Postflop stats
    cbet_flop    REAL DEFAULT 0,
    fold_to_cbet REAL DEFAULT 0,
    -- Bluff stats
    bluffs_seen      INTEGER DEFAULT 0,
    bluffs_caught    INTEGER DEFAULT 0,
    total_showdowns  INTEGER DEFAULT 0,
    -- Sizing tells
    avg_bluff_size   REAL DEFAULT 0,  -- as fraction of pot
    avg_value_size   REAL DEFAULT 0,
    -- Notes
    notes        TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS showdowns (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id      INTEGER REFERENCES hands(id),
    opponent     TEXT,
    their_cards  TEXT,           -- JSON if visible
    was_bluff    INTEGER,        -- 1 = bluff, 0 = value
    final_action TEXT,           -- last action before showdown
    pot_size     REAL,
    bet_size     REAL            -- size of their last bet
);

CREATE INDEX IF NOT EXISTS idx_actions_actor   ON actions(actor);
CREATE INDEX IF NOT EXISTS idx_actions_hand    ON actions(hand_id);
CREATE INDEX IF NOT EXISTS idx_showdowns_opp   ON showdowns(opponent);
"""


@dataclass
class OpponentProfile:
    name: str
    hands_played: int = 0
    vpip: float = 0.0
    pfr: float = 0.0
    three_bet: float = 0.0
    fold_to_3bet: float = 0.0
    cbet_flop: float = 0.0
    fold_to_cbet: float = 0.0
    bluffs_seen: int = 0
    bluffs_caught: int = 0
    total_showdowns: int = 0
    avg_bluff_size: float = 0.0
    avg_value_size: float = 0.0
    notes: str = ""

    @property
    def bluff_frequency(self) -> float:
        """Fraction of showdowns that were bluffs."""
        return self.bluffs_seen / max(self.total_showdowns, 1)

    @property
    def wtsd(self) -> float:
        """Went to showdown % (proxy for stickiness)."""
        return self.total_showdowns / max(self.hands_played, 1)

    @property
    def player_type(self) -> str:
        """Rough player classification."""
        if self.vpip < 0.15 and self.pfr < 0.12:
            return "nit"
        if self.vpip < 0.25 and self.pfr > 0.18:
            return "tag"      # tight-aggressive (solid)
        if self.vpip > 0.35 and self.pfr > 0.25:
            return "lag"      # loose-aggressive (tricky)
        if self.vpip > 0.40 and self.pfr < 0.15:
            return "fish"     # loose-passive (calling station)
        if self.bluff_frequency > 0.45:
            return "maniac"
        return "unknown"

    def exploit_note(self) -> str:
        """One-liner exploit recommendation."""
        pt = self.player_type
        if pt == "nit":
            return "fold to all pressure; only call with top 10%"
        if pt == "fish":
            return "value-bet thin; never bluff; call down light"
        if pt == "maniac":
            return f"call down wide (bluff_freq={self.bluff_frequency:.0%}); trap with strong hands"
        if pt == "tag":
            return "GTO baseline; exploit fold_to_cbet if >65%"
        if pt == "lag":
            return "tighten value range; don't bluff-catch light"
        return "collect more data"


class HandHistoryDB:
    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ── Hand lifecycle ────────────────────────────────────────────────────────
    def start_hand(self, hole_cards: list, num_players: int, position: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO hands (ts, hole_cards, final_board, pot_final, hero_result, num_players, hero_position) "
            "VALUES (?,?,?,?,?,?,?)",
            (time.time(), json.dumps(hole_cards), "[]", 0.0, 0.0, num_players, position)
        )
        self.conn.commit()
        return cur.lastrowid

    def finish_hand(self, hand_id: int, final_board: list, pot_final: float, hero_result: float):
        self.conn.execute(
            "UPDATE hands SET final_board=?, pot_final=?, hero_result=? WHERE id=?",
            (json.dumps(final_board), pot_final, hero_result, hand_id)
        )
        self.conn.commit()

    # ── Actions ───────────────────────────────────────────────────────────────
    def log_action(self, hand_id: int, street: str, actor: str,
                   action: str, amount: float, pot_before: float, board: list):
        self.conn.execute(
            "INSERT INTO actions (hand_id,street,actor,action,amount,pot_before,board_state) "
            "VALUES (?,?,?,?,?,?,?)",
            (hand_id, street, actor, action, amount, pot_before, json.dumps(board))
        )
        self.conn.commit()

    # ── Showdowns ─────────────────────────────────────────────────────────────
    def log_showdown(self, hand_id: int, opponent: str, their_cards: Optional[list],
                     was_bluff: bool, final_action: str, pot_size: float, bet_size: float):
        self.conn.execute(
            "INSERT INTO showdowns (hand_id,opponent,their_cards,was_bluff,final_action,pot_size,bet_size) "
            "VALUES (?,?,?,?,?,?,?)",
            (hand_id, opponent,
             json.dumps(their_cards) if their_cards else None,
             int(was_bluff), final_action, pot_size, bet_size)
        )
        self._refresh_opponent_stats(opponent)
        self.conn.commit()

    # ── Opponent registry ─────────────────────────────────────────────────────
    def upsert_opponent(self, name: str):
        self.conn.execute(
            "INSERT INTO opponents (name, first_seen, last_seen) VALUES (?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET last_seen=excluded.last_seen",
            (name, time.time(), time.time())
        )
        self.conn.commit()

    def increment_stat(self, name: str, stat: str, delta: float = 1.0):
        """Safely increment a numeric column in opponents table."""
        allowed = {
            "hands_played","vpip","pfr","three_bet","fold_to_3bet",
            "cbet_flop","fold_to_cbet","bluffs_seen","bluffs_caught","total_showdowns"
        }
        if stat not in allowed:
            raise ValueError(f"Unknown stat: {stat}")
        self.conn.execute(f"UPDATE opponents SET {stat}={stat}+? WHERE name=?", (delta, name))
        self.conn.commit()

    def _refresh_opponent_stats(self, name: str):
        """Recompute bluff sizing averages from showdown history."""
        rows = self.conn.execute(
            "SELECT was_bluff, bet_size, pot_size FROM showdowns WHERE opponent=? AND pot_size>0",
            (name,)
        ).fetchall()
        bluff_sizes = [r["bet_size"]/r["pot_size"] for r in rows if r["was_bluff"] and r["pot_size"]>0]
        value_sizes = [r["bet_size"]/r["pot_size"] for r in rows if not r["was_bluff"] and r["pot_size"]>0]
        avg_b = sum(bluff_sizes)/len(bluff_sizes) if bluff_sizes else 0.0
        avg_v = sum(value_sizes)/len(value_sizes) if value_sizes else 0.0
        self.conn.execute(
            "UPDATE opponents SET avg_bluff_size=?, avg_value_size=? WHERE name=?",
            (avg_b, avg_v, name)
        )

    def get_opponent(self, name: str) -> Optional[OpponentProfile]:
        row = self.conn.execute("SELECT * FROM opponents WHERE name=?", (name,)).fetchone()
        if not row:
            return None
        return OpponentProfile(**{k: row[k] for k in row.keys() if k in OpponentProfile.__dataclass_fields__})

    def get_all_opponents(self) -> list:
        rows = self.conn.execute("SELECT * FROM opponents ORDER BY hands_played DESC").fetchall()
        return [OpponentProfile(**{k: r[k] for k in r.keys() if k in OpponentProfile.__dataclass_fields__})
                for r in rows]

    def get_recent_actions(self, opponent: str, n: int = 20) -> list:
        """Get the last N actions from a specific player — useful for pattern detection."""
        rows = self.conn.execute(
            "SELECT a.street, a.action, a.amount, a.pot_before, a.board_state "
            "FROM actions a WHERE a.actor=? ORDER BY a.id DESC LIMIT ?",
            (opponent, n)
        ).fetchall()
        return [dict(r) for r in rows]
