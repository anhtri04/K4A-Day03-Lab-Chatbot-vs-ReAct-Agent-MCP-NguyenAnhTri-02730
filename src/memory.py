"""Short-term (SQLite, window of 20) + long-term (MEMORY.md) memory.

Arch Linux note:
- Uses only stdlib `sqlite3` (backed by system sqlite 3.53+ on Arch, verified).
- DB parent dir is auto-created; connection sets:
    PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;
  WAL avoids "database is locked" on quick successive REPL turns.
- All timestamps are UTC ISO-8601. No extra dependency (no aiosqlite / SQLAlchemy).
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'untitled',
    created_at TEXT NOT NULL,
    total_tokens INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    tool_calls_json TEXT,
    created_at TEXT NOT NULL,
    total_tokens INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, id);
"""


class ShortTermMemory:
    """SQLite-backed per-session message store with a sliding window of N latest."""

    def __init__(self, db_path: Path, window: int = 20, model: str = ""):
        self.db_path = Path(db_path).expanduser()
        self.window = window
        self.model = model
        # Arch: ensure parent exists (e.g. ./cli-agent/) before connect.
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            # Lightweight migration for DBs created before token columns.
            for table, col in (("sessions", "total_tokens"), ("messages", "total_tokens")):
                cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
                if col not in cols:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0")

    # -- sessions ---------------------------------------------------------
    def create_session(self, title: str = "untitled") -> str:
        sid = uuid.uuid4().hex[:12]
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
                (sid, title.strip() or "untitled", _utcnow()),
            )
        return sid

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT s.id, s.title, s.created_at,
                          COALESCE(s.total_tokens, 0) AS total_tokens,
                          (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS n_msgs,
                          (SELECT COALESCE(SUM(m.total_tokens), 0)
                             FROM messages m WHERE m.session_id = s.id) AS msg_tokens
                    FROM sessions s ORDER BY s.created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Single-session getter for status line: title + token total."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT s.id, s.title, s.created_at,
                          COALESCE(s.total_tokens, 0) AS total_tokens,
                          (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS n_msgs
                    FROM sessions s WHERE s.id = ?""",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def session_exists(self, session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return row is not None

    def delete_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    # -- messages ----------------------------------------------------------
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls_json: str | None = None,
        total_tokens: int | None = None,
    ) -> int:
        if not self.session_exists(session_id):
            raise ValueError(f"unknown session_id={session_id!r}")
        if total_tokens is None:
            try:
                from tokens import count_tokens  # local helper, tiktoken-exact

                total_tokens = count_tokens(content or "", self.model)
            except Exception:
                total_tokens = len(content or "") // 4
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_calls_json, created_at,"
                " total_tokens) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, role, content or "", tool_calls_json, _utcnow(), total_tokens),
            )
            conn.execute(
                "UPDATE sessions SET total_tokens = COALESCE(total_tokens, 0) + ? WHERE id = ?",
                (total_tokens, session_id),
            )
            return int(cur.lastrowid)

    def get_window(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Return the latest N messages, oldest-first (for LLM context)."""
        n = limit or self.window
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, tool_calls_json, created_at FROM messages"
                " WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, n),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def count(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()
        return int(row["c"])

    def recount_session(self, session_id: str) -> int:
        """Backfill per-message + session totals (e.g. after migration)."""
        try:
            from tokens import count_tokens
        except Exception:
            count_tokens = lambda t, m="": len(t or "") // 4  # type: ignore
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, content FROM messages WHERE session_id = ?", (session_id,)
            ).fetchall()
            total = 0
            for r in rows:
                n = count_tokens(r["content"] or "", self.model)
                conn.execute("UPDATE messages SET total_tokens = ? WHERE id = ?", (n, r["id"]))
                total += n
            conn.execute("UPDATE sessions SET total_tokens = ? WHERE id = ?", (total, session_id))
        return total


class LongTermMemory:
    """Markdown-file memory. Agent reads via read tool, writes via add_memory tool."""

    def __init__(self, path: Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(
                "# Long-Term Memory\n\n> Agent-appended facts about user/project.\n",
                encoding="utf-8",
            )

    def read(self, limit_chars: int = 4000) -> str:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        if len(text) > limit_chars:
            return text[-limit_chars:]  # keep most recent tail
        return text

    def append(self, reason_to_add: str, memory_to_add: str) -> str:
        reason = (reason_to_add or "").strip() or "note"
        memory = (memory_to_add or "").strip()
        if not memory:
            return "Skipped: empty memory_to_add."
        entry = f"\n## {_utcnow()} — {reason}\n{memory}\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(entry)
        return f"Saved to {self.path.name}: [{reason}] {memory[:120]}"
