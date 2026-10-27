"""Persistent memory: previous meetings + pending actions (SQLite)."""
import json
import os
import sqlite3
from datetime import date, datetime


class Memory:
    def __init__(self, path="data/meetings.db"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS meetings(
            id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, date TEXT, transcript TEXT,
            summary TEXT, topics TEXT, participants TEXT, decisions TEXT);
        CREATE TABLE IF NOT EXISTS tasks(
            id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id INTEGER, task TEXT, owner TEXT,
            due TEXT, priority TEXT DEFAULT 'medium', status TEXT DEFAULT 'open',
            created_at TEXT, updated_at TEXT);
        """)
        self.conn.commit()

    # ---- meetings ----
    def save_meeting(self, title, transcript, summary, topics, participants, decisions):
        cur = self.conn.execute(
            "INSERT INTO meetings(title,date,transcript,summary,topics,participants,decisions) VALUES(?,?,?,?,?,?,?)",
            (title, date.today().isoformat(), transcript, summary, json.dumps(topics),
             json.dumps(participants), json.dumps(decisions)))
        self.conn.commit()
        return cur.lastrowid

    def _meeting(self, r):
        d = dict(r)
        for k in ("topics", "participants", "decisions"):
            d[k] = json.loads(d[k] or "[]")
        return d

    def recent_meetings(self, n=5):
        rows = self.conn.execute("SELECT * FROM meetings ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [self._meeting(r) for r in rows]

    def get_meeting(self, mid):
        r = self.conn.execute("SELECT * FROM meetings WHERE id=?", (mid,)).fetchone()
        return self._meeting(r) if r else None

    def search_meetings(self, q, n=3):
        like = f"%{q}%"
        rows = self.conn.execute(
            "SELECT * FROM meetings WHERE title LIKE ? OR summary LIKE ? OR transcript LIKE ? ORDER BY id DESC LIMIT ?",
            (like, like, like, n)).fetchall()
        return [self._meeting(r) for r in rows]

    # ---- tasks ----
    def add_task(self, meeting_id, task, owner, due, priority="medium"):
        now = datetime.now().isoformat(timespec="seconds")
        cur = self.conn.execute(
            "INSERT INTO tasks(meeting_id,task,owner,due,priority,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
            (meeting_id, task, owner, due, priority, "open", now, now))
        self.conn.commit()
        return cur.lastrowid

    def list_tasks(self, status=None, owner=None, pending_only=False):
        q, args = "SELECT * FROM tasks WHERE 1=1", []
        if pending_only:
            q += " AND status IN ('open','in_progress')"
        elif status:
            q += " AND status=?"; args.append(status)
        if owner:
            q += " AND LOWER(owner)=LOWER(?)"; args.append(owner)
        q += " ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, COALESCE(due,'9999'), id"
        return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    def get_task(self, tid):
        r = self.conn.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()
        return dict(r) if r else None

    def update_task(self, tid, **fields):
        allowed = {k: v for k, v in fields.items() if k in ("owner", "due", "status", "priority", "task")}
        if not allowed or not self.get_task(tid):
            return False
        allowed["updated_at"] = datetime.now().isoformat(timespec="seconds")
        sets = ",".join(f"{k}=?" for k in allowed)
        self.conn.execute(f"UPDATE tasks SET {sets} WHERE id=?", (*allowed.values(), tid))
        self.conn.commit()
        return True

    def workload(self):
        rows = self.conn.execute(
            "SELECT owner, COUNT(*) c FROM tasks WHERE status IN ('open','in_progress') AND owner IS NOT NULL GROUP BY owner").fetchall()
        return {r["owner"]: r["c"] for r in rows}
