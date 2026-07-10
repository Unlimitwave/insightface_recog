"""识别/验证事件持久化与统计查询（SQLite，与底库同目录）。"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RecognitionEvent:
    """单条识别/验证事件记录。"""

    event_id: str
    request_id: str
    event_type: str  # identify | verify
    created_at: datetime
    matched: bool
    is_stranger: bool
    person_id: str | None
    display_name: str | None
    similarity: float | None
    threshold: float
    liveness_passed: bool | None
    latency_ms: float | None
    metadata: dict


@dataclass
class DailyEventStats:
    date: str
    total: int
    matched: int
    strangers: int
    verify_pass: int


@dataclass
class PersonActivity:
    person_id: str
    display_name: str
    event_count: int
    last_seen_at: datetime | None


@dataclass
class EventStatsSummary:
    period_days: int
    total_events: int
    identify_count: int
    verify_count: int
    matched_count: int
    stranger_count: int
    pass_rate: float
    daily: list[DailyEventStats]
    top_persons: list[PersonActivity]


# =============================================================================
# EventStore: 识别/验证事件审计日志
# =============================================================================
# 设计:
#   - 与 GalleryStore 共用 data_dir，独立 events.db，避免改动底库表结构
#   - identify / verify 成功后异步写入（同步 INSERT，开销 <1ms）
#   - 支持按时间/类型/人员/陌生人筛选，供 /v1/events 与 /v1/stats 查询
# =============================================================================


class EventStore:
    def __init__(self, data_dir: str, retention_days: int = 90) -> None:
        self.data_dir = data_dir
        self.retention_days = retention_days
        self.db_path = os.path.join(data_dir, "events.db")
        self._lock = threading.RLock()
        os.makedirs(data_dir, exist_ok=True)
        self._init_db()
        self._purge_expired()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    matched INTEGER NOT NULL,
                    is_stranger INTEGER NOT NULL DEFAULT 0,
                    person_id TEXT,
                    display_name TEXT,
                    similarity REAL,
                    threshold REAL NOT NULL,
                    liveness_passed INTEGER,
                    latency_ms REAL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);
                CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
                CREATE INDEX IF NOT EXISTS idx_events_person ON events(person_id);
                CREATE INDEX IF NOT EXISTS idx_events_stranger ON events(is_stranger);
                """
            )

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _purge_expired(self) -> None:
        if self.retention_days <= 0:
            return
        cutoff = (_utcnow() - timedelta(days=self.retention_days)).isoformat()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute("DELETE FROM events WHERE created_at < ?", (cutoff,))
                if cur.rowcount:
                    logger.info("Purged %d expired events (older than %d days)", cur.rowcount, self.retention_days)

    def record(
        self,
        *,
        request_id: str,
        event_type: str,
        matched: bool,
        is_stranger: bool,
        threshold: float,
        person_id: str | None = None,
        display_name: str | None = None,
        similarity: float | None = None,
        liveness_passed: bool | None = None,
        latency_ms: float | None = None,
        metadata: dict | None = None,
    ) -> RecognitionEvent:
        event_id = str(uuid.uuid4())
        now = _utcnow().isoformat()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO events (
                        event_id, request_id, event_type, created_at,
                        matched, is_stranger, person_id, display_name,
                        similarity, threshold, liveness_passed, latency_ms, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        request_id,
                        event_type,
                        now,
                        int(matched),
                        int(is_stranger),
                        person_id,
                        display_name,
                        similarity,
                        threshold,
                        int(liveness_passed) if liveness_passed is not None else None,
                        latency_ms,
                        meta_json,
                    ),
                )

        return RecognitionEvent(
            event_id=event_id,
            request_id=request_id,
            event_type=event_type,
            created_at=datetime.fromisoformat(now),
            matched=matched,
            is_stranger=is_stranger,
            person_id=person_id,
            display_name=display_name,
            similarity=similarity,
            threshold=threshold,
            liveness_passed=liveness_passed,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

    def list_events(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        event_type: str | None = None,
        person_id: str | None = None,
        is_stranger: bool | None = None,
        since: datetime | None = None,
    ) -> tuple[list[RecognitionEvent], int]:
        clauses: list[str] = []
        params: list = []

        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if person_id:
            clauses.append("person_id = ?")
            params.append(person_id)
        if is_stranger is not None:
            clauses.append("is_stranger = ?")
            params.append(int(is_stranger))
        if since:
            clauses.append("created_at >= ?")
            params.append(since.isoformat())

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._connect() as conn:
            total = int(
                conn.execute(f"SELECT COUNT(*) AS c FROM events {where}", params).fetchone()["c"]
            )
            rows = conn.execute(
                f"""
                SELECT * FROM events {where}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()

        return [self._row_to_event(r) for r in rows], total

    def stats_summary(self, days: int = 7, top_persons_limit: int = 20) -> EventStatsSummary:
        since = _utcnow() - timedelta(days=days)
        since_iso = since.isoformat()

        with self._connect() as conn:
            agg = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN event_type = 'identify' THEN 1 ELSE 0 END) AS identify_count,
                    SUM(CASE WHEN event_type = 'verify' THEN 1 ELSE 0 END) AS verify_count,
                    SUM(CASE WHEN matched = 1 THEN 1 ELSE 0 END) AS matched_count,
                    SUM(CASE WHEN is_stranger = 1 THEN 1 ELSE 0 END) AS stranger_count
                FROM events
                WHERE created_at >= ?
                """,
                (since_iso,),
            ).fetchone()

            daily_rows = conn.execute(
                """
                SELECT
                    substr(created_at, 1, 10) AS day,
                    COUNT(*) AS total,
                    SUM(CASE WHEN matched = 1 THEN 1 ELSE 0 END) AS matched,
                    SUM(CASE WHEN is_stranger = 1 THEN 1 ELSE 0 END) AS strangers,
                    SUM(CASE WHEN event_type = 'verify' AND matched = 1 THEN 1 ELSE 0 END) AS verify_pass
                FROM events
                WHERE created_at >= ?
                GROUP BY day
                ORDER BY day ASC
                """,
                (since_iso,),
            ).fetchall()

            person_rows = conn.execute(
                """
                SELECT
                    person_id,
                    display_name,
                    COUNT(*) AS event_count,
                    MAX(created_at) AS last_seen_at
                FROM events
                WHERE created_at >= ?
                  AND person_id IS NOT NULL
                  AND matched = 1
                GROUP BY person_id
                ORDER BY event_count DESC
                LIMIT ?
                """,
                (since_iso, top_persons_limit),
            ).fetchall()

        total = int(agg["total"] or 0)
        identify_count = int(agg["identify_count"] or 0)
        matched_count = int(agg["matched_count"] or 0)
        pass_rate = round(matched_count / identify_count, 4) if identify_count > 0 else 0.0

        daily = [
            DailyEventStats(
                date=r["day"],
                total=int(r["total"]),
                matched=int(r["matched"]),
                strangers=int(r["strangers"]),
                verify_pass=int(r["verify_pass"]),
            )
            for r in daily_rows
        ]

        top_persons = [
            PersonActivity(
                person_id=r["person_id"],
                display_name=r["display_name"] or r["person_id"],
                event_count=int(r["event_count"]),
                last_seen_at=datetime.fromisoformat(r["last_seen_at"]) if r["last_seen_at"] else None,
            )
            for r in person_rows
        ]

        return EventStatsSummary(
            period_days=days,
            total_events=total,
            identify_count=identify_count,
            verify_count=int(agg["verify_count"] or 0),
            matched_count=matched_count,
            stranger_count=int(agg["stranger_count"] or 0),
            pass_rate=pass_rate,
            daily=daily,
            top_persons=top_persons,
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> RecognitionEvent:
        return RecognitionEvent(
            event_id=row["event_id"],
            request_id=row["request_id"],
            event_type=row["event_type"],
            created_at=datetime.fromisoformat(row["created_at"]),
            matched=bool(row["matched"]),
            is_stranger=bool(row["is_stranger"]),
            person_id=row["person_id"],
            display_name=row["display_name"],
            similarity=row["similarity"],
            threshold=float(row["threshold"]),
            liveness_passed=bool(row["liveness_passed"]) if row["liveness_passed"] is not None else None,
            latency_ms=row["latency_ms"],
            metadata=json.loads(row["metadata_json"]),
        )
