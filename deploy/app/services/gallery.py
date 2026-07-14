"""1:N gallery: SQLite metadata + embeddings + FAISS vector index."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class StoredFace:
    face_id: str
    person_id: str
    faiss_id: int
    created_at: datetime


@dataclass
class PersonRecord:
    person_id: str
    display_name: str
    metadata: dict
    created_at: datetime
    updated_at: datetime
    face_count: int


@dataclass
class SearchHit:
    faiss_id: int
    person_id: str
    display_name: str
    similarity: float


# =============================================================================
# GalleryStore: 人脸底库的持久化存储与向量搜索
# =============================================================================
# 架构设计: SQLite(元数据) + FAISS IndexFlatIP(相似度搜索)
#   - SQLite:  持久化存储人员信息(persons 表)和人脸特征(embedding BLOB, faces 表)
#   - FAISS:   内存向量索引, 做高速近似搜索
#
# 为什么是内积(Inner Product)而不是 L2 距离?
#   特征向量在入库和查询时都会做 L2 归一化(模长=1),
#   归一化后: 内积 = 余弦相似度(Cosine Similarity), 值域 [-1, 1]
#   值越接近 1 说明两个人脸越相似, 0.5+ 通常视为匹配
#
# 线程安全: 所有写操作(persons/faces 增删改 + FAISS 索引重建)都在 _lock 保护下进行
# 删除限制: FAISS IndexFlatIP 不支持单条删除, 所以删除后人脸后需要_重建整个索引_
# =============================================================================

class GalleryStore:
    # -------------------------------------------------------------------
    # 初始化步骤:
    #  1. data_dir 不存在则创建(通常是 ./data/gallery)
    #  2. _init_db:    建 SQLite 表(persons + faces), 如果已有则跳过 CREATE
    #  3. 创建 FAISS 索引: IndexFlatIP(embedding_dim), 内积索引
    #  4. _rebuild_index_from_db: 从 SQLite 中恢复之前存储的向量, 重建 FAISS 索引
    #     这样即使服务重启, 之前注册的人脸也不会丢失
    # -------------------------------------------------------------------
    def __init__(self, data_dir: str, embedding_dim: int = 512) -> None:
        self.data_dir = data_dir
        self.embedding_dim = embedding_dim
        self.db_path = os.path.join(data_dir, "gallery.db")
        self._lock = threading.RLock()
        os.makedirs(data_dir, exist_ok=True)
        self._init_db()
        self.index = faiss.IndexFlatIP(embedding_dim)
        self._rebuild_index_from_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    person_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS faces (
                    face_id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL,
                    faiss_id INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
                """
            )

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -------------------------------------------------------------------
    # _rebuild_index_from_db: 从 SQLite 的 faces 表恢复 FAISS 索引
    # 调用时机: 启动时 / delete_person / delete_face
    # 步骤:
    #   ① 按 faiss_id 升序读取所有已存储的 embedding BLOB
    #   ② 将 BLOB(bytes) 转为 float32 向量
    #   ③ 检查 faiss_id 是否连续(0,1,2,...), 不连续则修复 UPDATE
    #   ④ 所有向量 stack 成矩阵, L2 归一化(再次保证模长=1)
    #   ⑤ faiss.Index.add() 一次性批量加入索引
    # 注意: 这一步是 O(N·D) + FAISS add 的开销, 人脸数量很大时重建会变慢
    # -------------------------------------------------------------------
    def _rebuild_index_from_db(self) -> None:
        with self._lock:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT face_id, faiss_id, embedding FROM faces ORDER BY faiss_id ASC"
                ).fetchall()
            if not rows:
                return
            vectors = []
            for i, row in enumerate(rows):
                vec = np.frombuffer(row["embedding"], dtype=np.float32).reshape(1, -1)
                vectors.append(vec)
                if int(row["faiss_id"]) != i:
                    with self._connect() as conn:
                        conn.execute(
                            "UPDATE faces SET faiss_id = ? WHERE face_id = ?",
                            (i, row["face_id"]),
                        )
            matrix = np.vstack(vectors).astype(np.float32)
            faiss.normalize_L2(matrix)
            self.index.add(matrix)
            logger.info("FAISS index rebuilt: %d vectors", self.index.ntotal)

    @property
    def total_faces(self) -> int:
        return int(self.index.ntotal)

    def person_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) AS c FROM persons").fetchone()["c"])

    def create_person(
        self,
        person_id: str,
        display_name: str,
        metadata: dict | None = None,
    ) -> PersonRecord:
        now = _utcnow().isoformat()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._lock:
            with self._connect() as conn:
                try:
                    conn.execute(
                        """
                        INSERT INTO persons (person_id, display_name, metadata_json, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (person_id, display_name, meta_json, now, now),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"person_id already exists: {person_id}") from exc
        return self.get_person(person_id)

    def get_person(self, person_id: str) -> PersonRecord:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM persons WHERE person_id = ?", (person_id,)
            ).fetchone()
            if row is None:
                raise KeyError(person_id)
            face_count = conn.execute(
                "SELECT COUNT(*) AS c FROM faces WHERE person_id = ?", (person_id,)
            ).fetchone()["c"]
        return PersonRecord(
            person_id=row["person_id"],
            display_name=row["display_name"],
            metadata=json.loads(row["metadata_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            face_count=int(face_count),
        )

    def list_persons(self, offset: int = 0, limit: int = 100) -> tuple[list[PersonRecord], int]:
        with self._connect() as conn:
            total = int(conn.execute("SELECT COUNT(*) AS c FROM persons").fetchone()["c"])
            rows = conn.execute(
                """
                SELECT p.*, (SELECT COUNT(*) FROM faces f WHERE f.person_id = p.person_id) AS face_count
                FROM persons p
                ORDER BY p.created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        items = [
            PersonRecord(
                person_id=r["person_id"],
                display_name=r["display_name"],
                metadata=json.loads(r["metadata_json"]),
                created_at=datetime.fromisoformat(r["created_at"]),
                updated_at=datetime.fromisoformat(r["updated_at"]),
                face_count=int(r["face_count"]),
            )
            for r in rows
        ]
        return items, total

    # -------------------------------------------------------------------
    # delete_person: 删除一个人员及其所有关联的人脸
    # SQLite 中 faces 表设置了 FOREIGN KEY ... ON DELETE CASCADE,
    # 所以删除 persons 行时, 对应的 faces 行会自动级联删除
    # 删除后调用 _rebuild_index_from_db 重建 FAISS 索引(因为无法从 IndexFlatIP 中单删)
    # -------------------------------------------------------------------
    def delete_person(self, person_id: str) -> None:
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute("DELETE FROM persons WHERE person_id = ?", (person_id,))
                if cur.rowcount == 0:
                    raise KeyError(person_id)
            self._rebuild_index_from_db()

    # -------------------------------------------------------------------
    # add_face: 将一张新的人脸特征加入底库
    # 步骤:
    #   ① 校验 embedding 维度(必须等于 embedding_dim, 如 512)
    #   ② L2 归一化: 确保向量模长=1, 使内积=余弦相似度
    #   ③ 生成 UUID 作为 face_id, 分配 faiss_id = 当前 index 中的向量总数
    #   ④ SQLite 写入: INSERT faces(embedding 以 BLOB 存储, 省空间)
    #      同时 UPDATE persons 的 updated_at 时间戳
    #   ⑤ FAISS 在线添加: self.index.add(vec) 追加到内存索引
    #   ⑥ 返回 StoredFace 记录
    # -------------------------------------------------------------------
    def add_face(self, person_id: str, embedding: np.ndarray) -> StoredFace:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.embedding_dim:
            raise ValueError(f"embedding dim {vec.shape[0]} != {self.embedding_dim}")
        faiss.normalize_L2(vec.reshape(1, -1))
        blob = vec.tobytes()

        with self._lock:
            self.get_person(person_id)
            face_id = str(uuid.uuid4())
            now = _utcnow().isoformat()
            faiss_id = int(self.index.ntotal)

            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO faces (face_id, person_id, faiss_id, embedding, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (face_id, person_id, faiss_id, blob, now),
                )
                conn.execute(
                    "UPDATE persons SET updated_at = ? WHERE person_id = ?",
                    (now, person_id),
                )

            self.index.add(vec.reshape(1, -1))

        return StoredFace(
            face_id=face_id,
            person_id=person_id,
            faiss_id=faiss_id,
            created_at=datetime.fromisoformat(now),
        )

    # -------------------------------------------------------------------
    # delete_face: 删除单张人脸(需要同时提供 person_id 和 face_id 做双重校验)
    # 同 delete_person, 由于 FAISS IndexFlatIP 不支持删除, 所以也需要重建索引
    # -------------------------------------------------------------------
    def delete_face(self, person_id: str, face_id: str) -> None:
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM faces WHERE face_id = ? AND person_id = ?",
                    (face_id, person_id),
                )
                if cur.rowcount == 0:
                    raise KeyError(face_id)
                now = _utcnow().isoformat()
                conn.execute(
                    "UPDATE persons SET updated_at = ? WHERE person_id = ?",
                    (now, person_id),
                )
            self._rebuild_index_from_db()

    # -------------------------------------------------------------------
    # verify_against_person: 1:1 比对 — 探针向量与该人员所有注册模板取最高相似度
    # 用于闸机「先验身份再开门」场景: POST /v1/verify?person_id=xxx
    # 返回: (最高相似度, 最佳匹配的 face_id)
    # -------------------------------------------------------------------
    def verify_against_person(
        self, person_id: str, query: np.ndarray
    ) -> tuple[float, str | None]:
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.embedding_dim:
            raise ValueError(f"embedding dim {query.shape[0]} != {self.embedding_dim}")
        faiss.normalize_L2(query.reshape(1, -1))

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT face_id, embedding FROM faces WHERE person_id = ?",
                (person_id,),
            ).fetchall()

        if not rows:
            raise ValueError(f"person has no enrolled faces: {person_id}")

        best_sim = -1.0
        best_face_id: str | None = None
        for row in rows:
            vec = np.frombuffer(row["embedding"], dtype=np.float32).reshape(-1)
            sim = float(np.dot(query, vec))
            if sim > best_sim:
                best_sim = sim
                best_face_id = row["face_id"]

        return best_sim, best_face_id

    # -------------------------------------------------------------------
    # search: 1:N 向量搜索 - 在底库中查找与 query 最相似的人脸
    # ================================================================
    # 完整流程:
    #
    # ① 空库短路: 如果 FAISS 索引中没有向量, 直接返回空列表
    #
    # ② 查询向量归一化: query 做 L2 归一化, 保证与底库向量在同一超球面上
    #
    # ③ FAISS 内积搜索:
    #    - k = min(top_k * 10, total_faces)
    #    - 为什么要 top_k * 10? 因为同一人可能有多张人脸在底库,
    #      我们需要足够多的候选项才能在步骤⑤中去重, 取 top_k*10 作为召回池
    #    - index.search() 返回 (similarities, indices):
    #      similarities = 归一化内积, 即余弦相似度, 越接近 1 越像
    #      indices = 每个候选项在 FAISS 中的 ID
    #
    # ④ SQLite 关联查询:
    #    - 用 faiss_id 去 faces 表 JOIN persons 表, 查到所属人员信息
    #    - 如果查不到(可能数据不一致), 跳过该候选项
    #
    # ⑤ 按人员去重(每人只保留一个最高分):
    #    - 遍历所有候选项, 同一 person_id 只保留 similarity 最高的那个
    #    - 这样保证最终结果中每个人最多出现一次
    #
    # ⑥ 排序 & 截断:
    #    - 按相似度降序排列
    #    - 取前 top_k 个返回
    #
    # 返回值: list[SearchHit], 每个包含 faiss_id, person_id, display_name, similarity
    # ================================================================
    # -------------------------------------------------------------------
    def search(self, query: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        if self.index.ntotal == 0:
            return []

        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        k = min(max(top_k * 10, top_k), self.index.ntotal)
        similarities, indices = self.index.search(query, k)

        with self._connect() as conn:
            hits: list[SearchHit] = []
            for sim, faiss_id in zip(similarities[0], indices[0]):
                if faiss_id < 0:
                    continue
                row = conn.execute(
                    """
                    SELECT f.person_id, p.display_name
                    FROM faces f
                    JOIN persons p ON p.person_id = f.person_id
                    WHERE f.faiss_id = ?
                    """,
                    (int(faiss_id),),
                ).fetchone()
                if row is None:
                    continue
                hits.append(
                    SearchHit(
                        faiss_id=int(faiss_id),
                        person_id=row["person_id"],
                        display_name=row["display_name"],
                        similarity=float(sim),
                    )
                )

        best_by_person: dict[str, SearchHit] = {}
        for hit in hits:
            prev = best_by_person.get(hit.person_id)
            if prev is None or hit.similarity > prev.similarity:
                best_by_person[hit.person_id] = hit

        candidates = list(best_by_person.values())
        candidates.sort(key=lambda h: h.similarity, reverse=True)
        return candidates[:top_k]
