"""
莉莉丝 — 向量化长期记忆存储

基于 sqlite-vec 的语义记忆系统，替代原有的关键词匹配方案。

架构:
  - memories 表: 记忆元数据（内容、类型、重要性、时间戳等）
  - memory_embeddings vec0 表: 向量索引，用于语义检索
  - 嵌入模型: all-MiniLM-L6-v2 (384维)，惰性加载

用法:
    from lilith_bot.memory_store import get_memory_store
    store = get_memory_store()
    store.add_memory("主人喜欢喝冰美式", "knowledge")
    results = store.search("主人的饮品偏好", limit=5)
"""

import os
import json
import struct
import sqlite3
import threading
from datetime import datetime
from typing import Optional, List, Dict

import sqlite_vec

# ─── 配置 ─────────────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_PROJECT_ROOT, "lilith_memory.db")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

# ─── 全局单例 ─────────────────────────────────────────────

_store_lock = threading.Lock()
_store_instance: Optional["LilithMemoryStore"] = None


def get_memory_store() -> "LilithMemoryStore":
    """获取全局唯一的 LilithMemoryStore 实例"""
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                _store_instance = LilithMemoryStore(DB_PATH)
    return _store_instance


# ─── 嵌入模型懒加载 ───────────────────────────────────────

_embedding_model = None
_embedding_lock = threading.Lock()


def _get_embedding_model():
    """惰性加载 SentenceTransformer 嵌入模型（线程安全）"""
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                print("[Memory] 加载嵌入模型 all-MiniLM-L6-v2 ...")
                from sentence_transformers import SentenceTransformer
                _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                print("[Memory] 嵌入模型加载完成 ✓")
    return _embedding_model


def preload_embedding_model():
    """预加载嵌入模型，避免首次调用卡顿"""
    from sentence_transformers import SentenceTransformer
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def encode_text(text: str) -> bytes:
    """将文本编码为 384 维 float32 向量（bytes）"""
    model = _get_embedding_model()
    vec = model.encode(text, normalize_embeddings=True)
    return _serialize_vec(vec)


def _serialize_vec(vec) -> bytes:
    """numpy array → sqlite-vec 兼容的 bytes"""
    return struct.pack(f"{len(vec)}f", *vec.astype("float32"))


# ─── 记忆存储类 ───────────────────────────────────────────

class LilithMemoryStore:
    """
    向量化长期记忆存储。

    表结构:
      memories (id, content, memory_type, importance, created_at,
                last_recalled_at, recall_count)
      memory_embeddings vec0 (embedding float[384])  -- rowid 与 memories.id 一一对应
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接（惰性创建）"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
        return self._conn

    def _init_db(self):
        """初始化数据库表结构"""
        conn = self._get_conn()

        # 记忆元数据表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL DEFAULT 'knowledge',
                importance REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                last_recalled_at TEXT,
                recall_count INTEGER NOT NULL DEFAULT 0
            )
        """)

        # 向量索引表（sqlite-vec）
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
                embedding float[{EMBEDDING_DIM}]
            )
        """)

        # 情绪快照表（为情绪引擎铺路）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mood_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mood_json TEXT NOT NULL,
                trigger_event TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
        """)

        conn.commit()

    # ─── CRUD ──────────────────────────────────────────────

    def add_memory(self, content: str, memory_type: str = "knowledge",
                   importance: float = 0.5) -> int:
        """
        添加一条记忆，自动生成向量。

        Args:
            content: 记忆内容（一句判断句）
            memory_type: knowledge | emotional | event | skill
            importance: 重要性 0~1

        Returns:
            新记忆的 id
        """
        embedding_bytes = encode_text(content)
        conn = self._get_conn()

        # 插入元数据
        cursor = conn.execute(
            "INSERT INTO memories (content, memory_type, importance) VALUES (?, ?, ?)",
            (content, memory_type, importance),
        )
        memory_id = cursor.lastrowid

        # 插入向量（rowid 必须与 memories.id 一致）
        conn.execute(
            "INSERT INTO memory_embeddings (rowid, embedding) VALUES (?, ?)",
            (memory_id, embedding_bytes),
        )
        conn.commit()
        return memory_id

    def add_memories_batch(self, items: List[dict]) -> List[int]:
        """
        批量添加记忆。

        Args:
            items: [{"content": ..., "type": ..., "importance": ...}, ...]
        """
        ids = []
        for item in items:
            mid = self.add_memory(
                content=item["content"],
                memory_type=item.get("type", "knowledge"),
                importance=item.get("importance", 0.5),
            )
            ids.append(mid)
        return ids

    def search(self, query: str, limit: int = 5) -> List[dict]:
        """
        语义搜索：根据查询文本找到最相关的记忆。

        Args:
            query: 查询文本（用户当前消息）
            limit: 返回条数

        Returns:
            [{"id": 42, "content": "...", "type": "knowledge", "distance": 0.123}, ...]
        """
        query_vec = encode_text(query)
        conn = self._get_conn()

        # sqlite-vec 向量检索
        rows = conn.execute(
            f"""
            SELECT
                m.id, m.content, m.memory_type, m.importance,
                m.created_at, m.last_recalled_at, m.recall_count,
                v.distance
            FROM memory_embeddings v
            JOIN memories m ON m.id = v.rowid
            WHERE v.embedding MATCH ?
            ORDER BY v.distance
            LIMIT ?
            """,
            (query_vec, limit),
        ).fetchall()

        results = []
        for row in rows:
            mid, content, mtype, imp, created, recalled, count, dist = row
            results.append({
                "id": mid,
                "content": content,
                "type": mtype,
                "importance": imp,
                "created_at": created,
                "last_recalled_at": recalled,
                "recall_count": count,
                "distance": dist,
            })
        return results

    def recall(self, query: str, limit: int = 5,
               distance_threshold: float = 0.8) -> List[str]:
        """
        召回记忆的便捷方法：返回格式化文本列表，并更新召回统计。

        Args:
            query: 用户当前消息
            limit: 最大召回数
            distance_threshold: 余弦距离阈值（越小越相关），超过此值不召回
        """
        results = self.search(query, limit)
        recalled = []
        recalled_ids = []

        for r in results:
            if r["distance"] < distance_threshold:
                recalled.append(f"[{r['type']}] {r['content']}")
                recalled_ids.append(r["id"])

        # 更新召回统计
        if recalled_ids:
            self._bump_recall(recalled_ids)

        return recalled

    def _bump_recall(self, memory_ids: List[int]):
        """更新记忆的召回计数与时间"""
        conn = self._get_conn()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        placeholders = ",".join("?" * len(memory_ids))
        conn.execute(
            f"UPDATE memories SET recall_count = recall_count + 1, "
            f"last_recalled_at = ? WHERE id IN ({placeholders})",
            [now] + memory_ids,
        )
        conn.commit()

    def get_all_memories(self, memory_type: str = None) -> List[dict]:
        """获取所有记忆（可按类型过滤）"""
        conn = self._get_conn()
        if memory_type:
            rows = conn.execute(
                "SELECT id, content, memory_type, importance, created_at, "
                "last_recalled_at, recall_count FROM memories WHERE memory_type = ? "
                "ORDER BY created_at DESC",
                (memory_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, content, memory_type, importance, created_at, "
                "last_recalled_at, recall_count FROM memories "
                "ORDER BY created_at DESC",
            ).fetchall()

        return [
            {
                "id": r[0], "content": r[1], "type": r[2],
                "importance": r[3], "created_at": r[4],
                "last_recalled_at": r[5], "recall_count": r[6],
            }
            for r in rows
        ]

    def count(self) -> int:
        """返回记忆总数"""
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def delete_memory(self, memory_id: int) -> bool:
        """删除一条记忆及其向量"""
        conn = self._get_conn()
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (memory_id,))
        conn.commit()
        return True

    # ─── 情绪快照 ──────────────────────────────────────────

    def save_mood(self, mood_data: dict, trigger: str = None):
        """保存情绪快照"""
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO mood_snapshots (mood_json, trigger_event) VALUES (?, ?)",
            (json.dumps(mood_data, ensure_ascii=False), trigger),
        )
        conn.commit()

    def latest_mood(self) -> Optional[dict]:
        """获取最近一次情绪快照"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT mood_json FROM mood_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    # ─── 统计与维护 ────────────────────────────────────────

    def stats(self) -> dict:
        """记忆库统计信息"""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        by_type = conn.execute(
            "SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type"
        ).fetchall()
        return {
            "total": total,
            "by_type": dict(by_type),
            "db_path": self.db_path,
        }

    def close(self):
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None


    # ═══════════════════════════════════════════════════════════
    # 里程碑管理
    # ═══════════════════════════════════════════════════════════

    def add_milestone(self, milestone: str) -> bool:
        """添加关系里程碑 (去重)"""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS milestones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL UNIQUE,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)
        try:
            conn.execute("INSERT INTO milestones (content) VALUES (?)", (milestone,))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_milestones(self) -> list:
        """获取所有里程碑"""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS milestones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL UNIQUE,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)
        rows = conn.execute(
            "SELECT content FROM milestones ORDER BY id ASC"
        ).fetchall()
        return [r[0] for r in rows]

    # ═══════════════════════════════════════════════════════════
    # 情感记忆查询
    # ═══════════════════════════════════════════════════════════

    def get_emotional_memories(self, limit: int = 10) -> list:
        """获取情感类记忆"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, content, created_at FROM memories "
            "WHERE memory_type = 'emotional' "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"id": r[0], "content": r[1], "created_at": r[2]}
            for r in rows
        ]

    def get_mood_snapshots(self, limit: int = 10) -> list:
        """获取最近的心情快照"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT mood_json, trigger_event, created_at "
            "FROM mood_snapshots ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"mood": json.loads(r[0]), "trigger": r[1], "time": r[2]}
            for r in rows
        ]
