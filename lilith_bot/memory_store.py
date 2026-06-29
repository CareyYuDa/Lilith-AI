"""
莉莉丝 — 向量化长期记忆存储

基于 sqlite-vec 的语义记忆系统，替代原有的关键词匹配方案。

架构:
  - memories 表: 记忆元数据（内容、类型、重要性、时间戳等）
  - memory_embeddings vec0 表: 向量索引，用于语义检索
  - activity_log 表: 短期活动日志（24h 环形，可归档到 memories）
  - person_knowledge 表: 莉莉丝对不同人的认知（portrait、tags、信任/亲密/熟悉度）
  - mood_snapshots 表: 情绪快照
  - milestones 表: 关系里程碑
  - 嵌入模型: all-MiniLM-L6-v2 (384维)，惰性加载

用法:
    from lilith_bot.memory_store import get_memory_store
    store = get_memory_store()
    store.add_memory("对方喜欢喝冰美式", "knowledge")
    results = store.search("对方的饮品偏好", limit=5)
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


def _get_embedding_model(timeout: float = 30):
    """惰性加载 SentenceTransformer 嵌入模型（线程安全，带超时）

    Args:
        timeout: 最大等待秒数，超过则返回 None（后续请求会继续尝试）

    Returns:
        模型实例，或 None（加载超时/失败）
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    with _embedding_lock:
        if _embedding_model is not None:
            return _embedding_model

        print("[Memory] 加载嵌入模型 all-MiniLM-L6-v2 ...")
        result = [None]
        error = [None]
        event = threading.Event()

        def _load():
            try:
                from sentence_transformers import SentenceTransformer
                result[0] = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                error[0] = e
            finally:
                event.set()

        t = threading.Thread(target=_load, daemon=True)
        t.start()
        loaded = event.wait(timeout=timeout)

        if loaded and result[0] is not None:
            _embedding_model = result[0]
            print("[Memory] 嵌入模型加载完成 ✓")
        else:
            msg = str(error[0]) if error[0] else f"加载超时（>{timeout}s）"
            print(f"[Memory] 嵌入模型加载失败: {msg}，下次请求会重试")
            return None

    return _embedding_model


def preload_embedding_model():
    """预加载嵌入模型（带超时，供 server.py 启动时调用）"""
    return _get_embedding_model(timeout=60)


def encode_text(text: str) -> bytes:
    """将文本编码为 384 维 float32 向量（bytes）"""
    model = _get_embedding_model()
    if model is None:
        raise RuntimeError("embedding model not loaded")
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

        # 活动日志表（短期意识流，24h 环形）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,          -- user | lilith_chat | lilith_channel | lilith_internal
                summary TEXT NOT NULL,         -- LLM 压缩后的摘要（1句话）
                detail TEXT,                   -- 原始文本（可选，调试用）
                person TEXT DEFAULT '对方',    -- 互动对象
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                created_date TEXT NOT NULL DEFAULT (date('now'))
            )
        """)
        # 兼容旧表：如果 person 列不存在则添加
        try:
            conn.execute("ALTER TABLE activity_log ADD COLUMN person TEXT DEFAULT '对方'")
        except Exception:
            pass  # 列已存在

        # 人物认知表（莉莉丝对不同人的认知）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS person_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT NOT NULL UNIQUE,
                portrait TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '[]',
                trust REAL NOT NULL DEFAULT 0.5,
                intimacy REAL NOT NULL DEFAULT 0.3,
                familiarity REAL NOT NULL DEFAULT 0.1,
                interaction_count INTEGER DEFAULT 0,
                last_topic TEXT,
                first_seen_at TEXT DEFAULT (datetime('now', 'localtime')),
                last_seen_at TEXT DEFAULT (datetime('now', 'localtime')),
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)

        # 给 memories 表加 person 和 mood_snapshot 字段（兼容旧表）
        for col in ('person', 'mood_snapshot'):
            try:
                conn.execute(f"ALTER TABLE memories ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass  # 字段已存在

        conn.commit()

    # ─── CRUD ──────────────────────────────────────────────

    def add_memory(self, content: str, memory_type: str = "knowledge",
                   importance: float = 0.5, person: str = None,
                   mood_data: dict = None) -> int:
        """
        添加一条记忆，自动生成向量。

        Args:
            content: 记忆内容（一句判断句）
            memory_type: knowledge | emotional | event | skill | daily_summary
            importance: 重要性 0~1
            person: 关联的人物（默认自动检测）
            mood_data: 添加记忆时的情绪快照 {"valence": 0.3, "arousal": 0.1, "dominance": 0.0, "mood_label": "平静"}

        Returns:
            新记忆的 id
        """
        embedding_bytes = encode_text(content)
        conn = self._get_conn()

        mood_json = json.dumps(mood_data, ensure_ascii=False) if mood_data else None

        # 插入元数据
        cursor = conn.execute(
            "INSERT INTO memories (content, memory_type, importance, person, mood_snapshot) VALUES (?, ?, ?, ?, ?)",
            (content, memory_type, importance, person, mood_json),
        )
        memory_id = cursor.lastrowid

        # 插入向量（rowid 必须与 memories.id 一致）
        conn.execute(
            "INSERT INTO memory_embeddings (rowid, embedding) VALUES (?, ?)",
            (memory_id, embedding_bytes),
        )
        conn.commit()
        return memory_id


    def search(self, query: str, limit: int = 5) -> List[dict]:
        """
        语义搜索：根据查询文本找到最相关的记忆。

        Args:
            query: 查询文本（用户当前消息）
            limit: 返回条数

        Returns:
            [{"id": 42, "content": "...", "type": "knowledge", "distance": 0.123}, ...]
        """
        try:
            query_vec = encode_text(query)
        except RuntimeError:
            return []  # embedding 模型未加载，返回空结果
        conn = self._get_conn()

        # sqlite-vec 向量检索
        rows = conn.execute(
            f"""
            SELECT
                m.id, m.content, m.memory_type, m.importance,
                m.created_at, m.last_recalled_at, m.recall_count,
                m.person, m.mood_snapshot,
                v.distance
            FROM memory_embeddings v
            JOIN memories m ON m.id = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (query_vec, limit),
        ).fetchall()

        results = []
        for row in rows:
            mid, content, mtype, imp, created, recalled, count, person, mood_json, dist = row
            mood_data = json.loads(mood_json) if mood_json else None
            results.append({
                "id": mid,
                "content": content,
                "type": mtype,
                "importance": imp,
                "created_at": created,
                "last_recalled_at": recalled,
                "recall_count": count,
                "person": person or "对方",
                "mood_snapshot": mood_data,
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

    def recall_with_mood(self, query: str, limit: int = 5,
                         distance_threshold: float = 0.8) -> tuple:
        """
        召回记忆并附带情绪快照（用于波动计算）。

        Returns:
            (recalled_texts, recalled_moods)
            recalled_texts: List[str] 格式化记忆文本
            recalled_moods: List[dict] 每条记忆对应的情绪快照
        """
        results = self.search(query, limit)
        recalled = []
        moods = []
        recalled_ids = []

        for r in results:
            if r["distance"] < distance_threshold:
                recalled.append(r["content"])
                moods.append(r["mood_snapshot"])
                recalled_ids.append(r["id"])

        if recalled_ids:
            self._bump_recall(recalled_ids)

        return recalled, moods

    @staticmethod
    def compute_mood_fluctuation(recalled_moods: List[dict],
                                  current_affection: dict = None) -> dict:
        """
        根据召回的多个记忆的情绪快照，计算对当前情绪的总波动影响。

        算法：
        1. 每条记忆的情绪快照对当前情绪产生一个 delta
        2. delta 的大小由该记忆的情绪强度 × 衰减因子决定
        3. 所有 delta 累加后限幅到 [-0.15, 0.15] 范围内

        Args:
            recalled_moods: recall_with_mood 返回的情绪快照列表（可能含 None）
            current_affection: 当前好感度状态（用于参考，暂未使用）

        Returns:
            {"valence_delta": 0.05, "arousal_delta": 0.02, "dominance_delta": 0.01}
        """
        if not recalled_moods or all(m is None for m in recalled_moods):
            return {"valence_delta": 0.0, "arousal_delta": 0.0, "dominance_delta": 0.0}

        total_v, total_a, total_d = 0.0, 0.0, 0.0
        count = 0

        for i, mood in enumerate(recalled_moods):
            if not mood:
                continue
            # 情感强度 = 愉悦度的绝对值 + 唤醒度的绝对值 的平均
            v = mood.get("valence", 0.0)
            a = mood.get("arousal", 0.0)
            d = mood.get("dominance", 0.0)
            intensity = (abs(v) + abs(a)) / 2.0

            # 衰减因子：越靠前的记忆（相关度越高）影响越大
            # index 0 最相关，decay = 0.3; index 4 decay = 0.1
            decay = max(0.1, 0.3 - i * 0.05)

            # 波动 = 情绪值 × 强度 × 衰减
            total_v += v * intensity * decay
            total_a += a * intensity * decay * 0.5   # arousal 影响减半
            total_d += d * intensity * decay * 0.3   # dominance 影响更小
            count += 1

        if count == 0:
            return {"valence_delta": 0.0, "arousal_delta": 0.0, "dominance_delta": 0.0}

        # 限幅
        def clamp(val, lo=-0.15, hi=0.15):
            return max(lo, min(hi, val))

        return {
            "valence_delta": clamp(total_v / count),
            "arousal_delta": clamp(total_a / count),
            "dominance_delta": clamp(total_d / count),
        }

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
        cols = "id, content, memory_type, importance, created_at, last_recalled_at, recall_count, person, mood_snapshot"
        if memory_type:
            rows = conn.execute(
                f"SELECT {cols} FROM memories WHERE memory_type = ? "
                "ORDER BY created_at DESC",
                (memory_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {cols} FROM memories "
                "ORDER BY created_at DESC",
            ).fetchall()

        return [
            {
                "id": r[0], "content": r[1], "type": r[2],
                "importance": r[3], "created_at": r[4],
                "last_recalled_at": r[5], "recall_count": r[6],
                "person": r[7] or "对方",
                "mood_snapshot": json.loads(r[8]) if r[8] else None,
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

    # ════════════════════════════════════════════
    # 活动日志（短期意识流）
    # ════════════════════════════════════════════

    def add_activity(self, source: str, summary: str, detail: str = None, person: str = None):
        """添加一条活动日志（LLM 压缩后存入）

        Args:
            source: user | lilith_chat | lilith_channel | lilith_internal
            summary: LLM 压缩后的 1 句话摘要
            detail: 原始文本（可选）
            person: 互动对象（默认自动检测）
        """
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO activity_log (source, summary, detail, person) VALUES (?, ?, ?, ?)",
            (source, summary[:200], detail[:500] if detail else None, person),
        )
        conn.commit()

    def get_recent_activities(self, hours: int = 24, limit: int = 50) -> list:
        """获取最近 N 小时的活动日志（含 id 用于逐条删除）"""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT id, source, summary, created_at, person
            FROM activity_log
            WHERE created_at >= datetime('now', 'localtime', ?)
            ORDER BY created_at DESC
            LIMIT ?
        """, (f'-{hours} hours', limit)).fetchall()
        return [
            {"id": r[0], "source": r[1], "summary": r[2], "time": r[3], "person": r[4]}
            for r in rows
        ]

    def delete_activity(self, activity_id: int) -> bool:
        """删除单条活动日志"""
        conn = self._get_conn()
        conn.execute("DELETE FROM activity_log WHERE id = ?", (activity_id,))
        conn.commit()
        return conn.total_changes > 0


    def add_or_update_person_knowledge(self, person_name: str, **kwargs) -> bool:
        """添加或更新对一个人的认知

        Args:
            person_name: 人名
            **kwargs: 可更新的字段 portrait, tags, trust, intimacy, familiarity, last_topic
        """
        conn = self._get_conn()
        now = "datetime('now', 'localtime')"

        # 检查是否存在
        existing = conn.execute(
            "SELECT id FROM person_knowledge WHERE person_name = ?",
            (person_name,),
        ).fetchone()

        if existing:
            # 更新
            sets = ["updated_at = " + now]
            params = []
            for k in ('portrait', 'tags', 'trust', 'intimacy', 'familiarity', 'last_topic'):
                if k in kwargs:
                    sets.append(f"{k} = ?")
                    params.append(kwargs[k])
            sets.append("interaction_count = interaction_count + 1")
            sets.append("last_seen_at = " + now)
            sql = "UPDATE person_knowledge SET " + ", ".join(sets) + " WHERE person_name = ?"
            params.append(person_name)
            conn.execute(sql, params)
        else:
            # 新建
            fields = ["person_name", "interaction_count"]
            vals = [person_name, 1]
            placeholders = ["?", "?"]
            for k in ('portrait', 'tags', 'trust', 'intimacy', 'familiarity', 'last_topic'):
                if k in kwargs:
                    fields.append(k)
                    vals.append(kwargs[k])
                    placeholders.append("?")
            sql = f"INSERT INTO person_knowledge ({', '.join(fields)}) VALUES ({', '.join(placeholders)})"
            conn.execute(sql, vals)

        conn.commit()
        return True

    def get_person_knowledge(self, person_name: str) -> Optional[dict]:
        """获取对一个人的认知"""
        conn = self._get_conn()
        row = conn.execute("""
            SELECT person_name, portrait, tags, trust, intimacy, familiarity,
                   interaction_count, last_topic, first_seen_at, last_seen_at
            FROM person_knowledge WHERE person_name = ?
        """, (person_name,)).fetchone()
        if not row:
            return None
        return {
            "person_name": row[0],
            "portrait": row[1],
            "tags": json.loads(row[2]) if row[2] else [],
            "trust": row[3],
            "intimacy": row[4],
            "familiarity": row[5],
            "interaction_count": row[6],
            "last_topic": row[7],
            "first_seen_at": row[8],
            "last_seen_at": row[9],
        }

    def list_known_persons(self) -> list:
        """列出莉莉丝认识的所有人"""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT person_name, portrait, trust, intimacy, interaction_count, last_seen_at
            FROM person_knowledge
            ORDER BY interaction_count DESC, last_seen_at DESC
        """).fetchall()
        return [
            {"name": r[0], "portrait": r[1], "trust": r[2],
             "intimacy": r[3], "interactions": r[4], "last_seen": r[5]}
            for r in rows
        ]



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


