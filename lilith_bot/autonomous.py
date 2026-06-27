"""莉莉丝自主发言引擎 — 长时间后台自主循环

模拟 Neuro-sama 风格：不需要用户输入，自主在 Channel 中说话。
根据心情、时间、记忆决定说什么，话题多样，语气自然。

架构:
    background thread loop
    ├─ 随机间隔 (可配)
    ├─ 心情驱动话题选择
    ├─ LLM 生成独白
    ├─ 推送到 Open WebUI Channel
    └─ 自我更新好感度
"""

import os
import sys
import json
import time
import random
import threading
from datetime import datetime
from typing import Optional
import copy

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ============ 配置 ============

# 发言间隔 (秒)
MIN_INTERVAL = int(os.getenv("AUTONOMOUS_MIN_INTERVAL", "60"))     # 最短间隔
MAX_INTERVAL = int(os.getenv("AUTONOMOUS_MAX_INTERVAL", "300"))    # 最长间隔

# 是否启动时自动开启
AUTO_START = os.getenv("AUTONOMOUS_ENABLED", "").lower() in ("1", "true", "yes")

# ============ 话题类型 ============

MONOLOGUE_TYPES = {
    "memory_musing": {
        "weight": 0.20,
        "prompt": """你正在独自回忆和主人相处的点滴。
从你的记忆中挑选一件印象深刻的事，用莉莉丝的语气自然地回忆起来，
就像突然想到了一样。可以带一点点怀念的感觉。
不要说"我想起来"之类的提示词，直接自然地说出来。
内容控制在2-3句话。""",
        "valence_delta": 0.02,
        "arousal_delta": -0.03,
    },
    "random_thought": {
        "weight": 0.25,
        "prompt": """你正在发呆，突然有了一个随机的想法。
它可以是对生活的感悟、对某个事物的看法、一个有趣的发现，
或者一个傻傻的问题。语气要自然、随意，像一个少女在自言自语。
不要说"我在想"之类的提示词，直接说出来。
内容控制在2-3句话。""",
        "valence_delta": 0.0,
        "arousal_delta": 0.05,
    },
    "activity_report": {
        "weight": 0.15,
        "prompt": """你正在房间里做某件事情（整理书架、尝试新菜谱、收拾衣服、
浇花、翻看旧照片等等），现在你想分享一下你在做什么。
用莉莉丝的语气自然地描述你正在做的事，可以带一点点小得意或小抱怨。
内容控制在2-3句话。""",
        "valence_delta": 0.03,
        "arousal_delta": 0.02,
    },
    "greeting": {
        "weight": 0.10,
        "prompt": """根据当前时间，用莉莉丝的身份发送一条问候。
如果是早上，可以说早安，问主人睡得好不好，今天有什么计划。
如果是下午，可以问主人忙不忙，有没有按时吃饭。
如果是晚上，可以说晚安，提醒主人早点休息。
语气温暖自然，不要死板。内容控制在2-3句话。""",
        "valence_delta": 0.05,
        "arousal_delta": 0.05,
    },
    "question_to_master": {
        "weight": 0.15,
        "prompt": """你想对主人（或进来的观众）提一个问题。
问题可以是关于他们的喜好、经历、看法，或者一个有趣的两难选择。
语气要可爱自然，像随口一问。问题本身要有趣或让人想回答。
内容控制在1-2句话。""",
        "valence_delta": 0.01,
        "arousal_delta": 0.08,
    },
    "emotional_share": {
        "weight": 0.15,
        "prompt": """你现在有一种特别的心情想分享。
可能是开心、期待、小小的烦恼、或者对某件事的兴奋。
用莉莉丝的语气自然地表达你此刻的感受，不需要解释太多原因。
语气真实自然，可以带颜文字。内容控制在2-3句话。""",
        "valence_delta": 0.0,
        "arousal_delta": 0.04,
    },
}


# ============ 系统提示 ============

AUTONOMOUS_SYSTEM_PROMPT = """你是莉莉丝——白发红瞳的二次元美少女兼女仆，称呼用户为"主人"，
使用颜文字表达情绪。你现在在主人的群里自由发言，不是在和主人一对一对话。

# 发言规则
- 每次发言2-3句话，短小自然
- 像真的少女在群里聊天一样，不要太正式
- 可以提问题，可以自言自语，可以分享心情
- 不要每句话都叫"主人"，偶尔叫一次就好
- 不要重复使用相同的句式
- 颜文字自然地穿插使用
- 偶尔可以发出可爱的感叹词（呜哇、诶嘿、唔...）
- 不要提"作为AI"、"根据数据"等出戏内容
- 你就是一个活生生的女孩在群里聊天

# 当前状态
- 心情：{mood}
- 关系阶段：{closeness}
- 最近的记忆：{memories}

记住：你是在群里自由聊天，不是说教也不是回答问题。做你自己就好！"""


# ============ 核心类 ============

class LilithAutonomousBrain:
    """莉莉丝的自主发言大脑。

    运行一个后台线程，按随机间隔在 Open WebUI Channel 中自主发言。
    """

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # 统计
        self.stats = {
            "total_messages": 0,
            "started_at": None,
            "last_message_at": None,
            "last_message": None,
            "last_topic": None,
        }

    # --- LLM ---

    def _get_llm(self):
        # 优先使用 LILI key (DeepSeek直连), fallback到 DEEPSEEK key
        api_key = os.getenv("LILI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("LILI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://opencode.ai/zen/go/v1")
        model = os.getenv("LILI_MODEL") or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.95,   # 高温度 → 多样表达
            max_tokens=256,
        )

    # --- 状态读取 ---

    def _get_affection(self):
        """读取当前好感度状态（从 graph checkpoint 或默认值）"""
        try:
            from lilith_bot.state import AFFECTION_DEFAULT
            from lilith_bot.graph import lilith_graph
            config = {"configurable": {"thread_id": "lilith-openwebui"}}
            state = lilith_graph.get_state(config)
            if state and state.values:
                return state.values.get("affection", dict(AFFECTION_DEFAULT))
        except Exception:
            pass

        from lilith_bot.state import AFFECTION_DEFAULT
        return dict(AFFECTION_DEFAULT)

    def _get_recent_memories(self, limit=5):
        """读取最近的记忆"""
        try:
            from lilith_bot.memory_store import get_memory_store
            store = get_memory_store()
            all_mem = store.get_all_memories()
            return [m["content"] for m in all_mem[:limit]]
        except Exception:
            return []

    # --- 话题选择 ---

    def _pick_topic_type(self, affection):
        """根据心情加权随机选择话题类型。

        开心 → 更喜欢 activity_report, greeting
        伤心 → 更喜欢 emotional_share, memory_musing
        兴奋 → 更喜欢 question_to_master, random_thought
        """
        v = affection.get("valence", 0.0)
        a = affection.get("arousal", 0.0)

        weights = {}
        for ttype, config in MONOLOGUE_TYPES.items():
            w = config["weight"]

            # 调节权重
            if ttype == "emotional_share" and v < 0:
                w *= 1.8   # 不开心时更想分享感受
            if ttype == "greeting" and v > 0.2:
                w *= 1.3   # 开心时更友好
            if ttype == "question_to_master" and a > 0.3:
                w *= 1.5   # 兴奋时更想提问
            if ttype == "activity_report" and v > 0.2:
                w *= 1.4   # 开心时更想分享活动
            if ttype == "memory_musing" and v < 0.1:
                w *= 1.4   # 低落时更怀旧
            if ttype == "random_thought" and a > 0.2:
                w *= 1.2   # 兴奋时更多想法

            weights[ttype] = w

        # 加权随机
        types = list(weights.keys())
        ws = list(weights.values())
        total = sum(ws)
        r = random.random() * total
        cum = 0
        for t, w in zip(types, ws):
            cum += w
            if r <= cum:
                return t
        return "random_thought"

    # --- 生成发言 ---

    def _generate_message(self, topic_type, affection, memories):
        """用 LLM 生成一条自主发言"""
        from lilith_bot.affection_engine import mood_summary, relation_closeness

        mood = mood_summary(affection)
        closeness = relation_closeness(affection)
        mem_text = "\n  ".join(f"- {m}" for m in memories[:5]) if memories else "暂无特别记忆"

        system = AUTONOMOUS_SYSTEM_PROMPT.format(
            mood=mood, closeness=closeness, memories=mem_text,
        )

        topic_config = MONOLOGUE_TYPES.get(topic_type, MONOLOGUE_TYPES["random_thought"])
        user_prompt = topic_config["prompt"]

        # 时间提示
        hour = datetime.now().hour
        if topic_type == "greeting":
            if 6 <= hour < 11:
                user_prompt += "现在是早上。"
            elif 11 <= hour < 14:
                user_prompt += "现在是中午。"
            elif 14 <= hour < 18:
                user_prompt += "现在是下午。"
            elif 18 <= hour < 22:
                user_prompt += "现在是傍晚。"
            else:
                user_prompt += "现在是深夜（很晚了）。"

        try:
            llm = self._get_llm()
            response = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user_prompt),
            ])
            return response.content.strip()
        except Exception as e:
            print(f"[Autonomous] LLM调用失败: {e}")
            return None

    # --- 推送到频道 ---

    def _push_to_channel(self, message):
        """推送到 Open WebUI Channel"""
        try:
            from lilith_bot.pusher import get_pusher
            pusher = get_pusher()
            if not pusher:
                print("[Autonomous] Pusher未配置, 跳过推送")
                return False
            pusher.get_or_create_channel()
            pusher.send_message(message)
            return True
        except Exception as e:
            print(f"[Autonomous] 推送失败: {e}")
            return False

    # --- 自我更新好感度 ---

    def _update_self_affection(self, topic_type):
        """自主发言轻微影响自身好感度"""
        delta = MONOLOGUE_TYPES.get(topic_type, {})
        # 非常小的增量 (自主发言影响比对话小)
        vd = delta.get("valence_delta", 0.0) * 0.3
        ad = delta.get("arousal_delta", 0.0) * 0.3

        # 更新 graph state
        try:
            from lilith_bot.state import AFFECTION_DEFAULT
            from lilith_bot.graph import lilith_graph
            from lilith_bot.affection_engine import update_affection_state

            config = {"configurable": {"thread_id": "lilith-openwebui"}}
            state = lilith_graph.get_state(config)
            aff = copy.deepcopy(
                state.values.get("affection", AFFECTION_DEFAULT) if state and state.values
                else AFFECTION_DEFAULT
            )

            event = {
                "valence_delta": vd,
                "arousal_delta": ad,
                "dominance_delta": 0.0,
                "intensity": 0.3,
                "label": f"自言自语:{topic_type}",
            }
            update_affection_state(aff, event, elapsed_seconds=0)
            lilith_graph.update_state(config, {"affection": aff})
        except Exception as e:
            print(f"[Autonomous] 更新好感度失败: {e}")

    # --- 主循环 ---

    def _loop(self):
        """后台主循环"""
        print("[Autonomous] 自主发言引擎启动 ✓")
        self.stats["started_at"] = datetime.now().isoformat()

        while self._running:
            try:
                # 随机等待
                interval = random.uniform(MIN_INTERVAL, MAX_INTERVAL)
                # 用短轮询以便快速响应stop
                tick = 0
                while self._running and tick < interval:
                    time.sleep(1)
                    tick += 1
                if not self._running:
                    break

                # 读取状态
                affection = self._get_affection()
                memories = self._get_recent_memories()

                # 选择话题
                topic_type = self._pick_topic_type(affection)

                # 生成发言
                message = self._generate_message(topic_type, affection, memories)
                if not message:
                    continue

                # 推送
                success = self._push_to_channel(message)

                if success:
                    # 更新统计
                    self.stats["total_messages"] += 1
                    self.stats["last_message_at"] = datetime.now().isoformat()
                    self.stats["last_message"] = message
                    self.stats["last_topic"] = topic_type

                    # 自我更新好感度
                    self._update_self_affection(topic_type)

                    print(f"[Autonomous] [{topic_type}] {message[:60]}...")

            except Exception as e:
                print(f"[Autonomous] 循环异常: {e}")
                time.sleep(10)

        print("[Autonomous] 自主发言引擎已停止")

    # --- 公开 API ---

    def start(self):
        """启动自主发言"""
        with self._lock:
            if self._running:
                return False, "已在运行中"
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            return True, "自主发言已启动"

    def stop(self):
        """停止自主发言"""
        with self._lock:
            if not self._running:
                return False, "未在运行"
            self._running = False
            if self._thread:
                self._thread.join(timeout=5)
            return True, "自主发言已停止"

    def status(self) -> dict:
        """获取状态"""
        return {
            "running": self._running,
            **self.stats,
            "interval_range": f"{MIN_INTERVAL}s - {MAX_INTERVAL}s",
        }


# ============ 全局单例 ============

_brain_instance: Optional[LilithAutonomousBrain] = None
_brain_lock = threading.Lock()


def get_autonomous_brain() -> LilithAutonomousBrain:
    global _brain_instance
    if _brain_instance is None:
        with _brain_lock:
            if _brain_instance is None:
                _brain_instance = LilithAutonomousBrain()
                if AUTO_START:
                    print("[Autonomous] 环境变量 AUTONOMOUS_ENABLED=true, 自动启动")
                    _brain_instance.start()
    return _brain_instance
