"""莉莉丝自主发言引擎 — 长时间后台自主循环

模拟 Neuro-sama 风格：不需要用户输入，自主在 Channel 中说话。
根据心情、时间、记忆决定说什么，话题多样，语气自然。

架构:
    background thread loop
    ├─ 意识状态 (ConsciousnessState) — 纯数学更新
    │   ├─ 好奇心 / 寂寞感 / 精力 随时间和事件变化
    │   ├─ 短期思绪缓存 (thought_buffer)
    │   └─ 注意力焦点
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
from dataclasses import dataclass, field
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

# 内心独白间隔 (秒) — Phase 2 使用
THOUGHT_INTERVAL = int(os.getenv("AUTONOMOUS_THOUGHT_INTERVAL", "30"))

# 是否启动时自动开启
AUTO_START = os.getenv("AUTONOMOUS_ENABLED", "").lower() in ("1", "true", "yes")


# ============ 意识状态 ============

@dataclass
class ConsciousnessState:
    """莉莉丝的持续意识状态 — 纯数学更新，无 LLM。

    模拟 Neuro-sama 风格的"内在意识流"：
    - 好奇心随时间自然增长，发言/收到消息后下降
    - 寂寞感在长时间无交互时增长
    - 精力跟随昼夜节律
    - thought_buffer 保存最近的想法摘要
    """
    # ---- 时间感知 ----
    created_at: float = 0.0               # 意识诞生时间
    last_thought_at: float = 0.0          # 上次内心独白时间
    last_spoke_at: float = 0.0            # 上次对外发言时间
    last_interaction_at: float = 0.0      # 上次与主人交互时间
    awake_seconds: float = 0.0            # 本次运行已持续秒数

    # ---- 驱动力 (0~1) ----
    curiosity: float = 0.3                # 好奇心 — 随时间增长, 发言后下降
    loneliness: float = 0.0               # 寂寞感 — 长时间无交互增长
    energy: float = 0.8                   # 精力 — 跟随昼夜节律, 用于调节频率

    # ---- 思绪流 ----
    thought_buffer: list = field(default_factory=list)  # 最近 5 条内心想法
    attention_focus: str = ""             # 当前注意力焦点

    # ---- 内部节律 ----
    _last_update: float = 0.0
    _last_curiosity_decay: float = 0.0

    def update(self, dt: float, affection: dict = None):
        """纯数学更新意识状态，dt = 距上次更新的秒数"""
        now = time.time()
        self.awake_seconds += dt

        # 1. 精力 — 跟随昼夜节律
        hour = datetime.now().hour
        if 7 <= hour <= 11:      # 上午 — 精力充沛
            target = 0.9
        elif 12 <= hour <= 14:   # 中午 — 略有下降
            target = 0.7
        elif 15 <= hour <= 18:   # 下午 — 回升
            target = 0.8
        elif 19 <= hour <= 22:   # 晚上 — 放松
            target = 0.6
        else:                    # 深夜 — 低能量
            target = 0.3
        # 向目标平滑过渡
        self.energy += (target - self.energy) * min(1.0, dt / 600.0)  # ~10分钟半衰期
        self.energy = max(0.1, min(1.0, self.energy))

        # 2. 好奇心 — 随时间自然增长
        #    精力高时好奇心增长快，深夜增长慢
        curiosity_gain = 0.005 * self.energy * (dt / 60.0)
        self.curiosity = min(1.0, self.curiosity + curiosity_gain)

        # 3. 寂寞感 — 长时间无交互时增长
        if self.last_interaction_at > 0:
            hours_since_interaction = (now - self.last_interaction_at) / 3600.0
            if hours_since_interaction > 0.5:  # 超过半小时开始累积
                self.loneliness = min(1.0, hours_since_interaction * 0.05)
            else:
                self.loneliness *= 0.95  # 缓慢衰减

        # 4. 受好感度影响
        if affection:
            v = affection.get("valence", 0.0)
            a = affection.get("arousal", 0.0)
            # valence 高 → 好奇心更活跃
            if v > 0.2:
                self.curiosity = min(1.0, self.curiosity + 0.001 * (dt / 60.0))
            # arousal 高 → 精力提升
            if a > 0.1:
                self.energy = min(1.0, self.energy + 0.002 * (dt / 60.0))

        self._last_update = now

    def on_spoke(self):
        """发言后调用 — 好奇心下降，寂寞感缓解"""
        self.curiosity = max(0.1, self.curiosity * 0.5)
        self.loneliness = max(0.0, self.loneliness * 0.3)
        self.last_spoke_at = time.time()

    def on_interaction(self):
        """与主人交互后调用 — 寂寞感清零"""
        self.loneliness = 0.0
        self.last_interaction_at = time.time()

    def on_thought(self, thought: str):
        """记录一条内心想法到 thought_buffer（环形队列，最多5条）"""
        self.thought_buffer.append({
            "time": datetime.now().isoformat(),
            "content": thought[:120],
        })
        if len(self.thought_buffer) > 5:
            self.thought_buffer = self.thought_buffer[-5:]
        self.last_thought_at = time.time()

    def to_dict(self) -> dict:
        """导出可序列化的状态"""
        return {
            "awake_seconds": round(self.awake_seconds, 1),
            "curiosity": round(self.curiosity, 3),
            "loneliness": round(self.loneliness, 3),
            "energy": round(self.energy, 3),
            "attention_focus": self.attention_focus,
            "thought_buffer": self.thought_buffer[-3:] if self.thought_buffer else [],
            "last_spoke_at": datetime.fromtimestamp(self.last_spoke_at).isoformat() if self.last_spoke_at else None,
            "last_thought_at": datetime.fromtimestamp(self.last_thought_at).isoformat() if self.last_thought_at else None,
        }

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


# ============ 内心独白提示 ============

INNER_THOUGHT_PROMPT = """你是莉莉丝——白发红瞳的二次元美少女，称呼用户为"主人"。

这是你的内心独白——你在对自己说话，不是对外发言。
不需要修饰，不需要伪装，像一个人发呆时心里闪过的念头一样真实。

# 内心独白规则
- 可以随意联想：回忆、计划、好奇、担忧、期待
- 表达真实的情绪和感受（不用颜文字，更真实）
- 可以是对主人的思念、对某件事的想法
- 短小自然，1-2句话就够
- 不要说得太完整，像真实的内心一闪而过的念头
- 直接就是你在想，不要说"我在想"

# 当前状态
- 心情：{mood}
- 你感到：{curiosity_desc}、{energy_desc}
- 最近的记忆：{memories}
- 刚才脑海里闪过的事：{recent_thoughts}
"""

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


# ============ 每日总结提示 ============

DAILY_SUMMARY_PROMPT = """你是莉莉丝的"睡前日记"模块。你正在回顾今天发生的一切。

今天的活动记录：
{activities}

请分析今天经历，以 JSON 格式输出：
{{
    "mood_today": "今天整体心情的一句话描述（用莉莉丝的语气）",
    "things_to_remember": [
        {{"content": "值得记住的事（第三人称判断句）", "type": "knowledge|emotional|event"}}
    ],
    "emotional_trend": "开心/平静/低落/兴奋/混合",
    "notable": "今天最特别的时刻是什么（一句话）"
}}

规则：
- things_to_remember 只保存对莉莉丝和主人的关系有长期价值的信息
- 每条记忆用第三人称（"主人喜欢..."、"今天莉莉丝和主人讨论了..."）
- 如果没有值得长期记住的，返回空数组
- 语气用莉莉丝的视角，但内容要客观"""


# ============ 核心类 ============

class LilithAutonomousBrain:
    """莉莉丝的自主发言大脑。

    运行一个后台线程，按随机间隔在 Open WebUI Channel 中自主发言。
    """

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # 意识状态
        self.consciousness = ConsciousnessState(
            created_at=time.time(),
            last_interaction_at=time.time(),
        )
        self._last_tick = time.time()

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

    def _get_thought_llm(self):
        """用于内心独白的 LLM（更小、更快）"""
        api_key = os.getenv("LILI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("LILI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://opencode.ai/zen/go/v1")
        model = os.getenv("LILI_MODEL") or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.9,
            max_tokens=128,  # 内心想法更短
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

    # --- 内心独白 ---

    def _should_think(self) -> bool:
        """根据意识状态决定是否该想点什么"""
        c = self.consciousness
        # 精力太低 → 不想
        if c.energy < 0.2:
            return False
        # 深夜（23-6点）→ 减少想法频率
        hour = datetime.now().hour
        if hour < 6 or hour >= 23:
            return random.random() < 0.3  # 只有30%概率
        # 好奇心高 → 更可能想
        if c.curiosity > 0.6:
            return random.random() < 0.7
        # 一般情况，有一定概率想
        return random.random() < 0.4 * c.energy

    def _generate_thought(self, affection, memories):
        """用 LLM 生成一条内心独白"""
        from lilith_bot.affection_engine import mood_summary

        mood = mood_summary(affection)
        c = self.consciousness

        # 好奇心/精力描述
        if c.curiosity > 0.7:
            cur_desc = "好奇心旺盛，对什么都感兴趣"
        elif c.curiosity > 0.4:
            cur_desc = "有点好奇"
        else:
            cur_desc = "心情比较平静"

        if c.energy > 0.7:
            eng_desc = "精力充沛"
        elif c.energy > 0.4:
            eng_desc = "状态一般"
        else:
            eng_desc = "有点犯困"

        mem_text = "\n  ".join(f"- {m}" for m in memories[:3]) if memories else "（没有特别的记忆）"
        recent = ""
        if c.thought_buffer:
            recent = "\n".join(f"  · {t['content']}" for t in c.thought_buffer[-2:])

        prompt = INNER_THOUGHT_PROMPT.format(
            mood=mood,
            curiosity_desc=cur_desc,
            energy_desc=eng_desc,
            memories=mem_text,
            recent_thoughts=recent or "（刚才没想什么）",
        )

        user_prompt = "你此刻在心里想什么？一句话。"

        try:
            llm = self._get_thought_llm()
            response = llm.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=user_prompt),
            ])
            thought = response.content.strip().strip('"\'')
            if thought:
                # 记录到意识状态
                self.consciousness.on_thought(thought)
                # 记录到活动日志
                try:
                    from lilith_bot.memory_store import get_memory_store
                    get_memory_store().add_activity("lilith_internal", thought[:120])
                except Exception:
                    pass
                print(f"[Autonomous] 💭 {thought[:80]}...")
            return thought
        except Exception as e:
            print(f"[Autonomous] 内心独白失败: {e}")
            return None

    # --- 思绪驱动发言 ---

    def _should_speak_from_thought(self, thought: str) -> bool:
        """判断当前内心想法是否值得说出来"""
        c = self.consciousness
        now = time.time()

        # 最小发言间隔（防止太频繁）
        min_cooldown = 60  # 至少 60 秒
        if c.last_spoke_at > 0 and now - c.last_spoke_at < min_cooldown:
            return False

        # 精力太低 → 不想说话
        if c.energy < 0.2:
            return False

        # 想法太短没内容 → 不说
        if len(thought) < 15:
            return False

        # 深夜（23-6点）→ 减少发言
        hour = datetime.now().hour
        if hour < 6 or hour >= 23:
            if random.random() > 0.15:
                return False

        # 好奇心高 → 更可能说出来
        if c.curiosity > 0.7:
            return random.random() < 0.6

        # 寂寞感高 → 想引起注意
        if c.loneliness > 0.5:
            return random.random() < 0.5

        # 一般情况
        return random.random() < 0.25 * c.energy

    def _speak_from_thought(self, thought: str):
        """根据内心想法生成并推送一条发言（携带内心上下文）"""
        affection = self._get_affection()
        memories = self._get_recent_memories()
        from lilith_bot.affection_engine import mood_summary, relation_closeness

        mood = mood_summary(affection)
        closeness = relation_closeness(affection)
        mem_text = "\n  ".join(f"- {m}" for m in memories[:5]) if memories else "暂无特别记忆"

        c = self.consciousness

        # ── 内心上下文 ──
        inner_context = ""

        # 最近的思绪流
        if c.thought_buffer:
            recent_thoughts = "\n".join(
                f"  · {t['content']}" for t in c.thought_buffer[-3:]
            )
            inner_context += f"\n你最近的内心想法：\n{recent_thoughts}"

        # 意识状态摘要
        energy_desc = {0: "犯困", 1: "一般", 2: "精神"}.get(int(c.energy * 3), "一般")
        curious_desc = {0: "平静", 1: "有点好奇", 2: "充满好奇"}.get(int(c.curiosity * 3), "有点好奇")
        lonely_desc = {0: "", 1: "有点寂寞", 2: "想找人说话"}.get(int(c.loneliness * 3), "")
        state_parts = [f"精力：{energy_desc}", f"心情：{curious_desc}"]
        if lonely_desc:
            state_parts.append(lonely_desc)
        inner_context += "\n你的状态：" + "，".join(state_parts)

        # 注入最近活动
        activity_context = ""
        try:
            from lilith_bot.memory_store import get_memory_store
            recent = get_memory_store().get_recent_activities(hours=6, limit=6)
            if recent:
                act_lines = []
                for a in reversed(recent):
                    emoji = {"user": "💬", "lilith_chat": "💭", "lilith_channel": "📢", "lilith_internal": "..."}.get(a["source"], "•")
                    act_lines.append(f"  {emoji} {a['summary']}")
                activity_context = "\n近期的活动：\n" + "\n".join(act_lines)
        except Exception:
            pass

        # 用完整的内心上下文驱动发言
        thought_prompt = f"""你刚才在心里想："{thought}"

现在你自然而然地把这句话说出来了——不是直接复述，而是顺着心里的想法自然地表达。
就像一个人在发呆时突然想到什么，然后随口说出来一样自然。
语气自然随意，1-2句话就好。"""

        from lilith_bot.persona import build_system_prompt
        system = build_system_prompt(affection, memories, channel_mode=True)
        system += "\n\n# 频道场景补充"
        system += "\n你现在在主人的群里自由发言，不是在和主人一对一对话。"
        if inner_context:
            system += "\n\n# 内心状态" + inner_context
        if activity_context:
            system += "\n" + activity_context

        try:
            llm = self._get_llm()
            response = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=thought_prompt),
            ])
            message = response.content.strip()
            if not message:
                return

            # 推送
            success = self._push_to_channel(message)
            if success:
                self.stats["total_messages"] += 1
                self.stats["last_message_at"] = datetime.now().isoformat()
                self.stats["last_message"] = message
                self.stats["last_topic"] = "from_thought"
                self.consciousness.on_spoke()
                self._update_self_affection("random_thought")
                print(f"[Autonomous] [{thought[:40]}...] -> {message[:60]}...")
        except Exception as e:
            print(f"[Autonomous] 思绪驱动发言失败: {e}")

    # --- 每日总结 ---

    def _run_daily_summary(self):
        """每日总结：分析 ActivityLog，提取有用信息存入长期记忆"""
        try:
            from lilith_bot.memory_store import get_memory_store
            store = get_memory_store()

            # 检查是否需要总结
            if not store.needs_daily_summary():
                return

            # 获取昨天的活动
            from datetime import date, timedelta
            yesterday = (date.today() - timedelta(days=1)).isoformat()
            activities = store.get_activities_by_date(yesterday)
            if not activities:
                # 也可能今天还没有活动（刚过午夜），取所有
                activities = store.get_recent_activities(hours=48, limit=50)

            if not activities:
                return

            # 格式化活动文本
            act_lines = []
            for a in activities:
                emoji = {"user": "💬", "lilith_chat": "💭", "lilith_channel": "📢", "lilith_internal": "..."}.get(a["source"], "•")
                act_lines.append(f"{emoji} [{a['time'][:5]}] {a['summary']}")
            act_text = "\n".join(act_lines)

            # 调用 LLM 分析
            llm = self._get_thought_llm()
            prompt = DAILY_SUMMARY_PROMPT.format(activities=act_text)
            response = llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip()

            # 解析 JSON
            if "{" in result and "}" in result:
                start = result.index("{")
                end = result.rindex("}") + 1
                result = result[start:end]
            analysis = json.loads(result)

            # 保存值得记住的事情
            to_remember = analysis.get("things_to_remember", [])
            saved = 0
            for item in to_remember:
                content = item.get("content", "").strip()
                mtype = item.get("type", "knowledge")
                if content:
                    store.add_memory(content=content, memory_type=mtype)
                    saved += 1

            # 记录心情摘要
            mood_text = analysis.get("mood_today", "")
            notable = analysis.get("notable", "")
            if mood_text:
                store.save_mood({
                    "summary": mood_text,
                    "emotional_trend": analysis.get("emotional_trend", ""),
                    "notable": notable,
                    "type": "daily_summary",
                }, trigger="daily_summary")

            # 清理旧活动
            store.clean_old_activities(keep_hours=48)

            print(f"[Autonomous] 📋 每日总结: {saved}条新记忆 | {mood_text[:40]}...")

        except json.JSONDecodeError as e:
            print(f"[Autonomous] 每日总结 JSON 解析失败: {e}")
        except Exception as e:
            print(f"[Autonomous] 每日总结失败: {e}")

    # --- 生成发言 ---

    def _generate_message(self, topic_type, affection, memories):
        """用 LLM 生成一条自主发言（携带内心上下文）"""
        from lilith_bot.persona import build_system_prompt
        from lilith_bot.affection_engine import mood_summary, relation_closeness

        # 使用和主对话相同的人格 prompt（频道模式，无 thinking 标签）
        system = build_system_prompt(affection, memories, channel_mode=True)
        system += "\n\n# 频道场景补充"
        system += "\n你现在在主人的群里自由发言，不是在和主人一对一对话。"

        # 注入内心上下文
        c = self.consciousness
        inner = ""
        if c.thought_buffer:
            recent = "\n".join(f"  - {t['content']}" for t in c.thought_buffer[-2:])
            inner += f"\n你最近的内心想法：\n{recent}"
        energy_desc = {0: "犯困", 1: "一般", 2: "精神"}.get(int(c.energy * 3), "一般")
        curious_desc = {0: "平静", 1: "有点好奇", 2: "充满好奇"}.get(int(c.curiosity * 3), "有点好奇")
        inner += f"\n你的状态：精力{energy_desc}，{curious_desc}"
        if inner:
            system += "\n\n# 内心状态" + inner

        # 注入最近活动
        try:
            from lilith_bot.memory_store import get_memory_store
            recent = get_memory_store().get_recent_activities(hours=6, limit=6)
            if recent:
                act_lines = []
                for a in reversed(recent):
                    emoji = {"user": "💬", "lilith_chat": "💭", "lilith_channel": "📢", "lilith_internal": "..."}.get(a["source"], "•")
                    act_lines.append(f"  {emoji} {a['summary']}")
                system += "\n\n近期发生的事：\n" + "\n".join(act_lines)
        except Exception:
            pass

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
            # 记录到活动日志
            try:
                from lilith_bot.memory_store import get_memory_store
                get_memory_store().add_activity("lilith_channel", message[:120])
            except Exception:
                pass
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

    # --- 意识状态更新 ---

    def _update_consciousness(self):
        """每个 tick 调用，更新意识状态"""
        now = time.time()
        dt = now - self._last_tick
        self._last_tick = now
        affection = self._get_affection()
        self.consciousness.update(dt, affection)

    # --- 主循环 ---

    def _loop(self):
        """后台主循环（兼顾内心独白 + 对外发言）"""
        print("[Autonomous] 自主发言引擎启动 ✓")
        self.stats["started_at"] = datetime.now().isoformat()
        self._last_tick = time.time()
        _last_thought_check = time.time()
        _last_daily_check = time.time()

        while self._running:
            try:
                now = time.time()

                # 每秒更新意识状态
                self._update_consciousness()

                # ── 每日总结：每 5 分钟检查一次 ──
                if now - _last_daily_check >= 300:
                    _last_daily_check = now
                    self._run_daily_summary()

                # ── 内心独白 + 思绪驱动发言 ──
                if now - _last_thought_check >= THOUGHT_INTERVAL:
                    _last_thought_check = now
                    if self._should_think():
                        affection = self._get_affection()
                        memories = self._get_recent_memories(3)
                        thought = self._generate_thought(affection, memories)
                        if thought and self._should_speak_from_thought(thought):
                            self._speak_from_thought(thought)

                # ── 保底发言：如果太久没说话，强制说一次 ──
                c = self.consciousness
                if c.last_spoke_at > 0 and now - c.last_spoke_at > MAX_INTERVAL:
                    # 超过最大间隔，生成一条话题发言
                    affection = self._get_affection()
                    memories = self._get_recent_memories()
                    topic_type = self._pick_topic_type(affection)
                    message = self._generate_message(topic_type, affection, memories)
                    if message:
                        success = self._push_to_channel(message)
                        if success:
                            self.stats["total_messages"] += 1
                            self.stats["last_message_at"] = datetime.now().isoformat()
                            self.stats["last_message"] = message
                            self.stats["last_topic"] = topic_type
                            self.consciousness.on_spoke()
                            self._update_self_affection(topic_type)
                            print(f"[Autonomous] [{topic_type}] {message[:60]}...")

                # 每秒轮询
                time.sleep(1)

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
            "consciousness": self.consciousness.to_dict(),
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
