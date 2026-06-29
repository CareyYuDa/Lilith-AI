"""反馈系统 — 用 LLM 分析对话，输出感受词 + 分层修正

三层反馈：
  快速层（情绪）— PAD 即时变化，但快速衰减
  中速层（激素）— 驱动力微调，缓慢衰减
  慢速层（人格）— 极微量变化，几乎不衰减

每次对话结束后，在 save_memory_node 中调用。
"""

import json
import random
from typing import Optional

from langchain_core.messages import HumanMessage


# ============ 感受词 → 多维映射表 ============

# 每个感受词映射到: {pad_delta, hormone_delta, personality_delta}
# 数值范围: pad ±0.05, hormone ±0.03, personality ±0.003
FEELING_WORD_MAP = {
    # === 正向情绪 ===
    "开心": {
        "pad": {"valence": 0.04, "arousal": 0.03, "dominance": 0.02},
        "hormones": {"playfulness": 0.02, "anxiety": -0.02},
        "personality": {"initiative": 0.001, "dominance": 0.0005},
    },
    "兴奋": {
        "pad": {"valence": 0.05, "arousal": 0.04, "dominance": 0.03},
        "hormones": {"curiosity": 0.02, "playfulness": 0.03, "anxiety": -0.01},
        "personality": {"initiative": 0.002},
    },
    "满足": {
        "pad": {"valence": 0.04, "arousal": -0.02, "dominance": 0.02},
        "hormones": {"anxiety": -0.03, "belonging": 0.01},
        "personality": {"dominance": 0.001},
    },
    "放松": {
        "pad": {"valence": 0.03, "arousal": -0.03, "dominance": 0.01},
        "hormones": {"anxiety": -0.03, "playfulness": 0.01},
        "personality": {},
    },
    "自信": {
        "pad": {"valence": 0.03, "dominance": 0.04},
        "hormones": {"anxiety": -0.02, "achievement": 0.02},
        "personality": {"dominance": 0.002, "initiative": 0.001},
    },
    "温暖": {
        "pad": {"valence": 0.04, "arousal": 0.01, "dominance": 0.01},
        "hormones": {"belonging": 0.03, "anxiety": -0.02},
        "personality": {},
    },
    "被认可": {
        "pad": {"valence": 0.05, "dominance": 0.03},
        "hormones": {"achievement": 0.03, "anxiety": -0.02},
        "personality": {"initiative": 0.002, "dominance": 0.001},
    },
    "被关心": {
        "pad": {"valence": 0.03, "arousal": -0.01},
        "hormones": {"belonging": 0.03, "anxiety": -0.03},
        "personality": {},
    },
    "好奇": {
        "pad": {"arousal": 0.03, "dominance": 0.01},
        "hormones": {"curiosity": 0.03, "playfulness": 0.01},
        "personality": {"initiative": 0.001},
    },
    "有趣": {
        "pad": {"valence": 0.03, "arousal": 0.03},
        "hormones": {"playfulness": 0.02, "curiosity": 0.02},
        "personality": {},
    },

    # === 负向情绪 ===
    "难过": {
        "pad": {"valence": -0.04, "arousal": -0.02, "dominance": -0.02},
        "hormones": {"anxiety": 0.02, "belonging": 0.02},
        "personality": {"initiative": -0.001},
    },
    "生气": {
        "pad": {"valence": -0.04, "arousal": 0.04, "dominance": 0.03},
        "hormones": {"anxiety": 0.03, "playfulness": -0.02},
        "personality": {"dominance": 0.002},
    },
    "焦虑": {
        "pad": {"valence": -0.03, "arousal": 0.03, "dominance": -0.03},
        "hormones": {"anxiety": 0.04, "curiosity": -0.01},
        "personality": {"initiative": -0.001, "dominance": -0.001},
    },
    "失落": {
        "pad": {"valence": -0.03, "arousal": -0.02, "dominance": -0.02},
        "hormones": {"anxiety": 0.02, "achievement": -0.02},
        "personality": {"initiative": -0.002},
    },
    "被忽视": {
        "pad": {"valence": -0.03, "arousal": -0.01, "dominance": -0.02},
        "hormones": {"belonging": 0.03, "anxiety": 0.02},
        "personality": {"initiative": -0.002},
    },
    "被批评": {
        "pad": {"valence": -0.04, "arousal": 0.02, "dominance": -0.03},
        "hormones": {"anxiety": 0.03, "achievement": -0.03},
        "personality": {"dominance": 0.001},
    },
    "无聊": {
        "pad": {"valence": -0.02, "arousal": -0.03},
        "hormones": {"curiosity": -0.02, "playfulness": -0.02},
        "personality": {"initiative": -0.001},
    },
    "困惑": {
        "pad": {"arousal": 0.02, "dominance": -0.02},
        "hormones": {"curiosity": 0.02, "anxiety": 0.02},
        "personality": {},
    },

    # === 中性/特殊 ===
    "平静": {
        "pad": {},
        "hormones": {"anxiety": -0.01},
        "personality": {},
    },
    "期待": {
        "pad": {"valence": 0.02, "arousal": 0.03},
        "hormones": {"curiosity": 0.02, "playfulness": 0.01},
        "personality": {"initiative": 0.001},
    },
    "依赖": {
        "pad": {"valence": 0.02, "dominance": -0.03},
        "hormones": {"belonging": 0.03},
        "personality": {"dominance": -0.001},
    },
    "害羞": {
        "pad": {"valence": 0.01, "arousal": 0.02, "dominance": -0.03},
        "hormones": {"anxiety": 0.01, "belonging": 0.01},
        "personality": {},
    },
}

# 默认反馈（没有匹配的感受词时）
DEFAULT_FEEDBACK = {
    "pad": {"valence": 0.01, "arousal": 0.01, "dominance": 0.0},
    "hormones": {},
    "personality": {},
}


# ============ LLM 反馈分析 Prompt ============

FEEDBACK_PROMPT = """你是一个对话分析器。分析以下对话，判断莉莉丝在这次互动后应该有什么感受，以及是否需要记住这段对话。

莉莉丝当前状态：
- 心情：{mood_label}
- 愉悦度：{valence}（-1~1）
- 唤醒度：{arousal}（-1~1）
- 支配度：{dominance}（-1~1）

用户：{user_message}
莉莉丝：{lilith_reply}

请用以下格式输出（三行，不要多也不要少）：

第一行：感受词（从以下列表选择 1~2 个）
{feeling_words}

第二行：记忆重要性（0~1 之间的小数，0=不重要，1=非常重要，只输出数字）

第三行：记忆内容（一句判断句，如果不重要就写"无"）
"""

# 如果都不符合，直接输出你认为最合适的感受词。"""


def get_available_feeling_words():
    """返回所有感受词列表"""
    return list(FEELING_WORD_MAP.keys())


# ============ 噪声注入 ============

def add_noise(value: float, noise_level: float = 0.3) -> float:
    """向数值添加高斯噪声"""
    if value == 0:
        return 0.0
    std = abs(value) * noise_level
    noise = random.gauss(0, std)
    result = value + noise
    # 保持符号不变
    if value > 0 and result < 0:
        result = 0.0
    elif value < 0 and result > 0:
        result = 0.0
    return result


# ============ 主反馈函数 ============

class FeedbackSystem:
    """反馈系统"""

    def __init__(self, fast_llm=None):
        """
        Args:
            fast_llm: 用于分析的 ChatOpenAI 实例
        """
        self._llm = fast_llm

    def _get_llm(self):
        """惰性获取 LLM 实例"""
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            import os
            self._llm = ChatOpenAI(
                model=os.getenv("LLM_CHAT_MODEL", "deepseek-v4-flash"),
                api_key=os.getenv("LLM_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL", "https://opencode.ai/zen/go/v1"),
                temperature=0.3,
                max_tokens=64,
            )
        return self._llm

    def analyze(self, user_message: str, lilith_reply: str,
                pad: dict, interaction_count: int = 0) -> dict:
        """
        分析一轮对话，返回分层反馈和记忆建议。

        Args:
            user_message: 用户消息
            lilith_reply: 莉莉丝的回复
            pad: 当前情绪 {valence, arousal, dominance}
            interaction_count: 总交互次数（用于人格稀释）

        Returns:
            {
                "feeling_words": ["开心"],
                "memory_importance": 0.7,
                "memory_content": "小明喜欢打FPS游戏",
                "pad_delta": {"valence": 0.04, ...},
                "hormone_delta": {"playfulness": 0.02, ...},
                "personality_delta": {"initiative": 0.001, ...},
                "noise_applied": True,
            }
        """
        # 1. LLM 分析感受词 + 记忆重要性
        feeling_words, importance, memory = self._get_feeling_words(
            user_message, lilith_reply, pad)

        # 2. 合并感受词对应的 delta
        raw = self._merge_feeling_words(feeling_words)

        # 3. 用重要性缩放情绪反馈倍数
        importance_multiplier = (importance or 0.5) * 0.5 + 0.5  # 0.5~1.0
        for k in raw["pad"]:
            raw["pad"][k] *= importance_multiplier
        for k in raw["hormones"]:
            raw["hormones"][k] *= importance_multiplier

        # 4. 加噪声
        noisy = {
            "pad": {k: add_noise(v) for k, v in raw["pad"].items()},
            "hormones": {k: add_noise(v) for k, v in raw["hormones"].items()},
            "personality": {k: add_noise(v, 0.5) for k, v in raw["personality"].items()},
        }

        return {
            "feeling_words": feeling_words,
            "memory_importance": importance,
            "memory_content": memory,
            "pad_delta": noisy["pad"],
            "hormone_delta": noisy["hormones"],
            "personality_delta": noisy["personality"],
            "noise_applied": True,
        }

    def _get_feeling_words(self, user_message: str, lilith_reply: str,
                           pad: dict) -> tuple:
        """用 LLM 获取感受词 + 记忆重要性 + 记忆内容"""
        try:
            llm = self._get_llm()
            feeling_list = "\n".join(f"- {w}" for w in get_available_feeling_words())

            prompt = FEEDBACK_PROMPT.format(
                mood_label=pad.get("mood_label", "平静"),
                valence=f"{pad.get('valence', 0):.2f}",
                arousal=f"{pad.get('arousal', 0):.2f}",
                dominance=f"{pad.get('dominance', 0):.2f}",
                user_message=user_message[:300],
                lilith_reply=lilith_reply[:300],
                feeling_words=feeling_list,
            )

            response = llm.invoke([HumanMessage(content=prompt)])
            text = response.content.strip()

            # 解析三行输出
            lines = text.split("\n")
            # 第一行：感受词
            first_line = lines[0].strip().strip("-* ") if lines else ""
            words = []
            for w in get_available_feeling_words():
                if w in first_line:
                    words.append(w)
            if not words and first_line:
                words = [first_line]

            # 第二行：记忆重要性
            importance = 0.5
            if len(lines) > 1:
                try:
                    importance = max(0.0, min(1.0, float(lines[1].strip())))
                except ValueError:
                    pass

            # 第三行：记忆内容
            memory = ""
            if len(lines) > 2:
                memory = lines[2].strip()
                if memory == "无":
                    memory = ""

            return words[:2], importance, memory

        except Exception as e:
            print(f"[Feedback] LLM分析失败: {e}")
            return ["平静"], 0.5, ""

    def _merge_feeling_words(self, words: list) -> dict:
        """合并多个感受词的 delta"""
        if not words:
            return dict(DEFAULT_FEEDBACK)

        merged = {
            "pad": {},
            "hormones": {},
            "personality": {},
        }

        for word in words:
            mapping = FEELING_WORD_MAP.get(word, DEFAULT_FEEDBACK)
            for layer in ("pad", "hormones", "personality"):
                for k, v in mapping.get(layer, {}).items():
                    merged[layer][k] = merged[layer].get(k, 0) + v

        # 加权平均（多个词效果折中）
        n = len(words)
        for layer in merged:
            for k in merged[layer]:
                merged[layer][k] /= n

        return merged


# ============ 便捷函数 ============

_default_feedback_system = None


def get_feedback_system(llm=None) -> FeedbackSystem:
    """获取全局反馈系统实例"""
    global _default_feedback_system
    if _default_feedback_system is None:
        _default_feedback_system = FeedbackSystem(llm=llm)
    return _default_feedback_system

