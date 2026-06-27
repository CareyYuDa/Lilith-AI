"""莉莉丝的对话状态定义 — LangGraph State Schema

阶段三：好感度系统重构 — PAD三维情绪 + 人格特质 + 关系里程碑。
"""

from typing import Annotated, TypedDict, Optional
from langgraph.graph.message import add_messages


class LilithState(TypedDict):
    """
    莉莉丝的对话状态。

    Attributes:
        messages: 对话消息历史。使用 add_messages reducer 自动合并新消息。
        persona: 当前激活的角色设定 prompt（默认使用 LILITH_SYSTEM_PROMPT）。
        long_term_memories: 长期记忆列表。每条记忆是一句独立的判断句。
        affection: 好感度系统完整状态。
        llm_type: LLM 类型 ("local" / "lili")，控制使用哪个后端。
    """
    messages: Annotated[list, add_messages]
    persona: str
    long_term_memories: list[str]      # 长期记忆条目
    affection: dict                    # 好感度系统 (见 AFFECTION_DEFAULT)
    llm_type: str                      # "local" | "lili"


# --- 好感度系统默认值 ---

AFFECTION_DEFAULT: dict = {
    # === PAD 三维情绪 (-1.0 ~ 1.0) ===
    "valence": 0.3,          # 愉悦度：负面↔正面
    "arousal": 0.1,          # 唤醒度：平静↔兴奋
    "dominance": 0.0,        # 支配度：顺从↔自信

    # === 衍生状态 ===
    "mood_label": "平静",     # PAD → 离散情绪标签
    "mood_intensity": 0.4,    # 情绪总强度 0~1
    "mood_emoji": "(。-ω-)",  # 情绪颜文字

    # === 人格特质 (0.0 ~ 1.0，相对稳定) ===
    "traits": {
        "playfulness": 0.8,   # 活泼
        "loyalty": 0.9,       # 忠诚
        "sensitivity": 0.6,   # 敏感
        "mischievousness": 0.3,  # 调皮
        "diligence": 0.8,     # 勤奋
    },

    # === 关系里程碑 ===
    "milestones": [],         # ["第一次一起写代码", "知道了主人的生日", ...]

    # === 统计 ===
    "interaction_count": 0,   # 总交互轮数
    "last_reflection_at": 0,  # 上次反思时的 interaction_count

    # === 近期心情历史（滑动窗口） ===
    "mood_history": [],       # [{label, intensity, interaction}, ...] 最近 20 条
}
