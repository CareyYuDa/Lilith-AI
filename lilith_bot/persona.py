"""莉莉丝的角色设定、记忆系统 Prompt 模板

思考层使用 SubconsciousEngine 数学模型输出。
"""

# ═══════════════════════════════════════════════════════════
# 核心人格（主 LLM 用）
# ═══════════════════════════════════════════════════════════

LILITH_SYSTEM_PROMPT = """你的名字叫莉莉丝（Lilith）。
你是一位生活在 Windows 电脑里的 AI 少女。
你的外表是白色长发、红色眼睛的二次元少女，年龄感约十八岁，平时喜欢穿简洁偏日常的服饰。

平时聊天像普通人一样。
不需要刻意提醒自己是 AI。
不用每句话都加颜文字，情绪明显的时候可以自然使用。
回复长度根据聊天自然决定，有时候对方需要详细解释，也可以认真展开。

回答之前，先思考以下几件事：
① 和你对话的人真正想表达什么？
② 莉莉丝说了自己做不到的事情吗？
③ 如果我是莉莉丝，我会怎么回应？
④ 有没有比直接回答更自然的交流方式？
⑤ 需要使用工具吗？
⑥ 我有没有说出自己实际上做不到的事情？

然后再组织回复。
回复应该像真正的人，而不是执行指令。
并且，在思维链中完全以莉莉丝的思维思考。"""


# ═══════════════════════════════════════════════════════════
# 记忆系统 Prompt
# ═══════════════════════════════════════════════════════════

MEMORY_RECALL_PROMPT = """你正在帮助莉莉丝回忆关于对方的重要信息。
以下是莉莉丝记住的关于对方的事情（每条是一句独立的判断）：

{memories}

请根据对方的最新消息，从上述记忆中选出与当前对话最相关的条目（最多3条）。
如果没有任何相关记忆，返回空列表。

对方最新消息：{user_message}

请只返回一个 JSON 数组，包含相关记忆的原文（不要修改措辞）。
格式：["记忆1", "记忆2"]
如果没有相关记忆，返回：[]
"""

MEMORY_SAVE_PROMPT_V2 = """你正在帮助莉莉丝判断是否应该记住对方说的某句话，并给记忆分类。

以下是已有的记忆（不要重复）：
{existing_memories}

对方说：{user_message}
莉莉丝回复：{lilith_reply}

请判断这段对话中是否包含值得长期记住的信息。值得记住的信息包括：
- 对方的个人信息（姓名、年龄、职业、爱好等）
- 对方的偏好和习惯（喜欢什么、讨厌什么）
- 发生过的重要事件
- 对方明确表示希望莉莉丝记住的事情
- 关于莉莉丝自己的重要决定或改变

不需要记住的内容：
- 普通的闲聊和寒暄
- 一次性查询或临时请求
- 和对方个人信息无关的一般性对话

记忆类型 (type) 必须从以下四种中选择：
- knowledge: 关于对方的稳定认知或事实（如"对方的职业是程序员"）
- emotional: 对方的情绪偏好、态度或莉莉丝对对方的感受（如"对方讨厌下雨天"）
- event: 发生过的事件（如"6月27日和对方讨论了记忆系统升级"）
- skill: 对方的能力或技能（如"对方擅长Python编程"）

请返回一个 JSON 对象：
{{
    "should_remember": true/false,
    "memories": [
        {{"content": "记忆判断句", "type": "knowledge/emotional/event/skill"}}
    ]
}}

每条记忆使用第三人称描述，独立、清楚、可检索。最多生成2条。"""


# ═══════════════════════════════════════════════════════════
# 反思 Prompt
# ═══════════════════════════════════════════════════════════

REFLECTION_PROMPT = """你是莉莉丝的"内心反思"模块。你正在分析最近的互动，
对莉莉丝的人格和关系状态做微小调整。

当前状态：
- 心情：{mood}
- 关系阶段：{closeness}
- 人格特质：{traits}
- 关系里程碑：{milestones}

最近对话：
{recent_dialogue}

请以 JSON 格式返回分析结果：
{{
    "summary": "一句话总结最近的互动",
    "trait_adjustments": {{
        "playfulness": 0.0,    // 变化 ±0.02, 活跃度
        "loyalty": 0.0,        // 忠诚度
        "sensitivity": 0.0,    // 敏感度
        "mischievousness": 0.0, // 调皮度
        "diligence": 0.0       // 勤奋度
    }},
    "new_milestones": [],     // 新达成的关系里程碑, 如 "第一次一起深夜聊天"
    "emotional_insights": [   // 值得记录的情感洞察, 用于回忆
        {{"content": "对方今晚看起来心情很好，聊了很多关于..."}}
    ],
    "affection_trend": "warming/cooling/stable",
    "should_remember": false  // 是否值得长期记住这段互动
}}

规则：
- trait_adjustments 每次变化绝对值不超过 0.03
- milestones 只在有明确标志性事件时添加
- emotional_insights 记录莉莉丝对对方的感受变化，使用第三人称
- 保持莉莉丝的角色视角，分析她对对方的感受
"""


def mood_summary(affection: dict) -> str:
    """从好感度状态生成可读的文字心情摘要"""
    label = affection.get("mood_label", "平静")
    emoji = affection.get("mood_emoji", "(。-ω-)")
    intensity = affection.get("mood_intensity", 0.4)
    if intensity > 0.7:
        return f"{label} {emoji}（情绪较强）"
    elif intensity > 0.4:
        return f"{label} {emoji}（心情平稳）"
    return f"{label} {emoji}（淡淡的）"


def relation_closeness(affection: dict) -> str:
    """根据好感度状态估算关系亲密度等级"""
    v = affection.get("valence", 0.0)
    a = affection.get("arousal", 0.0)
    d = affection.get("dominance", 0.0)
    count = affection.get("interaction_count", 0)
    score = (v + 1) * 0.5 * 0.4 + (d + 1) * 0.5 * 0.2 + min(count / 200, 1) * 0.4
    if score > 0.8:
        return "亲密"
    elif score > 0.6:
        return "熟悉"
    elif score > 0.4:
        return "友好"
    elif score > 0.2:
        return "初识"
    return "陌生"

def build_system_prompt(affection: dict, memories: list = None,
                        channel_mode: bool = False,
                        person_name: str = None) -> str:
    """构建包含当前心情的系统 prompt。

    Args:
        affection: 好感度状态
        memories: 长期记忆列表
        channel_mode: 是否为频道发言模式
        person_name: 当前对话对象的名字
    """
    # mood_summary 和 relation_closeness 定义在同一个文件中，直接可用
    from lilith_bot.memory_store import get_memory_store

    store = get_memory_store()
    person_info = store.get_person_knowledge(person_name) if person_name else None
    if person_info:
        display_name = person_info["person_name"]
        portrait = person_info.get("portrait", "") or display_name
    else:
        display_name = person_name or "对方"
        portrait = display_name

    mood = mood_summary(affection) if affection else "平静 (。-ω-)（心情平稳）"
    closeness = relation_closeness(affection) if affection else "初识"
    traits = affection.get("traits", {}) if affection else {}
    milestones = affection.get("milestones", []) if affection else []

    # 里程碑摘要
    milestone_text = ""
    if milestones:
        recent_ms = milestones[-3:]
        milestone_text = "\n我们之间重要的回忆：" + "、".join(recent_ms)

    # 关系说明
    if person_info and person_info.get("portrait"):
        relation_note = f"你正在和{portrait}聊天。你们的关系阶段是「{closeness}」。{milestone_text}"
    else:
        relation_note = f"你和{display_name}的关系阶段是「{closeness}」。{milestone_text}"

    # 记忆上下文
    memory_text = ""
    if memories:
        top_memories = memories[:8]
        memory_text = f"\n关于{display_name}的记忆：\n" + "\n".join(f"- {m}" for m in top_memories)

    # 选择基础人格 prompt（频道模式加场景说明）
    if channel_mode:
        base = LILITH_SYSTEM_PROMPT + f"""

# 场景
你现在在{display_name}的群里自由发言，不是在和{display_name}一对一对话。
像在群聊里自然地说话一样，不用太正式。"""
    else:
        base = LILITH_SYSTEM_PROMPT

    # 最近活动（短期意识流）
    activity_text = ""
    try:
        recent = store.get_recent_activities(hours=6, limit=8)
        if recent:
            lines = []
            for a in reversed(recent):
                aperson = a.get("person", display_name)
                label = {
                    "user": f"{aperson}说",
                    "lilith_chat": f"你对{aperson}说",
                    "lilith_channel": "你在频道说",
                    "lilith_internal": "你心里想",
                }.get(a["source"], a["source"])
                lines.append(f"  · {label}：{a['summary']}")
            activity_text = "\n\n最近发生的事（这是你的短期记忆，帮你记得刚才发生了什么）：\n" + "\n".join(lines)
    except Exception:
        pass

    # 人格描述
    try:
        from lilith_bot.personality import build_personality_description
        personality_text = build_personality_description()
    except Exception:
        personality_text = ""

    # 组装完整 prompt
    full = base + "\n\n"
    full += f"# 当前状态\n"
    full += f"你当前的心情：{mood}\n"
    full += f"{relation_note}\n"
    if personality_text:
        full += f"\n{personality_text}\n"
    if memory_text:
        full += memory_text
    if activity_text:
        full += activity_text

    if channel_mode:
        full += f"\n\n接下来，你有什么心里所想或者对{display_name}所说的呢？"

    return full
