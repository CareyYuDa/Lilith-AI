"""莉莉丝好感度引擎 — PAD情绪模型 + 衰减 + 昼夜节律 + 特质调节

方案四混合架构的"实时层"：纯数学运算，无 LLM 调用，低延迟。

核心机制:
  1. PAD 三维情绪 (Pleasure-Arousal-Dominance)
  2. 情感事件 → PAD 增量 (来自 affection_events.py)
  3. 自然衰减 (向 baseline 回归)
  4. 昼夜节律 (时间影响 arousal/valence)
  5. 人格特质调节 (trait 影响事件反应强度)
  6. PAD → 离散情绪标签映射
"""

import math
import time


# ============ 常量 ============

DECAY_RATE = 0.0003           # 衰减速率, ~5分钟半衰
BASELINE_PAD = (0.2, -0.05, 0.05)  # (valence, arousal, dominance) 情感休息态

CIRCADIAN_AROUSAL = {
    0: -0.3, 1: -0.4, 2: -0.4, 3: -0.4, 4: -0.3, 5: -0.2,
    6: 0.0,  7: 0.1,  8: 0.2,  9: 0.25, 10: 0.3, 11: 0.3,
    12: 0.2, 13: 0.15, 14: 0.1, 15: 0.2, 16: 0.3, 17: 0.35,
    18: 0.3, 19: 0.2, 20: 0.1, 21: 0.05, 22: -0.1, 23: -0.2,
}

CIRCADIAN_VALENCE = {
    0: 0.0, 1: -0.05, 2: -0.05, 3: -0.05, 4: 0.0, 5: 0.05,
    6: 0.05, 7: 0.1, 8: 0.1, 9: 0.1, 10: 0.1, 11: 0.1,
    12: 0.1, 13: 0.1, 14: 0.1, 15: 0.1, 16: 0.1, 17: 0.05,
    18: 0.05, 19: 0.05, 20: 0.05, 21: 0.0, 22: 0.0, 23: 0.0,
}

# (vc, ac, dc, radius, label, emoji)
PAD_EMOTIONS = [
    (0.8, 0.7, 0.5, 0.35, "兴奋", "ヽ(>∀<☆)ノ"),
    (0.6, 0.6, 0.3, 0.35, "开心", "(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧"),
    (0.7, -0.3, 0.3, 0.35, "满足", "(´∀｀)♡"),
    (0.5, -0.4, 0.2, 0.35, "放松", "(ᵕ—ᴗ—)"),
    (-0.6, 0.6, -0.3, 0.40, "生气", "(╬ Ò﹏Ó)"),
    (-0.3, 0.5, -0.4, 0.40, "焦虑", "(｡•́︿•̀｡)"),
    (-0.7, -0.4, -0.4, 0.35, "伤心", "(╥﹏╥)"),
    (-0.4, -0.3, -0.2, 0.40, "失落", "(っ- ‸ - ς)"),
    (0.4, 0.3, 0.7, 0.35, "自信", "(￣▽￣)b"),
    (0.1, 0.1, 0.6, 0.35, "从容", "( ͡° ͜ʖ ͡°)"),
    (-0.1, 0.3, -0.6, 0.40, "害羞", "(⁄⁄•⁄ω⁄•⁄⁄)"),
    (0.0, -0.2, -0.5, 0.40, "依赖", "(๑•́ ₃ •̀๑)"),
    (0.4, 0.2, 0.1, 0.25, "愉悦", "(●'◡'●)"),
    (0.1, 0.1, 0.0, 0.25, "平静", "(。-ω-)"),
    (-0.1, 0.2, 0.0, 0.30, "困惑", "(⊙.⊙)?"),
    (0.3, 0.5, 0.2, 0.30, "期待", "(✧ω✧)"),
]


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def get_circadian_adjustment(hour=None):
    if hour is None:
        hour = time.localtime().tm_hour
    return CIRCADIAN_VALENCE.get(hour, 0.0), CIRCADIAN_AROUSAL.get(hour, 0.0)


def apply_decay(affection, elapsed_seconds):
    rate = 1.0 - math.exp(-DECAY_RATE * elapsed_seconds)
    for dim, baseline in zip(["valence", "arousal", "dominance"], BASELINE_PAD):
        current = affection[dim]
        affection[dim] = current + rate * (baseline - current)
    return affection


def apply_circadian(affection, hour=None):
    dv, da = get_circadian_adjustment(hour)
    affection["_circadian_dv"] = dv
    affection["_circadian_da"] = da
    return affection


def apply_event(affection, event, traits=None):
    intensity = event.get("intensity", 0.5)
    vd = event.get("valence_delta", 0.0)
    ad = event.get("arousal_delta", 0.0)
    dd = event.get("dominance_delta", 0.0)

    if traits:
        sensitivity = traits.get("sensitivity", 0.5)
        playfulness = traits.get("playfulness", 0.5)
        intensity *= 0.5 + sensitivity
        if vd > 0:
            vd *= 1.0 + playfulness * 0.5
        if ad > 0:
            ad *= 1.0 + playfulness * 0.5

    affection["valence"] = clamp(affection["valence"] + vd * intensity, -1.0, 1.0)
    affection["arousal"] = clamp(affection["arousal"] + ad * intensity, -1.0, 1.0)
    affection["dominance"] = clamp(affection["dominance"] + dd * intensity, -1.0, 1.0)
    return affection


def compute_mood(affection):
    v = affection.get("valence", 0.0) + affection.get("_circadian_dv", 0.0)
    a = affection.get("arousal", 0.0) + affection.get("_circadian_da", 0.0)
    d = affection.get("dominance", 0.0)

    best_emo = ("平静", "(。-ω-)")
    best_dist = float("inf")
    for vc, ac, dc, radius, label, emoji in PAD_EMOTIONS:
        dist = math.sqrt((v - vc)**2 + (a - ac)**2 + (d - dc)**2)
        if dist / radius < best_dist:
            best_dist = dist / radius
            best_emo = (label, emoji)

    raw = math.sqrt(v**2 + a**2 + abs(d)**0.8) / math.sqrt(3)
    intensity = clamp(raw, 0.1, 1.0)
    return best_emo[0], intensity, best_emo[1]


def update_affection_state(affection, event=None, elapsed_seconds=0.0):
    """一站式好感度更新 (实时层入口, graph节点调用)"""
    if elapsed_seconds > 0:
        apply_decay(affection, elapsed_seconds)
    if event:
        apply_event(affection, event, affection.get("traits", {}))
    apply_circadian(affection)

    label, intensity, emoji = compute_mood(affection)
    affection["mood_label"] = label
    affection["mood_intensity"] = intensity
    affection["mood_emoji"] = emoji
    affection["interaction_count"] = affection.get("interaction_count", 0) + 1

    history = affection.get("mood_history", [])
    history.append({"label": label, "intensity": intensity,
                    "interaction": affection["interaction_count"]})
    if len(history) > 20:
        history = history[-20:]
    affection["mood_history"] = history
    return affection


def mood_trend(affection):
    history = affection.get("mood_history", [])
    if len(history) < 3:
        return "stable"
    vals = [h["intensity"] for h in history[-3:]]
    if vals[-1] > vals[0] + 0.15:
        return "rising"
    elif vals[-1] < vals[0] - 0.15:
        return "falling"
    return "stable"


def mood_summary(affection):
    label = affection.get("mood_label", "平静")
    emoji = affection.get("mood_emoji", "")
    intensity = affection.get("mood_intensity", 0.4)
    trend = mood_trend(affection)

    if intensity > 0.8:
        adverb = "非常"
    elif intensity > 0.5:
        adverb = "挺"
    else:
        adverb = "有点"

    trend_phrase = {"rising": "心情正在变好", "falling": "心情正在低落",
                    "stable": "心情平稳"}.get(trend, "")
    return f"{adverb}{label} {emoji}（{trend_phrase}）"


def relation_closeness(affection):
    count = affection.get("interaction_count", 0)
    milestones = affection.get("milestones", [])
    if count < 20 and len(milestones) == 0:
        return "初识"
    elif count < 100 or len(milestones) <= 2:
        return "熟悉"
    elif len(milestones) <= 5:
        return "亲密"
    else:
        return "挚友"
