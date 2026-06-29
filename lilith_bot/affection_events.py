"""莉莉丝好感度事件识别器 — 从用户消息中提取情感事件

无 LLM 调用，纯正则/关键词匹配。每个匹配产生一个 PAD 增量事件。
"""

import re


# ============ 事件定义: (正则, v_delta, a_delta, d_delta, intensity, label) ============

EVENT_PATTERNS = [
    # ---- 正面 ----
    (r"(真棒|好厉害|太好了|谢谢你|感谢|多谢|谢谢|辛苦[了啦]|做得好|干得好|优秀|天才|聪明|靠谱|可靠|有用|帮大忙)",
     0.15, 0.1, 0.1, 0.7, "被夸奖"),
    (r"(喜欢|爱你|爱死|最喜欢|好喜欢|可爱|萌|好看|漂亮)",
     0.25, 0.2, 0.05, 0.85, "被表白/喜爱"),
    (r"(累不累|休息一下|别太累|辛苦了|还好吗|没事吧|注意身体|早点睡|好好休息)",
     0.1, -0.1, 0.0, 0.6, "被关心"),
    (r"(抱抱|摸摸头|摸摸|贴贴|蹭蹭)",
     0.2, 0.15, 0.0, 0.7, "亲密互动"),
    (r"(听你的|交给你|你来决定|你说了算|相信你|信任你|你看着办|按你说的)",
     0.15, 0.05, 0.25, 0.7, "被信任"),
    (r"(哈哈哈|哈哈|笑死|好好笑|有趣|搞笑|乐|太逗了|绝了|笑)",
     0.15, 0.25, 0.05, 0.6, "开心"),
    (r"!{2,}|！{2,}",
     0.05, 0.2, 0.0, 0.5, "兴奋"),

    # ---- 负面 ----
    (r"(不对|错了|不对哦|不行|不好|太差|失望|不好用|没用|帮不上|不懂|笨|蠢)",
     -0.2, 0.1, -0.15, 0.7, "被批评"),
    (r"(算了|不用了|别说了|闭嘴|烦|别烦|别吵|走开|滚)",
     -0.3, 0.15, -0.2, 0.8, "被拒绝/驱赶"),
    (r"(难过|伤心|哭|抑郁|崩溃|绝望|难受|痛苦|不开心|心烦|郁闷|烦躁)",
     -0.15, -0.1, -0.05, 0.7, "难过"),
    (r"(生气|气愤|愤怒|怒|火大|气死|离谱|恶心|讨厌|厌烦)",
     -0.1, 0.3, 0.05, 0.7, "生气"),

    # ---- 其他 ----
    (r"(哇|天哪|我的天|不是吧|真的假的|居然|竟然|不会吧|什么鬼|震惊)",
     0.0, 0.35, -0.1, 0.75, "惊讶"),
    (r"(帮帮我|救我|救命|帮我|求求|拜托|好不好嘛|好不好|可以吗|行不行|能不能)",
     -0.05, 0.1, 0.05, 0.5, "求助"),
    (r"(我教|学一下|记住|记好了|来我告诉|听好了|听我说|你听着|注意|看好了)",
     0.05, 0.0, 0.15, 0.4, "教导"),
    (r"(送你|给你|礼物|惊喜|好东西|分享|尝一下|看这个|给你看)",
     0.3, 0.3, 0.1, 0.85, "收到礼物/分享"),
]


def detect_events(user_message):
    if not user_message:
        return []
    results = []
    for pattern, vd, ad, dd, intensity, label in EVENT_PATTERNS:
        if re.search(pattern, user_message):
            results.append({
                "valence_delta": vd, "arousal_delta": ad,
                "dominance_delta": dd, "intensity": intensity, "label": label,
            })
    return results


def merge_events(events):
    if not events:
        return None
    vd = sum(e["valence_delta"] * e["intensity"] for e in events)
    ad = sum(e["arousal_delta"] * e["intensity"] for e in events)
    dd = sum(e["dominance_delta"] * e["intensity"] for e in events)
    total_intensity = sum(e["intensity"] for e in events)
    labels = ",".join(e["label"] for e in events)
    avg_intensity = min(total_intensity / len(events), 1.0)
    vd = max(-0.5, min(0.5, vd / len(events)))
    ad = max(-0.5, min(0.5, ad / len(events)))
    dd = max(-0.5, min(0.5, dd / len(events)))
    return {
        "valence_delta": vd, "arousal_delta": ad,
        "dominance_delta": dd, "intensity": avg_intensity, "label": labels,
    }


def parse_user_message(user_message):
    """解析用户消息,返回合并后的情感事件 (供 graph 节点调用)"""
    return merge_events(detect_events(user_message))
