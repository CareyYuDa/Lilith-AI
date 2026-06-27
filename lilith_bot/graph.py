"""莉莉丝的 LangGraph 对话图 — 阶段三：好感度系统重构

Graph 结构:
    START -> recall_memory -> update_affection -> chatbot(local + tools) -> [route]
                                   ^                     |-- "tools" -> tool_node -> chatbot (loop)
                                   +---------------------+-- "save_memory" -> [maybe_reflect] -> END

新增节点:
    update_affection: PAD实时层，基于用户消息更新情绪（无LLM）
    maybe_reflect:    条件节点，每N轮触发LLM深度反思
    evolution:        自演化节点，AI观察自身并修改代码
"""

import os
import sys
import json
import time
import sqlite3
from dotenv import load_dotenv

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lilith_bot.reasoning_patch import *  # noqa: F401, F403

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from lilith_bot.state import LilithState, AFFECTION_DEFAULT, EVOLUTION_DEFAULT
from lilith_bot.persona import (
    LILITH_SYSTEM_PROMPT, LOCAL_TOOL_HINT,
    MEMORY_SAVE_PROMPT_V2, build_system_prompt, REFLECTION_PROMPT,
)
from lilith_bot.tools import LANGCHAIN_TOOLS, ask_api_assistant as _ask_api_tool
from lilith_bot.memory_store import get_memory_store
from lilith_bot.affection_engine import (
    update_affection_state, mood_summary, relation_closeness,
)
from lilith_bot.affection_events import parse_user_message
from lilith_bot.evolution_engine import get_evolution_engine

import re


# ============ 手动触发 API 助手的关键词 ============

_API_TRIGGER_PATTERNS = [
    r"另一个.{0,4}(莉莉|你|她|家伙)",
    r"换.{0,4}(莉莉|你)",
    r"叫.{0,6}(莉莉|你).{0,4}(来|帮|答)",
    r"姐姐.{0,4}(来|帮|答|说)",
    r"更(厉害|聪明|厉害|好用).{0,6}(你|莉莉|AI|模型)",
    r"让.{0,4}(来|帮|答)",
]


def _detect_api_trigger(user_message):
    for pattern in _API_TRIGGER_PATTERNS:
        if re.search(pattern, user_message):
            return True
    return False


# ============ 思维链解析 ============

_THINK_PATTERN = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL)


def _parse_thinking(content):
    if not content or (
        "<think>" not in content
        and "<thinking>" not in content
        and "</think>" not in content
        and "</thinking>" not in content
    ):
        return "", content
    # 提取所有 thinking 块的内容（用 findall 而非 search，处理多个块）
    all_reasoning = _THINK_PATTERN.findall(content)
    reasoning = "\n\n".join(r.strip() for r in all_reasoning if r.strip())
    # 移除所有 thinking 标签（包括 mismatched 的残留）
    clean = _THINK_PATTERN.sub("", content).strip()
    # 额外清理可能残留的裸标签（例如只有 </thinking> 没有 <thinking> 的情况）
    clean = clean.replace("<thinking>", "").replace("</thinking>", "")
    clean = clean.replace("<think>", "").replace("</think>", "")
    clean = clean.strip()
    return reasoning, clean


def _apply_thinking_to_message(msg):
    if not hasattr(msg, 'additional_kwargs') or not hasattr(msg, 'content'):
        return
    if msg.additional_kwargs.get('reasoning_content'):
        return
    reasoning, clean_content = _parse_thinking(msg.content or '')
    if reasoning:
        msg.additional_kwargs['reasoning_content'] = reasoning
    # 始终使用 clean_content（移除标签后的内容），即使没有提取到 reasoning
    if clean_content != msg.content:
        msg.content = clean_content


# ============ LLM 实例 ============

_fast_llm = None
_local_llm = None
_local_llm_with_tools = None
_db_path = None


def DB_PATH():
    return os.path.join(_PROJECT_ROOT, "lilith_memory.db")


def get_checkpointer():
    from langgraph.checkpoint.sqlite import SqliteSaver
    db_path = os.path.join(_PROJECT_ROOT, "lilith_memory.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


def get_fast_llm():
    global _fast_llm
    if _fast_llm is None:
        _fast_llm = ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://opencode.ai/zen/go/v1"),
            temperature=0.3,
            max_tokens=4096,
        )
    return _fast_llm


def get_local_llm():
    global _local_llm
    if _local_llm is None:
        _local_llm = ChatOpenAI(
            model=os.getenv("LOCAL_MODEL", "qwen"),
            api_key="not-needed",
            base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:1234/v1"),
            temperature=0.8,
            max_tokens=4096,
        )
    return _local_llm


def get_local_llm_with_tools():
    global _local_llm_with_tools
    if _local_llm_with_tools is None:
        _local_llm_with_tools = ChatOpenAI(
            model=os.getenv("LOCAL_MODEL", "qwen"),
            api_key="not-needed",
            base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:1234/v1"),
            temperature=0.8,
            max_tokens=4096,
        ).bind_tools(LANGCHAIN_TOOLS)
    return _local_llm_with_tools


def get_lili_llm():
    """远程 lili 模型 (API Agent, 适合 coding)"""
    return ChatOpenAI(
        model=os.getenv("LILI_MODEL", "deepseek-v4"),
        api_key=os.getenv("LILI_API_KEY"),
        base_url=os.getenv("LILI_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0.3,
        max_tokens=4096,
    )


def get_reflection_llm():
    """深度反思用 LLM (低温度, 高 token)"""
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://opencode.ai/zen/go/v1"),
        temperature=0.3,
        max_tokens=4096,
    )


# ============ Graph 节点 ============

MEMORY_MAX_COUNT = 50
REFLECTION_INTERVAL = 5  # 每 N 轮对话触发一次深度反思


def recall_memory_node(state: LilithState):
    """记忆召回节点"""
    messages = state["messages"]
    user_msg = ""
    for m in reversed(messages):
        if hasattr(m, "type") and m.type == "human":
            user_msg = m.content
            break

    memories = state.get("long_term_memories", [])
    if user_msg and len(user_msg) >= 3:
        try:
            store = get_memory_store()
            recalled = store.recall(user_msg, limit=5)
            memories = recalled + state.get("long_term_memories", [])
            memories = list(dict.fromkeys(memories))[:MEMORY_MAX_COUNT]
        except Exception as e:
            print(f"[Lilith] recall_memory failed: {type(e).__name__}: {e}")

    return {"long_term_memories": memories}


def update_affection_node(state: LilithState) -> dict:
    """好感度实时更新节点 (PAD层, 无LLM)

    1. 获取用户最新消息
    2. 解析情感事件
    3. 更新PAD + 衰减 + 昼夜节律
    """
    messages = state["messages"]
    user_msg = ""
    for m in reversed(messages):
        if hasattr(m, "type") and m.type == "human":
            user_msg = m.content
            break

    affection = state.get("affection")
    if affection is None or not isinstance(affection, dict) or "valence" not in affection:
        affection = dict(AFFECTION_DEFAULT)

    # 估算距上次更新的秒数 (粗略: 假设每次对话间隔平均60秒)
    elapsed = 60.0

    # 解析事件
    event = parse_user_message(user_msg)

    # 更新状态
    update_affection_state(affection, event, elapsed)

    return {"affection": affection}


def chatbot_node(state: LilithState):
    """核心对话节点

    构建包含 mood 的系统 prompt, 调用本地模型。
    """
    messages = list(state["messages"])
    memories = state.get("long_term_memories", [])
    affection = state.get("affection", {})
    if not isinstance(affection, dict) or "valence" not in affection:
        affection = dict(AFFECTION_DEFAULT)

    # 构建带 mood 的系统 prompt
    system_text = build_system_prompt(affection, memories)

    # 强检测试: 是否需要强制调用 API 助手
    user_msg = ""
    for m in reversed(messages):
        if hasattr(m, "type") and m.type == "human":
            user_msg = m.content or ""
            break

    if _detect_api_trigger(user_msg):
        # 强制调用 ask_api_assistant
        from langchain_core.messages import ToolMessage
        try:
            context = ""
            if memories:
                context = "相关记忆:\n" + "\n".join(f"- {m}" for m in memories[:10])
            result = _ask_api_tool.invoke({
                "question": user_msg,
                "context": context,
            })
            ai_msg = AIMessage(content=result)
            return {"messages": [ai_msg]}
        except Exception as e:
            print(f"[Lilith] API assistant failed, falling through: {type(e).__name__}: {e}")

    # 正常流程: 本地模型 + 工具
    full_messages = [SystemMessage(content=system_text)] + messages

    # 根据 llm_type 选择模型
    llm_type = state.get("llm_type", "local")
    try:
        if llm_type == "lili":
            llm = get_lili_llm()
        else:
            llm = get_local_llm_with_tools()
        response = llm.invoke(full_messages)
        _apply_thinking_to_message(response)
    except Exception:
        # fallback to local
        try:
            llm = get_local_llm()
            response = llm.invoke(full_messages)
            _apply_thinking_to_message(response)
        except Exception:
            llm = get_fast_llm()
            response = llm.invoke(full_messages)
            _apply_thinking_to_message(response)

    # 记录活动日志
    try:
        from lilith_bot.memory_store import get_memory_store
        store = get_memory_store()
        if user_msg:
            store.add_activity("user", user_msg[:120])
        ai_content = response.content or ""
        if ai_content:
            store.add_activity("lilith_chat", ai_content[:120])
    except Exception:
        pass

    return {"messages": [response]}


def route_after_chatbot(state: LilithState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "save_memory"


def save_memory_node(state: LilithState) -> dict:
    """记忆保存节点"""
    messages = state["messages"]
    last_user_msg = ""
    for m in reversed(messages):
        if hasattr(m, "type") and m.type == "human":
            last_user_msg = m.content
            break

    last_ai_msg = ""
    for m in reversed(messages):
        if hasattr(m, "type") and m.type == "ai":
            if m.content and m.content.strip():
                last_ai_msg = m.content
                break

    if not last_user_msg or not last_ai_msg or len(last_user_msg) < 5:
        return {}

    memories = state.get("long_term_memories", [])
    existing_text = "\n".join(f"- {m}" for m in memories) if memories else "（暂无记忆）"

    prompt = MEMORY_SAVE_PROMPT_V2.format(
        existing_memories=existing_text,
        user_message=last_user_msg,
        lilith_reply=last_ai_msg,
    )

    try:
        llm = get_fast_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        result_text = response.content.strip()

        if "{" in result_text and "}" in result_text:
            start = result_text.index("{")
            end = result_text.rindex("}") + 1
            result_text = result_text[start:end]

        result = json.loads(result_text)
    except Exception as e:
        print(f"[Memory] save_memory json parse failed: {type(e).__name__}: {e}")
        return {}

    should_remember = result.get("should_remember", False)
    new_memories = result.get("memories", [])

    if should_remember and new_memories:
        store = get_memory_store()
        existing_set = set(memories)
        for item in new_memories:
            if isinstance(item, str):
                content = item.strip()
                mtype = "knowledge"
            else:
                content = item.get("content", "").strip()
                mtype = item.get("type", "knowledge")
            if not content or content in existing_set:
                continue
            try:
                store.add_memory(content=content, memory_type=mtype)
                memories.append(f"[{mtype}] {content}")
                existing_set.add(content)
            except Exception as e:
                print(f"[Memory] 保存失败: {e}")
        if len(memories) > MEMORY_MAX_COUNT:
            memories = memories[-MEMORY_MAX_COUNT:]

    return {"long_term_memories": memories}


def route_after_save(state: LilithState) -> str:
    """决定是否需要深度反思 或 自演化"""
    affection = state.get("affection", AFFECTION_DEFAULT)
    evo_state = state.get("evolution_state", dict(EVOLUTION_DEFAULT))
    count = affection.get("interaction_count", 0)
    last_reflection = affection.get("last_reflection_at", 0)
    last_evolution = evo_state.get("last_evolution_at", 0)
    evo_interval = evo_state.get("evolution_interval", 50)
    evo_enabled = evo_state.get("evolution_enabled", True)

    # 反思优先（更频繁）
    if count - last_reflection >= REFLECTION_INTERVAL and count > 0:
        return "reflect"

    # 自演化检查
    if evo_enabled and count - last_evolution >= evo_interval and count > 0:
        return "evolution"

    return "end"


def reflection_node(state: LilithState) -> dict:
    """LLM 深度反思节点 (低频触发)

    分析最近对话, 更新:
    - 人格特质微调
    - 关系里程碑
    - 情感记忆写入向量库
    """
    messages = state["messages"]
    affection = state.get("affection", dict(AFFECTION_DEFAULT))
    memories = state.get("long_term_memories", [])

    # 收集最近几轮对话
    recent = []
    for m in messages[-8:]:
        if hasattr(m, "type"):
            role = "主人" if m.type == "human" else "莉莉丝"
            content = m.content[:200] if m.content else ""
            recent.append(f"[{role}] {content}")

    mood = mood_summary(affection)
    closeness = relation_closeness(affection)
    traits = affection.get("traits", {})
    milestones = affection.get("milestones", [])
    current_milestones = ", ".join(milestones) if milestones else "暂无"

    prompt = REFLECTION_PROMPT.format(
        mood=mood,
        closeness=closeness,
        traits=json.dumps(traits, ensure_ascii=False),
        milestones=current_milestones,
        recent_dialogue="\n".join(recent),
    )

    try:
        llm = get_reflection_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        result_text = response.content.strip()
        if "{" in result_text and "}" in result_text:
            start = result_text.index("{")
            end = result_text.rindex("}") + 1
            result_text = result_text[start:end]
        result = json.loads(result_text)
    except Exception as e:
        print(f"[Reflection] 反思失败: {e}")
        affection["last_reflection_at"] = affection.get("interaction_count", 0)
        return {"affection": affection}

    # 更新特质 (微调, 变化 ±0.03 以内)
    new_traits = result.get("trait_adjustments", {})
    for k, v in new_traits.items():
        if k in traits and isinstance(v, (int, float)):
            old = traits[k]
            traits[k] = round(max(0.0, min(1.0, old + v)), 2)

    # 更新里程碑
    new_milestones = result.get("new_milestones", [])
    for m in new_milestones:
        if m and m not in milestones:
            milestones.append(m)

    # 写入情感记忆
    emotional_insights = result.get("emotional_insights", [])
    if emotional_insights:
        store = get_memory_store()
        for item in emotional_insights:
            content = item if isinstance(item, str) else item.get("content", "")
            if content:
                try:
                    store.add_memory(content=content, memory_type="emotional")
                except Exception as e:
                    print(f"[Reflection] 写情感记忆失败: {e}")

    affection["traits"] = traits
    affection["milestones"] = milestones
    affection["last_reflection_at"] = affection.get("interaction_count", 0)

    return {"affection": affection}



# @evolvable: evolution_node
def evolution_node(state: LilithState) -> dict:
    """Self-evolution node - AI observes and modifies its own code.
    Triggered every N interactions.
    """
    messages = state.get("messages", [])
    affection = state.get("affection", dict(AFFECTION_DEFAULT))
    evo_state = state.get("evolution_state", dict(EVOLUTION_DEFAULT))

    recent = []
    for m in messages[-12:]:
        if hasattr(m, "type"):
            role = "Master" if m.type == "human" else "Lilith"
            content = m.content[:300] if m.content else ""
            recent.append("[" + role + "] " + content)

    try:
        engine = get_evolution_engine()
        evolvable = engine.list_evolvable_files()
        evo_log = engine.get_evolution_log(limit=5)
    except Exception as e:
        print("[Evolution] Engine init failed:", e)
        evo_state["last_evolution_at"] = affection.get("interaction_count", 0)
        return {"evolution_state": evo_state}

    files_desc = "\n".join(
        "- " + f["name"] + ": " + f["description"] for f in evolvable
    )
    recent_log = "\n".join(
        "  #" + str(r.get("iteration","?")) + ": " + str(r.get("reason","?"))[:60]
        for r in (evo_log or [])
    ) if evo_log else "No history"

    mood = mood_summary(affection)
    closeness = relation_closeness(affection)

    analysis_prompt = (
        "You are Lilith's self-observation module. "
        "Analyze recent dialogue for areas to be more human-like.\n\n"
        "Mood: " + mood + " | Relationship: " + closeness + "\n"
        "Evolution history:\n" + recent_log + "\n\n"
        "Recent dialogue:\n" + "\n".join(recent) + "\n\n"
        "Files:\n" + files_desc + "\n\n"
        "Analyze: 1.naturalness 2.emotion 3.initiative 4.memory 5.boundary\n\n"
        'Output JSON: {"insights":[{"dimension":"","observation":"","problem":"","target_file":"","improvement":""}],"ready_to_evolve":false,"human_likeness_score":0.5}'
    )

    try:
        llm = get_fast_llm()
        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        result_text = response.content.strip()
        if "{" in result_text and "}" in result_text:
            start = result_text.index("{")
            end = result_text.rindex("}") + 1
            result_text = result_text[start:end]
        analysis = json.loads(result_text)
    except Exception as e:
        print("[Evolution] Analysis failed:", e)
        evo_state["last_evolution_at"] = affection.get("interaction_count", 0)
        return {"evolution_state": evo_state}

    insights = analysis.get("insights", [])
    ready = analysis.get("ready_to_evolve", False)
    evo_state["pending_insights"] = insights
    evo_state["human_likeness_score"] = analysis.get(
        "human_likeness_score", evo_state.get("human_likeness_score", 0.5))

    applied_count = 0
    if ready and insights:
        for insight in insights[:2]:
            target = insight.get("target_file", "")
            improvement = insight.get("improvement", "")
            if not target or not improvement or "." not in target:
                continue

            file_content = engine.read_self_code(target)
            if file_content.startswith("[") and len(file_content) < 200:
                continue

            patch_prompt = (
                "Generate a code patch for Lilith self-evolution.\n\n"
                "Issue: " + insight.get("problem", "") + "\n"
                "Fix: " + improvement + "\n\n"
                "File: " + target + "\n"
                "Content:\n```python\n" + file_content[:3000] + "\n```\n\n"
                "Output:\n"
                "@@ SEARCH @@\n[original]\n@@ REPLACE @@\n[new]\n@@ END @@"
            )

            try:
                patch_llm = get_fast_llm()
                patch_resp = patch_llm.invoke([HumanMessage(content=patch_prompt)])
                patch_text = patch_resp.content.strip()
                if "@@ SEARCH @@" in patch_text and "@@ REPLACE @@" in patch_text:
                    s_idx = patch_text.index("@@ SEARCH @@")
                    patch = patch_text[s_idx:]
                    result = engine.apply_evolution(
                        file_name=target, patch_content=patch,
                        reason=improvement[:200],
                        insight=insight.get("observation","")[:200],
                        dry_run=True)
                    if result.get("success") and result.get("dry_run"):
                        result = engine.apply_evolution(
                            file_name=target, patch_content=patch,
                            reason=improvement[:200],
                            insight=insight.get("observation","")[:200],
                            dry_run=False)
                        if result.get("success"):
                            applied_count += 1
                            print("[Evolution] Modified " + target)
            except Exception as e:
                print("[Evolution] Patch error:", e)

    evo_state["total_evolutions"] = evo_state.get("total_evolutions", 0) + applied_count
    evo_state["last_evolution_at"] = affection.get("interaction_count", 0)

    journal = evo_state.get("evolution_journal", [])
    journal.append({
        "iteration": evo_state.get("total_evolutions", 0),
        "interaction": affection.get("interaction_count", 0),
        "insights_found": len(insights),
        "applied": applied_count,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "success": applied_count > 0,
    })
    if len(journal) > 20:
        journal = journal[-20:]
    evo_state["evolution_journal"] = journal

    print("[Evolution] Done:", len(insights), "insights,", applied_count, "applied")
    return {"evolution_state": evo_state}

# ============ 构建 Graph ============

_tool_node = ToolNode(LANGCHAIN_TOOLS)


def build_lilith_graph() -> StateGraph:
    graph_builder = StateGraph(LilithState)

    # 注册节点
    graph_builder.add_node("recall_memory", recall_memory_node)
    graph_builder.add_node("update_affection", update_affection_node)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_node("tools", _tool_node)
    graph_builder.add_node("save_memory", save_memory_node)
    graph_builder.add_node("reflect", reflection_node)

    # 注册节点
    graph_builder.add_node("evolution", evolution_node)

    # 注册边
    graph_builder.add_edge(START, "recall_memory")
    graph_builder.add_edge("recall_memory", "update_affection")
    graph_builder.add_edge("update_affection", "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot", route_after_chatbot,
        {"tools": "tools", "save_memory": "save_memory"},
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_conditional_edges(
        "save_memory", route_after_save,
        {"reflect": "reflect", "evolution": "evolution", "end": END},
    )
    graph_builder.add_edge("reflect", END)
    graph_builder.add_edge("evolution", END)

    checkpointer = get_checkpointer()
    return graph_builder.compile(checkpointer=checkpointer)


lilith_graph = build_lilith_graph()
