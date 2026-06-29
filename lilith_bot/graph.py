"""莉莉丝的 LangGraph 对话图"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lilith_bot.reasoning_patch import *  # noqa: F401, F403

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from lilith_bot.state import LilithState, AFFECTION_DEFAULT, EVOLUTION_DEFAULT
from lilith_bot.tools import LANGCHAIN_TOOLS
from lilith_bot.memory_store import get_memory_store
from lilith_bot.affection_engine import update_affection_state
from lilith_bot.affection_events import parse_user_message
from lilith_bot.evolution_engine import get_evolution_engine
from lilith_bot.trace_logger import write_log


# ============ LLM 实例 ============

_chat_llm = None
_chat_llm_with_tools = None


def DB_PATH():
    return os.path.join(_PROJECT_ROOT, "lilith_memory.db")


def get_checkpointer():
    from langgraph.checkpoint.sqlite import SqliteSaver
    db_path = DB_PATH()
    conn = __import__('sqlite3').connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


def get_chat_llm():
    """主对话 LLM（DeepSeek Flash）"""
    global _chat_llm
    if _chat_llm is None:
        _chat_llm = ChatOpenAI(
            model=os.getenv("LLM_CHAT_MODEL", "deepseek-v4-flash"),
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL", "https://opencode.ai/zen/go/v1"),
            temperature=0.8,
            max_tokens=4096,
            request_timeout=120,
        )
    return _chat_llm


def get_chat_llm_with_tools():
    """主对话 LLM（含工具绑定）"""
    global _chat_llm_with_tools
    if _chat_llm_with_tools is None:
        _chat_llm_with_tools = get_chat_llm().bind_tools(LANGCHAIN_TOOLS)
    return _chat_llm_with_tools


# ============ Graph 节点 ============

MEMORY_MAX_COUNT = 50
REFLECTION_INTERVAL = 5


def recall_memory_node(state: LilithState):
    """记忆召回节点（含情绪波动）"""
    messages = state["messages"]
    user_msg = ""
    for m in reversed(messages):
        if hasattr(m, "type") and m.type == "human":
            user_msg = m.content
            break
    memories = state.get("long_term_memories", [])
    affection = state.get("affection", {})
    if not isinstance(affection, dict) or "valence" not in affection:
        affection = dict(AFFECTION_DEFAULT)
    if user_msg and len(user_msg) >= 3:
        try:
            store = get_memory_store()
            recalled, recalled_moods = store.recall_with_mood(user_msg, limit=5)
            write_log("memory_recall", {
                "query": user_msg[:100],
                "recalled_count": len(recalled),
                "top_memories": [m[:120] for m in recalled[:3]],
            })
            memories = recalled + state.get("long_term_memories", [])
            memories = list(dict.fromkeys(memories))[:MEMORY_MAX_COUNT]
            fluctuation = store.compute_mood_fluctuation(recalled_moods, affection)
            if any(abs(v) > 0.001 for v in fluctuation.values()):
                for k in ("valence", "arousal", "dominance"):
                    affection[k] = max(-1.0, min(1.0, affection.get(k, 0.0) + fluctuation[k + "_delta"]))
        except Exception as e:
            print(f"[Lilith] recall_memory failed: {type(e).__name__}: {e}")
    # 记录用户活动
    try:
        store = get_memory_store()
        store.add_activity("user", user_msg[:200], person="对方")
    except Exception:
        pass
    return {"long_term_memories": memories, "affection": affection}


def update_affection_node(state: LilithState) -> dict:
    """好感度实时更新节点"""
    messages = state["messages"]
    user_msg = ""
    for m in reversed(messages):
        if hasattr(m, "type") and m.type == "human":
            user_msg = m.content
            break
    affection = state.get("affection")
    if not isinstance(affection, dict) or "valence" not in affection:
        affection = dict(AFFECTION_DEFAULT)
    event = parse_user_message(user_msg)
    update_affection_state(affection, event, elapsed_seconds=60.0)
    return {"affection": affection}


def chatbot_node(state: LilithState):
    """核心对话节点 — persona 系统提示"""
    messages = list(state["messages"])

    # 判断是否是工具调用后的第二次运行
    has_tool_messages = any(
        hasattr(m, "type") and "ToolMessage" in type(m).__name__
        for m in messages
    )

    if has_tool_messages:
        # 工具调用后：保留完整对话历史，只替换第一条为 system prompt
        system_prompt = ""
        try:
            from lilith_bot.persona import build_system_prompt
            affection = state.get("affection", {})
            if not isinstance(affection, dict) or "valence" not in affection:
                affection = dict(AFFECTION_DEFAULT)
            memories = state.get("long_term_memories", [])
            system_prompt = build_system_prompt(affection, memories)
        except Exception as e:
            print(f"[Lilith] build_system_prompt failed: {type(e).__name__}: {e}")
            system_prompt = "你是莉莉丝（Lilith），一个生活在 Windows 电脑里的 AI 少女。"
        full_messages = [SystemMessage(content=system_prompt)] + messages[1:]
    else:
        # 首次调用：构建 system prompt + 用户消息
        user_msg = ""
        for m in reversed(messages):
            if hasattr(m, "type") and m.type == "human":
                user_msg = m.content or ""
                break
        if not user_msg:
            user_msg = "..."

        affection = state.get("affection", {})
        if not isinstance(affection, dict) or "valence" not in affection:
            affection = dict(AFFECTION_DEFAULT)
        memories = state.get("long_term_memories", [])

        system_prompt = ""
        try:
            from lilith_bot.persona import build_system_prompt
            system_prompt = build_system_prompt(affection, memories)
        except Exception as e:
            print(f"[Lilith] build_system_prompt failed: {type(e).__name__}: {e}")
            system_prompt = "你是莉莉丝（Lilith），一个生活在 Windows 电脑里的 AI 少女。外表白色长发、红色眼睛。活泼温柔、有一点点调皮。"

        full_messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_msg)]

    # ③ 调用 LLM（优先带工具绑定，失败降级到纯文本）
    try:
        llm = get_chat_llm_with_tools()
        response = llm.invoke(full_messages)
        return {"messages": [response]}
    except Exception:
        try:
            llm = get_chat_llm()
            response = llm.invoke(full_messages)
            return {"messages": [response]}
        except Exception as e:
            print(f"[Lilith] chatbot failed: {e}")
            raise


def route_after_chatbot(state: LilithState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "save_memory"


def save_memory_node(state: LilithState) -> dict:
    """记忆保存节点 — 集成 FeedbackSystem"""
    messages = list(state["messages"])
    memories = state.get("long_term_memories", [])
    affection = state.get("affection", {})
    if not isinstance(affection, dict):
        affection = dict(AFFECTION_DEFAULT)

    # 提取用户消息和莉莉丝回复
    last_user_msg = ""
    last_ai_msg = ""
    for m in reversed(messages):
        if hasattr(m, "type") and m.type == "human" and not last_user_msg:
            last_user_msg = m.content or ""
        if hasattr(m, "type") and m.type == "ai" and not last_ai_msg:
            last_ai_msg = m.content or ""
        if last_user_msg and last_ai_msg:
            break

    if not last_user_msg or not last_ai_msg:
        return {}

    try:
        from lilith_bot.feedback_system import get_feedback_system
        fs = get_feedback_system()

        # LLM 分析：感受词 + 记忆重要性 + 记忆内容
        feedback = fs.analyze(
            user_message=last_user_msg,
            lilith_reply=last_ai_msg,
            pad=affection,
            interaction_count=affection.get("interaction_count", 0),
        )

        # 保存记忆（如果 LLM 认为重要）
        memory_content = feedback.get("memory_content", "")
        memory_importance = feedback.get("memory_importance", 0.0)
        saved = False
        if memory_content and memory_importance > 0.3:
            from lilith_bot.memory_store import get_memory_store
            store = get_memory_store()
            # 避免重复
            existing_set = set(memories)
            if memory_content not in existing_set:
                store.add_memory(
                    content=memory_content,
                    memory_type="knowledge",
                    importance=memory_importance,
                )
                memories.append(f"[knowledge] {memory_content}")
                saved = True
                if len(memories) > MEMORY_MAX_COUNT:
                    memories = memories[-MEMORY_MAX_COUNT:]

        write_log("memory_save", {
            "memory_content": memory_content[:200] if memory_content else "",
            "memory_importance": memory_importance,
            "saved": saved,
            "total_memories": len(memories),
        })

    except Exception as e:
        print(f"[Memory] save_memory_node failed: {type(e).__name__}: {e}")

    # 记录 AI 回复活动
    try:
        if last_ai_msg:
            from lilith_bot.memory_store import get_memory_store
            mstore = get_memory_store()
            mstore.add_activity("lilith_chat", last_ai_msg[:200], person="对方")
    except Exception:
        pass

    return {"long_term_memories": memories}


def route_after_save(state: LilithState) -> str:
    affection = state.get("affection", AFFECTION_DEFAULT)
    evo_state = state.get("evolution_state", dict(EVOLUTION_DEFAULT))
    count = affection.get("interaction_count", 0)
    last_reflection = affection.get("last_reflection_at", 0)
    last_evolution = evo_state.get("last_evolution_at", 0)
    evo_interval = evo_state.get("evolution_interval", 50)
    evo_enabled = evo_state.get("evolution_enabled", True)
    if count - last_reflection >= REFLECTION_INTERVAL and count > 0:
        return "reflect"
    if evo_enabled and count - last_evolution >= evo_interval and count > 0:
        return "evolution"
    return "end"


def reflection_node(state: LilithState) -> dict:
    """深度反思节点"""
    return {"affection": state.get("affection", dict(AFFECTION_DEFAULT))}


def evolution_node(state: LilithState) -> dict:
    """自演化节点"""
    messages = state.get("messages", [])
    affection = state.get("affection", dict(AFFECTION_DEFAULT))
    evo_state = state.get("evolution_state", dict(EVOLUTION_DEFAULT))
    try:
        engine = get_evolution_engine()
        evolvable = engine.list_evolvable_files()
        evo_log = engine.get_evolution_log(limit=5)
    except Exception as e:
        print("[Evolution] Engine init failed:", e)
        evo_state["last_evolution_at"] = affection.get("interaction_count", 0)
        return {"evolution_state": evo_state}
    # （演化逻辑保留，待新 prompt 系统适配）
    return {"evolution_state": evo_state}


# ============ 构建 Graph ============

_tool_node = ToolNode(LANGCHAIN_TOOLS)


def build_lilith_graph() -> StateGraph:
    graph_builder = StateGraph(LilithState)
    graph_builder.add_node("recall_memory", recall_memory_node)
    graph_builder.add_node("update_affection", update_affection_node)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_node("tools", _tool_node)
    graph_builder.add_node("save_memory", save_memory_node)
    graph_builder.add_node("reflect", reflection_node)
    graph_builder.add_node("evolution", evolution_node)
    graph_builder.add_edge(START, "recall_memory")
    graph_builder.add_edge("recall_memory", "update_affection")
    graph_builder.add_edge("update_affection", "chatbot")
    graph_builder.add_conditional_edges("chatbot", route_after_chatbot, {"tools": "tools", "save_memory": "save_memory"})
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_conditional_edges("save_memory", route_after_save, {"reflect": "reflect", "evolution": "evolution", "end": END})
    graph_builder.add_edge("reflect", END)
    graph_builder.add_edge("evolution", END)
    checkpointer = get_checkpointer()
    return graph_builder.compile(checkpointer=checkpointer)


lilith_graph = build_lilith_graph()
