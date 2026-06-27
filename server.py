"""莉莉丝 — OpenAI 兼容 API 服务器（双模型架构）

将 LangGraph 对话图包装为 OpenAI /v1/chat/completions 端点，
供 Open WebUI 作为"外部 OpenAI 连接"调用。

特性:
  - 双模型支持：lilith（本地 Qwen 9B 思维链） / lili（远程 API Agent，适合 coding）
  - 完整思维链（reasoning_content）支持（通过 monkey-patch 透传）
  - LangGraph 原生 ToolNode + bind_tools 工具调用
  - 流式 & 非流式响应（统一使用 graph.stream / graph.invoke）
  - LangGraph 记忆系统全保留（recall → chatbot ↔ tools → save_memory）
  - 自动过滤 Open WebUI 辅助请求（生成标题/标签/推荐等），不污染对话历史
  - 工具调用系统（Python/CMD/键鼠/截图/文件操作）

启动方式（激活 venv 后）:
    python server.py
    # 或指定端口
    python server.py --port 8000

然后在 Open WebUI 中:
    设置 → 外部连接 → OpenAI API
    Base URL: http://localhost:8000/v1
    API Key: lilith-local （随便填，不做鉴权）
    模型下拉可选: lilith / lili
"""

import os
import sys
import json
import time
import uuid
import argparse

# ─── 项目路径 ────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import httpx

from langchain_core.messages import HumanMessage

# 必须在导入 graph 之前应用 reasoning_content 补丁（graph.py 内部已导入）
from lilith_bot.graph import lilith_graph, DB_PATH
from lilith_bot.persona import LILITH_SYSTEM_PROMPT
from lilith_bot.tools import LANGCHAIN_TOOLS
from lilith_bot.state import AFFECTION_DEFAULT


# ─── 配置 ─────────────────────────────────────────────────
THREAD_ID = "lilith-openwebui"
MODEL_NAME = "lilith"
LOCAL_API_KEY = "lilith-local"

# API 配置（辅助请求直接调用，不走 graph）
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://opencode.ai/zen/go/v1")
CHAT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
REASONER_MODEL = os.getenv("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner")

# Open WebUI 辅助请求的特征前缀（这些请求不走对话记忆系统）
_AUXILIARY_PREFIXES = (
    "### Task:",
    "### Instruction:",
    "Generate a concise",
    "Suggest 3-5",
    "Generate 1-3 broad tags",
)




# 工具名称 -> 用户友好显示名
TOOL_DISPLAY_NAMES = {
    "ask_api_assistant": "API助手",
    "run_python": "Python代码",
    "run_cmd": "CMD命令",
    "screenshot": "截屏",
    "mouse_click": "鼠标点击",
    "mouse_move": "鼠标移动",
    "mouse_scroll": "鼠标滚轮",
    "type_text": "键盘输入",
    "press_key": "按键",
    "get_clipboard": "剪贴板读取",
    "set_clipboard": "剪贴板写入",
    "list_files": "文件列表",
    "read_file": "读取文件",
    "write_file": "写入文件",
    "system_info": "系统信息",
    "open_path": "打开文件/URL",
    "get_cursor_pos": "鼠标坐标",
    "window_list": "窗口列表",
}


app = FastAPI(title="Lilith API", version="0.4.0")


# ─── 辅助请求检测 ─────────────────────────────────────────

def _is_auxiliary_request(user_msg: str, messages: list) -> bool:
    """检测是否是 Open WebUI 的辅助请求"""
    msg_stripped = user_msg.strip()
    for prefix in _AUXILIARY_PREFIXES:
        if msg_stripped.startswith(prefix):
            return True

    for m in messages:
        if m.get("role") == "system":
            content = m.get("content", "")
            if isinstance(content, str):
                if any(kw in content for kw in [
                    "Generate a concise, 3-5 word title",
                    "Suggest 3-5 relevant follow-up",
                    "Generate 1-3 broad tags",
                    "title with an emoji",
                    "categorizing the main themes",
                    "follow-up questions or prompts",
                ]):
                    return True
    return False


# ─── 工具函数 ─────────────────────────────────────────────

def _extract_last_user_message(messages: list) -> str:
    """从 OpenAI 格式的 messages 数组中提取最后一条用户消息"""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                return "".join(
                    part.get("text", "") for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            return content
    return ""


def _get_state(config: dict) -> dict:
    """从 LangGraph checkpoint 获取当前状态"""
    try:
        st = lilith_graph.get_state(config)
        return st.values if st and st.values else {}
    except Exception:
        return {}


# ─── 辅助请求处理（不走 graph）───────────────────────────

def _call_auxiliary(messages: list, stream: bool):
    """辅助请求专用：用普通模型调用，不走记忆/工具系统"""
    api_url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": stream,
    }

    if stream:
        def generate():
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            with httpx.Client(timeout=httpx.Timeout(120.0, connect=15.0)) as client:
                with client.stream("POST", api_url, json=payload, headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                }) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            d = json.loads(data_str)
                            delta = d.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                chunk = {
                                    "id": chat_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": MODEL_NAME,
                                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                                }
                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        except json.JSONDecodeError:
                            continue

            final = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )
    else:
        with httpx.Client(timeout=httpx.Timeout(120.0, connect=15.0)) as client:
            resp = client.post(api_url, json=payload, headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            })
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            return {
                "id": chat_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }


# ─── 健康检查 ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "ok",
        "name": "Lilith API",
        "models": ["lilith", "lili"],
        "lilith_model": MODEL_NAME,
        "lili_model": os.getenv("LILI_MODEL", "deepseek-v4"),
        "reasoner_model": REASONER_MODEL,
        "reasoning_enabled": True,
        "tools_enabled": True,
        "tool_count": len(LANGCHAIN_TOOLS),
        "graph_version": "LangGraph native ToolNode",
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "lilith",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "lilith",
                "permission": [],
                "root": "lilith",
                "parent": None,
            },
            {
                "id": "lili",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "lilith",
                "permission": [],
                "root": "lili",
                "parent": None,
            },
        ],
    }




# ═══════════════════════════════════════════════════════════════
# 自主发言 API
# ═══════════════════════════════════════════════════════════════

from lilith_bot.autonomous import get_autonomous_brain

@app.get("/autonomous/status")
async def autonomous_status():
    """查看自主发言状态"""
    brain = get_autonomous_brain()
    return brain.status()

@app.post("/autonomous/start")
async def autonomous_start():
    """启动自主发言"""
    brain = get_autonomous_brain()
    ok, msg = brain.start()
    return {"success": ok, "message": msg}

@app.post("/autonomous/stop")
async def autonomous_stop():
    """停止自主发言"""
    brain = get_autonomous_brain()
    ok, msg = brain.stop()
    return {"success": ok, "message": msg}

@app.post("/autonomous/say")
async def autonomous_say_once(request: Request):
    """手动触发一次自主发言"""
    brain = get_autonomous_brain()
    affection = brain._get_affection()
    memories = brain._get_recent_memories()
    topic = brain._pick_topic_type(affection)
    msg = brain._generate_message(topic, affection, memories)
    if msg:
        brain._push_to_channel(msg)
        brain._update_self_affection(topic)
        brain.stats["total_messages"] += 1
        brain.stats["last_message"] = msg
        brain.stats["last_topic"] = topic
        return {"success": True, "message": msg, "topic": topic}
    return {"success": False, "message": "生成失败"}

# ─── 聊天核心 ─────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    model = body.get("model", MODEL_NAME)

    # 提取最后一条用户消息
    user_msg = _extract_last_user_message(messages)

    if not user_msg:
        return JSONResponse(
            {"error": {"message": "No user message found", "type": "invalid_request_error"}},
            status_code=400,
        )

    # ── 辅助请求检测 ──
    if _is_auxiliary_request(user_msg, messages):
        return _call_auxiliary(messages, stream)

    # 特殊命令
    if user_msg.strip().lower() == "/reset":
        if stream:
            return _stream_simple("莉莉丝: 对话已重置~ 莉莉丝会忘了之前的事哦 (´･ω･`)", model)
        return _make_response("莉莉丝: 对话已重置~ 莉莉丝会忘了之前的事哦 (´･ω･`)", model)

    config = {"configurable": {"thread_id": THREAD_ID}}
    sv = _get_state(config)

    # 构建 graph 输入（只传新消息，checkpoint 自动保留历史）
    state_in = {
        "messages": [HumanMessage(content=user_msg)],
        "persona": LILITH_SYSTEM_PROMPT,
        "long_term_memories": sv.get("long_term_memories", []),
        "affection": sv.get("affection", AFFECTION_DEFAULT),
        "llm_type": "lili" if model == "lili" else "local",
    }

    if stream:
        return _stream_graph_response(state_in, config, model)
    else:
        return await _sync_graph_response(state_in, config, model)


# ─── 流式响应（LangGraph 原生流式）────────────────────────

def _stream_graph_response(state_in: dict, config: dict, model: str):
    """
    流式响应 — 使用 graph.astream(stream_mode="messages") 统一输出思维链 + 正文。

    自动处理工具调用循环（由 LangGraph ToolNode 管理），流式输出 chatbot 节点的
    AIMessageChunk（包含 reasoning_content 和 content）。
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def _chunk(reasoning: str, content: str, finish: str = None):
        delta = {}
        if reasoning:
            delta["reasoning_content"] = reasoning
        if content:
            delta["content"] = content
        return {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    def generate():
        # 发送初始空 chunk（触发 Open WebUI 开始接收）
        stream_start = time.time()
        stream_timeout = 90.0
        # send initial empty chunk (triggers Open WebUI to start receiving)
        yield f"data: {json.dumps(_chunk('', ''), ensure_ascii=False)}\n\n"

        full_response = ""

        # 使用 graph.stream 同步流式输出（SqliteSaver 不支持异步）
        for msg_chunk, metadata in lilith_graph.stream(
            state_in, config, stream_mode="messages"
        ):
            node = metadata.get("langgraph_node", "")

            # ToolMessage 来自 tools 节点：输出工具完成提示（必须在 chatbot 过滤之前）
            if time.time() - stream_start > stream_timeout:
                print("[Lilith] Streaming response timed out (90s), aborting")
                break
            chunk_type_name = type(msg_chunk).__name__
            if "ToolMessage" in chunk_type_name:
                tool_done_hint = '\n✅ 工具调用完成\n\n'
                yield f"data: {json.dumps(_chunk('', tool_done_hint), ensure_ascii=False)}\n\n"
                continue
            if "AIMessage" not in chunk_type_name:
                continue

            # 工具调用：输出提示信息给用户
            tool_calls = getattr(msg_chunk, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tool_name = tc.get("name", "unknown")
                    else:
                        tool_name = getattr(tc, "name", "unknown")
                    friendly = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
                    hint = f"\n\n🔧 莉莉丝正在调用工具: {friendly}...\n\n"
                    yield f"data: {json.dumps(_chunk('', hint), ensure_ascii=False)}\n\n"
                continue

            # 提取 reasoning_content（来自 monkey-patch 透传）
            reasoning = ""
            if hasattr(msg_chunk, "additional_kwargs") and msg_chunk.additional_kwargs:
                reasoning = msg_chunk.additional_kwargs.get("reasoning_content", "")

            content = getattr(msg_chunk, "content", "") or ""

            if reasoning:
                yield f"data: {json.dumps(_chunk(reasoning, ''), ensure_ascii=False)}\n\n"
            if content:
                full_response += content
                yield f"data: {json.dumps(_chunk('', content), ensure_ascii=False)}\n\n"

        # 结束标记
        yield f"data: {json.dumps(_chunk('', '', 'stop'), ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ─── 非流式响应 ───────────────────────────────────────────

async def _sync_graph_response(state_in: dict, config: dict, model: str):
    """Non-streaming response with 60s timeout"""
    import asyncio
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(lilith_graph.invoke, state_in, config),
            timeout=60.0
        )
    except asyncio.TimeoutError:
        print('[Lilith] Graph invoke timed out (60s), returning fallback')
        fallback = "\u5410\u5410\u2026\u66f4\u65b0\u8fc7\u4e8e\u6162\u4e86\uff0c\u518d\u7b49\u7b49\u5427~"
        return _make_response(fallback, model)

    content = ""
    reasoning = ""
    for m in reversed(result.get("messages", [])):
        if hasattr(m, "type") and m.type == "ai":
            if m.content and m.content.strip():
                content = m.content
                if hasattr(m, "additional_kwargs") and m.additional_kwargs:
                    reasoning = m.additional_kwargs.get("reasoning_content", "")
                break

    return _make_response_with_reasoning(content, reasoning, model)


# ---


# ─── 响应格式化 ───────────────────────────────────────────

def _make_response(text: str, model: str):
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _make_response_with_reasoning(content: str, reasoning: str, model: str):
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ─── 简单流式（用于 /reset 等特殊命令）───────────────────

def _stream_simple(text: str, model: str):
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def generate():
        data = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        final_data = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ─── 启动 ─────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lilith OpenAI-compatible API Server")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    args = parser.parse_args()

    if not API_KEY or API_KEY.strip() in {"", "your_api_key_here"}:
        print("[Lilith] 请先配置 .env 中的 API Key！")
        sys.exit(1)

    # Preload embedding model to avoid first-request lag
    print("[Lilith]    Preloading embedding model...")
    try:
        from lilith_bot.memory_store import preload_embedding_model
        t0 = time.time()
        preload_embedding_model()
        print(f"[Lilith]   Embedding model loaded ({time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"[Lilith]   Embedding model preload failed: {e}")

    print(f"[Lilith] 🌸 API 服务器启动中... (v0.4.0 LangGraph 原生工具系统)")
    print(f"[Lilith]    地址: http://{args.host}:{args.port}")
    print(f"[Lilith]    模型: {MODEL_NAME} / lili")
    print(f"[Lilith]    思维链引擎: {REASONER_MODEL}")
    print(f"[Lilith]    工具数量: {len(LANGCHAIN_TOOLS)}")
    print(f"[Lilith]    记忆库: {DB_PATH}")
    print(f"[Lilith]")
    print(f"[Lilith]    可用工具:")
    for t in LANGCHAIN_TOOLS:
        print(f"[Lilith]      ✓ {t.name} — {t.description[:60]}")
    print(f"[Lilith]")
    print(f"[Lilith]    Graph 结构:")
    print(f"[Lilith]      START → recall_memory → chatbot → [route]")
    print(f"[Lilith]                                     ├── tools → chatbot (循环)")
    print(f"[Lilith]                                     └── save_memory → END")
    print(f"[Lilith]")
    print(f"[Lilith]    在 Open WebUI 中配置:")
    print(f"[Lilith]    设置 → 外部连接 → OpenAI API")
    print(f"[Lilith]    Base URL: http://{args.host}:{args.port}/v1")
    print(f"[Lilith]    API Key: {LOCAL_API_KEY}")
    print()
    uvicorn.run(app, host=args.host, port=args.port)
