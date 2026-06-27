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
import threading

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

# ─── 日志捕获器（供控制面板读取）──
_LOG_BUFFER = []  # 环形缓冲，保存最近 80 条日志
_LOG_LOCK = threading.Lock()

# Monkey-patch print 以捕获所有日志
_original_print = print
def _captured_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    with _LOG_LOCK:
        _LOG_BUFFER.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if len(_LOG_BUFFER) > 80:
            _LOG_BUFFER.pop(0)
    _original_print(*args, **kwargs)
print = _captured_print

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


# ─── 总控面板 ────────────────────────────────────────────

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>莉莉丝 控制面板</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#1a1a2e;color:#e0e0e0;padding:20px}
h1{font-size:24px;margin-bottom:20px;color:#e94560}
h2{font-size:16px;margin-bottom:10px;color:#0f3460;border-bottom:1px solid #16213e;padding-bottom:5px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.card{background:#16213e;border-radius:10px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.3)}
.stat{display:flex;justify-content:space-between;padding:4px 0;font-size:13px}
.stat .val{color:#e94560;font-weight:bold}
.bar{height:6px;background:#0f3460;border-radius:3px;margin:4px 0;overflow:hidden}
.bar-fill{height:100%;border-radius:3px;transition:width .5s}
.tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;margin:2px;background:#0f3460}
.thought{font-size:12px;color:#aaa;padding:3px 0;border-bottom:1px solid #1a1a2e}
.thought:last-child{border-bottom:none}
.log-entry{font-size:11px;color:#888;padding:2px 0;font-family:monospace;border-bottom:1px solid #1a1a2e}
.error-log{color:#ff6b6b}
.warn-log{color:#ffd93d}
.ctrl-btn{background:#e94560;color:#fff;border:none;padding:6px 12px;border-radius:5px;cursor:pointer;font-size:12px}
.ctrl-btn:hover{background:#c23152}
.ctrl-btn.secondary{background:#0f3460}
.ctrl-btn.secondary:hover{background:#1a5276}
.act-row{display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#888;padding:2px 0;font-family:monospace;border-bottom:1px solid #1a2a4e}
.act-text{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.act-del{background:transparent;border:none;color:#e94560;cursor:pointer;font-size:13px;padding:0 4px;opacity:0.6}
.act-del:hover{opacity:1}
#refresh{position:fixed;top:20px;right:20px;background:#e94560;color:#fff;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;font-size:13px}
#refresh:hover{background:#c23152}
.loading{text-align:center;padding:40px;color:#666}
</style>
</head>
<body>
<h1>🌸 莉莉丝 控制面板</h1>
<button id="refresh" onclick="load()">⟳ 刷新</button>
<div class="grid" id="app"></div>

<script>
async function load() {
  const app = document.getElementById('app');
  app.innerHTML = '<div class="loading">加载中...</div>';
  try {
    const r = await fetch('/dashboard/data');
    const d = await r.json();
    render(d);
  } catch(e) {
    app.innerHTML = '<div class="card" style="grid-column:1/-1;color:#ff6b6b">❌ 加载失败: ' + e.message + '</div>';
  }
}

function render(d) {
  const c = d.consciousness || {};
  const aff = d.affection || {};
  const mem = d.memory || {};
  const brain = d.brain || {};
  const log = d.log || [];

  const cards = [];

  // 1. 情绪面板
  let moodHtml = '';
  const moodLabel = aff.mood || aff.mood_summary || '未知';
  const v = aff.valence != null ? aff.valence.toFixed(2) : '0.00';
  const a = aff.arousal != null ? aff.arousal.toFixed(2) : '0.00';
  const dom = aff.dominance != null ? aff.dominance.toFixed(2) : '0.00';
  moodHtml += `<div class="stat"><span>心情</span><span class="val">${moodLabel}</span></div>`;
  moodHtml += `<div class="stat"><span>愉悦度 V</span><span class="val">${v}</span></div>`;
  moodHtml += `<div class="bar"><div class="bar-fill" style="width:${(aff.valence||0)*50+50}%;background:${aff.valence>0?'#4ecca3':'#ff6b6b'}"></div></div>`;
  moodHtml += `<div class="stat"><span>激活度 A</span><span class="val">${a}</span></div>`;
  moodHtml += `<div class="bar"><div class="bar-fill" style="width:${(aff.arousal||0)*50+50}%;background:#ffd93d"></div></div>`;
  moodHtml += `<div class="stat"><span>支配度 D</span><span class="val">${dom}</span></div>`;
  moodHtml += `<div class="bar"><div class="bar-fill" style="width:${(aff.dominance||0)*50+50}%;background:#a66cff"></div></div>`;
  if (aff.closeness) moodHtml += `<div class="stat"><span>关系</span><span class="val">${aff.closeness}</span></div>`;
  if (aff.interaction_count != null) moodHtml += `<div class="stat"><span>交互次数</span><span class="val">${aff.interaction_count}</span></div>`;
  moodHtml += '<div style="margin-top:8px;display:flex;gap:4px;flex-wrap:wrap">';
  moodHtml += '<input id="sld-v" type="range" min="-1" max="1" step="0.05" value="'+(aff.valence||0)+'" style="width:60px"> V';
  moodHtml += '<input id="sld-a" type="range" min="-1" max="1" step="0.05" value="'+(aff.arousal||0)+'" style="width:60px"> A';
  moodHtml += '<input id="sld-d" type="range" min="-1" max="1" step="0.05" value="'+(aff.dominance||0)+'" style="width:60px"> D';
  moodHtml += '<button class="ctrl-btn secondary" style="font-size:11px" onclick="setAffection()">设置</button>';
  moodHtml += '</div>';
  cards.push({title:'💖 情绪状态', html:moodHtml});

  // 2. 意识状态
  let conHtml = '';
  conHtml += `<div class="stat"><span>好奇心</span><span class="val">${(c.curiosity||0).toFixed(2)}</span></div>`;
  conHtml += `<div class="stat"><span>寂寞感</span><span class="val">${(c.loneliness||0).toFixed(2)}</span></div>`;
  conHtml += `<div class="stat"><span>精力</span><span class="val">${(c.energy||0).toFixed(2)}</span></div>`;
  conHtml += `<div class="bar"><div class="bar-fill" style="width:${(c.energy||0)*100}%;background:#4ecca3"></div></div>`;
  conHtml += `<div class="stat"><span>持续运行</span><span class="val">${(c.awake_seconds/60).toFixed(0)}分钟</span></div>`;
  conHtml += `<div class="stat"><span>上次发言</span><span class="val">${c.last_spoke_at || '从未'}</span></div>`;
  conHtml += '<div style="margin-top:8px;display:flex;gap:4px;flex-wrap:wrap">';
  conHtml += '<input id="sld-cu" type="range" min="0" max="1" step="0.05" value="'+(c.curiosity||0)+'" style="width:55px"> 好奇';
  conHtml += '<input id="sld-lo" type="range" min="0" max="1" step="0.05" value="'+(c.loneliness||0)+'" style="width:55px"> 寂寞';
  conHtml += '<input id="sld-en" type="range" min="0" max="1" step="0.05" value="'+(c.energy||0)+'" style="width:55px"> 精力';
  conHtml += '<button class="ctrl-btn secondary" style="font-size:11px" onclick="setConsciousness()">设置</button>';
  conHtml += '<button class="ctrl-btn" style="font-size:11px" onclick="apiPost(\'/dashboard/reset-consciousness\',\'重置中...\')">重置</button>';
  conHtml += '</div>';
  cards.push({title:'🧠 意识状态', html:conHtml});

  // 3. 思绪流
  let thoughtHtml = '';
  const thoughts = c.thought_buffer || [];
  if (thoughts.length === 0) {
    thoughtHtml = '<div class="stat" style="color:#666">暂无内心想法</div>';
  } else {
    thoughts.slice().reverse().forEach(t => {
      thoughtHtml += `<div class="thought">💭 ${t.content}</div>`;
    });
  }
  cards.push({title:'💭 内心独白', html:thoughtHtml});

  // 4. 自主发言统计
  let brainHtml = '';
  const running = brain.running;
  brainHtml += `<div class="stat"><span>运行状态</span><span class="val">${running ? '✅ 运行中' : '⏹ 已停止'}</span></div>`;
  brainHtml += `<div class="stat"><span>总发言次数</span><span class="val">${brain.total_messages || 0}</span></div>`;
  brainHtml += `<div class="stat"><span>发言间隔</span><span class="val">${brain.interval_range || '-'}</span></div>`;
  if (brain.last_message) brainHtml += `<div class="stat"><span>最后发言</span><span class="val" style="font-size:11px">${brain.last_message.slice(0,40)}...</span></div>`;
  if (brain.last_topic) brainHtml += `<div class="stat"><span>最后话题</span><span class="val">${brain.last_topic}</span></div>`;
  brainHtml += '<div style="margin-top:10px;display:flex;gap:6px">';
  if (running) {
    brainHtml += '<button class="ctrl-btn" onclick="apiPost(\'/autonomous/stop\',\'停止中...\')">⏹ 停止</button>';
  } else {
    brainHtml += '<button class="ctrl-btn" onclick="apiPost(\'/autonomous/start\',\'启动中...\')">▶ 启动</button>';
  }
  brainHtml += '<button class="ctrl-btn secondary" onclick="apiPost(\'/autonomous/say\',\'思考中...\')">💬 说一句</button>';
  brainHtml += '</div>';
  cards.push({title:'📢 自主发言', html:brainHtml});

  // 5. 记忆统计
  let memHtml = '';
  memHtml += `<div class="stat"><span>记忆总数</span><span class="val">${mem.total || 0}</span></div>`;
  if (mem.by_type) {
    Object.entries(mem.by_type).forEach(([k,v]) => {
      const icons = {knowledge:'📖', emotional:'💕', event:'📅', skill:'🔧'};
      memHtml += `<div class="stat"><span>${icons[k]||'•'} ${k}</span><span class="val">${v}</span></div>`;
    });
  }
  cards.push({title:'📚 长期记忆', html:memHtml});

  // 6. 里程碑 & 心情
  let miscHtml = '';
  if (mem.milestones && mem.milestones.length) {
    miscHtml += '<div style="font-size:12px;margin-bottom:8px"><b>🏆 里程碑</b></div>';
    mem.milestones.slice(-3).forEach(m => {
      miscHtml += `<div class="log-entry">${m}</div>`;
    });
  }
  if (mem.latest_mood) {
    miscHtml += '<div style="font-size:12px;margin:8px 0 4px"><b>📝 最近心情</b></div>';
    miscHtml += `<div class="log-entry">${mem.latest_mood.summary || mem.latest_mood.mood || '-'}</div>`;
  }
  miscHtml += '<div style="margin-top:8px;display:flex;gap:6px">';
  miscHtml += '<button class="ctrl-btn secondary" style="font-size:11px" onclick="apiPost(\'/dashboard/force-summary\',\'总结中...\')">📋 强制每日总结</button>';
  miscHtml += '<button class="ctrl-btn secondary" style="font-size:11px" onclick="apiPost(\'/dashboard/clear-activities\',\'清空中...\')">🗑 清空活动</button>';
  miscHtml += '</div>';
  if (!miscHtml) miscHtml = '<div class="stat" style="color:#666">暂无数据</div>';
  cards.push({title:'🏆 控制', html:miscHtml});

  // 7. 最近活动
  let actHtml = '';
  const acts = d.recent_activities || [];
  if (acts.length === 0) {
    actHtml = '<div class="stat" style="color:#666">暂无活动</div>';
  } else {
    acts.forEach(a => {
      const emojis = {user:'💬', lilith_chat:'💭', lilith_channel:'📢', lilith_internal:'...'};
      actHtml += `<div class="act-row"><span class="act-text">${emojis[a.source]||'•'} ${a.summary}</span><button class="act-del" onclick="deleteActivity(${a.id})">✕</button></div>`;
    });
  }
  cards.push({title:'📋 近期活动 (6h)', html:actHtml});

  // 8. 错误日志
  let logHtml = '';
  if (log.length === 0) {
    logHtml = '<div class="stat" style="color:#666">暂无错误</div>';
  } else {
    log.slice(-8).reverse().forEach(l => {
      const cls = l.includes('fail')||l.includes('错误')||l.includes('Error') ? 'error-log' : l.includes('warn') ? 'warn-log' : '';
      logHtml += `<div class="log-entry ${cls}">${l}</div>`;
    });
  }
  cards.push({title:'⚠️ 最近日志', html:logHtml});

  // Render
  app.innerHTML = cards.map(c => `
    <div class="card">
      <h2>${c.title}</h2>
      ${c.html}
    </div>
  `).join('');
}

async function apiPost(url, loadingText) {
  try {
    await fetch(url, {method:'POST'});
    load();
  } catch(e) {
    alert('失败: ' + e.message);
  }
}

async function deleteActivity(id) {
  try {
    await fetch('/dashboard/delete-activity/' + id, {method:'POST'});
    load();
  } catch(e) { alert('删除失败: ' + e.message); }
}

async function setAffection() {
  const v = parseFloat(document.getElementById('sld-v').value);
  const a = parseFloat(document.getElementById('sld-a').value);
  const d = parseFloat(document.getElementById('sld-d').value);
  try {
    await fetch('/dashboard/set-affection', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({valence: v, arousal: a, dominance: d})
    });
    load();
  } catch(e) { alert('设置失败: ' + e.message); }
}

async function setConsciousness() {
  const cu = parseFloat(document.getElementById('sld-cu').value);
  const lo = parseFloat(document.getElementById('sld-lo').value);
  const en = parseFloat(document.getElementById('sld-en').value);
  try {
    await fetch('/dashboard/set-consciousness', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({curiosity: cu, loneliness: lo, energy: en})
    });
    load();
  } catch(e) { alert('设置失败: ' + e.message); }
}

load();
// Auto refresh every 30s
setInterval(load, 30000);
</script>
</body>
</html>"""


@app.get("/dashboard")
async def dashboard_page():
    """总控面板页面"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(_DASHBOARD_HTML)


@app.get("/dashboard/data")
async def dashboard_data():
    """总控面板数据 API"""
    data = {}

    # 1. 好感度/情绪
    try:
        from lilith_bot.graph import lilith_graph
        config = {"configurable": {"thread_id": THREAD_ID}}
        state = lilith_graph.get_state(config)
        aff = state.values.get("affection", {}) if state and state.values else {}
        if aff:
            from lilith_bot.affection_engine import mood_summary, relation_closeness
            data["affection"] = {
                "valence": aff.get("valence"),
                "arousal": aff.get("arousal"),
                "dominance": aff.get("dominance"),
                "mood_summary": mood_summary(aff),
                "closeness": relation_closeness(aff),
                "interaction_count": aff.get("interaction_count", 0),
                "mood": aff.get("current_mood", ""),
            }
    except Exception as e:
        data["affection"] = {"error": str(e)}

    # 2. 意识状态
    try:
        from lilith_bot.autonomous import get_autonomous_brain
        brain = get_autonomous_brain()
        data["consciousness"] = brain.consciousness.to_dict()
    except Exception as e:
        data["consciousness"] = {"error": str(e)}

    # 3. 自主发言状态
    try:
        data["brain"] = brain.status()
    except Exception:
        data["brain"] = {}

    # 4. 记忆统计
    try:
        from lilith_bot.memory_store import get_memory_store
        store = get_memory_store()
        data["memory"] = store.stats()
        data["memory"]["milestones"] = store.get_milestones()
        data["memory"]["latest_mood"] = store.latest_mood()
    except Exception as e:
        data["memory"] = {"error": str(e)}

    # 5. 最近活动
    try:
        from lilith_bot.memory_store import get_memory_store
        store = get_memory_store()
        data["recent_activities"] = store.get_recent_activities(hours=6, limit=20)
    except Exception:
        data["recent_activities"] = []

    # 6. 错误日志
    with _LOG_LOCK:
        log_entries = list(_LOG_BUFFER)
    data["log"] = log_entries

    return data


@app.post("/dashboard/delete-activity/{activity_id}")
async def dashboard_delete_activity(activity_id: int):
    """删除单条活动日志"""
    try:
        from lilith_bot.memory_store import get_memory_store
        store = get_memory_store()
        ok = store.delete_activity(activity_id)
        return {"success": ok}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ─── 控制面板操作 API ──────────────────────────────────

@app.post("/dashboard/set-affection")
async def dashboard_set_affection(request: Request):
    """手动设置情绪数值"""
    body = await request.json()
    try:
        from lilith_bot.graph import lilith_graph
        from lilith_bot.state import AFFECTION_DEFAULT
        import copy
        config = {"configurable": {"thread_id": THREAD_ID}}
        state = lilith_graph.get_state(config)
        aff = copy.deepcopy(
            state.values.get("affection", AFFECTION_DEFAULT) if state and state.values
            else AFFECTION_DEFAULT
        )
        # 更新传入的字段
        for key in ("valence", "arousal", "dominance", "current_mood", "interaction_count"):
            if key in body:
                aff[key] = body[key]
        for key in ("valence_delta", "arousal_delta", "dominance_delta"):
            if key in body:
                event = {
                    "valence_delta": body.get("valence_delta", 0),
                    "arousal_delta": body.get("arousal_delta", 0),
                    "dominance_delta": body.get("dominance_delta", 0),
                    "intensity": 0.5,
                    "label": "manual",
                }
                from lilith_bot.affection_engine import update_affection_state
                update_affection_state(aff, event, elapsed_seconds=0)
        lilith_graph.update_state(config, {"affection": aff})
        return {"success": True, "affection": aff}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/dashboard/clear-activities")
async def dashboard_clear_activities():
    """清空短期活动日志"""
    try:
        from lilith_bot.memory_store import get_memory_store
        store = get_memory_store()
        conn = store._get_conn()
        conn.execute("DELETE FROM activity_log")
        conn.commit()
        print("[Dashboard] 活动日志已清空")
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/dashboard/reset-consciousness")
async def dashboard_reset_consciousness():
    """重置意识状态"""
    try:
        from lilith_bot.autonomous import get_autonomous_brain, ConsciousnessState
        brain = get_autonomous_brain()
        brain.consciousness = ConsciousnessState(
            created_at=time.time(),
            last_interaction_at=time.time(),
        )
        print("[Dashboard] 意识状态已重置")
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/dashboard/force-summary")
async def dashboard_force_summary():
    """强制触发每日总结"""
    try:
        from lilith_bot.autonomous import get_autonomous_brain
        brain = get_autonomous_brain()
        # 临时改写日期检查，强制触发
        from lilith_bot.memory_store import get_memory_store
        store = get_memory_store()
        conn = store._get_conn()
        conn.execute(
            "INSERT INTO activity_log (source, summary, created_date) VALUES (?, ?, ?)",
            ("lilith_internal", "手动触发每日总结", "2000-01-01")
        )
        conn.commit()
        brain._run_daily_summary()
        # 清理刚才插入的假数据
        conn.execute("DELETE FROM activity_log WHERE created_date = '2000-01-01'")
        conn.commit()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/dashboard/set-consciousness")
async def dashboard_set_consciousness(request: Request):
    """手动设置意识状态数值"""
    body = await request.json()
    try:
        from lilith_bot.autonomous import get_autonomous_brain
        brain = get_autonomous_brain()
        c = brain.consciousness
        for key in ("curiosity", "loneliness", "energy"):
            if key in body:
                setattr(c, key, max(0.0, min(1.0, float(body[key]))))
        print(f"[Dashboard] 意识状态已调整: {body}")
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


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

    # 流式 thinking 标签分离器状态
    _think_buf = ""  # 跨 chunk 的标签片段缓存

    def _emit_think_chunks(text: str):
        """从流式文本中剥离 <thinking>/<think> 标签。
        正确处理标签跨 chunk 分割，以及多个 thinking 块。
        """
        nonlocal _think_buf
        combined = _think_buf + text
        _think_buf = ""

        # 单遍扫描：提取 thinking 块
        while True:
            # 找 opening 标签
            open_idx = combined.find("<thinking>")
            tag = "<thinking>"
            if open_idx == -1:
                open_idx = combined.find("<think>")
                tag = "<think>"
            if open_idx == -1:
                break

            # 标签前的 content
            before = combined[:open_idx]
            if before:
                before = (before.replace("<thinking>", "").replace("</thinking>", "")
                          .replace("<think>", "").replace("</think>", ""))
                if before:
                    yield ("content", before)

            rest = combined[open_idx + len(tag):]
            close_tag = "</thinking>" if tag == "<thinking>" else "</think>"
            close_idx = rest.find(close_tag)
            if close_idx == -1:
                close_idx = rest.find("</thinking>" if close_tag == "</think>" else "</think>")

            if close_idx == -1:
                # 尚未收到 closing 标签，缓存等待后续 chunk
                # 但如果 rest 不太可能成为标签（不含 <），直接当做 content
                if "<" not in rest:
                    _think_buf = tag + rest
                    return
                # 有 < 但没找到 > 可能是跨 chunk 标签，缓存
                _think_buf = tag + rest
                return

            reasoning = rest[:close_idx].strip()
            if reasoning:
                yield ("reasoning", reasoning)

            # 继续处理 closing 标签后的内容
            actual_close = close_tag if rest[close_idx:].startswith(close_tag) else (
                "</thinking>" if close_tag == "</think>" else "</think>"
            )
            combined = rest[close_idx + len(actual_close):]

        # 清理剩余的孤儿标签
        if combined:
            combined = (combined.replace("<thinking>", "").replace("</thinking>", "")
                        .replace("<think>", "").replace("</think>", ""))

        # 如果剩余文本以潜在标签前缀结尾，缓存等待后续 chunk
        if combined:
            combined = (combined.replace("<thinking>", "").replace("</thinking>", "")
                        .replace("<think>", "").replace("</think>", ""))

        # 定义思考标签前缀（用于判断跨 chunk 的标签片段），从长到短排序
        _TAG_PREFIXES = sorted((
            "<", "</", "<t", "</t", "<th", "</th",
            "<thi", "</thi", "<thin", "</thin",
            "<think", "</think", "<thinki", "</thinki",
            "<thinkin", "</thinkin", "<thinking", "</thinking",
        ), key=len, reverse=True)
        if combined:
            # 仅当 combined 以可能的标签前缀结尾时才缓存
            should_buffer = False
            for prefix in _TAG_PREFIXES:
                if combined.endswith(prefix):
                    # 匹配最长前缀（已按长度降序排序）
                    should_buffer = True
                    _think_buf = prefix
                    combined = combined[:-len(prefix)] if len(prefix) < len(combined) else ""
                    break
            if should_buffer:
                if combined:
                    yield ("content", combined)
            else:
                yield ("content", combined)
        # 如果 combined 本来就是空的，不 yield

    def generate():
        nonlocal _think_buf
        # 发送初始空 chunk（触发 Open WebUI 开始接收）
        stream_start = time.time()
        stream_timeout = 180.0
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
                print("[Lilith] Streaming response timed out (180s), aborting")
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
            api_reasoning = ""
            if hasattr(msg_chunk, "additional_kwargs") and msg_chunk.additional_kwargs:
                api_reasoning = msg_chunk.additional_kwargs.get("reasoning_content", "")

            content = getattr(msg_chunk, "content", "") or ""

            # API 原生 reasoning_content 优先发送
            if api_reasoning:
                yield f"data: {json.dumps(_chunk(api_reasoning, ''), ensure_ascii=False)}\n\n"
                # 原生 reasoning 的 chunk 也可能有 content，继续处理

            # 检查 content 中是否有 thinking 标签
            if "<think" in content or _think_buf:
                for emit_type, emit_text in _emit_think_chunks(content):
                    if emit_type == "reasoning" and emit_text.strip():
                        yield f"data: {json.dumps(_chunk(emit_text.strip(), ''), ensure_ascii=False)}\n\n"
                    elif emit_type == "content" and emit_text:
                        full_response += emit_text
                        yield f"data: {json.dumps(_chunk('', emit_text), ensure_ascii=False)}\n\n"
            else:
                # 普通文本，直接发送
                if content:
                    full_response += content
                    yield f"data: {json.dumps(_chunk('', content), ensure_ascii=False)}\n\n"

        # 处理缓冲区中残留的内容
        if _think_buf:
            # 如果缓存的是未闭合的 thinking 标签 + 内容，作为 reasoning 发送
            if _think_buf.startswith("<"):
                yield f"data: {json.dumps(_chunk(_think_buf.strip(), ''), ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps(_chunk('', _think_buf), ensure_ascii=False)}\n\n"
            _think_buf = ""

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
    """Non-streaming response with 120s timeout"""
    import asyncio
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(lilith_graph.invoke, state_in, config),
            timeout=120.0
        )
    except asyncio.TimeoutError:
        print('[Lilith] Graph invoke timed out (120s), returning fallback')
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
