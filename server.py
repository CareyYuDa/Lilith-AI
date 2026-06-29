"""莉莉丝 — OpenAI 兼容 API 服务器（双模型架构）

将 LangGraph 对话图包装为 OpenAI /v1/chat/completions 端点，
供 Open WebUI 作为"外部 OpenAI 连接"调用。

特性:
  - 单模型（DeepSeek Flash）：lilith（日常聊天 + 工具调用 + 思维链）
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
    模型: lilith
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
from lilith_bot.tools import LANGCHAIN_TOOLS
from lilith_bot.state import AFFECTION_DEFAULT
from lilith_bot.trace_logger import set_trace_id, write_log, clear_trace_id


# ─── 配置 ─────────────────────────────────────────────────
THREAD_ID = "lilith-openwebui"
MODEL_NAME = "lilith"
LOCAL_API_KEY = "lilith-local"

# ─── 调试日志文件 ────────────────────────────────────────
_DEBUG_LOG = os.path.join(_ROOT, "lilith_debug.log")
def debug_log(msg: str):
    """追加写到日志文件（免去控制台查看）"""
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass

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
API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://opencode.ai/zen/go/v1")
CHAT_MODEL = os.getenv("LLM_CHAT_MODEL", "deepseek-v4-flash")

# Open WebUI 辅助请求的特征前缀（这些请求不走对话记忆系统）
_AUXILIARY_PREFIXES = (
    "### Task:",
    "### Instruction:",
    "Generate a concise",
    "Suggest 3-5",
    "Generate 1-3 broad tags",
)

TOOL_DISPLAY_NAMES = {
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
        "models": ["lilith"],
        "model": MODEL_NAME,
        "model": CHAT_MODEL,
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
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lilith Console</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#1a1a2e;color:#e0e0e0;padding:20px}
h1{font-size:24px;margin-bottom:20px;color:#e94560}
h2{font-size:15px;margin-bottom:8px;color:#0f3460;border-bottom:1px solid #16213e;padding-bottom:5px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}
.card{background:#16213e;border-radius:10px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.3)}
.scroll{max-height:400px;overflow-y:auto;overflow-x:hidden}
.scroll::-webkit-scrollbar{width:4px}
.scroll::-webkit-scrollbar-thumb{background:#0f3460;border-radius:2px}
.stat{display:flex;justify-content:space-between;padding:3px 0;font-size:13px;cursor:pointer;border-radius:3px}
.stat:hover{background:#1a2a4e}
.stat .val{color:#e94560;font-weight:bold}
.bar{height:6px;background:#0f3460;border-radius:3px;margin:2px 0 6px;overflow:hidden}
.bar-fill{height:100%;border-radius:3px}
.row{display:flex;justify-content:space-between;align-items:center;font-size:12px;padding:4px 6px;border-bottom:1px solid #1a2a4e;gap:4px;cursor:pointer;border-radius:3px}
.row:hover{background:#1a2a4e}
.row .info{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.row .del{background:transparent;border:none;color:#e94560;cursor:pointer;font-size:16px;padding:0;opacity:0.6}
.row .del:hover{opacity:1}
.inp{background:#0f3460;border:1px solid #1a5276;color:#fff;border-radius:4px;padding:4px 8px;font-size:12px;width:100%;margin:2px 0}
.inp:focus{outline:none;border-color:#e94560}
.sel{background:#0f3460;border:1px solid #1a5276;color:#fff;border-radius:4px;padding:4px 8px;font-size:12px;margin:2px 0}
#refresh{position:fixed;top:20px;right:20px;background:#e94560;color:#fff;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;font-size:13px;z-index:100}
.ctrl-btn{background:#e94560;color:#fff;border:none;padding:4px 10px;border-radius:4px;cursor:pointer;font-size:12px;margin:2px}
.ctrl-btn.s{background:#0f3460;border:1px solid #1a5276}
.ctrl-btn.s:hover{background:#1a5276}
[title]{text-decoration:underline 1px dotted #555}
.tab{font-size:12px;color:#aaa;margin:6px 0 2px}
.flex{display:flex;gap:4px;align-items:center;flex-wrap:wrap}
.modal-overlay{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.7);z-index:200;justify-content:center;align-items:center}
.modal-overlay.show{display:flex}
.modal{background:#16213e;border-radius:12px;padding:24px;min-width:340px;max-width:500px;max-height:80vh;overflow-y:auto;box-shadow:0 8px 32px rgba(0,0,0,0.5)}
.modal h3{color:#e94560;margin-bottom:12px;font-size:16px}
.modal label{font-size:12px;color:#aaa;display:block;margin-top:8px;margin-bottom:2px}
.modal .btns{display:flex;gap:8px;margin-top:16px;justify-content:flex-end}
</style>
</head>
<body>
<h1>&#x1F338; Lilith Console</h1>
<button id="refresh" onclick="loadData()">&#x21bb; Refresh</button>
<div class="grid" id="app"><div class="card"><div class="loading">Loading...</div></div></div>
<div class="modal-overlay" id="modal" onclick="if(event.target==this)closeModal()"><div class="modal" id="modalBody"></div></div>
<script>
var EDIT_DATA=null,DASH_DATA=null;
async function loadData(){const e=document.getElementById('app');try{const t=await fetch('/dashboard/data'),n=await t.json();DASH_DATA=n;renderCards(n)}catch(t){e.innerHTML='<div class="card" style="color:#ff6b6b">Error: '+t.message+'</div>'}}
function fmt(e){return null==e?'0.00':e.toFixed(2)}
function bar(e,t){return'<div class="bar"><div class="bar-fill" style="width:'+((e||0)*50+50)+'%;background:'+t+'"></div></div>'}
function renderCards(d){const e=d.affection||{},t=d.memories||[],n=d.activities||[],a=d.persons||[],i=d.log||[],r=d.personality||{},u=d.system_prompt||'',l=[];let s='';s+='<div class="stat" title="Mood"><span>Mood</span><span class="val">'+(e.mood_label||'?')+' '+(e.mood_emoji||'')+'</span></div>',s+='<div class="stat" title="Valence"><span>Valence V</span><span class="val">'+fmt(e.valence)+'</span></div>'+bar(e.valence,e.valence>0?'#4ecca3':'#ff6b6b'),s+='<div class="stat" title="Arousal"><span>Arousal A</span><span class="val">'+fmt(e.arousal)+'</span></div>'+bar(e.arousal,'#ffd93d'),s+='<div class="stat" title="Dominance"><span>Dominance D</span><span class="val">'+fmt(e.dominance)+'</span></div>'+bar(e.dominance,'#a66cff'),s+='<div class="stat"><span>Interactions</span><span class="val">'+(e.interaction_count||0)+'</span></div>',s+='<div class="flex" style="margin-top:8px">V<input id="sv" class="inp" style="width:55px" type="number" step="0.05" min="-1" max="1" value="'+(e.valence||0).toFixed(2)+'">A<input id="sa" class="inp" style="width:55px" type="number" step="0.05" min="-1" max="1" value="'+(e.arousal||0).toFixed(2)+'">D<input id="sd" class="inp" style="width:55px" type="number" step="0.05" min="-1" max="1" value="'+(e.dominance||0).toFixed(2)+'"><button class="ctrl-btn s" onclick="setAff()">Set</button></div>',l.push({title:'Emotion',html:s}),
s='<div class="stat" onclick="editPrompt()" style="cursor:pointer;color:#a66cff" title="Click to edit the system prompt">&#x2699; Edit System Prompt (worldview &amp; personality) <span style="float:right">&#x270E;</span></div><div style="font-size:11px;color:#666;margin-top:4px;padding:6px;background:#0f3460;border-radius:4px;max-height:120px;overflow-y:auto">'+(u||'').slice(0,200)+'...</div>',l.push({title:'Prompt',html:s}),s='<div class="scroll">';if(r.drives){s+='<div class="tab">Drives:</div>';for(const[e,t]of Object.entries(r.drives)){const n={dominance:'Dominance',autonomy:'Autonomy',belonging:'Belonging',achievement:'Achievement'};s+='<div class="stat" onclick="editDrive(\''+e+'\','+t.toFixed(2)+')" title="Click to edit"><span>'+(n[e]||e)+'</span><span class="val">'+t.toFixed(2)+'</span></div>'}}if(r.behavior){s+='<div class="tab">Behavior:</div>';for(const[e,t]of Object.entries(r.behavior))s+='<div class="stat" onclick="editBehavior(\''+e+'\','+t.toFixed(2)+')" title="Click to edit"><span>'+e+'</span><span class="val">'+t.toFixed(2)+'</span></div>'}if(r.emotion){s+='<div class="tab">Emotion:</div>';for(const[e,t]of Object.entries(r.emotion))s+='<div class="stat" onclick="editEmotion(\''+e+'\','+t.toFixed(2)+')" title="Click to edit"><span>'+e+'</span><span class="val">'+t.toFixed(2)+'</span></div>'}s||(s='<div class="stat" style="color:#666">No data</div>'),s+='</div>',l.push({title:'Personality',html:s}),s='<div class="scroll">',0===a.length?s='<div class="stat" style="color:#666">No persons</div>':a.forEach(e=>{s+='<div class="row" onclick="editPerson(\''+e.name+'\','+e.trust+','+e.intimacy+',\''+(e.portrait||'')+'\')" title="Click to edit"><span class="info"><b>'+e.name+'</b> trust:'+e.trust.toFixed(2)+' intm:'+e.intimacy.toFixed(2)+' ('+e.interactions+'x)</span></div>'}),s+='</div>',l.push({title:'Persons ('+a.length+')',html:s}),s='<div class="scroll">',0===t.length?s='<div class="stat" style="color:#666">No memories</div>':(t.forEach(e=>{let n=e.content;n.length>55&&(n=n.slice(0,55)+'..');const a={knowledge:'KN',emotional:'EM',event:'EV',skill:'SK',daily_summary:'SUM'};s+='<div class="row" onclick="editMemory('+e.id+',\''+(e.content||'').replace(/\x27/g,' ')+'\',\''+(e.type||'knowledge')+'\','+e.importance.toFixed(2)+',\''+(e.person||'')+'\')" title="Click to edit"><span class="info">['+(a[e.type]||e.type)+'] '+n+' <span style="color:#888">'+e.importance.toFixed(1)+'</span></span><button class="del" onclick="event.stopPropagation();delMem('+e.id+')">x</button></div>'}),s+='</div>'),l.push({title:'Memories ('+t.length+')',html:s}),s='<div class="scroll">',0===n.length?s='<div class="stat" style="color:#666">No activity</div>':(n.forEach(e=>{const t=e.person?e.person+': ':'',n=(e.summary||'').slice(0,50);s+='<div class="row" onclick="editActivity('+e.id+',\''+(e.summary||'').replace(/\x27/g,' ')+'\',\''+(e.source||'')+'\',\''+(e.person||'')+'\')" title="Click to edit"><span class="info">'+t+n+'</span><button class="del" onclick="event.stopPropagation();delAct('+e.id+')">x</button></div>'}),s+='<div style="margin-top:8px"><button class="ctrl-btn s" onclick="clearActs()">Clear All</button></div>'),s+='</div>',l.push({title:'Activity',html:s}),s='<div class="scroll">',0===i.length?s='<div class="stat" style="color:#666">No log</div>':(()=>{const e=Math.max(0,i.length-15);for(let t=i.length-1;t>=e;t--)s+='<div style="font-size:11px;font-family:monospace;color:#888;padding:2px 0;border-bottom:1px solid #1a2a4e">'+(i[t]||'').slice(0,70)+'</div>'})() ,s+='</div>',l.push({title:'System Log',html:s});let o='';for(let e=0;e<l.length;e++)o+='<div class="card"><h2>'+l[e].title+'</h2>'+l[e].html+'</div>';document.getElementById('app').innerHTML=o}
function showModal(e,t){document.getElementById('modalBody').innerHTML='<h3>'+e+'</h3>'+t+'<div class="btns"><button class="ctrl-btn s" onclick="closeModal()">Cancel</button></div>',document.getElementById('modal').classList.add('show')}
function closeModal(){document.getElementById('modal').classList.remove('show')}
function saveModal(e,t){fetch(e,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(t)}).then(()=>{closeModal();loadData()})}
function editDrive(e,t){showModal('Edit Drive: '+e,'<label>Value (0~1)</label><input id="me-val" class="inp" type="number" step="0.01" min="0" max="1" value="'+t.toFixed(2)+'"><div class="btns"><button class="ctrl-btn" onclick="saveDrive(\''+e+'\')">Save</button></div>')}
function saveDrive(e){saveModal('/dashboard/set-personality',{section:'drives',key:e,value:parseFloat(document.getElementById('me-val').value)})}
function editBehavior(e,t){showModal('Edit Behavior: '+e,'<label>Value (0~1)</label><input id="me-val" class="inp" type="number" step="0.01" min="0" max="1" value="'+t.toFixed(2)+'"><div class="btns"><button class="ctrl-btn" onclick="saveBehavior(\''+e+'\')">Save</button></div>')}
function saveBehavior(e){saveModal('/dashboard/set-personality',{section:'behavior',key:e,value:parseFloat(document.getElementById('me-val').value)})}
function editEmotion(e,t){showModal('Edit Emotion: '+e,'<label>Value (0~1)</label><input id="me-val" class="inp" type="number" step="0.01" min="0" max="1" value="'+t.toFixed(2)+'"><div class="btns"><button class="ctrl-btn" onclick="saveEmotion(\''+e+'\')">Save</button></div>')}
function saveEmotion(e){saveModal('/dashboard/set-personality',{section:'emotion',key:e,value:parseFloat(document.getElementById('me-val').value)})}
function editPerson(e,t,n,a){showModal('Edit Person: '+e,'<label>Name</label><input id="me-name" class="inp" value="'+(e||'')+'"><label>Portrait</label><input id="me-portrait" class="inp" value="'+(a||'')+'"><label>Trust (0~1)</label><input id="me-trust" class="inp" type="number" step="0.01" min="0" max="1" value="'+t.toFixed(2)+'"><label>Intimacy (0~1)</label><input id="me-intm" class="inp" type="number" step="0.01" min="0" max="1" value="'+n.toFixed(2)+'"><div class="btns"><button class="ctrl-btn" onclick="savePerson(\''+e+'\')">Save</button></div>')}
function savePerson(e){const t={orig_name:e,person_name:document.getElementById('me-name').value,portrait:document.getElementById('me-portrait').value,trust:parseFloat(document.getElementById('me-trust').value),intimacy:parseFloat(document.getElementById('me-intm').value)};saveModal('/dashboard/set-person',t)}
function editPrompt(){showModal('Edit System Prompt','<label>世界观/人格提示词</label><textarea id="me-prompt" class="inp" style="height:300px;font-size:12px;font-family:monospace">'+(DASH_DATA.system_prompt||'').replace(/`/g,'&#x60;').replace(/\$/g,'&#36;')+'</textarea><div class="btns"><button class="ctrl-btn" onclick="savePrompt()">Save</button></div>')}
function savePrompt(){saveModal('/dashboard/set-system-prompt',{prompt:document.getElementById('me-prompt').value})}
function editMemory(e,t,n,a,i){showModal('Edit Memory #'+e,'<label>Content</label><textarea id="me-content" class="inp" rows="3">'+(t||'')+'</textarea><label>Type</label><select id="me-type" class="sel"><option value="knowledge"'+(n=='knowledge'?' selected':'')+'>Knowledge</option><option value="emotional"'+(n=='emotional'?' selected':'')+'>Emotional</option><option value="event"'+(n=='event'?' selected':'')+'>Event</option><option value="skill"'+(n=='skill'?' selected':'')+'>Skill</option><option value="daily_summary"'+(n=='daily_summary'?' selected':'')+'>Daily Summary</option></select><label>Importance (0~1)</label><input id="me-imp" class="inp" type="number" step="0.1" min="0" max="1" value="'+a.toFixed(1)+'"><label>Person</label><input id="me-person" class="inp" value="'+(i||'对方')+'"><div class="btns"><button class="ctrl-btn" onclick="saveMemory('+e+')">Save</button></div>')}
function saveMemory(e){const t={content:document.getElementById('me-content').value,type:document.getElementById('me-type').value,importance:parseFloat(document.getElementById('me-imp').value),person:document.getElementById('me-person').value};saveModal('/dashboard/set-memory/'+e,t)}
function editActivity(e,t,n,a){showModal('Edit Activity #'+e,'<label>Summary</label><input id="me-summary" class="inp" value="'+(t||'')+'"><label>Source</label><select id="me-source" class="sel"><option value="user"'+(n=='user'?' selected':'')+'>User</option><option value="lilith_chat"'+(n=='lilith_chat'?' selected':'')+'>Lilith</option><option value="lilith_channel"'+(n=='lilith_channel'?' selected':'')+'>Channel</option><option value="lilith_internal"'+(n=='lilith_internal'?' selected':'')+'>Internal</option></select><label>Person</label><input id="me-person" class="inp" value="'+(a||'')+'"><div class="btns"><button class="ctrl-btn" onclick="saveActivity('+e+')">Save</button></div>')}
function saveActivity(e){const t={summary:document.getElementById('me-summary').value,source:document.getElementById('me-source').value,person:document.getElementById('me-person').value};saveModal('/dashboard/set-activity/'+e,t)}
async function setAff(){try{await fetch('/dashboard/set-affection',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({valence:parseFloat(document.getElementById('sv').value),arousal:parseFloat(document.getElementById('sa').value),dominance:parseFloat(document.getElementById('sd').value)})});loadData()}catch(e){}}
async function delMem(e){try{await fetch('/dashboard/delete-memory/'+e,{method:'POST'});loadData()}catch(e){}}
async function delAct(e){try{await fetch('/dashboard/delete-activity/'+e,{method:'POST'});loadData()}catch(e){}}
async function clearActs(){try{await fetch('/dashboard/clear-activities',{method:'POST'});loadData()}catch(e){}}
loadData();setInterval(loadData,30000);
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
            data["affection"] = {
                "valence": aff.get("valence"),
                "arousal": aff.get("arousal"),
                "dominance": aff.get("dominance"),
                "mood_label": aff.get("mood_label", "未知"),
                "mood_emoji": aff.get("mood_emoji", ""),
                "interaction_count": aff.get("interaction_count", 0),
            }
    except Exception as e:
        data["affection"] = {"error": str(e)}

    # 2. 自主发言状态
    try:
        from lilith_bot.autonomous import get_autonomous_brain
        brain = get_autonomous_brain()
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

    # 5. 长期记忆列表
    try:
        from lilith_bot.memory_store import get_memory_store
        store = get_memory_store()
        all_mems = store.get_all_memories()
        data["memories"] = [{
            "id": m["id"], "content": m["content"],
            "type": m["type"], "importance": m["importance"],
            "person": m.get("person", "对方"),
            "created_at": m.get("created_at", ""),
        } for m in all_mems]
    except Exception as e:
        data["memories"] = []

    # 6. 人物认知
    try:
        from lilith_bot.memory_store import get_memory_store
        store = get_memory_store()
        data["persons"] = store.list_known_persons()
    except Exception:
        data["persons"] = []

    # 7. 人格参数
    try:
        from lilith_bot.personality import get_personality_summary
        data["personality"] = get_personality_summary()
    except Exception:
        data["personality"] = {}

    # 8. 最近活动
    try:
        from lilith_bot.memory_store import get_memory_store
        store = get_memory_store()
        data["activities"] = store.get_recent_activities(hours=6, limit=20)
    except Exception:
        data["activities"] = []

    # 9. System Prompt（当前世界观/人格文本）
    try:
        from lilith_bot.persona import LILITH_SYSTEM_PROMPT
        data["system_prompt"] = LILITH_SYSTEM_PROMPT
    except Exception:
        data["system_prompt"] = ""

    # 10. 错误日志
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

@app.post("/dashboard/set-personality")
async def dashboard_set_personality(request: Request):
    data = await request.json()
    try:
        import lilith_bot.personality as p
        section = data.get("section")
        key = data.get("key")
        value = data.get("value")
        if section and key and value is not None:
            mapping = {
                "drives": {"dominance":"DOMINANCE_DRIVE","autonomy":"AUTONOMY_DRIVE","belonging":"BELONGING_DRIVE","achievement":"ACHIEVEMENT_DRIVE"},
                "behavior": {"initiative":"INITIATIVE_CHANCE","interruption":"INTERRUPTION_TENDENCY","agreement":"AGREEMENT_BIAS","debate":"DEBATE_TENDENCY","self_disclosure":"SELF_DISCLOSURE"},
                "emotion": {"sensitivity":"EMOTIONAL_SENSITIVITY","forgiveness":"FORGIVENESS_RATE","jealousy":"JEALOUSY_TENDENCY"},
            }
            const_name = mapping.get(section, {}).get(key)
            if const_name and hasattr(p, const_name):
                setattr(p, const_name, max(0.0, min(1.0, float(value))))
                return {"success": True, "updated": f"{const_name}={value}"}
        return {"success": False, "error": "Invalid params"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/dashboard/set-system-prompt")
async def dashboard_set_system_prompt(request: Request):
    """修改世界观/人格提示词"""
    try:
        body = await request.json()
        prompt = body.get("prompt", "")
        if not prompt or len(prompt) < 10:
            return {"success": False, "error": "Prompt too short"}
        import lilith_bot.persona as persona
        persona.LILITH_SYSTEM_PROMPT = prompt
        # 写回文件持久化
        import re
        path = os.path.join(_ROOT, "lilith_bot", "persona.py")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # 替换 LILITH_SYSTEM_PROMPT 的内容（保留引号和等号）
        new_content = re.sub(
            r'(LILITH_SYSTEM_PROMPT\s*=\s*""").*?(""")',
            lambda m: m.group(1) + prompt + m.group(2),
            content,
            flags=re.DOTALL,
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return {"success": True, "length": len(prompt)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/dashboard/delete-person")
async def dashboard_delete_person(request: Request):
    try:
        data = await request.json()
        name = data.get("name", "")
        if not name:
            return {"success": False, "error": "Name required"}
        from lilith_bot.memory_store import get_memory_store
        store = get_memory_store()
        conn = store._get_conn()
        conn.execute("DELETE FROM person_knowledge WHERE person_name = ?", (name,))
        conn.commit()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/dashboard/set-person")
async def dashboard_set_person(request: Request):
    data = await request.json()
    try:
        from lilith_bot.memory_store import get_memory_store
        store = get_memory_store()
        name = data.get("person_name", "").strip()
        orig = data.get("orig_name", name)
        if not name:
            return {"success": False, "error": "Name required"}
        if orig and orig != name:
            conn = store._get_conn()
            conn.execute("DELETE FROM person_knowledge WHERE person_name = ?", (orig,))
            conn.commit()
        store.add_or_update_person_knowledge(person_name=name, portrait=data.get("portrait",""), trust=float(data.get("trust",0.5)), intimacy=float(data.get("intimacy",0.3)))
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/dashboard/set-memory/{memory_id}")
async def dashboard_set_memory(memory_id: int, request: Request):
    data = await request.json()
    try:
        from lilith_bot.memory_store import get_memory_store
        store = get_memory_store()
        conn = store._get_conn()
        conn.execute("UPDATE memories SET content=?, memory_type=?, importance=?, person=? WHERE id=?", (data.get("content",""), data.get("type","knowledge"), float(data.get("importance",0.5)), data.get("person",""), memory_id))
        conn.commit()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/dashboard/set-activity/{activity_id}")
async def dashboard_set_activity(activity_id: int, request: Request):
    data = await request.json()
    try:
        from lilith_bot.memory_store import get_memory_store
        store = get_memory_store()
        conn = store._get_conn()
        conn.execute("UPDATE activity_log SET summary=?, source=?, person=? WHERE id=?", (data.get("summary",""), data.get("source","user"), data.get("person",""), activity_id))
        conn.commit()
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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # ── 全链路追踪：生成 trace_id ──
    trace_id = set_trace_id()
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    model = body.get("model", MODEL_NAME)

    # 提取最后一条用户消息
    user_msg = _extract_last_user_message(messages)

    # 记录 API 入口
    write_log("api_entry", {
        "user_msg": user_msg[:200],
        "stream": stream,
        "model": model,
        "message_count": len(messages),
    })

    if not user_msg:
        clear_trace_id()
        return JSONResponse(
            {"error": {"message": "No user message found", "type": "invalid_request_error"}},
            status_code=400,
        )

    # ── 辅助请求检测 ──
    if _is_auxiliary_request(user_msg, messages):
        write_log("api_entry", {"auxiliary": True, "user_msg": user_msg[:200]})
        clear_trace_id()
        return _call_auxiliary(messages, stream)

    # 特殊命令
    if user_msg.strip().lower() == "/reset":
        write_log("api_entry", {"reset": True})
        clear_trace_id()
        if stream:
            return _stream_simple("莉莉丝: 对话已重置~ 莉莉丝会忘了之前的事哦 (´･ω･`)", model)
        return _make_response("莉莉丝: 对话已重置~ 莉莉丝会忘了之前的事哦 (´･ω･`)", model)

    config = {"configurable": {"thread_id": THREAD_ID}}
    sv = _get_state(config)

    # 构建 graph 输入（只传新消息，checkpoint 自动保留历史）
    state_in = {
        "messages": [HumanMessage(content=user_msg)],
        "persona": "莉莉丝（Lilith）System Prompt（待SubconsciousEngine重构）",
        "long_term_memories": sv.get("long_term_memories", []),
        "affection": sv.get("affection", AFFECTION_DEFAULT),
        "llm_type": "local",
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
        delta = {"role": "assistant"}
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
        # 每次流式请求先清空旧日志
        try:
            open(_DEBUG_LOG, "w", encoding="utf-8").close()
        except Exception:
            pass
        debug_log("=== 新请求 ===")

        # 发送初始空 chunk（触发 Open WebUI 开始接收）
        stream_start = time.time()
        stream_timeout = 180.0
        yield f"data: {json.dumps(_chunk('', ''), ensure_ascii=False)}\n\n"

        full_response = ""

        # 使用 graph.stream 同步流式输出（SqliteSaver 不支持异步）
        try:
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
                    # 记录工具执行结果
                    tool_content = getattr(msg_chunk, "content", "") or ""
                    write_log("tool_call", {"result": tool_content[:300]})
                    tool_done_hint = '\n✅ 工具调用完成\n\n'
                    yield f"data: {json.dumps(_chunk('', tool_done_hint), ensure_ascii=False)}\n\n"
                    continue
                if "AIMessage" not in chunk_type_name:
                    continue

                # 工具调用：输出提示信息给用户 + 记录日志
                tool_calls = getattr(msg_chunk, "tool_calls", None)
                if tool_calls:
                    tool_log = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            tool_name = tc.get("name", "unknown")
                            tool_args = tc.get("args", {})
                        else:
                            tool_name = getattr(tc, "name", "unknown")
                            tool_args = getattr(tc, "args", {})
                        tool_log.append({"name": tool_name, "args": tool_args})
                        friendly = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
                        hint = f"\n\n🔧 莉莉丝正在调用工具: {friendly}...\n\n"
                        yield f"data: {json.dumps(_chunk('', hint), ensure_ascii=False)}\n\n"
                    write_log("tool_call", {"tools": tool_log})
                    continue

                # 提取 reasoning_content（来自 monkey-patch 透传）
                api_reasoning = ""
                if hasattr(msg_chunk, "additional_kwargs") and msg_chunk.additional_kwargs:
                    api_reasoning = msg_chunk.additional_kwargs.get("reasoning_content", "")

                content = getattr(msg_chunk, "content", "") or ""

                # API 原生 reasoning_content 优先发送
                if api_reasoning:
                    yield f"data: {json.dumps(_chunk(api_reasoning, ''), ensure_ascii=False)}\n\n"

                # 检查 content 中是否有 thinking 标签
                if "<think" in content or _think_buf:
                    for emit_type, emit_text in _emit_think_chunks(content):
                        if emit_type == "reasoning" and emit_text.strip():
                            yield f"data: {json.dumps(_chunk(emit_text.strip(), ''), ensure_ascii=False)}\n\n"
                        elif emit_type == "content" and emit_text:
                            full_response += emit_text
                            yield f"data: {json.dumps(_chunk('', emit_text), ensure_ascii=False)}\n\n"
                else:
                    if content:
                        full_response += content
                        yield f"data: {json.dumps(_chunk('', content), ensure_ascii=False)}\n\n"

        except Exception as e:
            print(f"[Lilith] Stream error: {type(e).__name__}: {e}")
            err_text = "\n\n[Error: " + type(e).__name__ + "]"
            yield f"data: {json.dumps(_chunk('', err_text), ensure_ascii=False)}\n\n"

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

        # 记录 LLM 完整输出
        write_log("llm_output", {"content": full_response[:500], "content_length": len(full_response)})

        # 清空 trace_id
        clear_trace_id()

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
        write_log("llm_output", {"error": "timeout", "timeout": 120})
        clear_trace_id()
        fallback = "\u5410\u5410\u2026\u66f4\u65b0\u8fc7\u4e8e\u6162\u4e86\uff0c\u518d\u7b49\u7b49\u5427~"
        return _make_response(fallback, model)
    except Exception as e:
        print(f'[Lilith] Graph invoke error: {type(e).__name__}: {e}')
        write_log("llm_output", {"error": type(e).__name__, "msg": str(e)[:200]})
        clear_trace_id()
        return _make_response(f"抱歉，出了点问题: {type(e).__name__}", model)

    content = ""
    reasoning = ""
    for m in reversed(result.get("messages", [])):
        if hasattr(m, "type") and m.type == "ai":
            if m.content and m.content.strip():
                content = m.content
                if hasattr(m, "additional_kwargs") and m.additional_kwargs:
                    reasoning = m.additional_kwargs.get("reasoning_content", "")
                break

    write_log("llm_output", {"content": content[:500], "content_length": len(content)})
    clear_trace_id()
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
    print(f"[Lilith]    模型: {CHAT_MODEL}")
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
