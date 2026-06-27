"""莉莉丝 Chat — Gradio 前端
暗色主题 · 粉色点缀 · 双标签（聊天 / 记忆&状态）
"""

import os, sys, uuid

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from langchain_core.messages import HumanMessage
from lilith_bot.graph import lilith_graph, DB_PATH
from lilith_bot.persona import LILITH_SYSTEM_PROMPT

# ═══════════════════════════════════════════════════════
#  色板
# ═══════════════════════════════════════════════════════
_BG     = "#0f0f1a"
_SURFACE= "#1a1a2e"
_PINK   = "#ff8fab"
_MUTED  = "#8890b0"
_BORDER = "#2a2a4a"
_TEXT   = "#e0e0ff"
_PINK_DIM = "rgba(255,143,171,0.12)"

# ═══════════════════════════════════════════════════════
#  主题 — 只用确认存在的属性
# ═══════════════════════════════════════════════════════
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.pink,
    neutral_hue=gr.themes.colors.slate,
).set(
    # 全局背景
    body_background_fill_dark       = _BG,
    body_background_fill            = _BG,
    background_fill_primary_dark    = _BG,
    background_fill_primary         = _BG,
    background_fill_secondary_dark  = _SURFACE,
    background_fill_secondary       = _SURFACE,
    body_text_color_dark            = _TEXT,
    body_text_color                 = _TEXT,
    body_text_color_subdued_dark    = _MUTED,
    body_text_color_subdued         = _MUTED,
    # 区块
    block_background_fill_dark      = _SURFACE,
    block_background_fill           = _SURFACE,
    block_border_color_dark         = _BORDER,
    block_border_color              = _BORDER,
    block_label_text_color_dark     = _MUTED,
    block_label_text_color          = _MUTED,
    block_title_text_color_dark     = _PINK,
    block_title_text_color          = _PINK,
    # 面板
    panel_background_fill_dark      = _BG,
    panel_background_fill           = _BG,
    panel_border_color_dark         = _BORDER,
    panel_border_color              = _BORDER,
    # 边框
    border_color_primary_dark       = _BORDER,
    border_color_primary            = _BORDER,
    # 输入框
    input_background_fill_dark      = _BG,
    input_background_fill           = _BG,
    input_border_color_dark         = _BORDER,
    input_border_color              = _BORDER,
    input_placeholder_color_dark    = _MUTED,
    input_placeholder_color         = _MUTED,
    # 按钮
    button_primary_background_fill_dark  = _PINK,
    button_primary_background_fill       = _PINK,
    button_primary_text_color_dark       = "#fff",
    button_primary_text_color            = "#fff",
    button_secondary_background_fill_dark = "transparent",
    button_secondary_background_fill      = "transparent",
    button_secondary_border_color_dark    = _BORDER,
    button_secondary_border_color         = _BORDER,
    button_secondary_text_color_dark      = _MUTED,
    button_secondary_text_color           = _MUTED,
    # 滑块
    slider_color_dark               = _PINK,
    slider_color                    = _PINK,
    # 代码
    code_background_fill_dark       = "rgba(0,0,0,0.3)",
    code_background_fill            = "rgba(0,0,0,0.3)",
)

# ═══════════════════════════════════════════════════════
#  CSS — 处理 theme 覆盖不到的部分
# ═══════════════════════════════════════════════════════
CSS = f"""
footer {{ display: none !important; }}

/* 全局字体 */
body, .gradio-container {{
    font-family: 'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif !important;
    background: {_BG} !important;
}}
.gradio-container,
.gradio-container > div,
.gradio-container .main,
.gradio-container .wrap,
.gradio-container .contain {{
    background: transparent !important;
}}

/* ── 标签栏 ── */
.tab-nav {{
    border-bottom: 1px solid {_BORDER} !important;
    gap: 0 !important;
    padding: 0 18px !important;
    margin: 0 !important;
    background: {_SURFACE} !important;
}}
.tab-nav button {{
    font-size: 14px !important;
    font-weight: 500 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 12px 22px !important;
    background: transparent !important;
    transition: all .2s !important;
}}
.tab-nav button:hover {{ color: {_PINK} !important; }}
.tab-nav button.selected {{
    color: {_PINK} !important;
    border-bottom: 2px solid {_PINK} !important;
    font-weight: 600 !important;
}}

/* ── 聊天区: 彻底去白底 ── */
.chatbot,
.chatbot > div,
.chatbot > div > div,
.chatbot .message-wrap,
.chatbot .message,
.chatbot .panel {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
/* 用户气泡 */
.chatbot .message.user {{
    background: {_PINK_DIM} !important;
    border-radius: 18px !important;
    border-bottom-right-radius: 6px !important;
    max-width: 82% !important;
}}
/* AI 气泡 */
.chatbot .message.assistant {{
    background: {_SURFACE} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 18px !important;
    border-bottom-left-radius: 6px !important;
    max-width: 82% !important;
}}
/* 聊天文字颜色 */
.chatbot .message,
.chatbot .message p,
.chatbot .message span {{
    color: {_TEXT} !important;
}}

/* ── 输入栏 ── */
.input-bar {{
    border-top: 1px solid {_BORDER} !important;
    background: {_SURFACE} !important;
    padding: 10px 18px 14px !important;
    border-radius: 0 !important;
}}
.input-bar textarea {{
    border: 1px solid {_BORDER} !important;
    border-radius: 24px !important;
    background: {_BG} !important;
    color: {_TEXT} !important;
    resize: none !important;
}}
.input-bar textarea::placeholder {{ color: {_MUTED} !important; }}
.input-bar textarea:focus {{
    border-color: {_PINK} !important;
    box-shadow: none !important;
}}
.input-bar button {{
    border-radius: 24px !important;
}}

/* 清空按钮 */
.btn-clear {{
    border: 1px solid {_BORDER} !important;
    background: transparent !important;
    color: {_MUTED} !important;
    border-radius: 14px !important;
    font-size: 12px !important;
    padding: 4px 14px !important;
    transition: all .2s !important;
}}
.btn-clear:hover {{
    border-color: {_PINK} !important;
    color: {_PINK} !important;
}}

/* ── 状态卡片 ── */
.status-card {{
    background: {_SURFACE} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 12px !important;
    padding: 24px !important;
}}
.card-title {{
    font-size: 16px !important;
    font-weight: 700 !important;
    color: {_PINK} !important;
    margin: 0 0 4px 0 !important;
}}
.card-desc {{
    font-size: 12px !important;
    color: {_MUTED} !important;
    margin: 0 0 16px 0 !important;
}}
.memory-box textarea {{
    font-size: 13px !important;
    line-height: 1.8 !important;
    color: #c0c0d8 !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 8px !important;
    background: {_BG} !important;
}}
.mood-box input {{
    color: #c0c0d8 !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 8px !important;
    background: {_BG} !important;
}}
.db-path {{
    font-size: 11px !important;
    color: {_MUTED} !important;
    font-family: monospace !important;
    margin-top: 20px !important;
}}

/* ── 滚动条 ── */
::-webkit-scrollbar {{ width: 5px !important; }}
::-webkit-scrollbar-thumb {{ background: {_BORDER} !important; border-radius: 3px !important; }}
::-webkit-scrollbar-track {{ background: transparent !important; }}
"""

# ═══════════════════════════════════════════════════════
#  JS — 强制暗色模式
# ═══════════════════════════════════════════════════════
JS_FORCE_DARK = """
() => {
    document.documentElement.classList.add('dark');
    document.body.classList.add('dark');
    document.body.style.background = '#0f0f1a';
}
"""

# ═══════════════════════════════════════════════════════
#  后端交互
# ═══════════════════════════════════════════════════════

def _to_chat_format(state_values: dict) -> list:
    out = []
    pending = None
    for m in state_values.get("messages", []):
        if not hasattr(m, "type"):
            continue
        if m.type == "human":
            pending = m.content
        elif m.type == "ai" and pending is not None:
            out.append({"role": "user", "content": pending})
            out.append({"role": "assistant", "content": m.content})
            pending = None
    return out


def _snapshot(state_values: dict) -> tuple:
    mems = state_values.get("long_term_memories", [])
    aff  = state_values.get("affection", {})
    text = "\n".join(f"· {m}" for m in mems) if mems else "暂无记忆"
    return text, aff.get("score", 50), aff.get("mood", "平静")


def _empty_snapshot() -> tuple:
    return "暂无记忆", 50, "平静"


def on_send(message: str, history, thread_id: str):
    if not message or not message.strip():
        return history, *_empty_snapshot(), thread_id

    cfg = {"configurable": {"thread_id": thread_id}}

    try:
        st = lilith_graph.get_state(cfg)
        sv = st.values if st and st.values else {}
    except Exception:
        sv = {}

    msgs = list(sv.get("messages", []))
    msgs.append(HumanMessage(content=message))

    state_in = {
        "messages": msgs,
        "persona": LILITH_SYSTEM_PROMPT,
        "long_term_memories": sv.get("long_term_memories", []),
        "affection": sv.get("affection", {"score": 50, "mood": "平静"}),
    }

    try:
        result = lilith_graph.invoke(state_in, cfg)
    except Exception as e:
        mem, score, mood = _snapshot(sv)
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"出错了: {e}"},
        ], mem, score, mood, thread_id

    return _to_chat_format(result), *_snapshot(result), thread_id


def on_reset(thread_id: str):
    return [], *_empty_snapshot(), str(uuid.uuid4())[:8]


# ═══════════════════════════════════════════════════════
#  界面
# ═══════════════════════════════════════════════════════

def build() -> gr.Blocks:
    with gr.Blocks(
        title="Lilith",
        fill_height=True,
    ) as demo:
        tid = gr.State(str(uuid.uuid4())[:8])

        with gr.Tabs():
            # ── 聊天 ──
            with gr.Tab("聊天"):
                chat = gr.Chatbot(
                    value=[],
                    height="72vh",
                    layout="bubble",
                    placeholder="和莉莉丝开始对话吧",
                )
                with gr.Row(elem_classes="input-bar"):
                    inp = gr.Textbox(
                        placeholder="输入消息…",
                        scale=9, show_label=False,
                        container=False, lines=1, max_lines=4,
                    )
                    btn_send = gr.Button("发送", scale=1, variant="primary")
                with gr.Row(elem_classes="input-bar"):
                    btn_clear = gr.Button("清空对话", elem_classes="btn-clear")

            # ── 记忆 & 状态 ──
            with gr.Tab("记忆 & 状态"):
                with gr.Row():
                    with gr.Column(elem_classes="status-card"):
                        gr.HTML(f'<h3 class="card-title">记忆</h3>')
                        gr.HTML(f'<p class="card-desc">莉莉丝记住的关于主人的信息</p>')
                        mem = gr.Textbox(
                            value="暂无记忆", lines=16, max_lines=22,
                            interactive=False, show_label=False,
                            elem_classes="memory-box",
                        )
                    with gr.Column(elem_classes="status-card"):
                        gr.HTML(f'<h3 class="card-title">好感度</h3>')
                        gr.HTML(f'<p class="card-desc">莉莉丝对主人的关系状态</p>')
                        aff = gr.Slider(
                            0, 100, value=50, interactive=False,
                            label="好感值", show_label=True,
                            elem_classes="aff-slider",
                        )
                        mood = gr.Textbox(
                            value="平静", interactive=False,
                            label="当前心情", elem_classes="mood-box",
                        )
                        gr.HTML(f'<div class="db-path">数据库: {DB_PATH}</div>')

        # 事件
        btn_send.click(
            on_send, [inp, chat, tid],
            [chat, mem, aff, mood, tid],
        ).then(lambda: "", outputs=[inp])

        inp.submit(
            on_send, [inp, chat, tid],
            [chat, mem, aff, mood, tid],
        ).then(lambda: "", outputs=[inp])

        btn_clear.click(
            on_reset, [tid],
            [chat, mem, aff, mood, tid],
        )

    return demo


# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY") == "your_deepseek_api_key_here":
        print("请先配置 .env 中的 DEEPSEEK_API_KEY")
        exit(1)
    build().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=theme,
        css=CSS,
        js=JS_FORCE_DARK,
    )
