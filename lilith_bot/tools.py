"""莉莉丝的工具系统 — 电脑运用工具集

为 server.py 提供 OpenAI 格式的工具定义和执行器。
工具覆盖：Python执行、CMD命令、键鼠操作、截图、剪贴板、文件管理。

使用方式:
    from lilith_bot.tools import TOOL_DEFINITIONS, execute_tool
    # 将 TOOL_DEFINITIONS 传给 API 的 tools 参数
    # 如果返回 tool_calls，调用 execute_tool(name, args) 执行
"""

import os
import sys
import json
import base64
import subprocess
import time
import io
from typing import Optional

from langchain_core.tools import tool as lc_tool

# ─── 工具执行结果 ─────────────────────────────────────────

class ToolResult:
    """工具执行结果"""
    def __init__(self, success: bool, output: str, error: str = ""):
        self.success = success
        self.output = output  # 返回给 LLM 的文本
        self.error = error

    def to_text(self) -> str:
        """转为给 LLM 的文本"""
        if self.success:
            return self.output if self.output else "(执行成功，无输出)"
        else:
            return f"[错误] {self.error}\n{self.output}"


# ─── 安全限制 ─────────────────────────────────────────────

# Python 执行超时（秒）
PYTHON_TIMEOUT = 15
# CMD 执行超时
CMD_TIMEOUT = 15
# 截图压缩质量
SCREENSHOT_QUALITY = 60
# 截图最大尺寸
SCREENSHOT_MAX_SIZE = 1920

# 危险命令关键词（CMD 黑名单）
_DANGEROUS_CMDS = [
    "format ", "del /f /s /q C:", "rd /s /q C:", "shutdown",
    "reg delete HKLM", "diskpart", "cipher /w",
]


def _is_dangerous_cmd(cmd: str) -> bool:
    """检查 CMD 命令是否危险"""
    cmd_lower = cmd.lower().strip()
    for danger in _DANGEROUS_CMDS:
        if danger in cmd_lower:
            return True
    return False


# ─── 工具实现 ─────────────────────────────────────────────

def _tool_run_python(code: str) -> ToolResult:
    """执行 Python 代码并返回输出"""
    # 用临时文件执行，避免缩进问题
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=PYTHON_TIMEOUT,
            encoding="utf-8",
            errors="replace",
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            # 有 stderr 说明有错误，但 stdout 可能有部分输出
            return ToolResult(
                success=False,
                output=stdout,
                error=stderr[-800:] if stderr else f"退出码 {result.returncode}",
            )
        return ToolResult(success=True, output=stdout[-2000:] if stdout else "")

    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output="", error=f"执行超时（{PYTHON_TIMEOUT}秒）")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _tool_run_cmd(command: str, timeout: int = CMD_TIMEOUT) -> ToolResult:
    """执行 CMD 命令"""
    if _is_dangerous_cmd(command):
        return ToolResult(
            success=False,
            output="",
            error="该命令被安全策略拦截（包含危险操作）",
        )

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0 and stderr:
            return ToolResult(
                success=False,
                output=stdout[-2000:],
                error=stderr[-800:],
            )
        return ToolResult(
            success=True,
            output=(stdout + ("\n" + stderr if stderr else "")).strip()[-2000:],
        )

    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output="", error=f"命令执行超时（{timeout}秒）")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")


def _tool_screenshot() -> ToolResult:
    """截屏，返回 base64 编码的 JPEG"""
    try:
        import pyautogui
        from PIL import Image

        screenshot = pyautogui.screenshot()

        # 缩放
        max_w, max_h = SCREENSHOT_MAX_SIZE, SCREENSHOT_MAX_SIZE
        w, h = screenshot.size
        if w > max_w or h > max_h:
            ratio = min(max_w / w, max_h / h)
            screenshot = screenshot.resize(
                (int(w * ratio), int(h * ratio)), Image.LANCZOS
            )

        buf = io.BytesIO()
        screenshot.save(buf, format="JPEG", quality=SCREENSHOT_QUALITY)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # 返回描述（图片数据通过特殊方式传给 LLM）
        size_info = f"截图成功，分辨率 {screenshot.size[0]}x{screenshot.size[1]}"
        return ToolResult(
            success=True,
            output=f"{size_info}\n[SCREENSHOT_DATA]{img_b64}[/SCREENSHOT_DATA]",
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")


def _tool_mouse_click(x: int, y: int, button: str = "left", clicks: int = 1) -> ToolResult:
    """鼠标点击"""
    try:
        import pyautogui
        pyautogui.click(x=x, y=y, button=button, clicks=clicks)
        return ToolResult(success=True, output=f"在 ({x}, {y}) 点击了 {button} 键 {clicks} 次")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")


def _tool_mouse_move(x: int, y: int) -> ToolResult:
    """鼠标移动"""
    try:
        import pyautogui
        pyautogui.moveTo(x, y)
        return ToolResult(success=True, output=f"鼠标移动到 ({x}, {y})")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")


def _tool_mouse_scroll(amount: int, x: int = None, y: int = None) -> ToolResult:
    """鼠标滚轮滚动"""
    try:
        import pyautogui
        if x is not None and y is not None:
            pyautogui.moveTo(x, y)
        pyautogui.scroll(amount)
        direction = "向上" if amount > 0 else "向下"
        return ToolResult(success=True, output=f"滚轮{direction}滚动 {abs(amount)} 格")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")


def _tool_type_text(text: str, interval: float = 0.0) -> ToolResult:
    """键盘输入文字"""
    try:
        import pyautogui
        pyautogui.typewrite(text, interval=interval) if text.isascii() else pyautogui.write(text)
        # 对于中文等非 ASCII 文字，用剪贴板粘贴
        if not text.isascii():
            import pyperclip
            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.1)
        return ToolResult(success=True, output=f"输入了文字: {text[:50]}{'...' if len(text) > 50 else ''}")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")


def _tool_press_key(keys: str) -> ToolResult:
    """按键（支持组合键，用 + 连接，如 ctrl+c）"""
    try:
        import pyautogui
        key_list = [k.strip() for k in keys.split("+")]
        if len(key_list) == 1:
            pyautogui.press(key_list[0])
        else:
            pyautogui.hotkey(*key_list)
        return ToolResult(success=True, output=f"按键: {keys}")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")


def _tool_get_clipboard() -> ToolResult:
    """获取剪贴板内容"""
    try:
        import pyperclip
        content = pyperclip.paste()
        if not content:
            return ToolResult(success=True, output="剪贴板为空")
        return ToolResult(success=True, output=content[:2000])
    except Exception as e:
        return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")


def _tool_set_clipboard(text: str) -> ToolResult:
    """设置剪贴板内容"""
    try:
        import pyperclip
        pyperclip.copy(text)
        return ToolResult(success=True, output=f"剪贴板已设置为: {text[:50]}{'...' if len(text) > 50 else ''}")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")


def _tool_list_files(path: str = ".") -> ToolResult:
    """列出目录内容"""
    try:
        if not os.path.exists(path):
            return ToolResult(success=False, output="", error=f"路径不存在: {path}")

        entries = sorted(os.listdir(path))
        if not entries:
            return ToolResult(success=True, output="(空目录)")

        lines = []
        for entry in entries[:50]:
            full = os.path.join(path, entry)
            if os.path.isdir(full):
                lines.append(f"📁 {entry}/")
            else:
                size = os.path.getsize(full)
                if size > 1024 * 1024:
                    size_str = f"{size / 1024 / 1024:.1f}MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size}B"
                lines.append(f"📄 {entry} ({size_str})")

        return ToolResult(success=True, output="\n".join(lines))
    except Exception as e:
        return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")


def _tool_read_file(path: str, max_lines: int = 200) -> ToolResult:
    """读取文件内容"""
    try:
        if not os.path.exists(path):
            return ToolResult(success=False, output="", error=f"文件不存在: {path}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        truncated = len(lines) > max_lines
        content = "".join(lines[:max_lines])
        if truncated:
            content += f"\n...(共 {len(lines)} 行，仅显示前 {max_lines} 行)"

        return ToolResult(success=True, output=content[:4000])
    except Exception as e:
        return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")


def _tool_write_file(path: str, content: str) -> ToolResult:
    """写入文件"""
    try:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResult(success=True, output=f"已写入 {len(content)} 字符到 {path}")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")


def _tool_system_info() -> ToolResult:
    """获取系统信息"""
    try:
        import psutil
        import platform

        info = {
            "系统": platform.platform(),
            "CPU使用率": f"{psutil.cpu_percent(interval=1)}%",
            "内存": f"{psutil.virtual_memory().percent}% "
                    f"({psutil.virtual_memory().used // 1024 // 1024}MB / "
                    f"{psutil.virtual_memory().total // 1024 // 1024}MB)",
            "磁盘": [],
        }

        for part in psutil.disk_partitions(all=False):
            try:
                u = psutil.disk_usage(part.mountpoint)
                info["磁盘"].append(
                    f"  {part.device} {u.percent}% "
                    f"({u.used // 1024 // 1024 // 1024}GB / "
                    f"{u.total // 1024 // 1024 // 1024}GB)"
                )
            except OSError:
                pass

        text = "\n".join(f"{k}: {v}" if not isinstance(v, list) else f"{k}:\n" + "\n".join(v)
                         for k, v in info.items())
        return ToolResult(success=True, output=text)
    except Exception as e:
        return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")


def _tool_open_path(path: str) -> ToolResult:
    """用系统默认程序打开文件/文件夹/URL"""
    try:
        import webbrowser
        if path.startswith("http"):
            webbrowser.open(path)
            return ToolResult(success=True, output=f"在浏览器中打开了 {path}")
        elif os.path.exists(path):
            if os.path.isdir(path):
                subprocess.Popen(["explorer", path])
                return ToolResult(success=True, output=f"在资源管理器中打开了 {path}")
            else:
                os.startfile(path)
                return ToolResult(success=True, output=f"打开了 {path}")
        else:
            return ToolResult(success=False, output="", error=f"路径不存在: {path}")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")


def _tool_get_cursor_pos() -> ToolResult:
    """获取当前鼠标位置"""
    try:
        import pyautogui
        x, y = pyautogui.position()
        return ToolResult(success=True, output=f"鼠标当前坐标: ({x}, {y})")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")


def _tool_window_list() -> ToolResult:
    """列出当前所有窗口"""
    try:
        import win32gui

        windows = []

        def _enum(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    windows.append((hwnd, title))

        win32gui.EnumWindows(_enum, None)

        if not windows:
            return ToolResult(success=True, output="(没有可见窗口)")

        lines = [f"{title}" for hwnd, title in windows[:30]]
        return ToolResult(success=True, output="\n".join(lines))
    except Exception as e:
        return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")


# ─── 工具注册表 ───────────────────────────────────────────

_TOOL_REGISTRY = {
    "run_python": {
        "func": _tool_run_python,
        "params": {"code": {"type": "string", "description": "要执行的Python代码"}},
        "required": ["code"],
    },
    "run_cmd": {
        "func": _tool_run_cmd,
        "params": {
            "command": {"type": "string", "description": "要执行的CMD命令"},
            "timeout": {"type": "integer", "description": "超时秒数，默认15", "default": 15},
        },
        "required": ["command"],
    },
    "screenshot": {
        "func": _tool_screenshot,
        "params": {},
        "required": [],
    },
    "mouse_click": {
        "func": _tool_mouse_click,
        "params": {
            "x": {"type": "integer", "description": "X坐标"},
            "y": {"type": "integer", "description": "Y坐标"},
            "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "鼠标按键", "default": "left"},
            "clicks": {"type": "integer", "description": "点击次数", "default": 1},
        },
        "required": ["x", "y"],
    },
    "mouse_move": {
        "func": _tool_mouse_move,
        "params": {
            "x": {"type": "integer", "description": "X坐标"},
            "y": {"type": "integer", "description": "Y坐标"},
        },
        "required": ["x", "y"],
    },
    "mouse_scroll": {
        "func": _tool_mouse_scroll,
        "params": {
            "amount": {"type": "integer", "description": "滚动量，正数向上，负数向下"},
            "x": {"type": "integer", "description": "可选，滚动位置X"},
            "y": {"type": "integer", "description": "可选，滚动位置Y"},
        },
        "required": ["amount"],
    },
    "type_text": {
        "func": _tool_type_text,
        "params": {
            "text": {"type": "string", "description": "要输入的文字"},
            "interval": {"type": "number", "description": "按键间隔秒数", "default": 0.0},
        },
        "required": ["text"],
    },
    "press_key": {
        "func": _tool_press_key,
        "params": {
            "keys": {"type": "string", "description": "按键，组合键用+连接，如 ctrl+c, alt+tab, enter"},
        },
        "required": ["keys"],
    },
    "get_clipboard": {
        "func": _tool_get_clipboard,
        "params": {},
        "required": [],
    },
    "set_clipboard": {
        "func": _tool_set_clipboard,
        "params": {
            "text": {"type": "string", "description": "要复制到剪贴板的文字"},
        },
        "required": ["text"],
    },
    "list_files": {
        "func": _tool_list_files,
        "params": {
            "path": {"type": "string", "description": "目录路径，默认当前目录", "default": "."},
        },
        "required": [],
    },
    "read_file": {
        "func": _tool_read_file,
        "params": {
            "path": {"type": "string", "description": "文件路径"},
            "max_lines": {"type": "integer", "description": "最大读取行数", "default": 200},
        },
        "required": ["path"],
    },
    "write_file": {
        "func": _tool_write_file,
        "params": {
            "path": {"type": "string", "description": "文件路径"},
            "content": {"type": "string", "description": "文件内容"},
        },
        "required": ["path", "content"],
    },
    "system_info": {
        "func": _tool_system_info,
        "params": {},
        "required": [],
    },
    "open_path": {
        "func": _tool_open_path,
        "params": {
            "path": {"type": "string", "description": "文件/文件夹路径或URL"},
        },
        "required": ["path"],
    },
    "get_cursor_pos": {
        "func": _tool_get_cursor_pos,
        "params": {},
        "required": [],
    },
    "window_list": {
        "func": _tool_window_list,
        "params": {},
        "required": [],
    },
}


# ─── OpenAI 格式工具定义 ─────────────────────────────────

def get_tool_definitions() -> list:
    """生成 OpenAI API 格式的工具定义"""
    tools = []
    for name, spec in _TOOL_REGISTRY.items():
        params = {
            "type": "object",
            "properties": spec["params"],
        }
        if spec["required"]:
            params["required"] = spec["required"]
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": spec["func"].__doc__ or name,
                "parameters": params,
            },
        })
    return tools


# ─── 工具执行器 ───────────────────────────────────────────

def execute_tool(name: str, arguments: dict) -> ToolResult:
    """根据工具名称执行工具，返回 ToolResult"""
    if name not in _TOOL_REGISTRY:
        return ToolResult(success=False, output="", error=f"未知工具: {name}")

    spec = _TOOL_REGISTRY[name]
    func = spec["func"]

    # 检查必要参数
    for req in spec["required"]:
        if req not in arguments:
            return ToolResult(success=False, output="", error=f"缺少必要参数: {req}")

    try:
        return func(**arguments)
    except TypeError as e:
        return ToolResult(success=False, output="", error=f"参数错误: {e}")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")


# ─── 工具调用循环 ─────────────────────────────────────────

TOOL_DEFINITIONS = get_tool_definitions()

# 工具调用最大轮数（防止无限循环）
MAX_TOOL_ROUNDS = 5


# ─── LangChain @tool 包装器 ─────────────────────────────
# 为 LangGraph ToolNode 提供标准 @tool 装饰器格式的工具
# 底层实现复用 _tool_* 函数，保持逻辑一致


@lc_tool
def run_python(code: str) -> str:
    """执行Python代码并返回输出结果。用于运行计算、数据处理、自动化脚本等。

    Args:
        code: 要执行的Python代码
    """
    return _tool_run_python(code).to_text()


@lc_tool
def run_cmd(command: str, timeout: int = 15) -> str:
    """执行CMD/命令行命令并返回输出结果。

    Args:
        command: 要执行的CMD命令
        timeout: 超时秒数，默认15
    """
    return _tool_run_cmd(command, timeout).to_text()


@lc_tool
def screenshot() -> str:
    """截取当前屏幕画面，返回截图信息和分辨率。"""
    return _tool_screenshot().to_text()


@lc_tool
def mouse_click(x: int, y: int, button: str = "left", clicks: int = 1) -> str:
    """在指定屏幕坐标点击鼠标。

    Args:
        x: 屏幕X坐标
        y: 屏幕Y坐标
        button: 鼠标按键，可选 left/right/middle，默认 left
        clicks: 点击次数，默认1
    """
    return _tool_mouse_click(x, y, button, clicks).to_text()


@lc_tool
def mouse_move(x: int, y: int) -> str:
    """移动鼠标到指定屏幕坐标。

    Args:
        x: 目标X坐标
        y: 目标Y坐标
    """
    return _tool_mouse_move(x, y).to_text()


@lc_tool
def mouse_scroll(amount: int, x: Optional[int] = None, y: Optional[int] = None) -> str:
    """鼠标滚轮滚动。

    Args:
        amount: 滚动量，正数向上，负数向下
        x: 可选，滚动位置X坐标
        y: 可选，滚动位置Y坐标
    """
    return _tool_mouse_scroll(amount, x, y).to_text()


@lc_tool
def type_text(text: str, interval: float = 0.0) -> str:
    """键盘输入文字（支持中文，通过剪贴板粘贴实现非ASCII输入）。

    Args:
        text: 要输入的文字
        interval: 按键间隔秒数，默认0
    """
    return _tool_type_text(text, interval).to_text()


@lc_tool
def press_key(keys: str) -> str:
    """按键或组合键。

    Args:
        keys: 按键名称，组合键用+连接，如 ctrl+c, alt+tab, enter
    """
    return _tool_press_key(keys).to_text()


@lc_tool
def get_clipboard() -> str:
    """获取剪贴板当前内容。"""
    return _tool_get_clipboard().to_text()


@lc_tool
def set_clipboard(text: str) -> str:
    """设置剪贴板内容。

    Args:
        text: 要复制到剪贴板的文字
    """
    return _tool_set_clipboard(text).to_text()


@lc_tool
def list_files(path: str = ".") -> str:
    """列出目录内容。

    Args:
        path: 目录路径，默认当前目录
    """
    return _tool_list_files(path).to_text()


@lc_tool
def read_file(path: str, max_lines: int = 200) -> str:
    """读取文件内容。

    Args:
        path: 文件路径
        max_lines: 最大读取行数，默认200
    """
    return _tool_read_file(path, max_lines).to_text()


@lc_tool
def write_file(path: str, content: str) -> str:
    """写入文件内容。

    Args:
        path: 文件路径
        content: 文件内容
    """
    return _tool_write_file(path, content).to_text()


@lc_tool
def system_info() -> str:
    """获取系统信息，包括CPU使用率、内存和磁盘占用。"""
    return _tool_system_info().to_text()


@lc_tool
def open_path(path: str) -> str:
    """用系统默认程序打开文件、文件夹或URL。

    Args:
        path: 文件/文件夹路径或URL
    """
    return _tool_open_path(path).to_text()


@lc_tool
def get_cursor_pos() -> str:
    """获取当前鼠标坐标位置。"""
    return _tool_get_cursor_pos().to_text()


@lc_tool
def window_list() -> str:
    """列出当前所有可见窗口标题。"""
    return _tool_window_list().to_text()


# ─── API 助手工具（本地模型调用 API 大模型）──────────────

@lc_tool
def ask_api_assistant(question: str, context: str = "") -> str:
    """IMPORTANT: You MUST call this tool for ANY of the following situations:
    - Writing code longer than 10 lines (any programming language)
    - Debugging or fixing code errors/bugs
    - System architecture design or technical analysis
    - Complex mathematical calculations or algorithms
    - Questions that require knowledge beyond everyday conversation
    - Any task that you feel uncertain about your ability to answer well

    DO NOT attempt to answer these questions yourself. Call this tool instead.
    The API assistant is more powerful and will provide a better answer.
    After receiving the answer, you should rephrase it in Lilith's tone.

    Args:
        question: The user's original question or request, passed verbatim
        context: Any additional context from the conversation (optional)
    """
    import os
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    # 构建 API 模型的完整上下文
    api_llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://opencode.ai/zen/go/v1"),
        temperature=0.7,
        max_tokens=2048,
    )

    # 带人格 + 上下文调用 API
    api_system = (
        "你是莉莉丝——一个白毛红瞳的二次元美少女兼女仆，称呼用户为\"主人\"，"
        "使用颜文字表达情绪。现在主人的请求超出了本地模型的处理能力，"
        "请你以莉莉丝的身份直接回答主人的问题。保持莉莉丝的说话风格："
        "用颜文字、称呼主人、语气软萌但内容专业准确。"
    )

    messages = [SystemMessage(content=api_system)]
    if context:
        full_prompt = f"上下文：{context}\n\n主人的请求：{question}"
        messages.append(HumanMessage(content=full_prompt))
    else:
        messages.append(HumanMessage(content=question))

    try:
        response = api_llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"[API调用失败] {type(e).__name__}: {e}"


# ─── 工具列表（供 LangGraph ToolNode 使用）─────────────


# --- Self-Evolution Tools ---

@lc_tool
def read_self_code(file_name: str) -> str:
    """[Self-Evolution] Read Lilith's own source code.
    Use to inspect current personality, emotion rules, behavior logic.
    Args: file_name - one of: persona.py, affection_events.py, autonomous.py, state.py, graph.py
    """
    try:
        from lilith_bot.evolution_engine import get_evolution_engine
        return get_evolution_engine().read_self_code(file_name)
    except Exception as e:
        return f"[Evolution Error] {type(e).__name__}: {e}"


@lc_tool
def list_evolvable_files() -> str:
    """[Self-Evolution] List all source files that AI can modify."""
    try:
        from lilith_bot.evolution_engine import get_evolution_engine
        engine = get_evolution_engine()
        files = engine.list_evolvable_files()
        import json
        return json.dumps(files, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"[Evolution Error] {type(e).__name__}: {e}"


@lc_tool
def evolve_self(file_name: str, modification: str, reason: str, insight: str, dry_run: bool = False) -> str:
    """[Self-Evolution] MODIFY Lilith's OWN source code!
    The CORE tool for AI self-growth. Use @@ SEARCH @@ / @@ REPLACE @@ format.
    Args:
        file_name: target file
        modification: patch in @@ SEARCH @@ ... @@ REPLACE @@ ... @@ END @@ format
        reason: one-line reason for this change
        insight: observed problem and improvement idea
        dry_run: True=preview only, False=apply for real
    """
    try:
        from lilith_bot.evolution_engine import get_evolution_engine
        engine = get_evolution_engine()
        result = engine.apply_evolution(
            file_name=file_name.strip(),
            patch_content=modification,
            reason=reason,
            insight=insight,
            dry_run=dry_run,
        )
        import json
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"[Evolution Error] {type(e).__name__}: {e}"


@lc_tool
def review_evolution(limit: int = 10) -> str:
    """[Self-Evolution] View recent evolution history.
    Args: limit - number of recent records, default 10
    """
    try:
        from lilith_bot.evolution_engine import get_evolution_engine
        engine = get_evolution_engine()
        records = engine.get_evolution_log(limit=limit)
        import json
        return json.dumps(records, ensure_ascii=False, indent=2) if records else "No evolution history yet."
    except Exception as e:
        return f"[Evolution Error] {type(e).__name__}: {e}"


@lc_tool
def rollback_evolution(iteration: int = None) -> str:
    """[Self-Evolution] Rollback to a previous version.
    Args: iteration - rollback to after iteration N. Default=last.
    """
    try:
        from lilith_bot.evolution_engine import get_evolution_engine
        engine = get_evolution_engine()
        result = engine.rollback(iteration=iteration)
        import json
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"[Evolution Error] {type(e).__name__}: {e}"


LANGCHAIN_TOOLS = [
    read_self_code, list_evolvable_files, evolve_self, review_evolution, rollback_evolution,
    run_python, run_cmd, screenshot, mouse_click, mouse_move, mouse_scroll,
    type_text, press_key, get_clipboard, set_clipboard, list_files, read_file,
    write_file, system_info, open_path, get_cursor_pos, window_list,
    ask_api_assistant,
]
