"""莉莉丝全链路追踪日志 — JSONL 格式

每条请求生成一个 trace_id，各个步骤依次写入 JSON 行到 trace 文件。
线程安全，零依赖，不影响主流程性能。
"""

import os
import json
import time
import threading
from datetime import datetime

_TRACE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "traces")
_TRACE_FILE = None
_TRACE_FILE_LOCK = threading.Lock()

# 线程本地存储 — 存放当前请求的 trace_id
_trace_local = threading.local()


def _ensure_trace_file():
    global _TRACE_FILE
    if _TRACE_FILE is None:
        with _TRACE_FILE_LOCK:
            if _TRACE_FILE is None:
                os.makedirs(_TRACE_DIR, exist_ok=True)
                date_str = datetime.now().strftime("%Y%m%d")
                _TRACE_FILE = os.path.join(_TRACE_DIR, f"trace_{date_str}.jsonl")
    return _TRACE_FILE


def set_trace_id(trace_id: str = None):
    """设置当前线程的 trace_id（省略则自动生成）"""
    if trace_id is None:
        import uuid
        trace_id = uuid.uuid4().hex[:16]
    _trace_local.trace_id = trace_id
    return trace_id


def get_trace_id() -> str:
    """获取当前线程的 trace_id"""
    return getattr(_trace_local, "trace_id", None)


def clear_trace_id():
    """清空当前线程的 trace_id"""
    if hasattr(_trace_local, "trace_id"):
        del _trace_local.trace_id


def write_log(step: str, data: dict):
    """写入一条追踪日志

    Args:
        step: 步骤名，如 'api_entry', 'llm_input', 'tool_call'
        data: 该步骤的结构化数据
    """
    trace_id = get_trace_id()
    if not trace_id:
        return  # 没有 trace_id 就不写

    record = {
        "trace_id": trace_id,
        "step": step,
        "ts": datetime.now().isoformat(timespec="milliseconds"),
        "data": _sanitize(data),
    }

    try:
        path = _ensure_trace_file()
        line = json.dumps(record, ensure_ascii=False)
        with _TRACE_FILE_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        pass  # 日志写入失败不影响主流程


def _sanitize(obj):
    """递归地截断过长字符串，确保可 JSON 序列化"""
    if isinstance(obj, str):
        return obj[:2000] if len(obj) > 2000 else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(i) for i in obj]
    # 处理非标准类型（如 LangChain 的 Message 对象）
    if hasattr(obj, "__dict__"):
        return _sanitize(obj.__dict__)
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)[:500]
