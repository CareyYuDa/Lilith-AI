"""DeepSeek reasoning_content 透传补丁

LangChain 的 ChatOpenAI 默认丢弃 DeepSeek/OpenAI o1 系列的 reasoning_content 字段。
此模块 monkey-patch _convert_delta_to_message_chunk，
使 reasoning_content 通过 AIMessageChunk.additional_kwargs["reasoning_content"] 传递。

只需 import 本模块即可生效：
    from lilith_bot.reasoning_patch import *  # noqa: F401,F403
    # 或
    import lilith_bot.reasoning_patch  # noqa: F401
"""

from langchain_openai.chat_models import base as _oa_base
from langchain_core.messages import AIMessageChunk

# 保存原始函数
_original = _oa_base._convert_delta_to_message_chunk


def _patched_convert(_dict, default_class):
    """在原始转换基础上，透传 reasoning_content 到 additional_kwargs"""
    chunk = _original(_dict, default_class)
    rc = _dict.get("reasoning_content", "")
    if rc and isinstance(chunk, AIMessageChunk):
        if not chunk.additional_kwargs:
            chunk.additional_kwargs = {}
        chunk.additional_kwargs["reasoning_content"] = rc
    return chunk


# 应用补丁
_oa_base._convert_delta_to_message_chunk = _patched_convert