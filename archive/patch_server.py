# -*- coding: utf-8 -*-
"""Patch server.py to add tool call notifications in streaming."""

path = r"D:\Lilith\Lilith\server.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add tool display name mapping after the _AUXILIARY_PREFIXES section
tool_map_insertion = '''

# 工具名称 -> 用户友好显示名
TOOL_DISPLAY_NAMES = {
    "ask_api_assistant": "API\u52a9\u624b",
    "run_python": "Python\u4ee3\u7801",
    "run_cmd": "CMD\u547d\u4ee4",
    "screenshot": "\u622a\u5c4f",
    "mouse_click": "\u9f20\u6807\u70b9\u51fb",
    "mouse_move": "\u9f20\u6807\u79fb\u52a8",
    "mouse_scroll": "\u9f20\u6807\u6eda\u8f6e",
    "type_text": "\u952e\u76d8\u8f93\u5165",
    "press_key": "\u6309\u952e",
    "get_clipboard": "\u526a\u8d34\u677f\u8bfb\u53d6",
    "set_clipboard": "\u526a\u8d34\u677f\u5199\u5165",
    "list_files": "\u6587\u4ef6\u5217\u8868",
    "read_file": "\u8bfb\u53d6\u6587\u4ef6",
    "write_file": "\u5199\u5165\u6587\u4ef6",
    "system_info": "\u7cfb\u7edf\u4fe1\u606f",
    "open_path": "\u6253\u5f00\u6587\u4ef6/URL",
    "get_cursor_pos": "\u9f20\u6807\u5750\u6807",
    "window_list": "\u7a97\u53e3\u5217\u8868",
}


'''

# Find the app = FastAPI line and insert before it
old_fastapi = 'app = FastAPI(title="Lilith API", version="0.4.0")'
if old_fastapi in content:
    content = content.replace(old_fastapi, tool_map_insertion + old_fastapi, 1)
    print("Step 1: Added TOOL_DISPLAY_NAMES mapping OK")
else:
    print("Step 1: FAILED")

# 2. Modify _stream_graph_response to output tool call notifications
old_tool_skip = (
    '            # \u8df3\u8fc7\u6709 tool_calls \u7684 chunk\uff08\u5de5\u5177\u8c03\u7528\u9636\u6bb5\uff0c\u4e0d\u8f93\u51fa\u7ed9\u7528\u6237\uff09\n'
    '            tool_calls = getattr(msg_chunk, "tool_calls", None)\n'
    '            if tool_calls:\n'
    '                continue\n'
)

new_tool_skip = (
    '            # \u5de5\u5177\u8c03\u7528\uff1a\u8f93\u51fa\u63d0\u793a\u4fe1\u606f\u7ed9\u7528\u6237\n'
    '            tool_calls = getattr(msg_chunk, "tool_calls", None)\n'
    '            if tool_calls:\n'
    '                for tc in tool_calls:\n'
    '                    if isinstance(tc, dict):\n'
    '                        tool_name = tc.get("name", "unknown")\n'
    '                    else:\n'
    '                        tool_name = getattr(tc, "name", "unknown")\n'
    '                    friendly = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)\n'
    '                    hint = f"\\n\\n\U0001f527 \u8389\u8389\u4e1d\u6b63\u5728\u8c03\u7528\u5de5\u5177: {friendly}...\\n\\n"\n'
    '                    yield f"data: {json.dumps(_chunk(\'\', hint), ensure_ascii=False)}\\n\\n"\n'
    '                continue\n'
)

if old_tool_skip in content:
    content = content.replace(old_tool_skip, new_tool_skip, 1)
    print("Step 2: Added tool call notification OK")
else:
    print("Step 2: FAILED - old tool skip block not found")
    # Debug
    for i, line in enumerate(content.split('\n')):
        if 'tool_calls' in line and 'continue' in content.split('\n')[min(i+1,len(content.split(chr(10)))-1)]:
            pass  # skip noisy debug
        if 'tool_calls' in line and 'getattr' in line:
            print(f"  Line {i+1}: {repr(line)}")

# 3. Also capture ToolMessage from tools node to show completion
old_toolmsg_skip = (
    '            chunk_type_name = type(msg_chunk).__name__\n'
    '            if "AIMessage" not in chunk_type_name:\n'
    '                continue\n'
)

new_toolmsg_skip = (
    '            # ToolMessage \u6765\u81ea tools \u8282\u70b9\uff1a\u8f93\u51fa\u5de5\u5177\u5b8c\u6210\u63d0\u793a\n'
    '            chunk_type_name = type(msg_chunk).__name__\n'
    '            if "ToolMessage" in chunk_type_name:\n'
    '                yield f"data: {json.dumps(_chunk(\'\', \'\\n\u2705 \\u5de5\\u5177\\u8c03\\u7528\\u5b8c\\u6210\\n\\n\'), ensure_ascii=False)}\\n\\n"\n'
    '                continue\n'
    '            if "AIMessage" not in chunk_type_name:\n'
    '                continue\n'
)

if old_toolmsg_skip in content:
    content = content.replace(old_toolmsg_skip, new_toolmsg_skip, 1)
    print("Step 3: Added tool completion notification OK")
else:
    print("Step 3: FAILED - chunk type check block not found")
    for i, line in enumerate(content.split('\n')):
        if 'chunk_type_name' in line and 'AIMessage' in line:
            print(f"  Line {i+1}: {repr(line)}")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("\nDone - all changes written to server.py")