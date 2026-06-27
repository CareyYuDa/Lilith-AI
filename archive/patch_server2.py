# -*- coding: utf-8 -*-
"""Fix server.py streaming logic ordering."""

path = r"D:\Lilith\Lilith\server.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# The problem: ToolMessage check is AFTER the chatbot filter, so it never runs.
# Fix: restructure the filter logic

old_block = (
    '            # \u53ea\u8f93\u51fa chatbot \u8282\u70b9\u7684 AIMessageChunk\n'
    '            if node != "chatbot":\n'
    '                continue\n'
    '\n'
    '            # ToolMessage \u6765\u81ea tools \u8282\u70b9\uff1a\u8f93\u51fa\u5de5\u5177\u5b8c\u6210\u63d0\u793a\n'
    '            chunk_type_name = type(msg_chunk).__name__\n'
    '            if "ToolMessage" in chunk_type_name:\n'
    "                yield f\"data: {json.dumps(_chunk('', '\\n\\u2705 \\u5de5\\u5177\\u8c03\\u7528\\u5b8c\\u6210\\n\\n'), ensure_ascii=False)}\\n\\n\"\n"
    '                continue\n'
    '            if "AIMessage" not in chunk_type_name:\n'
    '                continue\n'
)

new_block = (
    '            # tools \u8282\u70b9\u7684 ToolMessage\uff1a\u8f93\u51fa\u5de5\u5177\u5b8c\u6210\u63d0\u793a\n'
    '            chunk_type_name = type(msg_chunk).__name__\n'
    '            if "ToolMessage" in chunk_type_name:\n'
    "                yield f\"data: {json.dumps(_chunk('', '\\n\\u2705 \\u5de5\\u5177\\u8c03\\u7528\\u5b8c\\u6210\\n\\n'), ensure_ascii=False)}\\n\\n\"\n"
    '                continue\n'
    '\n'
    '            # \u975e chatbot \u8282\u70b9\u4e14\u975e ToolMessage \u7684\u5176\u4ed6\u6d88\u606f\uff0c\u8df3\u8fc7\n'
    '            if node != "chatbot":\n'
    '                continue\n'
    '\n'
    '            # \u53ea\u5904\u7406 AIMessage\n'
    '            if "AIMessage" not in chunk_type_name:\n'
    '                continue\n'
)

if old_block in content:
    content = content.replace(old_block, new_block, 1)
    print("Fix: Restructured streaming filter logic OK")
else:
    print("Fix: FAILED - old block not found")
    for i, line in enumerate(content.split('\n')):
        if 'node != "chatbot"' in line:
            print(f"  Line {i+1}: {repr(line)}")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)