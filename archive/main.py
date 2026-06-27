"""莉莉丝 — 命令行聊天入口

阶段二：含记忆系统。支持 SQLite 持久化、记忆检索/保存、好感度跟踪。
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

# 检查 API Key
if not os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY") == "your_deepseek_api_key_here":
    print("⚠️  请先配置 .env 文件中的 DEEPSEEK_API_KEY！")
    print("   编辑 D:\\Lilith\\Lilith\\.env 填入你的 DeepSeek API Key")
    sys.exit(1)

from langchain_core.messages import HumanMessage
from lilith_bot.graph import lilith_graph, get_checkpointer, DB_PATH
from lilith_bot.persona import LILITH_SYSTEM_PROMPT

# 线程 ID
THREAD_ID = "lilith-console"


def print_lilith(text: str):
    """美化输出莉莉丝的回复"""
    print(f"\n🌸 \033[95m莉莉丝\033[0m: {text}\n")


def print_separator():
    print("─" * 50)


def show_status(state: dict):
    """展示记忆和好感度状态"""
    memories = state.get("long_term_memories", [])
    affection = state.get("affection", {"score": 50, "mood": "平静"})

    print(f"  💾 记忆: {len(memories)}条  |  💕 好感: {affection['score']}/100 ({affection['mood']})")


def main():
    print_separator()
    print("🌸 莉莉丝 LangGraph 版 v0.2.0 — 阶段二：记忆系统")
    print(f"   📂 记忆库: {DB_PATH}")
    print("   输入消息开始聊天，输入 /quit 退出，/reset 重置对话")
    print("   /mem 查看记忆，/aff 查看好感度")
    print_separator()

    config = {"configurable": {"thread_id": THREAD_ID}}
    checkpointer = get_checkpointer()

    # 尝试恢复之前的会话状态
    try:
        previous_state = lilith_graph.get_state(config)
        if previous_state and previous_state.values:
            messages = previous_state.values.get("messages", [])
            memories = previous_state.values.get("long_term_memories", [])
            affection = previous_state.values.get("affection", {"score": 50, "mood": "平静"})
            print(f"🔄 恢复了之前的会话 ({len(messages)}条消息, {len(memories)}条记忆)")
            show_status({"long_term_memories": memories, "affection": affection})
        else:
            messages = []
            memories = []
            affection = {"score": 50, "mood": "平静"}
    except Exception:
        messages = []
        memories = []
        affection = {"score": 50, "mood": "平静"}

    while True:
        try:
            user_input = input("\n💬 \033[94m主人\033[0m: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n莉莉丝: 主人要走了吗... 下次见哦 (´；ω；`)")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("\n莉莉丝: 主人拜拜~ 莉莉丝会想你的 (｡･ω･｡)ﾉ♡")
            break

        if user_input.lower() == "/reset":
            messages = []
            memories = []
            affection = {"score": 50, "mood": "平静"}
            print("🔄 对话和记忆已全部重置！")
            continue

        if user_input.lower() == "/mem":
            if memories:
                print("\n💾 \033[93m莉莉丝的记忆:\033[0m")
                for i, m in enumerate(memories, 1):
                    print(f"  {i}. {m}")
            else:
                print("\n💾 莉莉丝还没有关于主人的记忆呢... (´·ω·`)")
            continue

        if user_input.lower() == "/aff":
            aff = affection
            print(f"\n💕 \033[95m好感度: {aff['score']}/100\033[0m — 莉莉丝现在感觉: {aff.get('mood', '平静')}")
            continue

        # 构建输入状态
        input_state = {
            "messages": messages + [HumanMessage(content=user_input)],
            "persona": LILITH_SYSTEM_PROMPT,
            "long_term_memories": memories,
            "affection": affection,
        }

        try:
            result = lilith_graph.invoke(input_state, config)

            # 更新本地状态
            messages = result["messages"]
            memories = result.get("long_term_memories", memories)
            affection = result.get("affection", affection)

            # 输出最后一条 AI 消息
            for m in reversed(messages):
                if hasattr(m, "type") and m.type == "ai":
                    print_lilith(m.content)
                    break

            # 显示状态摘要
            show_status({"long_term_memories": memories, "affection": affection})

        except Exception as e:
            print(f"\n❌ 出错了: {e}")
            # 回退
            messages = messages[:-1] if messages else []


if __name__ == "__main__":
    main()
