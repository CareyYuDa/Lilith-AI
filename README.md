# Lilith Bot

Lilith Bot is an OpenAI-compatible chat service built on FastAPI and LangGraph.
It powers **莉莉丝 (Lilith)** — a silver-haired, red-eyed anime AI girl who lives on your Windows PC.
Designed for Open WebUI, with tool calling, a PAD 3D emotion engine, vector long-term memory, and self-evolution capabilities.

## Features

- **OpenAI-compatible** `/v1/chat/completions` endpoint (streaming + non-streaming).
- **Single model**: `lilith` — DeepSeek Flash with tool calling and reasoning chain.
- **LangGraph workflow**: `recall_memory → update_affection → chatbot ↔ tools → save_memory → [reflect | evolution | END]`.
- **PAD 3D emotion engine**: Pleasure-Arousal-Dominance model with circadian rhythm, decay, and personality traits — pure math, no LLM calls.
- **Vector long-term memory**: SQLite + `sqlite-vec` + `all-MiniLM-L6-v2` (384-dim) semantic search.
- **Self-reflection**: LLM deep-reflection every N turns — adjusts personality traits, discovers relationship milestones.
- **Self-evolution**: AI observes conversations and modifies its own code (via `evolution_engine.py`) every 50 interactions.
- **Desktop tools (23 tools)**: Python/CMD execution, screenshots, mouse/keyboard control, clipboard, file management, system info.
- **Full reasoning chain** (`reasoning_content`) support via monkey-patch.
- **Web dashboard**: Real-time emotion/memory/personality editor at `/dashboard`.

## Quick Start

1. Copy `.env.example` (or use existing `.env`), fill in the required API keys and model endpoints.
2. Start the service:

```powershell
cd D:\Lilith\Lilith
venv\Scripts\python.exe server.py --port 8000
```

3. Or use the system tray app: double-click **Lilith.exe**.

4. In Open WebUI, add an OpenAI-compatible connection:

```text
Base URL: http://localhost:8000/v1
API Key: lilith-local
Models: lilith
```

## Project Structure

```
Lilith/
├── server.py                    # FastAPI OpenAI-compatible API server
├── lilith_tray.py               # Windows system tray launcher
├── Lilith.exe                   # Compiled tray launcher (PyInstaller)
├── lilith_bot/                  # Core package
│   ├── __init__.py              # Package metadata
│   ├── graph.py                 # LangGraph conversation workflow (StateGraph)
│   ├── state.py                 # TypedDict state schema (LilithState)
│   ├── persona.py               # Persona definition, system prompt builder
│   ├── tools.py                 # 23 desktop tool definitions (LangChain @tool)
│   ├── memory_store.py          # Vector long-term memory store (sqlite-vec)
│   ├── affection_engine.py      # PAD 3D emotion engine (pure math)
│   ├── affection_events.py      # Emotion event detection from user messages
│   ├── feedback_system.py       # LLM-powered conversation analysis
│   ├── personality.py           # Four-quadrant personality drive model
│   ├── evolution_engine.py      # Self-evolution safety layer & patch engine
│   ├── autonomous.py            # Autonomous speech (skeleton)
│   ├── reasoning_patch.py       # reasoning_content monkey-patch for LangChain
│   └── trace_logger.py          # JSONL full-chain trace logging
├── pyproject.toml               # Project metadata & dependencies
├── langgraph.json               # LangGraph CLI configuration
├── .env                         # Local runtime secrets (gitignored)
└── .gitignore
```

## Dashboard

Open `http://localhost:8000/dashboard` in your browser to see:

- **Emotion**: PAD values, mood label, interaction count — click to edit
- **Prompt**: Current system prompt (worldview & personality) — click to edit
- **Personality**: Four-quadrant drives (dominance/autonomy/belonging/achievement), behavior, emotion — click to edit
- **Persons**: Known persons and their intimacy/trust scores
- **Memories**: All stored long-term memories with importance
- **Activity**: Recent conversation activity log
- **System Log**: Recent console output

## Notes

- `.env` and `lilith_memory.db` contain local runtime state and should not be committed to a public repository.
- Some desktop tools require optional dependencies from `.[desktop]` (pyautogui, pywin32, etc.).
- The bundled `venv` is local runtime state; recreate it from `pyproject.toml` when moving the project.
- Evolution history is stored in `evolutions/` (gitignored).
