# Lilith Bot

Lilith Bot is an OpenAI-compatible chat service built on FastAPI and LangGraph. It is designed to be used from Open WebUI, with two model modes, tool calling, and a SQLite-backed long-term memory store.

## Features

- OpenAI-compatible `/v1/chat/completions` endpoint.
- Two exposed models:
  - `lilith`: local LM Studio compatible model, configured with `LOCAL_BASE_URL`.
  - `lili`: remote API model, configured with `LILI_BASE_URL`.
- LangGraph workflow with memory recall, chatbot/tool loop, and memory saving.
- Long-term memory with SQLite, `sqlite-vec`, and `sentence-transformers`.
- Optional desktop tools for Python/CMD execution, screenshots, mouse/keyboard control, clipboard, files, system info, and window listing.
- Streaming and non-streaming responses.

## Quick Start

1. Copy `.env.example` to `.env`, then fill in the required API keys and model endpoints.
2. Start the service:

```powershell
cd D:\Lilith\Lilith
venv\Scripts\python.exe server.py --port 8000
```

3. In Open WebUI, add an OpenAI-compatible connection:

```text
Base URL: http://localhost:8000/v1
API Key: lilith-local
Models: lilith, lili
```

You can also run `start.bat` to launch both Lilith API and Open WebUI.

## Test Client

```powershell
venv\Scripts\python.exe call_lili.py "Hello" --model lilith
venv\Scripts\python.exe call_lili.py "Write a small Python script" --model lili --stream
```

## Important Files

- `server.py`: FastAPI OpenAI-compatible wrapper.
- `lilith_bot/graph.py`: LangGraph conversation workflow.
- `lilith_bot/memory_store.py`: vector memory store.
- `lilith_bot/tools.py`: tool definitions and executors.
- `lilith_bot/persona.py`: persona and memory prompts.
- `lilith_bot/pusher.py`: optional Open WebUI Channel push integration.

## Notes

- `.env` and `lilith_memory.db` contain local runtime state and should not be committed to a public repository.
- Some desktop tools require optional dependencies from `.[desktop]`.
- The bundled `venv` is local runtime state; recreate it from `pyproject.toml` when moving the project.
