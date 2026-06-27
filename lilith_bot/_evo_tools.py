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

