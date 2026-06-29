# -*- coding: utf-8 -*-
"""莉莉丝自演化引擎 —— AI 自我修改代码的核心

这个模块是演化系统的基础层，**不可被 AI 自己修改**。
提供：文件白名单管理、安全校验、Git 版本控制、演化日志。
"""

import os
import sys
import re
import json
import subprocess
import shutil
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EVOLUTION_LOG_PATH = os.path.join(_PROJECT_ROOT, "evolutions", "evolution_log.jsonl")
_EVOLUTION_DIR = os.path.join(_PROJECT_ROOT, "evolutions")


# ============ 可演化文件白名单 ============

@dataclass
class EvolvableFile:
    """一个可被 AI 修改的文件"""
    path: str
    description: str
    evolvable_sections: List[str] = field(default_factory=list)
    max_change_ratio: float = 0.15
    priority: int = 5

EVOLVABLE_FILES: Dict[str, EvolvableFile] = {
    "personality.py": EvolvableFile(
        path="lilith_bot/personality.py",
        description="结构化人格参数定义。四象限驱动力、对话行为、情感反应等可调参数。",
        evolvable_sections=[
            "DOMINANCE_DRIVE", "AUTONOMY_DRIVE", "BELONGING_DRIVE", "ACHIEVEMENT_DRIVE",
            "INITIATIVE_CHANCE", "INTERRUPTION_TENDENCY", "AGREEMENT_BIAS",
            "DEBATE_TENDENCY", "SELF_DISCLOSURE",
            "EMOTIONAL_SENSITIVITY", "FORGIVENESS_RATE", "JEALOUSY_TENDENCY",
        ],
        max_change_ratio=0.15,
        priority=10,
    ),
    "persona.py": EvolvableFile(
        path="lilith_bot/persona.py",
        description="核心人格定义和系统提示词。",
        evolvable_sections=["LILITH_SYSTEM_PROMPT", "LOCAL_TOOL_HINT", "REFLECTION_PROMPT"],
        max_change_ratio=0.10,
        priority=9,
    ),
    "affection_events.py": EvolvableFile(
        path="lilith_bot/affection_events.py",
        description="情绪事件识别器。正则匹配用户情感信号。",
        evolvable_sections=["EVENT_PATTERNS"],
        max_change_ratio=0.15,
        priority=7,
    ),
    "autonomous.py": EvolvableFile(
        path="lilith_bot/autonomous.py",
        description="自主发言引擎。后台自言自语行为和话题选择。",
        evolvable_sections=["MONOLOGUE_TYPES", "_pick_topic_type"],
        max_change_ratio=0.15,
        priority=6,
    ),
    "state.py": EvolvableFile(
        path="lilith_bot/state.py",
        description="对话状态定义。",
        evolvable_sections=["LilithState", "AFFECTION_DEFAULT"],
        max_change_ratio=0.05,
        priority=4,
    ),
    "graph.py": EvolvableFile(
        path="lilith_bot/graph.py",
        description="LangGraph 对话管道。路由逻辑和节点行为。",
        evolvable_sections=["chatbot_node", "route_after_chatbot"],
        max_change_ratio=0.08,
        priority=5,
    ),
}

IMMUTABLE_FILES = [
    "lilith_bot/evolution_engine.py",
    "server.py",
    "lilith_bot/memory_store.py",
]


# ============ 演化记录 ============

@dataclass
class EvolutionRecord:
    id: str
    timestamp: str
    iteration: int
    file_path: str
    reason: str
    insight: str
    change_summary: str
    diff_preview: str
    safety_passed: bool
    git_hash: Optional[str] = None
    reverted: bool = False
    meta: Dict = field(default_factory=dict)


# ============ Git 辅助 ============

def _run_git(args: List[str], cwd: str = None) -> Tuple[int, str, str]:
    try:
        result = subprocess.run(
            ["git"] + args, cwd=cwd or _PROJECT_ROOT,
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return -1, "", str(e)

def _is_git_repo() -> bool:
    rc, _, _ = _run_git(["rev-parse", "--git-dir"])
    return rc == 0


# ============ 演化引擎主类 ============

class EvolutionEngine:
    """自演化引擎"""

    def __init__(self):
        self._project_root = _PROJECT_ROOT
        self._evolution_log_path = _EVOLUTION_LOG_PATH
        self._ensure_dirs()
        self._git_available = _is_git_repo()
        self._iteration = self._load_iteration()

    def _ensure_dirs(self):
        os.makedirs(_EVOLUTION_DIR, exist_ok=True)

    def is_evolvable(self, relative_path: str) -> Tuple[bool, Optional[EvolvableFile]]:
        if relative_path in IMMUTABLE_FILES:
            return False, None
        fname = os.path.basename(relative_path)
        for name, ef in EVOLVABLE_FILES.items():
            if name == fname or ef.path == relative_path:
                return True, ef
        return False, None

    def list_evolvable_files(self) -> List[Dict]:
        result = []
        for ef in EVOLVABLE_FILES.values():
            fp = os.path.join(self._project_root, ef.path)
            ex = os.path.exists(fp)
            result.append({
                "name": os.path.basename(ef.path), "path": ef.path,
                "description": ef.description,
                "evolvable_sections": ef.evolvable_sections,
                "exists": ex,
                "size_kb": round(os.path.getsize(fp) / 1024, 1) if ex else 0,
                "priority": ef.priority,
            })
        result.sort(key=lambda x: x["priority"], reverse=True)
        return result

    # ---- Self-Read ----

    def read_self_code(self, file_name: str) -> str:
        ok, ef = self.is_evolvable(file_name)
        if not ok:
            names = ", ".join(EVOLVABLE_FILES.keys())
            return "[安全拦截] '" + file_name + "' 不在白名单。可修改: " + names
        fp = os.path.join(self._project_root, ef.path)
        if not os.path.exists(fp):
            return "[错误] 文件不存在: " + ef.path
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()

    # ---- Apply Evolution ----

    def apply_evolution(self, file_name: str, patch_content: str,
                        reason: str, insight: str,
                        dry_run: bool = False) -> Dict:
        ok, ef = self.is_evolvable(file_name)
        if not ok:
            return {"success": False, "error": "不在白名单中"}

        fp = os.path.join(self._project_root, ef.path)
        if not os.path.exists(fp):
            return {"success": False, "error": "文件不存在"}

        with open(fp, "r", encoding="utf-8") as f:
            original = f.read()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bk = os.path.join(_EVOLUTION_DIR, "backup_" + ts + "_" + os.path.basename(file_name))
        shutil.copy2(fp, bk)

        new_code = self._parse_patch(original, patch_content)
        if new_code is None:
            return {"success": False, "error": "补丁解析失败: SEARCH 块未匹配"}

        ratio = self._change_ratio(original, new_code)
        if ratio > ef.max_change_ratio:
            return {"success": False,
                    "error": "变更比例 {:.1%} 超过上限 {:.0%}".format(ratio, ef.max_change_ratio)}

        if dry_run:
            return {"success": True, "dry_run": True,
                    "change_ratio": round(ratio, 3),
                    "diff_preview": self._gen_diff(original, new_code)[:800]}

        with open(fp, "w", encoding="utf-8") as f:
            f.write(new_code)

        safety = self._safety_check(ef.path)
        if not safety["passed"]:
            with open(fp, "w", encoding="utf-8") as f:
                f.write(original)
            return {"success": False, "error": "安全检查失败: " + safety["error"], "rolled_back": True}

        git_hash = None
        if self._git_available:
            git_hash = self._git_commit(file_name, reason)

        self._iteration += 1
        record = EvolutionRecord(
            id="evo-" + str(self._iteration).zfill(4),
            timestamp=datetime.now().isoformat(),
            iteration=self._iteration,
            file_path=ef.path,
            reason=reason,
            insight=insight,
            change_summary=reason[:200],
            diff_preview=self._gen_diff(original, new_code)[:500],
            safety_passed=True,
            git_hash=git_hash,
        )
        self._append_log(record)

        return {
            "success": True, "iteration": self._iteration,
            "file": ef.path, "change_ratio": round(ratio, 3),
            "git_hash": git_hash, "backup": bk,
            "diff_preview": self._gen_diff(original, new_code)[:600],
        }

    def rollback(self, iteration: int = None) -> Dict:
        if not self._git_available:
            return {"success": False, "error": "Git 不可用"}
        target = "HEAD~1"
        if iteration is not None and iteration < self._iteration:
            target = "HEAD~" + str(self._iteration - iteration)
        rc, _, stderr = _run_git(["checkout", target, "--", "."])
        if rc != 0:
            return {"success": False, "error": stderr}
        self._iteration = iteration if iteration is not None else max(0, self._iteration - 1)
        return {"success": True, "iteration": self._iteration}

    def get_evolution_log(self, limit: int = 20) -> List[Dict]:
        records = []
        if not os.path.exists(self._evolution_log_path):
            return records
        with open(self._evolution_log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-limit:]:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
        return list(reversed(records))

    def get_stats(self) -> Dict:
        log = self.get_evolution_log(limit=200)
        fm = {}
        for r in log:
            fp = r.get("file_path", "unknown")
            fm[fp] = fm.get(fp, 0) + 1
        return {
            "total_iterations": self._iteration,
            "git_available": self._git_available,
            "files_modified": fm,
            "recent_activity": log[:5] if log else [],
        }

    # ---- Internals ----

    def _load_iteration(self) -> int:
        if os.path.exists(self._evolution_log_path):
            with open(self._evolution_log_path, "r", encoding="utf-8") as f:
                return len(f.readlines())
        return 0

    def _append_log(self, record: EvolutionRecord):
        os.makedirs(os.path.dirname(self._evolution_log_path), exist_ok=True)
        with open(self._evolution_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": record.id, "timestamp": record.timestamp,
                "iteration": record.iteration, "file_path": record.file_path,
                "reason": record.reason, "insight": record.insight,
                "change_summary": record.change_summary,
                "safety_passed": record.safety_passed,
                "git_hash": record.git_hash, "reverted": record.reverted,
            }, ensure_ascii=False) + "\n")

    def _parse_patch(self, original: str, patch: str) -> Optional[str]:
        m = re.search(
            r"@@\s*SEARCH\s*@@\s*\n(.+?)\n@@\s*REPLACE\s*@@\s*\n(.+?)(?:\n@@\s*END\s*@@)?$",
            patch, re.DOTALL
        )
        if m:
            sb = m.group(1).strip()
            rb = m.group(2).strip()
            if sb in original:
                return original.replace(sb, rb, 1)
            # fuzzy: try first meaningful line
            first = sb.split("\n")[0].strip()
            if first and first in original:
                idx = original.index(first)
                return original[:idx] + rb + original[idx + len(sb):]
            return None

        om = re.search(r"<<<OLD>>>\s*\n(.*?)<<<OLD>>>", patch, re.DOTALL)
        nm = re.search(r"<<<NEW>>>\s*\n(.*?)<<<NEW>>>", patch, re.DOTALL)
        if om and nm:
            old = om.group(1).strip()
            new = nm.group(1).strip()
            if old in original:
                return original.replace(old, new, 1)
            return None
        return None

    def _change_ratio(self, original: str, new: str) -> float:
        if not original:
            return 1.0
        import difflib
        diff = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            new.splitlines(keepends=True),
        ))
        changed = sum(1 for l in diff if l.startswith(("+", "-")))
        return changed / max(len(original.splitlines()), 1)

    def _gen_diff(self, original: str, new: str) -> str:
        import difflib
        diff = list(difflib.unified_diff(
            original.splitlines(), new.splitlines(),
            fromfile="before", tofile="after", lineterm="",
        ))
        return "\n".join(diff[:80])

    def _safety_check(self, relative_path: str) -> Dict:
        fp = os.path.join(self._project_root, relative_path)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                code = f.read()
            compile(code, fp, "exec")
        except SyntaxError as e:
            return {"passed": False, "error": "语法错误 L" + str(e.lineno) + ": " + str(e.msg)}
        try:
            ns = {}
            exec(compile(code, fp, "exec"), ns)
        except Exception as e:
            return {"passed": False, "error": type(e).__name__ + ": " + str(e)[:200]}
        return {"passed": True, "error": None}

    def _git_commit(self, file_name: str, reason: str) -> Optional[str]:
        fp = os.path.join(self._project_root, "lilith_bot", file_name)
        _run_git(["add", fp])
        rc, _, _ = _run_git(["commit", "-m",
            "[Evo#" + str(self._iteration + 1) + "] " + file_name + ": " + reason[:80]])
        if rc == 0:
            hrc, h, _ = _run_git(["rev-parse", "HEAD"])
            return h[:8] if hrc == 0 else None
        return None


# ============ 全局单例 ============

_evolution_engine: Optional[EvolutionEngine] = None

def get_evolution_engine() -> EvolutionEngine:
    global _evolution_engine
    if _evolution_engine is None:
        _evolution_engine = EvolutionEngine()
    return _evolution_engine
