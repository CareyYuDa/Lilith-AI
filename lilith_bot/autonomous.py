"""莉莉丝自主发言引擎 — 骨架"""
import os
import threading


# ============ 配置 ============

MIN_INTERVAL = int(os.getenv("AUTONOMOUS_MIN_INTERVAL", "60"))
MAX_INTERVAL = int(os.getenv("AUTONOMOUS_MAX_INTERVAL", "300"))

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LilithAutonomousBrain:
    """自主发言引擎（骨架，待重构）"""

    def __init__(self):
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        pass

    def stop(self):
        pass

    def status(self) -> dict:
        return {"running": self._running, "mode": "skeleton"}


_brain_instance = None
_brain_lock = threading.Lock()


def get_autonomous_brain() -> LilithAutonomousBrain:
    global _brain_instance
    if _brain_instance is None:
        with _brain_lock:
            if _brain_instance is None:
                _brain_instance = LilithAutonomousBrain()
    return _brain_instance
