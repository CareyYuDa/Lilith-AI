"""Lilith 莉莉丝 — 系统托盘守护程序"""

import os, sys, io, time, subprocess, threading, webbrowser
from pathlib import Path
from PIL import Image, ImageDraw

# ── 路径 ──
if getattr(sys, 'frozen', False):
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parent
VENV = ROOT / "venv"
SERVER_PORT = 8000
WEBUI_PORT = 8080
SERVER_SCRIPT = ROOT / "server.py"
OPEN_WEBUI_EXE = VENV / "Scripts" / "open-webui.exe"
PYTHON_EXE = VENV / "Scripts" / "python.exe"

_server_proc = None
_webui_proc = None

def _create_icon():
    """简单粉色圆形图标（和测试一样可靠）"""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, 60, 60], fill=(255, 120, 160, 255))
    draw.ellipse([14, 14, 50, 50], fill=(255, 180, 200, 255))
    return img

def _kill_port(port):
    try:
        r = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        for line in r.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.strip().split()
                if parts:
                    subprocess.run(["taskkill", "/f", "/pid", parts[-1]], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    except Exception:
        pass

def _cleanup_ports():
    _kill_port(SERVER_PORT)
    _kill_port(WEBUI_PORT)

def _wait_port(port, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = subprocess.run(["curl", "-s", f"http://localhost:{port}/"], capture_output=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
            if r.returncode == 0:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def _start_services():
    global _server_proc, _webui_proc
    _cleanup_ports()
    _server_proc = subprocess.Popen([str(PYTHON_EXE), str(SERVER_SCRIPT), "--port", str(SERVER_PORT)], cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    server_ok = _wait_port(SERVER_PORT, 20)
    _webui_proc = subprocess.Popen([str(OPEN_WEBUI_EXE), "serve", "--port", str(WEBUI_PORT)], cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    webui_ok = _wait_port(WEBUI_PORT, 60)
    return server_ok, webui_ok

def _stop_services():
    global _server_proc, _webui_proc
    for proc in [_server_proc, _webui_proc]:
        if proc and proc.poll() is None:
            try: proc.terminate(); proc.wait(5)
            except Exception: pass
    _server_proc = None
    _webui_proc = None
    _cleanup_ports()

def _restart_services():
    _stop_services()
    _start_services()

def run_tray():
    import pystray
    
    def on_open(icon):
        webbrowser.open(f"http://localhost:{WEBUI_PORT}/")
    
    def on_restart(icon):
        threading.Thread(target=_restart_services, daemon=True).start()
    
    def on_exit(icon):
        icon.stop()
        _stop_services()
    
    def on_status(icon):
        parts = []
        parts.append("API: " + ("Running" if _server_proc and _server_proc.poll() is None else "Stopped"))
        parts.append("WebUI: " + ("Running" if _webui_proc and _webui_proc.poll() is None else "Stopped"))
        icon.notify("\n".join(parts), "Lilith Status")
    
    icon = pystray.Icon(
        "lilith",
        _create_icon(),
        "Lilith",
        menu=pystray.Menu(
            pystray.MenuItem("Open Lilith", on_open, default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Status", on_status),
            pystray.MenuItem("Restart", on_restart),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Exit", on_exit),
        ),
    )
    
    # Background service startup
    def _startup():
        try:
            server_ok, webui_ok = _start_services()
            on_open(icon)
            msg = []
            msg.append("API: " + ("Ready" if server_ok else "Timeout"))
            msg.append("WebUI: " + ("Ready" if webui_ok else "Timeout"))
            time.sleep(0.5)
            icon.notify("\n".join(msg), "Lilith Ready")
        except Exception as e:
            time.sleep(0.5)
            try: icon.notify(f"Error: {e}", "Lilith")
            except Exception: pass
    
    threading.Thread(target=_startup, daemon=True).start()
    icon.run()

if __name__ == "__main__":
    # PyInstaller --noconsole mode: sys.stdout is None, guard against it
    if sys.stdout is not None:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if sys.stderr is not None:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    run_tray()
