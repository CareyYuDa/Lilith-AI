@echo off
chcp 65001 >nul
title Lilith 一键启动

set ROOT=D:\Lilith\Lilith
set VENV=%ROOT%\venv
set SERVER_PORT=8000
set WEBUI_PORT=8080
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo.
echo  ╔══════════════════════════════════════╗
echo  ║     🌸 Lilith 莉莉丝 一键启动 🌸      ║
echo  ╚══════════════════════════════════════╝
echo.

:: ── 关闭旧进程 ──
echo [1/4] 清理旧进程...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%SERVER_PORT%" 2^>nul') do (
    taskkill /f /pid %%a 2>nul
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%WEBUI_PORT%" 2^>nul') do (
    taskkill /f /pid %%a 2>nul
)
echo       已清理占用端口 %SERVER_PORT% 和 %WEBUI_PORT% 的进程

:: ── 启动 Lilith API server ──
echo [2/4] 启动 Lilith API Server (port %SERVER_PORT%)...
start "Lilith API" /min "%VENV%\Scripts\python.exe" "%ROOT%\server.py" --port %SERVER_PORT%

:: 等待 server 就绪
echo       等待 API Server 就绪...
set /a COUNT=0
:wait_server
timeout /t 1 /nobreak >nul
set /a COUNT+=1
curl -s http://localhost:%SERVER_PORT%/ >nul 2>&1
if errorlevel 1 (
    if %COUNT% lss 20 goto wait_server
    echo       [警告] API Server 启动超时，继续...
) else (
    echo       API Server 就绪 ✓
)

:: ── 启动 Open WebUI ──
echo [3/4] 启动 Open WebUI (port %WEBUI_PORT%)...
start "Open WebUI" /min "%VENV%\Scripts\open-webui.exe" serve --port %WEBUI_PORT%

:: 等待 WebUI 就绪
echo       等待 Open WebUI 就绪...
set /a COUNT=0
:wait_webui
timeout /t 1 /nobreak >nul
set /a COUNT+=1
curl -s http://localhost:%WEBUI_PORT%/ >nul 2>&1
if errorlevel 1 (
    if %COUNT% lss 60 goto wait_webui
    echo       [警告] Open WebUI 启动超时
) else (
    echo       Open WebUI 就绪 ✓
)

:: ── 打开浏览器 ──
echo [4/4] 打开浏览器...
start "" http://localhost:%WEBUI_PORT%/

echo.
echo ╔══════════════════════════════════════╗
echo ║   ✨ 启动完成！享受和莉莉丝的对话吧 ✨ ║
echo ║   API:  http://localhost:8000          ║
echo ║   WebUI: http://localhost:8080         ║
echo ╚══════════════════════════════════════╝
echo.
pause
