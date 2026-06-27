@echo off
chcp 65001 >nul
title Lilith 一键关停

echo.
echo  正在关闭 Lilith 相关服务...

for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" 2^>nul') do (
    taskkill /f /pid %%a 2>nul
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8080" 2^>nul') do (
    taskkill /f /pid %%a 2>nul
)

echo  已关停！莉莉丝等你下次召唤哦 (｡•́︿•̀｡)👋
echo.
pause
