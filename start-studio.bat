@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run install-windows.bat first.
  pause
  exit /b 1
)
start "Broadcast Studio" .venv\Scripts\pythonw.exe studio.py
if errorlevel 1 .venv\Scripts\python.exe studio.py
