@echo off
REM Builds BroadcastStudio.exe on the Windows PC. Run install-windows.bat first.
call .venv\Scripts\activate
pip install pyinstaller
pyinstaller --noconfirm --windowed --name BroadcastStudio --collect-all mediapipe studio.py
echo.
echo Build is in dist\BroadcastStudio\
echo First launch still downloads the matting weights.
echo.