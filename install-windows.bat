@echo off
REM Windows PC install. Needs Python 3.10+ and an NVIDIA driver for the CUDA wheel.
py -3 -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
echo.
echo Start the editor with:  .venv\Scripts\python studio.py
echo OBS Virtual Camera must be installed once if you want Zoom or Discord to see it.
echo.