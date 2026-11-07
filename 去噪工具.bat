@echo off
rem Opens the denoise web app. Set PYW to your own pythonw.exe if it is not on PATH.
set "PYW=pythonw"
if exist "H:\ComfyUI-latest\.venv\Scripts\pythonw.exe" set "PYW=H:\ComfyUI-latest\.venv\Scripts\pythonw.exe"
start "" "%PYW%" "%~dp0denoise_app.py"
