@echo off
rem Start the Anime Denoise GUI (a local web page) without a console window.
cd /d "%~dp0"
set PYTHONUTF8=1
set "PY=pythonw"
if exist "..\..\.venv\Scripts\pythonw.exe" set "PY=..\..\.venv\Scripts\pythonw.exe"
if exist "..\.venv\Scripts\pythonw.exe" set "PY=..\.venv\Scripts\pythonw.exe"
if exist ".venv\Scripts\pythonw.exe" set "PY=.venv\Scripts\pythonw.exe"
start "" "%PY%" -X utf8 server.py
