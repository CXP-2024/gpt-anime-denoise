@echo off
rem Opens the denoise web app with the first Python that can run it: DENOISE_PYTHON (full path to a
rem python.exe), .venv next to this file, then the py launcher and python on PATH. When none can,
rem setup_env.py installs the dependencies into .venv in this window first.
rem Keep this file ASCII: cmd misreads UTF-8 batch files.
rem
rem "setup_env.py --probe" runs inside each candidate and prints PYEXE=<its path> only when the imports
rem work and, on a computer with an NVIDIA or supported AMD card, torch can really run on that card
rem (unless setup_env.py already found that only the CPU works on this card and driver). Otherwise it
rem says why it was skipped. DENOISE_PYTHON is always used as given, CPU build or not.
setlocal
set "PYEXE="
set "PROBE=auto"
call :find
if defined PYEXE goto launch

set "BASEPY="
call :base py -3
if not defined BASEPY call :base python
if not defined BASEPY goto nopython
"%BASEPY%" "%~dp0setup_env.py" || goto failed
rem Accept what setup installed even when it had to fall back to the CPU build; it says why.
set "PROBE=any"
call :find
if not defined PYEXE goto failed

:launch
rem pythonw keeps the app running without a console window; setup_env.py --launch shows a message box
rem with the error if the app fails to start
for %%P in ("%PYEXE%") do set "PYW=%%~dpPpythonw.exe"
if not exist "%PYW%" set "PYW=%PYEXE%"
start "" "%PYW%" "%~dp0setup_env.py" --launch
if errorlevel 1 goto failed
exit /b 0

:find
if defined DENOISE_PYTHON call :probe any "%DENOISE_PYTHON%"
if not defined PYEXE if exist "%~dp0.venv\Scripts\python.exe" call :probe %PROBE% "%~dp0.venv\Scripts\python.exe"
if not defined PYEXE where py >nul 2>nul && call :probe %PROBE% py -3
if not defined PYEXE where python >nul 2>nul && call :probe %PROBE% python
exit /b 0

:probe
rem %1 = auto or any, the rest = the command that starts that Python
set "MODE=%~1"
shift /1
set "CMD=%1 %2"
if "%~2"=="" set "CMD=%1"
for /f "usebackq tokens=1,* delims==" %%A in (`call %CMD% "%~dp0setup_env.py" --probe %MODE%`) do if "%%A"=="PYEXE" if exist "%%B" set "PYEXE=%%B"
exit /b 0

:base
rem a Python that really runs, for creating .venv (a broken py launcher is skipped)
for /f "usebackq delims=" %%P in (`call %* -c "import sys; print(sys.executable)" 2^>nul`) do if exist "%%P" set "BASEPY=%%P"
exit /b 0

:nopython
echo Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH", then double-click this file again.
pause
exit /b 1

:failed
echo.
echo Setup did not finish, see the messages above.
pause
exit /b 1
