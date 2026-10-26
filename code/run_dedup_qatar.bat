@echo off
set "PYTHON=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if exist "%~dp0..\venv\Scripts\python.exe" set "PYTHON=%~dp0..\venv\Scripts\python.exe"

cd /d "%~dp0.."
set TBX_GRAY=1
set PYTHONUNBUFFERED=1
set PYTHONFAULTHANDLER=1
echo ===== DEDUP QATAR START %date% %time% ===== > dedup_qatar.log
%PYTHON% dedup_qatar_sensitivity.py --epochs 15 --seeds 42,43,44,45,46 >> dedup_qatar.log 2>&1
echo EXIT_DEDUP_QATAR=%ERRORLEVEL% >> dedup_qatar.log
echo ===== DEDUP QATAR DONE %date% %time% ===== >> dedup_qatar.log
