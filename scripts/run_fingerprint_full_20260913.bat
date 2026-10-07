@echo off
set "PYTHON=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if exist "%~dp0..\venv\Scripts\python.exe" set "PYTHON=%~dp0..\venv\Scripts\python.exe"

cd /d "%~dp0.."
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
echo ===== FINGERPRINT FULL START %date% %time% ===== > fingerprint_full_run.log
%PYTHON% fingerprint_overlap_full_20260913.py >> fingerprint_full_run.log 2>&1
echo EXIT_FP=%ERRORLEVEL% >> fingerprint_full_run.log
echo ===== FINGERPRINT FULL DONE %date% %time% ===== >> fingerprint_full_run.log
