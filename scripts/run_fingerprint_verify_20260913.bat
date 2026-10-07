@echo off
set "PYTHON=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if exist "%~dp0..\venv\Scripts\python.exe" set "PYTHON=%~dp0..\venv\Scripts\python.exe"

cd /d "%~dp0.."
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
echo ===== FINGERPRINT VERIFY START %date% %time% ===== > fingerprint_verify_run.log
%PYTHON% fingerprint_verify_20260913.py >> fingerprint_verify_run.log 2>&1
echo EXIT_VERIFY=%ERRORLEVEL% >> fingerprint_verify_run.log
echo ===== FINGERPRINT VERIFY DONE %date% %time% ===== >> fingerprint_verify_run.log
