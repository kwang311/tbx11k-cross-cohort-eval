@echo off
set "PYTHON=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if exist "%~dp0..\venv\Scripts\python.exe" set "PYTHON=%~dp0..\venv\Scripts\python.exe"

cd /d "%~dp0.."
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
%PYTHON% count_manifest_tags_wsl_20260913.py > manifest_counts_run.log 2>&1
echo EXIT_MANIFEST_COUNTS=%ERRORLEVEL% >> manifest_counts_run.log
