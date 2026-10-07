@echo off
set "PYTHON=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if exist "%~dp0..\venv\Scripts\python.exe" set "PYTHON=%~dp0..\venv\Scripts\python.exe"

cd /d "%~dp0.."
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set TBX_GRAY=1
echo ===== EXPORT FEATURES START %date% %time% ===== > export_features_run.log
%PYTHON% export_lowlevel_features_20260913.py >> export_features_run.log 2>&1
echo EXIT_EXPORT=%ERRORLEVEL% >> export_features_run.log
echo ===== EXPORT FEATURES DONE %date% %time% ===== >> export_features_run.log
