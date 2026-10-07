@echo off
set "PYTHON=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if exist "%~dp0..\venv\Scripts\python.exe" set "PYTHON=%~dp0..\venv\Scripts\python.exe"


cd /d "%~dp0.."

%PYTHON% stats_cross_cells.py --prefix resnet50_full > cells_ci_resnet50_full.log 2>&1

echo EXIT=%ERRORLEVEL% >> cells_ci_resnet50_full.log

