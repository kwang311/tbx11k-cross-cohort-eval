@echo off
set "PYTHON=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if exist "%~dp0..\venv\Scripts\python.exe" set "PYTHON=%~dp0..\venv\Scripts\python.exe"

rem Paper 2 statistics artifact (B1/B2/S4) - all CIs, restricted-AUC score, decision rule
set TBX_GRAY=1
cd /d "%~dp0.."
%PYTHON% stats_paper2.py > stats_paper2_20260913.log 2>&1
echo EXIT=%ERRORLEVEL% >> stats_paper2_20260913.log
