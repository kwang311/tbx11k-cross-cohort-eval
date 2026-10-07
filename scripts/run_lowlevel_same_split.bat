@echo off
set "PYTHON=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if exist "%~dp0..\venv\Scripts\python.exe" set "PYTHON=%~dp0..\venv\Scripts\python.exe"

set TBX_GRAY=1
cd /d "%~dp0.."
%PYTHON% lowlevel_same_split.py > lowlevel_same_split_20260911.log 2>&1
echo EXIT=%ERRORLEVEL% >> lowlevel_same_split_20260911.log
echo DONE lowlevel_same_split 20260911 >> lowlevel_same_split_20260911.log
