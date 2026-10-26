@echo off
set "PYTHON=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if exist "%~dp0..\venv\Scripts\python.exe" set "PYTHON=%~dp0..\venv\Scripts\python.exe"


cd /d "%~dp0.."

%PYTHON% diag_resnet18_gpu.py > diag_resnet18_gpu.log 2>&1

echo EXIT=%ERRORLEVEL% >> diag_resnet18_gpu.log

