@echo off
set "PYTHON=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if exist "%~dp0..\venv\Scripts\python.exe" set "PYTHON=%~dp0..\venv\Scripts\python.exe"

rem TBX11K 4-class TB subtype (angle 2): baseline ResNet-50, 10 seeds
cd /d "%~dp0.."
echo ===== TBX11K 4class RUN START %date% %time% ===== >> train_run.log
%PYTHON% train.py --model baseline --epochs 15 --batch-size 16 --seeds 42,43,44,45,46,47,48,49,50,51 >> train_run.log 2>&1
echo ===== ALL DONE %date% %time% ===== >> train_run.log
