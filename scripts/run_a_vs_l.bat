@echo off
set "PYTHON=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if exist "%~dp0..\venv\Scripts\python.exe" set "PYTHON=%~dp0..\venv\Scripts\python.exe"

rem TBX11K active vs latent binary (angle 2 deef dive): baseline, 10 seeds
set TBX_TASK=a_vs_l
cd /d "%~dp0.."
echo ===== A_VS_L RUN START %date% %time% ===== >> train_a_vs_l.log
%PYTHON% train.py --model baseline --epochs 15 --batch-size 16 --seeds 42,43,44,45,46,47,48,49,50,51 >> train_a_vs_l.log 2>&1
echo ===== ALL DONE %date% %time% ===== >> train_a_vs_l.log
