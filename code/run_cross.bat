@echo off
set "PYTHON=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if exist "%~dp0..\venv\Scripts\python.exe" set "PYTHON=%~dp0..\venv\Scripts\python.exe"

rem Cross-cohort TB screening (angle 1): train each cohort -> test all cohorts
cd /d "%~dp0.."
echo ===== CROSS RUN START %date% %time% ===== >> train_cross.log
%PYTHON% train_cross_cohort.py --seeds 42,43,44,45,46 --epochs 15 >> train_cross.log 2>&1
echo ===== ALL DONE %date% %time% ===== >> train_cross.log
