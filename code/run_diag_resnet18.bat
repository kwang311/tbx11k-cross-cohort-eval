@echo off
set "PYTHON=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if exist "%~dp0..\venv\Scripts\python.exe" set "PYTHON=%~dp0..\venv\Scripts\python.exe"

cd /d "%~dp0.."
set PYTHONFAULTHANDLER=1
set CUDA_LAUNCH_BLOCKING=1
echo ===== DIAG resnet18 seed42 epoch1 %date% %time% ===== > diag_resnet18_run.log
%PYTHON% train_cross_cohort_ext.py --backbone resnet18 --tbset full --npz 0 --seeds 42 --epochs 1 --prefix resnet18_diag >> diag_resnet18_run.log 2>&1
echo EXIT_DIAG=%ERRORLEVEL% >> diag_resnet18_run.log
