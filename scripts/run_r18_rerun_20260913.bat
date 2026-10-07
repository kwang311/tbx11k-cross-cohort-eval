@echo off
set "PYTHON=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if exist "%~dp0..\venv\Scripts\python.exe" set "PYTHON=%~dp0..\venv\Scripts\python.exe"

cd /d "%~dp0.."
set TBX_GRAY=1
echo ===== R18 RERUN START %date% %time% ===== >> addon_queue_20260913.log
%PYTHON% train_cross_cohort_ext.py --backbone resnet18 --tbset full --npz 1 --prefix resnet18_full --seeds 42,43,44,45,46 --epochs 15 > ext_resnet18_full.log 2>&1
echo EXIT_RESNET18_RERUN=%ERRORLEVEL% >> addon_queue_20260913.log
%PYTHON% stats_cross_cells.py --prefix resnet50_full > cells_ci_resnet50_full.log 2>&1
echo EXIT_CELLS_R50=%ERRORLEVEL% >> addon_queue_20260913.log
%PYTHON% stats_cross_cells.py --prefix effnetb0_full > cells_ci_effnetb0_full.log 2>&1
echo EXIT_CELLS_EFF=%ERRORLEVEL% >> addon_queue_20260913.log
%PYTHON% stats_cross_cells.py --prefix resnet50_active > cells_ci_resnet50_active.log 2>&1
echo EXIT_CELLS_ACT=%ERRORLEVEL% >> addon_queue_20260913.log
%PYTHON% stats_cross_cells.py --prefix resnet50_latent > cells_ci_resnet50_latent.log 2>&1
echo EXIT_CELLS_LAT=%ERRORLEVEL% >> addon_queue_20260913.log
echo ===== R18 RERUN DONE %date% %time% ===== >> addon_queue_20260913.log
