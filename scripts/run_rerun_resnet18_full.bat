@echo off
cd /d G:\Xray\tbx11k_proj
set TBX_GRAY=1
set PYTHONFAULTHANDLER=1
set PYTHONUNBUFFERED=1
echo ===== RERUN resnet18_full START %date% %time% ===== > rerun_resnet18_full.log
G:\GitHub\venv_pytorch\Scripts\python.exe train_cross_cohort_ext.py --backbone resnet18 --tbset full --npz 1 --prefix resnet18_full --seeds 42,43,44,45,46 --epochs 15 >> rerun_resnet18_full.log 2>&1
echo EXIT_RESNET18_FULL=%ERRORLEVEL% >> rerun_resnet18_full.log
echo ===== RERUN resnet18_full DONE %date% %time% ===== >> rerun_resnet18_full.log
