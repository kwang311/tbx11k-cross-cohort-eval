@echo off
cd /d G:\Xray\tbx11k_proj
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set TBX_GRAY=1
echo ===== EXPORT FEATURES START %date% %time% ===== > export_features_run.log
G:\GitHub\venv_pytorch\Scripts\python.exe export_lowlevel_features_20260913.py >> export_features_run.log 2>&1
echo EXIT_EXPORT=%ERRORLEVEL% >> export_features_run.log
echo ===== EXPORT FEATURES DONE %date% %time% ===== >> export_features_run.log
