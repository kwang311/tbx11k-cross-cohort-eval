@echo off
cd /d G:\Xray\tbx11k_proj
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
G:\GitHub\venv_pytorch\Scripts\python.exe count_manifest_tags_wsl_20260913.py > manifest_counts_run.log 2>&1
echo EXIT_MANIFEST_COUNTS=%ERRORLEVEL% >> manifest_counts_run.log
