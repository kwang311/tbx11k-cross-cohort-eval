@echo off
cd /d G:\Xray\tbx11k_proj
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
echo ===== AUDIT RERUN START %date% %time% > audit_rerun_run.log
G:\GitHub\venv_pytorch\Scripts\python.exe audit_tbx11k.py > output\audit_tbx11k_20260913.txt 2>&1
echo EXIT_AUDIT_RERUN=%ERRORLEVEL% >> audit_rerun_run.log
type output\audit_tbx11k_20260913.txt >> audit_rerun_run.log
echo ===== DONE %date% %time% >> audit_rerun_run.log
