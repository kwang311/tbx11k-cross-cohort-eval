@echo off
cd /d G:\Xray\tbx11k_proj
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
echo ===== AUDIT TBX11K START %date% %time% ===== > audit_tbx11k_20260913_run.log
G:\GitHub\venv_pytorch\Scripts\python.exe audit_tbx11k.py > output\audit_tbx11k_20260913.txt 2>&1
echo EXIT_AUDIT=%ERRORLEVEL% >> audit_tbx11k_20260913_run.log
type output\audit_tbx11k_20260913.txt >> audit_tbx11k_20260913_run.log
echo ===== AUDIT DONE %date% %time% ===== >> audit_tbx11k_20260913_run.log
