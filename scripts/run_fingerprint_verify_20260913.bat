@echo off
cd /d G:\Xray\tbx11k_proj
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
echo ===== FINGERPRINT VERIFY START %date% %time% ===== > fingerprint_verify_run.log
G:\GitHub\venv_pytorch\Scripts\python.exe fingerprint_verify_20260913.py >> fingerprint_verify_run.log 2>&1
echo EXIT_VERIFY=%ERRORLEVEL% >> fingerprint_verify_run.log
echo ===== FINGERPRINT VERIFY DONE %date% %time% ===== >> fingerprint_verify_run.log
