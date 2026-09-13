@echo off
cd /d G:\Xray\tbx11k_proj
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
echo ===== FINGERPRINT FULL START %date% %time% ===== > fingerprint_full_run.log
G:\GitHub\venv_pytorch\Scripts\python.exe fingerprint_overlap_full_20260913.py >> fingerprint_full_run.log 2>&1
echo EXIT_FP=%ERRORLEVEL% >> fingerprint_full_run.log
echo ===== FINGERPRINT FULL DONE %date% %time% ===== >> fingerprint_full_run.log
