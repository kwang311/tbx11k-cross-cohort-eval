@echo off
cd /d G:\Xray\tbx11k_proj
set TBX_GRAY=1
set PYTHONUNBUFFERED=1
set PYTHONFAULTHANDLER=1
echo ===== DEDUP QATAR START %date% %time% ===== > dedup_qatar_20260913.log
G:\GitHub\venv_pytorch\Scripts\python.exe dedup_qatar_sensitivity_20260913.py --epochs 15 --seeds 42,43,44,45,46 >> dedup_qatar_20260913.log 2>&1
echo EXIT_DEDUP_QATAR=%ERRORLEVEL% >> dedup_qatar_20260913.log
echo ===== DEDUP QATAR DONE %date% %time% ===== >> dedup_qatar_20260913.log
