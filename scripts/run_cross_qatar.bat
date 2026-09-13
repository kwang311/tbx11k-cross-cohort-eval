@echo off
rem Qatar-inclusive cross-cohort matrix (Paper 2 case study) — grayscale, same seeds/epochs as the main matrix.
rem Outputs go to output_cross\*_qatar.json so the primary 3-cohort matrix is never overwritten.
set TBX_GRAY=1
cd /d G:\Xray\tbx11k_proj
echo ===== CROSS-GRAY-QATAR RUN START %date% %time% ===== >> train_cross_qatar.log
G:\GitHub\venv_pytorch\Scripts\python.exe train_cross_cohort_qatar.py --seeds 42,43,44,45,46 --epochs 15 >> train_cross_qatar.log 2>&1
echo EXIT=%ERRORLEVEL% >> train_cross_qatar.log
echo ===== ALL DONE %date% %time% ===== >> train_cross_qatar.log
