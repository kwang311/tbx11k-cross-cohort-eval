@echo off
rem Cross-cohort TB screening, GRAYSCALE fix (TBX_GRAY=1) to remove chroma shortcut
set TBX_GRAY=1
cd /d G:\Xray\tbx11k_proj
echo ===== CROSS-GRAY RUN START %date% %time% ===== >> train_cross_gray.log
G:\GitHub\venv_pytorch\Scripts\python.exe train_cross_cohort.py --seeds 42,43,44,45,46 --epochs 15 >> train_cross_gray.log 2>&1
echo ===== ALL DONE %date% %time% ===== >> train_cross_gray.log
