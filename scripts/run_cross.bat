@echo off
rem Cross-cohort TB screening (angle 1): train each cohort -> test all cohorts
cd /d G:\Xray\tbx11k_proj
echo ===== CROSS RUN START %date% %time% ===== >> train_cross.log
G:\GitHub\venv_pytorch\Scripts\python.exe train_cross_cohort.py --seeds 42,43,44,45,46 --epochs 15 >> train_cross.log 2>&1
echo ===== ALL DONE %date% %time% ===== >> train_cross.log
