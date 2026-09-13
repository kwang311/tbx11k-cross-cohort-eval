@echo off
rem TBX11K active vs latent binary (angle 2 deef dive): baseline, 10 seeds
set TBX_TASK=a_vs_l
cd /d G:\Xray\tbx11k_proj
echo ===== A_VS_L RUN START %date% %time% ===== >> train_a_vs_l.log
G:\GitHub\venv_pytorch\Scripts\python.exe train.py --model baseline --epochs 15 --batch-size 16 --seeds 42,43,44,45,46,47,48,49,50,51 >> train_a_vs_l.log 2>&1
echo ===== ALL DONE %date% %time% ===== >> train_a_vs_l.log
