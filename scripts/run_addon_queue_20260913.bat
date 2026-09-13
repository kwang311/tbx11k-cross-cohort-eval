@echo off
cd /d G:\Xray\tbx11k_proj
echo ===== ADDON QUEUE START %date% %time% ===== >> addon_queue_20260913.log
set TBX_GRAY=1
G:\GitHub\venv_pytorch\Scripts\python.exe lowlevel_nonlinear_control.py > lowlevel_nonlinear_control_20260913.log 2>&1
echo EXIT_NONLINEAR=%ERRORLEVEL% >> addon_queue_20260913.log
G:\GitHub\venv_pytorch\Scripts\python.exe train_cross_cohort_ext.py --backbone resnet50 --tbset full --npz 1 --prefix resnet50_full --seeds 42,43,44,45,46 --epochs 15 > ext_resnet50_full.log 2>&1
echo EXIT_RESNET50_FULL=%ERRORLEVEL% >> addon_queue_20260913.log
G:\GitHub\venv_pytorch\Scripts\python.exe train_cross_cohort_ext.py --backbone resnet18 --tbset full --npz 1 --prefix resnet18_full --seeds 42,43,44,45,46 --epochs 15 > ext_resnet18_full.log 2>&1
echo EXIT_RESNET18_FULL=%ERRORLEVEL% >> addon_queue_20260913.log
G:\GitHub\venv_pytorch\Scripts\python.exe train_cross_cohort_ext.py --backbone efficientnet_b0 --tbset full --npz 1 --prefix effnetb0_full --seeds 42,43,44,45,46 --epochs 15 > ext_effnetb0_full.log 2>&1
echo EXIT_EFFNETB0_FULL=%ERRORLEVEL% >> addon_queue_20260913.log
G:\GitHub\venv_pytorch\Scripts\python.exe train_cross_cohort_ext.py --backbone resnet50 --tbset active --npz 1 --prefix resnet50_active --seeds 42,43,44,45,46 --epochs 15 > ext_resnet50_active.log 2>&1
echo EXIT_RESNET50_ACTIVE=%ERRORLEVEL% >> addon_queue_20260913.log
G:\GitHub\venv_pytorch\Scripts\python.exe train_cross_cohort_ext.py --backbone resnet50 --tbset latent --npz 1 --prefix resnet50_latent --seeds 42,43,44,45,46 --epochs 15 > ext_resnet50_latent.log 2>&1
echo EXIT_RESNET50_LATENT=%ERRORLEVEL% >> addon_queue_20260913.log
echo ===== ADDON QUEUE DONE %date% %time% ===== >> addon_queue_20260913.log
