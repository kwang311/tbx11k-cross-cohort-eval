@echo off
cd /d G:\Xray\tbx11k_proj
G:\GitHub\venv_pytorch\Scripts\python.exe diag_resnet18_gpu.py > diag_resnet18_gpu.log 2>&1
echo EXIT=%ERRORLEVEL% >> diag_resnet18_gpu.log
