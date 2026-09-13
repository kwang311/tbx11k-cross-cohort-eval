@echo off
cd /d G:\Xray\tbx11k_proj
G:\GitHub\venv_pytorch\Scripts\python.exe stats_cross_cells.py --prefix resnet50_full > cells_ci_resnet50_full.log 2>&1
echo EXIT=%ERRORLEVEL% >> cells_ci_resnet50_full.log
