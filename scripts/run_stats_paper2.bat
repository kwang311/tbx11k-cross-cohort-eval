@echo off
rem Paper 2 statistics artifact (B1/B2/S4) - all CIs, restricted-AUC score, decision rule
set TBX_GRAY=1
cd /d G:\Xray\tbx11k_proj
G:\GitHub\venv_pytorch\Scripts\python.exe stats_paper2.py > stats_paper2_20260913.log 2>&1
echo EXIT=%ERRORLEVEL% >> stats_paper2_20260913.log
