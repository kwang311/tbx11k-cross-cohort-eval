@echo off
set TBX_GRAY=1
cd /d G:\Xray\tbx11k_proj
G:\GitHub\venv_pytorch\Scripts\python.exe lowlevel_same_split.py > lowlevel_same_split_20260911.log 2>&1
echo EXIT=%ERRORLEVEL% >> lowlevel_same_split_20260911.log
echo DONE lowlevel_same_split 20260911 >> lowlevel_same_split_20260911.log
