@echo off
cd /d G:\Xray\tbx11k_proj
G:\GitHub\venv_pytorch\Scripts\python.exe smoke_ext.py > smoke_ext.log 2>&1
echo EXIT=%ERRORLEVEL% >> smoke_ext.log
