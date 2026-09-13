@echo off
cd /d G:\Xray\tbx11k_proj
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set TBX_GRAY=1
echo ===== CROSS INDEX START %date% %time% > materials_reply_run.log
G:\GitHub\venv_pytorch\Scripts\python.exe export_cross_pred_index_wsl_20260913.py >> materials_reply_run.log 2>&1
echo EXIT_CROSS_INDEX=%ERRORLEVEL% >> materials_reply_run.log
echo ===== MD5 LISTS START %date% %time% >> materials_reply_run.log
G:\GitHub\venv_pytorch\Scripts\python.exe make_md5_lists_wsl_20260913.py >> materials_reply_run.log 2>&1
echo EXIT_MD5=%ERRORLEVEL% >> materials_reply_run.log
echo ===== DONE %date% %time% >> materials_reply_run.log
