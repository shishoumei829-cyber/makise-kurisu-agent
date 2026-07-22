@echo off
set HF_HOME=E:\amadeus_finetune\hf-cache
set HF_HUB_CACHE=E:\amadeus_finetune\hf-cache\hub
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_XET_HIGH_PERFORMANCE=1
set HF_ENDPOINT=https://hf-mirror.com
cd /d D:\Amadeus_Trae\Amadeus_Project
echo [%date% %time%] starting
E:\amadeus_finetune\.venv-finetune\Scripts\python.exe -m pip install -q huggingface_hub hf_transfer
powershell -ExecutionPolicy Bypass -File scripts\finetune\run_train.ps1
echo [%date% %time%] finished exit=%ERRORLEVEL%
