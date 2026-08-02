@echo off
set BASE=https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct/resolve/master
set DIR=E:\amadeus_finetune\models\Qwen2.5-7B-Instruct

start "" /b curl.exe -L -C - --retry 30 --retry-delay 5 --connect-timeout 30 --max-time 0 -o "%DIR%\model-00002-of-00004.safetensors" "%BASE%/model-00002-of-00004.safetensors" 2>"D:\Amadeus_Trae\Amadeus_Project\scripts\finetune\v6-shard2.err.log"
start "" /b curl.exe -L -C - --retry 30 --retry-delay 5 --connect-timeout 30 --max-time 0 -o "%DIR%\model-00003-of-00004.safetensors" "%BASE%/model-00003-of-00004.safetensors" 2>"D:\Amadeus_Trae\Amadeus_Project\scripts\finetune\v6-shard3.err.log"
start "" /b curl.exe -L -C - --retry 30 --retry-delay 5 --connect-timeout 30 --max-time 0 -o "%DIR%\model-00004-of-00004.safetensors" "%BASE%/model-00004-of-00004.safetensors" 2>"D:\Amadeus_Trae\Amadeus_Project\scripts\finetune\v6-shard4.err.log"
