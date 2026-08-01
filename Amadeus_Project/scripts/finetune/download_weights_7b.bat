@echo off
setlocal
set MODEL_DIR=E:\amadeus_finetune\models\Qwen2.5-7B-Instruct
set BASE=https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct/resolve/master

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

for %%F in (
  model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors
  model-00004-of-00004.safetensors
) do (
  echo [download-7b] %%F
  curl.exe -L -C - --retry 30 --retry-delay 5 --connect-timeout 30 --max-time 0 -o "%MODEL_DIR%\%%F" "%BASE%/%%F"
  if errorlevel 1 exit /b 1
)

echo [download-7b] all shards ready
exit /b 0
