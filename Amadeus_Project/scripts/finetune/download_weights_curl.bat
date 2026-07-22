@echo off
setlocal
set MODEL_DIR=E:\amadeus_finetune\models\Qwen2.5-3B-Instruct
set MIRROR=https://hf-mirror.com/Qwen/Qwen2.5-3B-Instruct/resolve/main
set LOG=E:\amadeus_finetune\train.log

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

echo [%date% %time%] curl download start >> "%LOG%"

for %%F in (model-00001-of-00002.safetensors model-00002-of-00002.safetensors) do (
  echo [curl] downloading %%F ...
  echo [%date% %time%] curl %%F >> "%LOG%"
  curl.exe -L -C - --retry 50 --retry-delay 5 --connect-timeout 30 --max-time 0 -o "%MODEL_DIR%\%%F" "%MIRROR%/%%F"
  if errorlevel 1 (
    echo [curl] FAILED %%F exit=%ERRORLEVEL%
    echo [%date% %time%] curl FAILED %%F >> "%LOG%"
    exit /b 1
  )
  echo [curl] ok %%F
  echo [%date% %time%] curl ok %%F >> "%LOG%"
)

echo [%date% %time%] curl download done >> "%LOG%"
echo [curl] all weights ready
exit /b 0
