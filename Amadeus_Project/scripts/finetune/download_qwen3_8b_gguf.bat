@echo off
curl.exe -L -C - --retry 30 --retry-delay 5 --connect-timeout 30 --max-time 0 ^
  -o "E:\amadeus_finetune\exports\Qwen3-8B-Q4_K_M.gguf" ^
  "https://www.modelscope.cn/models/Qwen/Qwen3-8B-GGUF/resolve/master/Qwen3-8B-Q4_K_M.gguf" ^
  2>"D:\Amadeus_Trae\model_artifacts\qwen3-8b-modelscope.err.log"
