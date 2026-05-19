# Amadeus: Ollama + NVIDIA 诊断（8GB 显存排障用）
# 用法: powershell -ExecutionPolicy Bypass -File .\scripts\diag_ollama_gpu.ps1
$ErrorActionPreference = 'Continue'
Write-Host "=== ollama ps ===" -ForegroundColor Cyan
try { ollama ps 2>&1 } catch { Write-Host $_.Exception.Message }

Write-Host "`n=== nvidia-smi (摘要) ===" -ForegroundColor Cyan
try { nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv 2>&1 } catch { Write-Host "nvidia-smi 不可用: $($_.Exception.Message)" }

Write-Host "`n=== 与 Ollama 相关的进程环境（当前 shell）===" -ForegroundColor Cyan
Write-Host "OLLAMA_NUM_GPU=$env:OLLAMA_NUM_GPU"
Write-Host "OLLAMA_FLASH_ATTENTION=$env:OLLAMA_FLASH_ATTENTION"
Write-Host "OLLAMA_KEEP_ALIVE=$env:OLLAMA_KEEP_ALIVE"

Write-Host "`n=== ollama show kurisu:latest ===" -ForegroundColor Cyan
try { ollama show kurisu:latest 2>&1 } catch { Write-Host $_.Exception.Message }

Write-Host "`n=== 建议 ===" -ForegroundColor Yellow
Write-Host "若 ollama ps 最右列显示 100% CPU：模型在 CPU 上跑，请见 docs/GPU_OLLAMA_SOVITS.md"
Write-Host "将本脚本完整输出复制给维护者即可。"
