# 释放 npm / 临时空间（需 PowerShell）。磁盘满时 ENOSPC 可先跑本脚本再安装依赖。
$ErrorActionPreference = 'Continue'

Write-Host "=== 磁盘剩余 ===" -ForegroundColor Cyan
Get-PSDrive -PSProvider FileSystem | ForEach-Object {
  $freeGb = [math]::Round($_.Free / 1GB, 2)
  Write-Host ("{0}: {1} GB free" -f $_.Name, $freeGb)
}

Write-Host "`n=== npm cache clean ===" -ForegroundColor Cyan
npm cache clean --force

$temp = [System.IO.Path]::GetTempPath()
Write-Host "`n=== 用户 TEMP 体积（仅提示，不自动删）: $temp ===" -ForegroundColor Cyan
if (Test-Path $temp) {
  $sum = (Get-ChildItem $temp -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
  if ($null -ne $sum) {
    Write-Host ("约 {0:N2} GB" -f ($sum / 1GB))
  }
}

Write-Host "`n建议: 清空回收站；卸载不用的游戏/模型；Ollama 模型可移到大盘。" -ForegroundColor Yellow
Write-Host "然后在 Amadeus_Project 执行: npm install --omit=dev --no-audit --no-fund" -ForegroundColor Yellow
