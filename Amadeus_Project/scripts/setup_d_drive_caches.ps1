# 把「可搬迁」的用户数据默认放到 D 盘（需 PowerShell）。
# 已装在 C:\Program Files 里的软件本体不能靠本脚本搬家，只能卸载后重装到 D 或在新装时选路径。
#Requires -Version 5.1
$ErrorActionPreference = 'Stop'

$root = 'D:\AmadeusUserData'
$dirs = @{
  OLLAMA_MODELS      = Join-Path $root 'ollama-models'
  NPM_CONFIG_CACHE   = Join-Path $root 'npm-cache'
  PIP_CACHE_DIR      = Join-Path $root 'pip-cache'
  HF_HOME            = Join-Path $root 'huggingface'
  TRANSFORMERS_CACHE = Join-Path $root 'huggingface\transformers'
}

foreach ($d in $dirs.Values) {
  New-Item -ItemType Directory -Force -Path $d | Out-Null
}

function Set-UserEnv([string]$Name, [string]$Value) {
  [Environment]::SetEnvironmentVariable($Name, $Value, 'User')
  Set-Item -Path "Env:$Name" -Value $Value
}

foreach ($kv in $dirs.GetEnumerator()) {
  Set-UserEnv $kv.Key $kv.Value
  Write-Host "[OK] $([string]$kv.Key) -> $($kv.Value)"
}

$srcOllama = Join-Path $env:USERPROFILE '.ollama'
$dstOllama = $dirs.OLLAMA_MODELS
$hasModels = Test-Path (Join-Path $srcOllama 'models')
$dstEmpty = -not (Test-Path (Join-Path $dstOllama 'models'))

if ($hasModels -and $dstEmpty) {
  Write-Host ''
  Write-Host '检测到 C 盘 Ollama 模型目录，是否复制到 D 盘？复制前请先退出 Ollama（托盘退出）。' -ForegroundColor Yellow
  $ans = Read-Host '输入 Y 复制并保留 C 盘副本；M 复制后删除 C 盘 models；N 跳过'
  if ($ans -eq 'Y' -or $ans -eq 'M') {
    & robocopy $srcOllama $dstOllama /E /COPY:DAT /R:1 /W:1 /NFL /NDL /NJH /NJS | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy 失败，退出码 $LASTEXITCODE" }
    Write-Host "[OK] 已复制到 $dstOllama"
    if ($ans -eq 'M') {
      Remove-Item (Join-Path $srcOllama 'models') -Recurse -Force -ErrorAction SilentlyContinue
      Write-Host '[OK] 已删除 C 盘 .ollama\models（请重启 Ollama 后用 ollama list 验证）'
    }
  }
}

Write-Host ''
Write-Host '完成。请注销或重启 Ollama 服务后生效。' -ForegroundColor Cyan
Write-Host '新软件：设置 -> 系统 -> 存储 -> 高级 -> 保存新内容的位置 -> 选 D 盘。'
