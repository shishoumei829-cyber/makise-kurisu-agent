# Amadeus 开机自启：声带(SoVITS) + 后端 + 打开界面
# 由 Startup\Amadeus.lnk → autostart-amadeus.vbs 调用；无 pause，可重复运行。

$ErrorActionPreference = 'Continue'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$logDir = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logFile = Join-Path $logDir 'autostart.log'

function Write-Log([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
  Add-Content -LiteralPath $logFile -Value $line -Encoding UTF8
  Write-Host $line
}

function Test-TcpPort([string]$HostName, [int]$Port) {
  $client = [System.Net.Sockets.TcpClient]::new()
  try {
    $task = $client.ConnectAsync($HostName, $Port)
    if (-not $task.Wait(500)) { return $false }
    return $client.Connected
  } catch {
    return $false
  } finally {
    $client.Dispose()
  }
}

function Test-HttpOk([string]$Url) {
  try {
    $r = Invoke-WebRequest -Uri $Url -TimeoutSec 2 -UseBasicParsing
    return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500)
  } catch {
    return $false
  }
}

Write-Log 'Amadeus autostart begin'

# ── 0) 等 Ollama（若已设开机自启，略等即可）──
if (-not (Test-TcpPort '127.0.0.1' 11434)) {
  Write-Log 'Waiting for Ollama on 11434...'
  $ollamaDeadline = (Get-Date).AddSeconds(45)
  while ((Get-Date) -lt $ollamaDeadline) {
    if (Test-TcpPort '127.0.0.1' 11434) { break }
    Start-Sleep -Seconds 1
  }
  if (Test-TcpPort '127.0.0.1' 11434) {
    Write-Log 'Ollama is up'
  } else {
    Write-Log 'Ollama not detected yet; backend will still start'
  }
}

# ── 1) 声带 GPT-SoVITS (9880) ──
$sovitsOk = Test-TcpPort '127.0.0.1' 9880
if ($sovitsOk) {
  Write-Log 'SoVITS already up on 9880'
} else {
  Write-Log 'Starting SoVITS (声带)...'
  try {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'start-sovits.ps1')
    if ($LASTEXITCODE -eq 0) {
      Write-Log 'SoVITS started via start-sovits.ps1'
      $sovitsOk = $true
    }
  } catch {
    Write-Log ("start-sovits.ps1 failed: {0}" -f $_.Exception.Message)
  }
}

if (-not $sovitsOk) {
  $desk = [Environment]::GetFolderPath('Desktop')
  $voiceBat = Get-ChildItem -LiteralPath $desk -Filter '*.bat' -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like '*声带*' -or $_.Length -eq 403 } |
    Select-Object -First 1
  # 更稳：按内容匹配 api_v2
  if (-not $voiceBat) {
    foreach ($f in (Get-ChildItem -LiteralPath $desk -Filter '*.bat' -ErrorAction SilentlyContinue)) {
      $raw = Get-Content -LiteralPath $f.FullName -Raw -ErrorAction SilentlyContinue
      if ($raw -and $raw -match 'api_v2\.py|GPT-SoVITS') {
        $voiceBat = $f
        break
      }
    }
  }
  if ($voiceBat) {
    Write-Log ("Fallback desktop voice bat: {0}" -f $voiceBat.FullName)
    Start-Process -FilePath 'cmd.exe' -ArgumentList @('/c', "`"$($voiceBat.FullName)`"") -WorkingDirectory $voiceBat.DirectoryName -WindowStyle Minimized
    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline) {
      if (Test-TcpPort '127.0.0.1' 9880) { $sovitsOk = $true; break }
      Start-Sleep -Seconds 1
    }
  } else {
    Write-Log 'No desktop 启动声带.bat found; TTS may be unavailable'
  }
}

# ── 2) 唯一桌面客户端（Electron 自己复用或启动后端）──
$electron = Join-Path $projectRoot 'node_modules\electron\dist\electron.exe'
if (Test-Path -LiteralPath $electron) {
  Start-Process -FilePath $electron -ArgumentList '.' -WorkingDirectory $projectRoot -WindowStyle Hidden
  Write-Log 'Opened Electron desktop client'
} else {
  Write-Log 'Electron missing; run npm install before enabling autostart'
}
Write-Log 'Amadeus autostart done'
