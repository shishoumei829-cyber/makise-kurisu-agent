param(
  [int]$TimeoutSeconds = 75
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $projectRoot '.env'

if (Test-Path -LiteralPath $envFile) {
  foreach ($line in Get-Content -LiteralPath $envFile -Encoding UTF8) {
    $text = $line.Trim()
    if (-not $text -or $text.StartsWith('#')) { continue }
    $eq = $text.IndexOf('=')
    if ($eq -lt 1) { continue }
    $key = $text.Substring(0, $eq).Trim()
    $value = $text.Substring($eq + 1).Trim().Trim('"').Trim("'")
    if (-not [Environment]::GetEnvironmentVariable($key, 'Process')) {
      [Environment]::SetEnvironmentVariable($key, $value, 'Process')
    }
  }
}

$sovitsRoot = [Environment]::GetEnvironmentVariable('AMADEUS_SOVITS_ROOT', 'Process')
$sovitsUrl = [Environment]::GetEnvironmentVariable('AMADEUS_SOVITS_URL', 'Process')
if (-not $sovitsUrl) { $sovitsUrl = 'http://127.0.0.1:9880' }
$uri = [Uri]$sovitsUrl
$port = if ($uri.Port -gt 0) { $uri.Port } else { 9880 }

function Test-TcpPort {
  param([string]$HostName, [int]$Port)
  $client = [System.Net.Sockets.TcpClient]::new()
  try {
    $task = $client.ConnectAsync($HostName, $Port)
    if (-not $task.Wait(600)) { return $false }
    return $client.Connected
  } catch {
    return $false
  } finally {
    $client.Dispose()
  }
}

if (Test-TcpPort -HostName '127.0.0.1' -Port $port) {
  Write-Host "[OK] GPT-SoVITS is already listening on $port."
  exit 0
}

if (-not $sovitsRoot -or -not (Test-Path -LiteralPath $sovitsRoot)) {
  throw "AMADEUS_SOVITS_ROOT is missing or invalid: $sovitsRoot"
}

$python = Join-Path $sovitsRoot 'runtime\python.exe'
$apiScript = Join-Path $sovitsRoot 'api_v2.py'
$config = Join-Path $sovitsRoot 'GPT_SoVITS\configs\tts_infer.yaml'
foreach ($required in @($python, $apiScript, $config)) {
  if (-not (Test-Path -LiteralPath $required)) {
    throw "GPT-SoVITS file not found: $required"
  }
}

$logDir = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stdout = Join-Path $logDir 'sovits.out.log'
$stderr = Join-Path $logDir 'sovits.err.log'
$arguments = @(
  $apiScript,
  '-a', '127.0.0.1',
  '-p', [string]$port,
  '-c', $config
)

Start-Process -FilePath $python `
  -ArgumentList $arguments `
  -WorkingDirectory $sovitsRoot `
  -WindowStyle Hidden `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr | Out-Null

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
  if (Test-TcpPort -HostName '127.0.0.1' -Port $port) {
    Write-Host "[OK] GPT-SoVITS is ready on $port."
    exit 0
  }
  Start-Sleep -Milliseconds 750
}

throw "GPT-SoVITS did not become ready within $TimeoutSeconds seconds. Check $stderr"
