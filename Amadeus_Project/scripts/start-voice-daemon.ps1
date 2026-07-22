param(
  [string]$Speaker = "owner",
  [int]$Device = -1,
  [string]$Backend = "http://127.0.0.1:3002"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $root ".venv-voice\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
  Write-Host "[voice] creating .venv-voice with Python 3.12"
  try {
    py -3.12 -m venv (Join-Path $root ".venv-voice")
  } catch {
    python -m venv (Join-Path $root ".venv-voice")
  }
  & $venvPython -m pip install -U pip
  & $venvPython -m pip install -r (Join-Path $root "voice_daemon\requirements.txt")
}

$argsList = @("voice_daemon\amadeus_voice_daemon.py", "listen", "--speaker", $Speaker, "--backend", $Backend)
if ($Device -ge 0) {
  $argsList += @("--device", "$Device")
}

Set-Location $root
& $venvPython @argsList
