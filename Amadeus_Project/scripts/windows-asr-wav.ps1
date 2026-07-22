param(
  [Parameter(Mandatory = $true)][string]$WavPath,
  [string]$Culture = 'zh-CN'
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech

if (-not (Test-Path -LiteralPath $WavPath)) {
  Write-Output (@{ ok = $false; error = "missing wav: $WavPath" } | ConvertTo-Json -Compress)
  exit 2
}

$cultureInfo = [System.Globalization.CultureInfo]::GetCultureInfo($Culture)
$engine = $null
try {
  $engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine $cultureInfo
  $engine.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
  $engine.SetInputToWaveFile((Resolve-Path -LiteralPath $WavPath).Path)
  $result = $engine.Recognize()
  $text = if ($result -and $result.Text) { [string]$result.Text.Trim() } else { '' }
  Write-Output (@{ ok = $true; text = $text; engine = 'windows-speech' } | ConvertTo-Json -Compress)
  exit 0
} catch {
  Write-Output (@{ ok = $false; error = $_.Exception.Message; engine = 'windows-speech' } | ConvertTo-Json -Compress)
  exit 4
} finally {
  if ($engine) { $engine.Dispose() }
}
