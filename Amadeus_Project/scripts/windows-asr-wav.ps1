param(
  [Parameter(Mandatory = $true)][string]$WavPath,
  [string]$Culture = 'zh-CN'
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech

# 强制 UTF-8，避免中文 Windows 默认 GBK 导致 Node 侧乱码
$utf8 = New-Object System.Text.UTF8Encoding $false
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Write-AsrJson([hashtable]$obj) {
  # text 用 UTF-8 base64，JSON 本体保持 ASCII，彻底避开控制台编码
  if ($obj.ContainsKey('text') -and $null -ne $obj['text']) {
    $bytes = $utf8.GetBytes([string]$obj['text'])
    $obj['text_b64'] = [Convert]::ToBase64String($bytes)
    $obj.Remove('text')
  }
  $json = ($obj | ConvertTo-Json -Compress)
  [Console]::Out.WriteLine($json)
}

if (-not (Test-Path -LiteralPath $WavPath)) {
  Write-AsrJson @{ ok = $false; error = "missing wav: $WavPath"; engine = 'windows-speech' }
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
  Write-AsrJson @{ ok = $true; text = $text; engine = 'windows-speech' }
  exit 0
} catch {
  Write-AsrJson @{ ok = $false; error = $_.Exception.Message; engine = 'windows-speech' }
  exit 4
} finally {
  if ($engine) { $engine.Dispose() }
}
