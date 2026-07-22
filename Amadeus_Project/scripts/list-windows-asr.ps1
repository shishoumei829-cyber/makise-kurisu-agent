Add-Type -AssemblyName System.Speech
try {
  $list = @([System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers() | ForEach-Object { $_.Culture.Name })
  if ($list.Count -eq 0) { Write-Output 'NONE' } else { Write-Output ($list -join ',') }
} catch {
  Write-Output ("FAIL:" + $_.Exception.Message)
}
