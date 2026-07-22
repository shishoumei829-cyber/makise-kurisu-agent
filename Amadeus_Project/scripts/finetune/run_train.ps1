# 预下载基座 + LoRA 训练（缓存与输出在 E 盘）
$Log = "E:\amadeus_finetune\train.log"
$env:HF_HOME = "E:\amadeus_finetune\hf-cache"
$env:HF_HUB_CACHE = "E:\amadeus_finetune\hf-cache\hub"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_XET_HIGH_PERFORMANCE = "1"
$env:HF_ENDPOINT = "https://hf-mirror.com"
Remove-Item Env:HF_HUB_ENABLE_HF_TRANSFER -ErrorAction SilentlyContinue
$Py = "E:\amadeus_finetune\.venv-finetune\Scripts\python.exe"
Set-Location "D:\Amadeus_Trae\Amadeus_Project"

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Add-Content -Path $Log -Value $line -Encoding UTF8
    Write-Host $msg
}

function Run-Py([string]$script) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Py
    $psi.Arguments = "-u `"$script`""
    $psi.WorkingDirectory = (Get-Location).Path
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    $proc = [System.Diagnostics.Process]::Start($psi)
    while (-not $proc.StandardOutput.EndOfStream) { Log $proc.StandardOutput.ReadLine() }
    while (-not $proc.StandardError.EndOfStream) { Log $proc.StandardError.ReadLine() }
    $proc.WaitForExit()
    return $proc.ExitCode
}

Log "[start] run_train.ps1"
Log "[1/3] download small files via HF mirror ..."
$code = Run-Py "scripts/finetune/download_hf_model.py"
if ($code -ne 0) { Log "[warn] HF small-file download exit=$code, continue if weights exist" }

Log "[2/3] download weight shards via curl (resume) ..."
$curlBat = Join-Path (Get-Location) "scripts\finetune\download_weights_curl.bat"
$curlProc = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", "`"$curlBat`"") -NoNewWindow -PassThru -Wait
if ($curlProc.ExitCode -ne 0) { Log "[error] curl download failed exit=$($curlProc.ExitCode)"; exit $curlProc.ExitCode }

Log "[3/3] LoRA train ..."
$code = Run-Py "scripts/finetune/train_lora.py"
Log "[end] exit=$code"
exit $code
