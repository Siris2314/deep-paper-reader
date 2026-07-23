param(
    [ValidateSet("Show", "Apply", "Reset")]
    [string]$Action = "Show",
    [ValidateSet("f16", "q8_0", "q4_0")]
    [string]$KvCacheType = "q8_0",
    [ValidateSet("0", "1")]
    [string]$FlashAttention = "1",
    [ValidateRange(2048, 262144)]
    [int]$ContextLength = 8192,
    [ValidateRange(1, 8)]
    [int]$MaxLoadedModels = 2,
    [ValidateRange(1, 8)]
    [int]$NumParallel = 1,
    [string]$KeepAlive = "15m"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BackupPath = Join-Path $ProjectRoot "agent_memory\ollama_windows_env_backup.json"
$Profile = [ordered]@{
    OLLAMA_FLASH_ATTENTION  = $FlashAttention
    OLLAMA_KV_CACHE_TYPE    = $KvCacheType
    OLLAMA_MAX_LOADED_MODELS = [string]$MaxLoadedModels
    OLLAMA_NUM_PARALLEL     = [string]$NumParallel
    OLLAMA_CONTEXT_LENGTH   = [string]$ContextLength
    OLLAMA_KEEP_ALIVE       = $KeepAlive
}

function Show-Profile {
    $rows = foreach ($name in $Profile.Keys) {
        [pscustomobject]@{
            Variable = $name
            Recommended = $Profile[$name]
            UserValue = [Environment]::GetEnvironmentVariable($name, "User")
            CurrentProcess = [Environment]::GetEnvironmentVariable($name, "Process")
        }
    }
    $rows | Format-Table -AutoSize
    Write-Host ""
    Write-Host "Loaded Ollama models:"
    & ollama ps
}

if ($Action -eq "Apply") {
    if (-not (Test-Path -LiteralPath $BackupPath)) {
        $backup = [ordered]@{}
        foreach ($name in $Profile.Keys) {
            $backup[$name] = [Environment]::GetEnvironmentVariable($name, "User")
        }
        $backupDirectory = Split-Path -Parent $BackupPath
        New-Item -ItemType Directory -Path $backupDirectory -Force | Out-Null
        $backup | ConvertTo-Json | Set-Content -LiteralPath $BackupPath -Encoding utf8
    }
    foreach ($name in $Profile.Keys) {
        [Environment]::SetEnvironmentVariable($name, $Profile[$name], "User")
    }
    Write-Host "Applied the Ollama GPU profile to your user environment."
    Write-Host "Fully quit Ollama from the Windows tray and start it again before benchmarking."
    Write-Host "The previous values are backed up at $BackupPath"
    Write-Host ""
    Show-Profile
    exit 0
}

if ($Action -eq "Reset") {
    if (-not (Test-Path -LiteralPath $BackupPath)) {
        throw "No profile backup exists at $BackupPath"
    }
    $backup = Get-Content -Raw -LiteralPath $BackupPath | ConvertFrom-Json
    foreach ($name in $Profile.Keys) {
        $previous = $backup.$name
        [Environment]::SetEnvironmentVariable($name, $previous, "User")
    }
    Remove-Item -LiteralPath $BackupPath -Force
    Write-Host "Restored the previous Ollama user environment."
    Write-Host "Fully quit and restart Ollama for the restored values to take effect."
    exit 0
}

Show-Profile
