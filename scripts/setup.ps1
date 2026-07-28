param(
    [switch]$Dev,
    [switch]$Observability,
    [switch]$SkipHardware
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
& $Python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 'Python 3.11 or newer is required.')"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating .venv..."
    & $Python -m venv .venv
}

$VenvPython = (Resolve-Path ".\.venv\Scripts\python.exe").Path
& $VenvPython -m pip install --upgrade pip
$Extras = @()
if ($Dev) { $Extras += "dev" }
if ($Observability) { $Extras += "observability" }
$InstallTarget = if ($Extras.Count -gt 0) { ".[$($Extras -join ',')]" } else { "." }
& $VenvPython -m pip install -e $InstallTarget

if (-not (Test-Path ".\.env")) {
    Copy-Item -LiteralPath ".\.env.example" -Destination ".\.env"
    Write-Host "Created .env from .env.example."
}

if (-not $SkipHardware) {
    & $VenvPython -m paper_agent.cli hardware --apply
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "1. Add optional API keys to .env."
Write-Host "2. Install the models reported by the hardware check with 'ollama pull <model>'."
Write-Host "3. Run .\scripts\run_paper_reader_ui.ps1"
