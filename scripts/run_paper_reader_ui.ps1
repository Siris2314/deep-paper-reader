$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Project environment not found. Run '.\scripts\setup.ps1' first."
}

$Python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
if (-not (Test-Path ".\agent_memory\hardware_profile.json")) {
    & $Python -m paper_agent.cli hardware --apply
    if ($LASTEXITCODE -ne 0) {
        throw "Hardware setup failed. Run 'paper-agent hardware' for diagnostics."
    }
}

& $Python ui/paper_reader_server.py
