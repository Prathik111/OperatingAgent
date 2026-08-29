<#
Run the native agent from the OperatingAgent folder.

    .\scripts\run-agent.ps1 "read README.md and tell me what this project is"
    .\scripts\run-agent.ps1 "write a summary to notes.txt" -Dir .\scratch -Ask

Checks the key and the install first, because the two things most likely to bite
on a first run are a missing GROQ_API_KEY and a plain `uv sync` (which installs
nothing here - the workspace root is virtual, so members need --all-packages).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Message,

    # The ONLY folder the agent may read or write. Defaults to a scratch folder
    # rather than the repo root, so a first run can't touch your source.
    [string]$Dir = ".\scratch",

    [string]$Model = "llama-3.3-70b",
    [int]$MaxTurns = 10,
    [double]$Temperature = 0.0,

    # Prompt before each tool instead of auto-approving.
    [switch]$Ask,

    # Skip the sync check when you know the environment is good.
    [switch]$NoSync
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is not on PATH. Install it: https://docs.astral.sh/uv/getting-started/installation/"
}

# The key can come from the shell or from a .env at the repo root; the agent
# loads .env itself, and an exported variable wins over the file.
$envFile = Join-Path $root ".env"
if (-not $env:GROQ_API_KEY -and -not (Test-Path $envFile)) {
    Write-Host "No Groq API key found." -ForegroundColor Yellow
    Write-Host "  This session only:  `$env:GROQ_API_KEY = 'gsk_...'"
    Write-Host "  Or persist it:      setx GROQ_API_KEY gsk_..."
    Write-Host "  Or create $envFile containing:  GROQ_API_KEY=gsk_..."
    exit 1
}

if (-not $NoSync) {
    # --all-packages is required: the root is a virtual workspace root with no
    # dependencies, so a bare `uv sync` leaves agent-native uninstalled.
    Write-Host "==> uv sync --all-packages" -ForegroundColor Cyan
    uv sync --all-packages
    if ($LASTEXITCODE -ne 0) { Write-Error "uv sync failed." }
}

if (-not (Test-Path $Dir)) {
    Write-Host "==> creating working folder $Dir" -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $Dir -Force | Out-Null
}
$resolved = (Resolve-Path $Dir).Path

Write-Host "==> agent may only touch: $resolved" -ForegroundColor Cyan
Write-Host ""

$agentArgs = @(
    "run", "agent-native",
    "-m", $Message,
    "--dir", $resolved,
    "--model", $Model,
    "--max-turns", $MaxTurns,
    "--temperature", $Temperature
)
if ($Ask) { $agentArgs += "--ask" }

& uv @agentArgs
exit $LASTEXITCODE
