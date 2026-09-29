# Install the app into the conda environment from environment.yml (Windows).
# -Compute cuda needs the CUDA Toolkit 12.5 or a later 12.x release.
param(
    [ValidateSet("cpu", "cuda")]
    [string]$Compute = "cpu",
    [switch]$SkipModelDownload
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path $PSScriptRoot
$conda = if ($env:CONDA_EXE) { $env:CONDA_EXE } else { "conda" }

function Invoke-CondaCommand {
    & $conda @args
    if ($LASTEXITCODE -ne 0) {
        throw "conda $args failed with exit code $LASTEXITCODE."
    }
}

function Invoke-EnvironmentPython {
    Invoke-CondaCommand run --no-capture-output --name offline-voice-chat python @args
}

Invoke-CondaCommand env update --file (Join-Path $repositoryRoot "environment.yml") --prune

# The CPU and CUDA wheels share a version number, so force the chosen one.
$wheelBackend = if ($Compute -eq "cuda") { "cu125" } else { "cpu" }
Invoke-EnvironmentPython -m pip install --force-reinstall --no-deps --only-binary=:all: `
    --index-url "https://abetlen.github.io/llama-cpp-python/whl/$wheelBackend" `
    "llama-cpp-python==0.3.35"
Invoke-EnvironmentPython -m pip install --editable "${repositoryRoot}[setup]"

if (-not $SkipModelDownload) {
    Invoke-EnvironmentPython (Join-Path $PSScriptRoot "download_models.py")
}

Write-Host "`nSetup complete. Start with:`n  .\scripts\run_offline_voice_chat.cmd"
