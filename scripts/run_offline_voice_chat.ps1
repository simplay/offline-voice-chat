# Start the app from its conda environment. App options pass through.
[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$GpuLayers = -1,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArguments = @()
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path $PSScriptRoot
$conda = if ($env:CONDA_EXE) { $env:CONDA_EXE } else { "conda" }

$environments = (& $conda env list --json | Out-String | ConvertFrom-Json).envs
$environment = $environments | Where-Object { (Split-Path $_ -Leaf) -eq "offline-voice-chat" } |
    Select-Object -First 1
if (-not $environment) {
    throw "The conda environment offline-voice-chat is missing. Run .\scripts\setup_windows.cmd first."
}

$arguments = @(
    "--gpu-layers", $GpuLayers,
    "--models-dir", (Join-Path $repositoryRoot "models\moonshine"),
    "--files-dir", (Join-Path $repositoryRoot "files"),
    "--system-prompts-dir", (Join-Path $repositoryRoot "system-prompts")
)

# Without a downloaded model, --help and --backend realtime still work.
$model = Join-Path $repositoryRoot "models\Qwen3.5-4B-Q4_K_M.gguf"
if (Test-Path -LiteralPath $model) {
    $arguments += @("--model", $model)
}

& (Join-Path $environment "Scripts\offline-voice-chat.exe") @arguments @AppArguments
exit $LASTEXITCODE
