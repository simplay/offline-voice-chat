#!/usr/bin/env bash
# Start the app from its conda environment. App options pass through; --cpu
# is short for --gpu-layers 0.
set -Eeuo pipefail

REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-conda}"

arguments=(
    --gpu-layers -1
    --models-dir "${REPOSITORY_DIR}/models/moonshine"
    --files-dir "${REPOSITORY_DIR}/files"
    --system-prompts-dir "${REPOSITORY_DIR}/system-prompts"
)

# Without a downloaded model, --help and --backend realtime still work.
model="${REPOSITORY_DIR}/models/Qwen3.5-4B-Q4_K_M.gguf"
if [[ -f "$model" ]]; then
    arguments+=(--model "$model")
fi

if [[ "$(uname -s)" == Linux ]]; then
    # Conda's ALSA setup lacks the desktop PulseAudio/PipeWire devices.
    if [[ -z "${ALSA_CONFIG_PATH:-}" && -f /usr/share/alsa/alsa.conf ]]; then
        export ALSA_CONFIG_PATH=/usr/share/alsa/alsa.conf
    fi
    if [[ -z "${ALSA_PLUGIN_DIR:-}" && -d /usr/lib/x86_64-linux-gnu/alsa-lib ]]; then
        export ALSA_PLUGIN_DIR=/usr/lib/x86_64-linux-gnu/alsa-lib
    fi
    # Before the user's options, so --no-echo-cancel still wins.
    arguments+=(--menu --echo-cancel)
fi

for argument in "$@"; do
    if [[ "$argument" == --cpu ]]; then
        arguments+=(--gpu-layers 0)
    else
        arguments+=("$argument")
    fi
done

# shellcheck source=/dev/null
source "$("$CONDA" info --base)/etc/profile.d/conda.sh"
conda activate offline-voice-chat
# exec keeps Ctrl-C delivery and the app's exit status.
exec offline-voice-chat "${arguments[@]}"
