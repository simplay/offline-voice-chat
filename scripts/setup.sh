#!/usr/bin/env bash
# Install the app into the conda environment from environment.yml (Linux and macOS).
set -Eeuo pipefail

REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-conda}"

case "${1:-}" in
    '') download_models=true ;;
    --skip-model-download) download_models=false ;;
    *) printf 'Usage: bash scripts/setup.sh [--skip-model-download]\n' >&2; exit 2 ;;
esac

run() {
    "$CONDA" run --no-capture-output --name offline-voice-chat "$@"
}

"$CONDA" env update --file "${REPOSITORY_DIR}/environment.yml" --prune

if [[ "$(uname -s)" == Darwin ]]; then
    # Build llama.cpp from source so it runs on the Apple GPU.
    export CMAKE_ARGS="-DGGML_METAL=ON ${CMAKE_ARGS:-}"
fi

if [[ -n "${CMAKE_ARGS:-}" ]]; then
    # A source build, e.g. CMAKE_ARGS=-DGGML_CUDA=ON; a cached CPU wheel must not win.
    llama_options=(--no-binary llama-cpp-python --force-reinstall --no-cache-dir)
else
    llama_options=(--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu)
fi

run python -m pip install "${llama_options[@]}" --editable "${REPOSITORY_DIR}[setup]"

if "$download_models"; then
    run python "${REPOSITORY_DIR}/scripts/download_models.py"
fi

printf '\nSetup complete. Start with:\n  bash scripts/run_offline_voice_chat.sh\n'
