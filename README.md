# Offline Voice Chat

A local voice agent using Moonshine for speech and a GGUF language model through llama.cpp. It supports interruption, saved memory, and command execution.

## Prerequisites

Install [Anaconda](https://www.anaconda.com/download) or Miniconda, and have a microphone and speakers.
Supported systems: Linux (x86-64), macOS 15 or newer on Apple Silicon, and Windows (x86-64).
Run the commands below from the repository root.

## Installation

Setup creates the conda environment `offline-voice-chat` from `environment.yml`, installs the app, and downloads the models.

```bash
# Linux and macOS
bash scripts/setup.sh
```

```powershell
# Windows
.\scripts\setup_windows.cmd
```

- On macOS, llama.cpp is built with Metal, so install the Xcode Command Line Tools first (`xcode-select --install`).
- For an NVIDIA GPU on Linux, install the CUDA toolkit and a C++ compiler, then run `CMAKE_ARGS='-DGGML_CUDA=ON' bash scripts/setup.sh`.
- For an NVIDIA GPU on Windows, install CUDA Toolkit 12.5 or a later 12.x release, then run `.\scripts\setup_windows.cmd -Compute cuda`.
- On Linux, echo cancellation needs `pactl` and the PipeWire echo module (`sudo apt install pulseaudio-utils libspa-0.2-modules`).

## Use the App

```bash
# Linux and macOS
bash scripts/run_offline_voice_chat.sh

# Use the CPU only
bash scripts/run_offline_voice_chat.sh --cpu

# List all options
bash scripts/run_offline_voice_chat.sh --help
```

```powershell
# Windows (CPU)
.\scripts\run_offline_voice_chat.cmd -GpuLayers 0
```

AI-assisted tools were used in the development of this project.

## License

The code is licensed under the [MIT License](LICENSE).
