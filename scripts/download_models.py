"""Download verified demo assets using the current Python environment."""

from __future__ import annotations

import hashlib
from pathlib import Path

MODEL_REPOSITORY = "lmstudio-community/Qwen3.5-4B-GGUF"
MODEL_FILENAME = "Qwen3.5-4B-Q4_K_M.gguf"
MODEL_SHA256 = "25082a7dd3776cc3c741c6347d3bd04523f05796607b3fbc32fa3a25dfa1418c"


def verify_model(path: Path) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)

    if digest.hexdigest() != MODEL_SHA256:
        raise RuntimeError(f"GGUF checksum mismatch: {path}")


def main() -> None:
    from huggingface_hub import hf_hub_download

    model_dir = Path(__file__).resolve().parents[1] / "models"
    speech_dir = model_dir / "moonshine"
    speech_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_FILENAME}...", flush=True)
    path = hf_hub_download(
        repo_id=MODEL_REPOSITORY, filename=MODEL_FILENAME, local_dir=model_dir
    )
    print("Verifying the GGUF checksum...", flush=True)
    verify_model(Path(path))
    cache_speech_models(speech_dir)


def cache_speech_models(speech_dir: Path) -> None:
    # AgentFlow imports microphone/audio code. Use download APIs so setup also
    # works on a server without a running PulseAudio/PipeWire session.
    from moonshine_voice.download import download_tts_assets, get_model_for_language

    print("Caching default English speech models...", flush=True)
    get_model_for_language("en", cache_root=speech_dir)
    # Match AgentFlow's per-user TTS cache; --models-dir controls its ASR cache.
    download_tts_assets("en")


if __name__ == "__main__":
    main()
