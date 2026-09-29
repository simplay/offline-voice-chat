"""Read editable system prompts from the checkout's system-prompts directory."""

from pathlib import Path


def default_system_prompts_dir() -> Path:
    # The setup scripts install this checkout in editable mode.
    return Path(__file__).resolve().parents[2] / "system-prompts"


def load_system_prompt(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exception:
        raise ValueError(
            f"could not read system prompt {path}: {exception}"
        ) from exception

    if not text:
        raise ValueError(f"system prompt is empty: {path}")

    return text


def load_default_system_prompt() -> str:
    return load_system_prompt(default_system_prompts_dir() / "default.txt")


def list_system_prompts(directory: Path) -> list[Path]:
    try:
        return sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() == ".txt"
            ),
            key=lambda path: path.name.casefold(),
        )
    except OSError as exception:
        raise ValueError(
            f"could not list system prompts in {directory}: {exception}"
        ) from exception
