from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from offline_voice_chat.config import parse_args

REPOSITORY = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "requires Bash")
class BashLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        outside = Path(directory.name)
        self.checkout = outside / "checkout with spaces"
        (self.checkout / "scripts").mkdir(parents=True)
        (self.checkout / "files").mkdir()
        (self.checkout / "models").mkdir()
        shutil.copytree(REPOSITORY / "system-prompts", self.checkout / "system-prompts")
        self.launcher = self.checkout / "scripts" / "run_offline_voice_chat.sh"
        shutil.copyfile(REPOSITORY / "scripts/run_offline_voice_chat.sh", self.launcher)
        self.model = self.checkout / "models" / "Qwen3.5-4B-Q4_K_M.gguf"
        self.model.touch()
        self.arguments_file = outside / "arguments"

        # A fake conda whose "activate" puts the app stub on PATH.
        conda_base = outside / "conda base"
        environment_bin = outside / "environment" / "bin"
        (conda_base / "etc/profile.d").mkdir(parents=True)
        environment_bin.mkdir(parents=True)
        (conda_base / "etc/profile.d/conda.sh").write_text(
            f'conda() {{ export PATH="{environment_bin}:$PATH"; }}\n'
        )
        self.write_script(outside / "conda", f'echo "{conda_base}"\n')
        self.write_script(
            environment_bin / "offline-voice-chat",
            'printf "%s\\0" "$@" > "$CHATBOT_TEST_OUTPUT"\nexit 17\n',
        )
        self.platform_commands = outside / "platform commands"
        self.platform_commands.mkdir()
        self.environment = {
            **os.environ,
            "CONDA_EXE": str(outside / "conda"),
            "CHATBOT_TEST_OUTPUT": str(self.arguments_file),
            "PATH": f"{self.platform_commands}{os.pathsep}{os.environ['PATH']}",
        }

    def write_script(self, path: Path, body: str) -> None:
        path.write_text("#!/usr/bin/env bash\n" + body)
        path.chmod(0o755)

    def launch(self, *args: str, platform: str = "Linux") -> list[str]:
        self.write_script(self.platform_commands / "uname", f'echo "{platform}"\n')
        result = subprocess.run(
            ["bash", str(self.launcher), *args],
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 17, result.stderr)
        output = self.arguments_file.read_text(encoding="utf-8")
        return output.rstrip("\0").split("\0")

    def test_defaults_resolve_from_the_checkout_and_preserve_exit_status(self) -> None:
        config = parse_args(self.launch())
        self.assertEqual(config.model_path, self.model.resolve())
        self.assertEqual(config.files_dir, (self.checkout / "files").resolve())
        self.assertEqual(config.gpu_layers, -1)

    def test_realtime_starts_without_a_downloaded_model(self) -> None:
        self.model.unlink()
        config = parse_args(self.launch("--backend", "realtime"))
        self.assertIsNone(config.model_path)

    def test_linux_adds_the_menu_and_user_flags_win(self) -> None:
        linux = parse_args(self.launch("--cpu", "--no-echo-cancel"))
        self.assertTrue(linux.menu)
        self.assertFalse(linux.echo_cancel)
        self.assertEqual(linux.gpu_layers, 0)

        macos = parse_args(self.launch(platform="Darwin"))
        self.assertFalse(macos.menu)


if __name__ == "__main__":
    unittest.main()
