from __future__ import annotations

import contextlib
import io
import sys
import unittest
from unittest import mock

from fakes import FakeLlama, FakeLlamaCpp, load_llama_model

from offline_voice_chat.cli import _run_local
from offline_voice_chat.config import AppConfig


class ComputeTests(unittest.TestCase):
    def run_local(self, llama_cpp: FakeLlamaCpp, config: AppConfig):
        """Run the local backend with a fake llama_cpp and a fake voice loop."""

        errors = io.StringIO()

        with (
            mock.patch.dict(sys.modules, {"llama_cpp": llama_cpp}),
            mock.patch("offline_voice_chat.cli.VoiceChatApplication") as application,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(errors),
        ):
            application.return_value.can_close_resources = True
            _run_local(config, None)

        return application, errors.getvalue()

    def test_cpu_mode_disables_all_offloading_in_a_gpu_build(self):
        llama_cpp = FakeLlamaCpp(FakeLlama())
        model = load_llama_model(llama_cpp)
        options = llama_cpp.model_options
        self.assertTrue(model.gpu_offload_available)
        self.assertEqual(options["n_gpu_layers"], 0)
        self.assertFalse(options["offload_kqv"])
        self.assertFalse(options["op_offload"])
        self.assertEqual(options["n_threads"], 4)

    def test_gpu_mode_preserves_full_and_partial_layer_requests(self):
        for layers in (-1, 12):
            with self.subTest(layers=layers):
                llama_cpp = FakeLlamaCpp(FakeLlama())
                load_llama_model(llama_cpp, gpu_layers=layers)
                options = llama_cpp.model_options
                self.assertEqual(options["n_gpu_layers"], layers)
                self.assertTrue(options["offload_kqv"])
                self.assertTrue(options["op_offload"])
