"""Unit tests for LocalLlamaCppProbeBackend."""

import pytest
from unittest.mock import MagicMock, patch

from backend.app.services.module1.counterfactual_probe import ProbeBackend
from backend.app.services.module1.llamacpp_probe import LocalLlamaCppProbeBackend


def test_probe_backend_subclass():
    assert issubclass(LocalLlamaCppProbeBackend, ProbeBackend)


def test_memory_check_warns_under_3gb():
    with patch("psutil.virtual_memory") as mock_mem, patch("os.path.exists", return_value=True):
        mock_mem.return_value.available = 1.2 * (1024 ** 3)
        mock_mem.return_value.total = 8.0 * (1024 ** 3)
        with patch("llama_cpp.Llama"):
            backend = LocalLlamaCppProbeBackend(model_path="dummy.gguf")
            assert backend.model_name == "Qwen2.5-3B-Instruct (llama.cpp GGUF)"


def test_perplexity_calculation_from_logprobs():
    with patch("psutil.virtual_memory") as mock_mem, patch("os.path.exists", return_value=True):
        mock_mem.return_value.available = 4.0 * (1024 ** 3)
        mock_mem.return_value.total = 8.0 * (1024 ** 3)
        with patch("llama_cpp.Llama") as mock_llama_cls:
            mock_instance = MagicMock()
            # Simulate prompt logprobs: [None, -1.0, -1.5, -0.5] -> mean -1.0 -> exp(1.0) ~ 2.718
            mock_instance.create_completion.return_value = {
                "choices": [{
                    "logprobs": {
                        "token_logprobs": [None, -1.0, -1.5, -0.5]
                    }
                }]
            }
            mock_llama_cls.return_value = mock_instance

            backend = LocalLlamaCppProbeBackend(model_path="dummy.gguf")
            ppl = backend.perplexity("Test prompt")

            assert isinstance(ppl, float)
            assert ppl == pytest.approx(2.71828, rel=1e-3)
            assert len(backend.call_latencies) == 1
