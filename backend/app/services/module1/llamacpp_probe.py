"""Local llama.cpp-backed counterfactual probe implementation for Module 1.

Implements the ProbeBackend interface from counterfactual_probe.py
using llama-cpp-python and GGUF quantization (e.g. Qwen2.5-3B-Instruct Q4_K_M).

Unlike the Groq API (which rejects logprobs and relies on an explanation effort
proxy), this backend computes GENUINE token-level cross-entropy perplexity:
    PPL = exp(- 1/N * sum(log P(w_i | w_{<i})))
using local log-probability outputs from llama.cpp.

Hardware Safety:
    - Pre-flight RAM verification prevents OS thrashing on low-memory machines.
    - Small context window (n_ctx=512) minimizes KV cache allocation.
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil

from .counterfactual_probe import ProbeBackend

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Qwen2.5-3B-Instruct"
DEFAULT_GGUF_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"


class LocalLlamaCppProbeBackend(ProbeBackend):
    """Local GGUF SLM backend for counterfactual cross-entropy perplexity probing."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 512,
        n_threads: Optional[int] = None,
        min_ram_gb: float = 1.0,
        warn_ram_gb: float = 3.0,
        verbose: bool = False,
    ) -> None:
        try:
            import llama_cpp
            self._llama_module = llama_cpp
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for LocalLlamaCppProbeBackend. "
                "Install via: pip install llama-cpp-python"
            )

        # 1. Resolve model file path
        if model_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
            candidate = os.path.join(repo_root, "data", "models", DEFAULT_GGUF_FILENAME)
            candidate_q3 = os.path.join(repo_root, "data", "models", "qwen2.5-3b-instruct-q3_k_m.gguf")
            if os.path.exists(candidate):
                model_path = candidate
            elif os.path.exists(candidate_q3):
                model_path = candidate_q3
            else:
                model_path = candidate

        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_threads = n_threads or max(1, (os.cpu_count() or 4) - 1)
        self._model_name = f"{DEFAULT_MODEL_NAME} (llama.cpp GGUF)"
        self._llm = None
        self._verbose = verbose

        # Performance and latency telemetry
        self.call_latencies: List[float] = []
        self.peak_rss_mb: float = 0.0

        # 2. Check system memory headroom
        self._check_memory(min_ram_gb, warn_ram_gb)

        # 3. Load model weights
        self._load_model()

    def _check_memory(self, min_ram_gb: float, warn_ram_gb: float) -> None:
        mem = psutil.virtual_memory()
        avail_gb = mem.available / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)

        if avail_gb < warn_ram_gb:
            logger.warning(
                "[LocalLlamaCppProbeBackend] Available system RAM (%.2f GB / %.2f GB) is below "
                "recommended %.1f GB threshold. Operating in memory-constrained mode.",
                avail_gb, total_gb, warn_ram_gb
            )
            print(
                f"[RAM WARNING] Available system RAM is {avail_gb:.2f} GB (recommended: >= {warn_ram_gb:.1f} GB). "
                f"Using n_ctx={self._n_ctx} to minimize footprint."
            )

        if avail_gb < min_ram_gb:
            logger.error(
                "[LocalLlamaCppProbeBackend] Critical: Available RAM (%.2f GB) is less than "
                "minimum required %.1f GB.", avail_gb, min_ram_gb
            )
            # Check if swap/pagefile exists to avoid hard crash
            swap = psutil.swap_memory()
            if swap.free / (1024 ** 3) < 2.0:
                raise RuntimeError(
                    f"Insufficient RAM to safely load 3B SLM model: only {avail_gb:.2f} GB free "
                    f"(minimum required: {min_ram_gb:.1f} GB). Close other applications and retry."
                )

    def _load_model(self) -> None:
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"GGUF model not found at '{self._model_path}'. "
                f"Run 'python scripts/download_slm.py' to download Qwen2.5-3B-Instruct."
            )

        t0 = time.perf_counter()
        # Initialize Llama with logits_all=True (required for logprobs extraction)
        self._llm = self._llama_module.Llama(
            model_path=self._model_path,
            n_ctx=self._n_ctx,
            n_threads=self._n_threads,
            logits_all=True,
            verbose=self._verbose,
        )
        load_time = time.perf_counter() - t0
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
        logger.info(
            "[LocalLlamaCppProbeBackend] Loaded %s in %.2fs (Process RSS: %.1f MB)",
            os.path.basename(self._model_path), load_time, rss_mb
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def explain(self, prompt: str) -> str:
        """Generate a short textual explanation for the prompt."""
        t0 = time.perf_counter()
        res = self._llm.create_completion(
            prompt=prompt,
            max_tokens=80,
            temperature=0.2,
            stop=["\n\n", "Question:"],
        )
        dur = time.perf_counter() - t0
        self.call_latencies.append(dur)
        text = res["choices"][0]["text"].strip()
        return text

    def perplexity(self, prompt: str) -> float:
        """Compute genuine cross-entropy perplexity using local token log probabilities.
        
        Evaluates:
            PPL = exp(- 1/N * sum_{i=1}^N log P(w_i | w_{<i}))
        where log P are the exact token log-probabilities returned by llama.cpp.
        """
        t0 = time.perf_counter()
        # max_tokens=0 with echo=True evaluates prompt tokens without generating new tokens
        res = self._llm.create_completion(
            prompt=prompt,
            max_tokens=0,
            echo=True,
            logprobs=1,
            temperature=0.0,
        )
        dur = time.perf_counter() - t0
        self.call_latencies.append(dur)

        rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)

        choice = res["choices"][0]
        logprobs_info = choice.get("logprobs")
        if not logprobs_info or "token_logprobs" not in logprobs_info:
            logger.warning("[LocalLlamaCppProbeBackend] No token_logprobs returned; fallback PPL=10.0")
            return 10.0

        raw_logprobs = logprobs_info["token_logprobs"]
        # Filter out leading None (the first prompt token has no prior context)
        valid_logprobs = [lp for lp in raw_logprobs if lp is not None and not math.isnan(lp)]

        if not valid_logprobs:
            return 10.0

        # Mean negative log likelihood per token
        mean_nll = -float(np.mean(valid_logprobs))
        # Clamp to avoid numerical overflow
        mean_nll = max(-20.0, min(20.0, mean_nll))
        ppl = float(np.exp(mean_nll))

        logger.debug(
            "[LocalLlamaCppProbeBackend] prompt_len=%d, n_tokens=%d, mean_nll=%.3f, PPL=%.2f, dur=%.3fs",
            len(prompt), len(valid_logprobs), mean_nll, ppl, dur
        )
        return ppl
