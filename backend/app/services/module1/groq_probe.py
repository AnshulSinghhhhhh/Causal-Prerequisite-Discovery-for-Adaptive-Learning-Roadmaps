"""Groq-backed counterfactual probe implementation for Module 1.

Implements the ProbeBackend interface from counterfactual_probe.py
using Groq hosted LLM API (e.g. qwen/qwen3.8-27b).

Note on Perplexity vs. Explanation Effort Proxy:
    Groq completions API does not expose full per-token log probabilities.
    To avoid misleading claims, this probe measures counterfactual difficulty via
    an explanation effort proxy (relative elaboration word count when a prerequisite
    concept is prohibited), with a fallback method perplexity() that issues a warning
    and delegates to explanation_effort_proxy().
"""
from __future__ import annotations

import logging
import os
import re
import time
import warnings
from typing import Optional

from dotenv import load_dotenv

from .counterfactual_probe import ProbeBackend

load_dotenv()
logger = logging.getLogger(__name__)


class GroqProbeBackend(ProbeBackend):
    """Live Groq API backend for counterfactual directional probing."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen/qwen3.8-27b",
        max_tokens: int = 80,
        temperature: float = 0.2,
        requests_per_minute: int = 25,
    ) -> None:
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("groq is required for GroqProbeBackend: pip install groq")

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY environment variable or api_key parameter is required")

        self._client = Groq(api_key=key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._min_interval = 60.0 / requests_per_minute
        self._last_call: float = 0.0
        self._model_name = model
        self._cache: dict[str, str] = {}

    @property
    def model_name(self) -> str:
        return self._model_name

    def _throttle(self) -> None:
        """Rate-limit to prevent exceeding requests_per_minute."""
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def explain(self, prompt: str) -> str:
        """Generate an explanation response from Groq LLM."""
        if prompt in self._cache:
            return self._cache[prompt]
        self._throttle()
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            text = resp.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("[GroqProbeBackend] Call failed for prompt: %s", exc)
            text = f"[error: {exc}]"
        self._cache[prompt] = text
        return text

    def explanation_effort_proxy(self, prompt: str) -> float:
        """Measure explanation difficulty via length/effort proxy.
        
        When an LLM explains concept V without mentioning prerequisite U,
        more circumlocution and words are required if U is truly foundational.
        """
        text = self.explain(prompt)
        words = len(re.findall(r"\w+", text))
        # Clamp at 5 words to prevent zero/underflow
        words = max(words, 5)
        baseline_words = 40.0
        effort = 10.0 * (words / baseline_words)
        return effort

    def perplexity(self, prompt: str) -> float:
        """Estimate perplexity using explanation effort proxy under Groq API constraints."""
        warnings.warn(
            "Groq completions API does not expose native token logprobs; using explanation effort proxy.",
            UserWarning,
            stacklevel=2,
        )
        return self.explanation_effort_proxy(prompt)
