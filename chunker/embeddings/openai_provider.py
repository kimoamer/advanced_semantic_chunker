"""
OpenAI embedding provider.

Supports text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002.
Includes a token-bucket rate limiter to prevent 429 errors.
"""

from __future__ import annotations

import os
import threading
import time
from typing import List, Optional

import numpy as np

from chunker.embeddings.base import BaseEmbeddingProvider


# Dimension map for known OpenAI models
_OPENAI_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class _TokenBucketRateLimiter:
    """
    Simple token-bucket rate limiter (thread-safe).

    Parameters
    ----------
    requests_per_minute : int
        Maximum API calls allowed per minute.
    """

    def __init__(self, requests_per_minute: int) -> None:
        self._rpm = requests_per_minute
        self._tokens = float(requests_per_minute)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until one request token is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self._rpm,
                    self._tokens + elapsed * (self._rpm / 60.0),
                )
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            time.sleep(0.05)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider backed by the OpenAI Embeddings API.

    Requires the ``openai`` package and a valid API key.

    Batches requests to minimise API calls (up to 2 047 texts per call).
    A token-bucket rate limiter prevents 429 errors on high-throughput jobs.

    Parameters
    ----------
    model_name : str
        OpenAI embedding model (default: ``"text-embedding-3-small"``).
    api_key : str, optional
        API key; falls back to the ``OPENAI_API_KEY`` environment variable.
    batch_size : int
        Texts per API call — capped at 2 047 (default: 2 047).
    max_retries : int
        Retries on transient errors (default: 3).
    requests_per_minute : int
        Rate-limit cap in requests/min (default: 500). Set to 0 to disable.
    """

    MAX_BATCH_SIZE = 2047

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 2047,
        max_retries: int = 3,
        requests_per_minute: int = 500,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.batch_size = min(batch_size, self.MAX_BATCH_SIZE)
        self.max_retries = max_retries
        self._client = None
        self._rate_limiter: Optional[_TokenBucketRateLimiter] = (
            _TokenBucketRateLimiter(requests_per_minute)
            if requests_per_minute and requests_per_minute > 0
            else None
        )

    def _get_client(self):
        """Lazy-load the OpenAI client."""
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAIEmbeddingProvider. "
                "Install it with: pip install openai"
            )

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set it via api_key parameter "
                "or OPENAI_API_KEY environment variable."
            )

        self._client = OpenAI(api_key=self.api_key)
        return self._client

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts using OpenAI API with batching.

        When the number of texts is within the batch_size limit, all texts
        are embedded in a single API call for optimal performance. For larger
        batches, multiple API calls are made automatically.

        Parameters
        ----------
        texts : List[str]
            Input texts to embed.

        Returns
        -------
        np.ndarray
            Embedding matrix of shape (len(texts), embedding_dim).
        """
        if not texts:
            dim = self.get_dimension()
            return np.array([]).reshape(0, dim)

        client = self._get_client()
        all_embeddings = []

        # Process in batches (single call if within limit)
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Replace empty strings with a space (OpenAI rejects empty strings)
            batch = [t if t.strip() else " " for t in batch]

            # Respect rate limit before each API call
            if self._rate_limiter is not None:
                self._rate_limiter.acquire()

            response = client.embeddings.create(
                model=self.model_name,
                input=batch,
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        result = self.embed([text])
        return result[0]

    def get_dimension(self) -> int:
        """Return embedding dimension for the configured model."""
        if self.model_name in _OPENAI_DIMS:
            return _OPENAI_DIMS[self.model_name]
        # Fallback: embed a test string
        result = self.embed(["dimension test"])
        return result.shape[1]
