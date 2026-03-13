"""
Abstract base class for embedding providers.

All embedding backends must implement this interface so that
chunking strategies remain decoupled from the embedding source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEmbeddingProvider(ABC):
    """
    Pluggable embedding backend interface.

    Implementations should handle:
      - Batching for efficiency
      - Normalization (L2) if the model doesn't do it natively
      - Caching (optional)
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Parameters
        ----------
        texts : List[str]
            Input texts to embed.

        Returns
        -------
        np.ndarray
            Embedding matrix of shape (len(texts), embedding_dim).
        """
        ...

    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Returns
        -------
        np.ndarray
            1-D embedding vector of shape (embedding_dim,).
        """
        ...

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance = 1 - cosine_similarity."""
        return 1.0 - BaseEmbeddingProvider.cosine_similarity(a, b)

    @staticmethod
    def pairwise_cosine_distances(embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine distances between consecutive embedding pairs.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (n, dim).

        Returns
        -------
        np.ndarray
            Shape (n-1,) — distance between embedding[i] and embedding[i+1].
        """
        if len(embeddings) < 2:
            return np.array([])

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid div-by-zero
        normed = embeddings / norms

        # Consecutive dot products
        similarities = np.sum(normed[:-1] * normed[1:], axis=1)
        distances = 1.0 - similarities
        return distances
