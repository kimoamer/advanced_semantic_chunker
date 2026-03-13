"""
SentenceTransformer embedding provider.

Recommended models for EN/AR bilingual use (from research):
  - BAAI/bge-m3           (best multilingual, 1024-d)
  - intfloat/multilingual-e5-large  (strong AR support, 1024-d)
  - CAMeL-Lab/GATE-AraBERT-v1      (Arabic-optimized, 768-d)
  - jinaai/jina-embeddings-v3       (long-context, 1024-d)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from chunker.embeddings.base import BaseEmbeddingProvider


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """
    Embedding provider backed by sentence-transformers library.

    Supports any HuggingFace model compatible with SentenceTransformer.
    Handles batching, normalization, and device placement automatically.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = False,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.show_progress = show_progress
        self._model = None
        self._device = device
        self._dimension: Optional[int] = None

    def _load_model(self):
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerProvider. "
                "Install it with: pip install sentence-transformers"
            )

        kwargs = {}
        if self._device:
            kwargs["device"] = self._device

        self._model = SentenceTransformer(self.model_name, **kwargs)
        # Cache dimension
        test_emb = self._model.encode(["test"], normalize_embeddings=self.normalize)
        self._dimension = test_emb.shape[1]

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts in a single model call.

        The model processes all texts together, using internal batching
        for memory efficiency. This ensures optimal performance for
        batch operations.

        Parameters
        ----------
        texts : List[str]
            Input texts to embed.

        Returns
        -------
        np.ndarray
            Embedding matrix of shape (len(texts), embedding_dim).
        """
        self._load_model()

        if not texts:
            return np.array([]).reshape(0, self._dimension)

        # Process all texts in a single call with internal batching
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        result = self.embed([text])
        return result[0]

    def get_dimension(self) -> int:
        """Return embedding vector dimension."""
        self._load_model()
        return self._dimension
