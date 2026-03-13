"""
Async wrapper for SemanticChunker.

Enables integration with async frameworks (FastAPI, asyncio, aiohttp)
without blocking the event loop.  All heavy work is delegated to a
thread-pool via ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from chunker.config import ChunkerConfig
from chunker.embeddings.base import BaseEmbeddingProvider
from chunker.models import Chunk

__all__ = ["AsyncSemanticChunker"]


class AsyncSemanticChunker:
    """
    Async-compatible wrapper around :class:`~chunker.core.SemanticChunker`.

    Wraps the synchronous ``chunk()`` and ``chunk_batch()`` calls in
    ``asyncio.to_thread`` so they run in a worker thread without
    blocking the async event loop.

    Parameters
    ----------
    config : ChunkerConfig, optional
        Configuration for the underlying chunker.
    embedding_provider : BaseEmbeddingProvider, optional
        Custom embedding provider to use.

    Examples
    --------
    >>> import asyncio
    >>> from chunker import AsyncSemanticChunker, ChunkerConfig
    >>>
    >>> async def main():
    ...     chunker = AsyncSemanticChunker()
    ...     chunks = await chunker.chunk("Your document text here...")
    ...     for c in chunks:
    ...         print(c.text[:80])
    >>>
    >>> asyncio.run(main())
    """

    def __init__(
        self,
        config: Optional[ChunkerConfig] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
    ) -> None:
        # Import here to avoid circular-import issues at module level
        from chunker.core import SemanticChunker

        self._chunker = SemanticChunker(
            config=config,
            embedding_provider=embedding_provider,
        )

    # ── Public async API ──────────────────────────────────────────────────

    async def chunk(
        self,
        text: str,
        source_file: str = "",
        document_id: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Async version of :meth:`SemanticChunker.chunk`.

        Runs the synchronous chunker in a thread pool so the event loop
        is never blocked.

        Parameters
        ----------
        text : str
            Document text to chunk.
        source_file : str
            Optional source filename for metadata.
        document_id : str
            Optional document identifier.
        extra_metadata : dict, optional
            Extra key/value pairs attached to every chunk's metadata.

        Returns
        -------
        List[Chunk]
            Chunks with rich metadata.
        """
        return await asyncio.to_thread(
            self._chunker.chunk,
            text,
            source_file,
            document_id,
            extra_metadata,
        )

    async def chunk_batch(
        self,
        texts: List[str],
        source_files: Optional[List[str]] = None,
        parallel: bool = False,
        num_workers: Optional[int] = None,
    ) -> List[List[Chunk]]:
        """
        Async version of batch processing.

        Parameters
        ----------
        texts : List[str]
            Documents to process.
        source_files : List[str], optional
            Corresponding source filenames.
        parallel : bool
            Use parallel processing within the thread (default: False).
        num_workers : int, optional
            Number of worker threads for parallel mode.

        Returns
        -------
        List[List[Chunk]]
            One list of chunks per input document.
        """
        from chunker.batch_processor import BatchProcessor

        processor = BatchProcessor(self._chunker)

        return await asyncio.to_thread(
            processor.process_batch,
            texts,
            source_files,
            parallel,
            num_workers,
        )

    # ── Metrics / cache passthrough ───────────────────────────────────────

    def get_metrics(self):
        """Return metrics from the underlying synchronous chunker."""
        if self._chunker._metrics_collector is not None:
            return self._chunker._metrics_collector.get_summary()
        return None

    @property
    def config(self) -> ChunkerConfig:
        """Expose the underlying configuration."""
        return self._chunker.config
