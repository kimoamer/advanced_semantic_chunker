"""
Batch processing optimizer for the Semantic Chunker.

Provides optimized batch operations for processing multiple documents with:
- Progress callback support
- Parallel processing with concurrent.futures
- Memory-efficient streaming for large batches
"""

from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING, Callable, Iterator, List, Literal, Optional, Tuple

if TYPE_CHECKING:
    from chunker.core import SemanticChunker
    from chunker.models import Chunk

__all__ = ["BatchProcessor"]


class BatchProcessor:
    """Optimized batch processing for multiple documents."""

    def __init__(
        self,
        chunker: "SemanticChunker",
        batch_size: int = 32,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        executor_type: Literal["thread", "process"] = "thread",
    ):
        """
        Initialize batch processor.

        Parameters
        ----------
        chunker : SemanticChunker
            The chunker instance to use for processing.
        batch_size : int, optional
            Number of documents to process in each batch (default: 32).
        progress_callback : Callable[[int, int], None], optional
            Callback function called with (current, total) after each document.

        Examples
        --------
        >>> from chunker import SemanticChunker, ChunkerConfig
        >>> from chunker.batch_processor import BatchProcessor
        >>> 
        >>> config = ChunkerConfig()
        >>> chunker = SemanticChunker(config)
        >>> 
        >>> def progress(current, total):
        ...     print(f"Processing {current}/{total}")
        >>> 
        >>> processor = BatchProcessor(chunker, progress_callback=progress)
        >>> texts = ["Document 1", "Document 2", "Document 3"]
        >>> results = processor.process_batch(texts)
        >>>
        >>> # Use process-based parallelism for CPU-bound workloads
        >>> processor = BatchProcessor(chunker, executor_type="process")
        >>> results = processor.process_batch(texts, parallel=True, num_workers=4)
        """
        self.chunker = chunker
        self.batch_size = batch_size
        self.progress_callback = progress_callback
        self.executor_type = executor_type

    def process_batch(
        self,
        texts: List[str],
        source_files: Optional[List[str]] = None,
        parallel: bool = False,
        num_workers: Optional[int] = None,
    ) -> List[List["Chunk"]]:
        """
        Process multiple documents with optimizations.

        Parameters
        ----------
        texts : List[str]
            List of document texts to process.
        source_files : List[str], optional
            Corresponding source filenames for each document.
        parallel : bool, optional
            Whether to use parallel processing (default: False).
        num_workers : int, optional
            Number of worker threads for parallel processing.
            If None, uses the default from concurrent.futures.

        Returns
        -------
        List[List[Chunk]]
            List of chunk lists, one per document.

        Examples
        --------
        >>> processor = BatchProcessor(chunker)
        >>> texts = ["Doc 1", "Doc 2", "Doc 3"]
        >>> results = processor.process_batch(texts)
        >>> len(results)
        3
        
        >>> # With parallel processing
        >>> results = processor.process_batch(texts, parallel=True, num_workers=4)
        """
        if not texts:
            return []

        # Ensure source_files matches texts length
        if source_files is None:
            source_files = [""] * len(texts)
        elif len(source_files) < len(texts):
            source_files = source_files + [""] * (len(texts) - len(source_files))

        if parallel:
            return self._process_parallel(texts, source_files, num_workers)
        else:
            return self._process_sequential(texts, source_files)

    def _process_sequential(
        self, texts: List[str], source_files: List[str]
    ) -> List[List["Chunk"]]:
        """Process documents sequentially with progress tracking."""
        results = []
        total = len(texts)

        for i, text in enumerate(texts):
            source_file = source_files[i] if i < len(source_files) else ""
            chunks = self.chunker.chunk(text, source_file=source_file)
            results.append(chunks)

            # Invoke progress callback if provided
            if self.progress_callback:
                self.progress_callback(i + 1, total)

        return results

    def _process_parallel(
        self, texts: List[str], source_files: List[str], num_workers: Optional[int]
    ) -> List[List["Chunk"]]:
        """Process documents in parallel using concurrent.futures."""
        results = [None] * len(texts)
        total = len(texts)
        completed = 0

        def process_single(index: int, text: str, source_file: str) -> Tuple[int, List["Chunk"]]:
            """Process a single document and return its index with results."""
            chunks = self.chunker.chunk(text, source_file=source_file)
            return index, chunks

        executor_cls = (
            concurrent.futures.ProcessPoolExecutor
            if self.executor_type == "process"
            else concurrent.futures.ThreadPoolExecutor
        )

        with executor_cls(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_single, i, text, source_files[i]): i
                for i, text in enumerate(texts)
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                index, chunks = future.result()
                results[index] = chunks
                completed += 1

                # Invoke progress callback if provided
                if self.progress_callback:
                    self.progress_callback(completed, total)

        return results

    def process_with_progress(
        self,
        texts: List[str],
        source_files: Optional[List[str]] = None,
    ) -> Iterator[Tuple[int, List["Chunk"]]]:
        """
        Process batch with progress reporting as an iterator.

        Yields results one at a time as they are processed, allowing
        for streaming processing of large batches.

        Parameters
        ----------
        texts : List[str]
            List of document texts to process.
        source_files : List[str], optional
            Corresponding source filenames for each document.

        Yields
        ------
        Tuple[int, List[Chunk]]
            Tuple of (index, chunks) for each processed document.

        Examples
        --------
        >>> processor = BatchProcessor(chunker)
        >>> texts = ["Doc 1", "Doc 2", "Doc 3"]
        >>> 
        >>> for index, chunks in processor.process_with_progress(texts):
        ...     print(f"Processed document {index}: {len(chunks)} chunks")
        """
        if source_files is None:
            source_files = [""] * len(texts)
        elif len(source_files) < len(texts):
            source_files = source_files + [""] * (len(texts) - len(source_files))

        for i, text in enumerate(texts):
            source_file = source_files[i] if i < len(source_files) else ""
            chunks = self.chunker.chunk(text, source_file=source_file)
            yield i, chunks
