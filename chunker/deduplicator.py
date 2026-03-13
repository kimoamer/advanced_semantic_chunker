"""
Chunk deduplication utility.

Removes duplicate chunks from a list or across a batch of documents
using the SHA-256 content hash stored on each Chunk object.
"""

from __future__ import annotations

from typing import List

__all__ = ["ChunkDeduplicator"]


class ChunkDeduplicator:
    """
    Remove duplicate chunks by content hash.

    Each :class:`~chunker.models.Chunk` already carries a ``content_hash``
    property (SHA-256 of its text).  This class uses that hash to
    deduplicate chunks either within a single document or across an entire
    batch.

    Examples
    --------
    >>> from chunker import SemanticChunker
    >>> from chunker.deduplicator import ChunkDeduplicator
    >>>
    >>> chunker = SemanticChunker()
    >>> # Same paragraph appears in two documents
    >>> doc1_chunks = chunker.chunk(text1)
    >>> doc2_chunks = chunker.chunk(text2)
    >>>
    >>> dedup = ChunkDeduplicator()
    >>> unique = dedup.deduplicate_batch([doc1_chunks, doc2_chunks])
    """

    def deduplicate(self, chunks: "List") -> "List":
        """
        Remove duplicate chunks from a single document's chunk list.

        The *first* occurrence of each unique content hash is kept;
        subsequent duplicates are dropped.  Order is preserved.

        Parameters
        ----------
        chunks : List[Chunk]
            Chunks from a single document.

        Returns
        -------
        List[Chunk]
            Deduplicated list, same type as input.
        """
        seen: set = set()
        result = []
        for chunk in chunks:
            h = chunk.content_hash
            if h not in seen:
                seen.add(h)
                result.append(chunk)
        return result

    def deduplicate_batch(self, batch: "List[List]") -> "List[List]":
        """
        Remove duplicate chunks across a batch of documents.

        A chunk that appears in document A will be removed from document B
        (and any later documents) if it has the same content hash.

        Parameters
        ----------
        batch : List[List[Chunk]]
            One list of chunks per document.

        Returns
        -------
        List[List[Chunk]]
            Deduplicated batch; same structure as input.
        """
        seen: set = set()
        result = []
        for doc_chunks in batch:
            unique = []
            for chunk in doc_chunks:
                h = chunk.content_hash
                if h not in seen:
                    seen.add(h)
                    unique.append(chunk)
            result.append(unique)
        return result

    def count_duplicates(self, chunks: "List") -> int:
        """
        Count the number of duplicate chunks in a list.

        Parameters
        ----------
        chunks : List[Chunk]
            Chunks to inspect.

        Returns
        -------
        int
            Number of chunks that are duplicates (would be removed).
        """
        seen: set = set()
        duplicates = 0
        for chunk in chunks:
            h = chunk.content_hash
            if h in seen:
                duplicates += 1
            else:
                seen.add(h)
        return duplicates
