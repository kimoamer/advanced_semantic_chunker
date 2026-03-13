"""
Fixed-size chunking strategy.

Simplest and fastest strategy. Splits by character or token count
with configurable overlap. Useful as a baseline or when document
structure is highly inconsistent.
"""

from __future__ import annotations

from typing import List

from chunker.config import ChunkerConfig
from chunker.strategies.base import BaseStrategy


class FixedSizeStrategy(BaseStrategy):
    """
    Fixed-size chunker — splits text into equal-sized pieces.

    Each chunk targets config.chunk_size tokens with config.chunk_overlap
    tokens of overlap between consecutive chunks.
    """

    def chunk(self, sentences: List[str], text: str = "") -> List[str]:
        """Split into fixed-size chunks with overlap."""
        source = text if text else " ".join(sentences)

        if not source.strip():
            return []

        target_chars = self.config.chunk_size * 4   # ~4 chars/token
        overlap_chars = self.config.chunk_overlap * 4

        chunks: List[str] = []
        start = 0

        while start < len(source):
            end = start + target_chars

            # Try to break at a word boundary
            if end < len(source):
                # Look for whitespace near the boundary
                search_start = max(end - 50, start)
                space_idx = source.rfind(" ", search_start, end + 50)
                if space_idx > start:
                    end = space_idx

            chunk = source[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start forward by (chunk_size - overlap) chars
            start = end - overlap_chars
            if start <= (end - target_chars):
                # Prevent infinite loop
                start = end

        return [c for c in chunks if c.strip()]
