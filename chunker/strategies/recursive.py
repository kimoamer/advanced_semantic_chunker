"""
Recursive character splitting strategy.

Production standard (per Google, OpenAI benchmarks):
  - Uses a hierarchy of separators (paragraphs → sentences → words)
  - Falls back to smaller separators only when chunks exceed target size
  - Often outperforms semantic chunking for generalized use cases
"""

from __future__ import annotations

from typing import List, Optional

from chunker.config import ChunkerConfig
from chunker.models import Language
from chunker.strategies.base import BaseStrategy


class RecursiveStrategy(BaseStrategy):
    """
    Recursive character text splitter.

    Algorithm:
      1. Try to split on the highest-level separator (e.g., double newline)
      2. If any resulting piece exceeds chunk_size, recursively split
         using the next separator in the hierarchy
      3. Continue until all pieces fit within chunk_size
      4. Apply overlap between consecutive chunks
    """

    def __init__(
        self,
        config: ChunkerConfig,
        language: Language = Language.ENGLISH,
    ):
        super().__init__(config)
        self.language = language
        self.separators = (
            config.recursive_separators_ar
            if language == Language.ARABIC
            else config.recursive_separators_en
        )

    def chunk(self, sentences: List[str], text: str = "") -> List[str]:
        """
        Recursively split text using separator hierarchy.

        Note: This strategy works on the raw text rather than pre-segmented
        sentences, for more natural boundary detection.
        """
        # Use raw text if available, otherwise join sentences
        source = text if text else " ".join(sentences)

        if not source.strip():
            return []

        chunks = self._recursive_split(source, self.separators)
        chunks = self._validate_chunks(chunks)
        chunks = self._apply_overlap(chunks, sentences)

        return chunks

    def _recursive_split(
        self, text: str, separators: List[str], depth: int = 0
    ) -> List[str]:
        """
        Core recursive splitting logic.

        Tries each separator in order. When a split produces pieces
        within the target size, they become chunks. Oversized pieces
        are recursively split with the next separator.
        """
        if self._estimate_tokens(text) <= self.config.chunk_size:
            return [text.strip()] if text.strip() else []

        if not separators:
            # No more separators — force split by character count
            return self._force_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        parts = text.split(separator)

        # Remove empty parts
        parts = [p for p in parts if p.strip()]

        if len(parts) <= 1:
            # This separator didn't help — try the next one
            return self._recursive_split(text, remaining_separators, depth + 1)

        # Build chunks by accumulating parts
        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = (
                current + separator + part if current else part
            )

            if self._estimate_tokens(candidate) <= self.config.chunk_size:
                current = candidate
            else:
                # Flush current
                if current.strip():
                    chunks.append(current.strip())

                # Check if this part alone exceeds the limit
                if self._estimate_tokens(part) > self.config.chunk_size:
                    # Recursively split this oversized part
                    sub_chunks = self._recursive_split(
                        part, remaining_separators, depth + 1
                    )
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _force_split(self, text: str) -> List[str]:
        """Last-resort: split by approximate token count."""
        target_chars = self.config.chunk_size * 4  # ~4 chars/token
        chunks = []
        start = 0

        while start < len(text):
            end = start + target_chars

            # Try to break at a space
            if end < len(text):
                space_idx = text.rfind(" ", start, end)
                if space_idx > start:
                    end = space_idx

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end

        return chunks
