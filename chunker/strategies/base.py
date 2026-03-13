"""
Abstract base class for all chunking strategies.

Every strategy receives pre-processed sentences and must return
a list of text chunks. The base class provides shared utilities
for overlap application and chunk validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from chunker.config import ChunkerConfig


class BaseStrategy(ABC):
    """
    Abstract base for chunking strategies.

    Subclasses implement `chunk()` which receives a list of sentences
    and returns a list of chunk strings. The core orchestrator handles
    sentence segmentation, metadata, and post-processing.
    """

    def __init__(self, config: ChunkerConfig):
        self.config = config

    @abstractmethod
    def chunk(self, sentences: List[str], text: str = "") -> List[str]:
        """
        Split content into chunks.

        Parameters
        ----------
        sentences : List[str]
            Pre-segmented sentences from the language processor.
        text : str
            Original full text (some strategies need it).

        Returns
        -------
        List[str]
            List of chunk text strings.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable strategy name in snake_case."""
        import re as _re
        raw = self.__class__.__name__.replace("Strategy", "")
        # CamelCase → snake_case
        return _re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", raw).lower()

    # ── Shared Utilities ──────────────────────────────────────

    def _estimate_tokens(self, text: str) -> int:
        """Quick token estimate (≈ 3.0 chars / token for mixed EN/AR)."""
        return max(1, int(len(text) / 3.0))

    def _apply_overlap(self, chunks: List[str], sentences: List[str]) -> List[str]:
        """
        Apply sentence-level overlap between adjacent chunks.

        Uses the configured overlap token count to pull trailing sentences
        from the previous chunk into the start of the next chunk.
        """
        if self.config.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]

            # Find overlap sentences from end of previous chunk
            prev_sentences = self._split_to_sentences_fast(prev_chunk)
            overlap_text = ""
            overlap_tokens = 0

            for sent in reversed(prev_sentences):
                sent_tokens = self._estimate_tokens(sent)
                if overlap_tokens + sent_tokens > self.config.chunk_overlap:
                    break
                overlap_text = sent + " " + overlap_text
                overlap_tokens += sent_tokens

            if overlap_text.strip():
                overlapped.append(overlap_text.strip() + " " + current_chunk)
            else:
                overlapped.append(current_chunk)

        return overlapped

    def _split_to_sentences_fast(self, text: str) -> List[str]:
        """Quick sentence split for overlap computation."""
        import re
        parts = re.split(r"(?<=[.!?؟])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge chunks that are below min_chunk_size with their neighbor."""
        if not chunks:
            return chunks

        merged = []
        buffer = ""

        for chunk in chunks:
            if buffer:
                candidate = buffer + " " + chunk
            else:
                candidate = chunk

            if self._estimate_tokens(candidate) >= self.config.min_chunk_size:
                merged.append(candidate)
                buffer = ""
            else:
                buffer = candidate

        # Handle remaining buffer
        if buffer:
            if merged:
                merged[-1] = merged[-1] + " " + buffer
            else:
                merged.append(buffer)

        return merged

    def _split_oversized_chunks(self, chunks: List[str]) -> List[str]:
        """Split chunks that exceed max_chunk_size."""
        result = []
        for chunk in chunks:
            if self._estimate_tokens(chunk) <= self.config.max_chunk_size:
                result.append(chunk)
            else:
                # Split at sentence boundaries
                sentences = self._split_to_sentences_fast(chunk)
                current = ""
                for sent in sentences:
                    candidate = (current + " " + sent).strip() if current else sent
                    if self._estimate_tokens(candidate) > self.config.max_chunk_size:
                        if current:
                            result.append(current)
                        current = sent
                    else:
                        current = candidate
                if current:
                    result.append(current)

        return result

    def _validate_chunks(self, chunks: List[str]) -> List[str]:
        """Apply min/max size constraints."""
        chunks = self._merge_small_chunks(chunks)
        chunks = self._split_oversized_chunks(chunks)
        return [c for c in chunks if c.strip()]
