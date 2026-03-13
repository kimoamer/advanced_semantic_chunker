"""
Sentence-aware chunking strategy.

Recommended by Arabic RAG benchmarks as the best strategy for
structured and semantically dense Arabic datasets (arxiv research).

Groups complete sentences to create chunks, never breaks mid-sentence.
"""

from __future__ import annotations

from typing import List

from chunker.config import ChunkerConfig
from chunker.strategies.base import BaseStrategy


class SentenceStrategy(BaseStrategy):
    """
    Sentence-aware chunker — groups complete sentences into chunks.

    Algorithm:
      1. Receive pre-segmented sentences
      2. Accumulate sentences until chunk_size is reached
      3. Start a new chunk (never splitting mid-sentence)
      4. Apply overlap and validation
    """

    def chunk(self, sentences: List[str], text: str = "") -> List[str]:
        """Group sentences into right-sized chunks."""
        if not sentences:
            return []

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)

            # If this single sentence exceeds max, it becomes its own chunk
            if sent_tokens > self.config.max_chunk_size:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0
                chunks.append(sent)
                continue

            # Check if adding this sentence would exceed chunk_size
            if current_tokens + sent_tokens > self.config.chunk_size:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                current_sentences = [sent]
                current_tokens = sent_tokens
            else:
                current_sentences.append(sent)
                current_tokens += sent_tokens

        # Flush remaining
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        chunks = self._validate_chunks(chunks)
        chunks = self._apply_overlap(chunks, sentences)

        return chunks
