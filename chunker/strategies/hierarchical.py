"""
Hierarchical chunking strategy.

Creates multi-level chunks (sections → paragraphs → sentences)
that can be merged at retrieval time (LlamaIndex AutoMergingRetriever pattern).

Best for very large, structured documents like research papers,
technical manuals, and legal documents.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from chunker.config import ChunkerConfig
from chunker.strategies.base import BaseStrategy


class HierarchicalStrategy(BaseStrategy):
    """
    Hierarchical multi-level chunker.

    Algorithm:
      1. Detect document sections (by headers / double newlines)
      2. Within each section, split into paragraph-level chunks
      3. Within each paragraph, ensure sentence boundaries are respected
      4. Each chunk carries its hierarchy level for later merging
    """

    def chunk(self, sentences: List[str], text: str = "") -> List[str]:
        """
        Build hierarchical chunks.

        Returns flattened list of chunks. Hierarchy info is tracked
        via metadata in the core orchestrator.
        """
        source = text if text else " ".join(sentences)

        if not source.strip():
            return []

        # ── Level 1: Split into sections ──
        sections = self._split_sections(source)

        # ── Level 2: Split sections into paragraphs ──
        all_chunks: List[str] = []
        for section_title, section_text in sections:
            paragraphs = self._split_paragraphs(section_text)

            for para in paragraphs:
                para_tokens = self._estimate_tokens(para)

                if para_tokens <= self.config.chunk_size:
                    # Paragraph fits → it's a chunk
                    chunk = para.strip()
                    if section_title:
                        chunk = f"[{section_title}] {chunk}"
                    if chunk.strip():
                        all_chunks.append(chunk.strip())
                else:
                    # ── Level 3: Split paragraph into sentence groups ──
                    para_sentences = self._split_to_sentences_fast(para)
                    current = ""
                    current_tokens = 0

                    for sent in para_sentences:
                        sent_tokens = self._estimate_tokens(sent)
                        if current_tokens + sent_tokens > self.config.chunk_size:
                            if current.strip():
                                chunk_text = current.strip()
                                if section_title:
                                    chunk_text = f"[{section_title}] {chunk_text}"
                                all_chunks.append(chunk_text)
                            current = sent
                            current_tokens = sent_tokens
                        else:
                            current = (current + " " + sent).strip()
                            current_tokens += sent_tokens

                    if current.strip():
                        chunk_text = current.strip()
                        if section_title:
                            chunk_text = f"[{section_title}] {chunk_text}"
                        all_chunks.append(chunk_text)

        all_chunks = self._validate_chunks(all_chunks)
        all_chunks = self._apply_overlap(all_chunks, sentences)

        return all_chunks

    def _split_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Split document into (title, content) section pairs.

        Detects Markdown headers (# ... ######) and also treats
        triple-newline separators as section boundaries.
        """
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

        sections: List[Tuple[str, str]] = []
        last_end = 0
        last_title = ""

        for match in header_pattern.finditer(text):
            # Content before this header belongs to previous section
            content = text[last_end : match.start()].strip()
            if content:
                sections.append((last_title, content))

            last_title = match.group(2).strip()
            last_end = match.end()

        # Remaining content
        content = text[last_end:].strip()
        if content:
            sections.append((last_title, content))

        # If no headers found, split on triple newlines
        if len(sections) <= 1 and sections:
            parts = re.split(r"\n{3,}", sections[0][1])
            if len(parts) > 1:
                sections = [("", part.strip()) for part in parts if part.strip()]

        return sections if sections else [("", text)]

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs (double newline separated)."""
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]
