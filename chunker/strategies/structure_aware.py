"""
Structure-aware chunking strategy — the recommended default.

Implements research-backed chunking for RAG:

  1. Headings are HARD boundaries — every heading starts a new chunk.
  2. Chunks are built ONLY within a single section (never cross headings).
  3. Target 180-300 tokens per chunk, hard max ~350.
  4. Never split mid-sentence.
  5. Overlap defaults to 0; never crosses headings.
  6. Tiny tails (< min_chunk_size) merge into previous chunk (same section).
  7. Rich metadata per chunk: heading_path, hierarchy_level, char offsets.

Sources:
  - Databricks Community best practices
  - Haystack split_threshold pattern
  - Unstructured.io chunking guidance
  - arXiv systematic study on chunk size impact
"""

from __future__ import annotations

import re
from typing import List

from chunker.config import ChunkerConfig
from chunker.document_tree import (
    DocumentSection,
    NodeType,
    build_sections,
    parse_document_tree,
)
from chunker.strategies.base import BaseStrategy


class StructureAwareStrategy(BaseStrategy):
    """
    Structure-first chunker with heading-based hard boundaries.

    Pipeline:
      0. Parse document into a tree of typed nodes.
      1. Group nodes into heading-delimited sections.
      2. For each section, collect atomic text units (sentences / structural blocks).
      3. Pack units into chunks (respecting max_chunk_size on the JOINED text).
      4. Merge tiny tail fragments within the same section.
      5. Optionally apply sentence-level overlap (never across headings).
    """

    def __init__(self, config: ChunkerConfig):
        super().__init__(config)

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def chunk(self, sentences: List[str], text: str = "") -> List[str]:
        """Entry point — parses structure then packs chunks within sections."""
        source = text if text else " ".join(sentences)
        if not source.strip():
            return []

        # Step 0-1: Parse → sections
        nodes = parse_document_tree(source)
        sections = build_sections(nodes)

        # Step 2-3: Build chunks per section
        raw_chunks: List[dict] = []
        for section in sections:
            raw_chunks.extend(self._chunk_section(section))

        # Step 4: Merge tiny tails (same section only)
        raw_chunks = self._merge_small_tails(raw_chunks)

        # Step 5: Overlap (never across headings)
        if self.config.chunk_overlap > 0:
            raw_chunks = self._apply_controlled_overlap(raw_chunks)

        # Expose metadata for the core orchestrator
        self._section_metadata = raw_chunks
        return [c["text"] for c in raw_chunks]

    @property
    def section_metadata(self) -> List[dict]:
        """Metadata dicts from last chunk() call."""
        return getattr(self, "_section_metadata", [])

    # ──────────────────────────────────────────────────────────────
    # Step 2-3: Build chunks for ONE section
    # ──────────────────────────────────────────────────────────────

    def _chunk_section(self, section: DocumentSection) -> List[dict]:
        """
        Build chunks within a single heading-delimited section.

        Even if the section has no body content (heading-only), we still
        emit that heading as its own chunk so it is never silently dropped
        and never drifts into an adjacent section's chunk.
        """
        meta_base = {
            "heading_path": list(section.heading_path),
            "section_title": section.heading_title,
            "hierarchy_level": section.heading_level,
            "start_char": section.start_char,
        }

        # -- Collect atomic units --
        units: List[str] = []

        # Always lead with the heading text itself (if present)
        heading_text = getattr(section, "heading_text", "")
        if heading_text:
            units.append(heading_text)

        for node in section.nodes:
            if node.node_type == NodeType.PARAGRAPH:
                for sent in self._split_sentences(node.content):
                    s = sent.strip()
                    if s:
                        units.append(s)
            elif node.node_type == NodeType.LIST:
                # Keep the whole list as one unit if it fits, else split
                if self._tok(node.content) <= self.config.max_chunk_size:
                    units.append(node.content)
                else:
                    for item in node.list_items:
                        if item.strip():
                            units.append(item.strip())
            else:
                # CODE, TABLE — keep intact as a single unit
                if node.content.strip():
                    units.append(node.content.strip())

        if not units:
            return []

        # -- Pack units into chunks --
        return self._pack_units(units, meta_base)

    # ──────────────────────────────────────────────────────────────
    # Packing: the core sizing loop
    # ──────────────────────────────────────────────────────────────

    def _pack_units(self, units: List[str], meta_base: dict) -> List[dict]:
        """
        Pack a list of text strings into chunks that respect max_chunk_size.

        KEY DIFFERENCE from previous version:
          Token estimation is computed on the *actual joined text* after
          every append — not by summing individual unit estimates.  This
          prevents the silent size creep caused by join separators.
        """
        chunks: List[dict] = []
        buf: List[str] = []

        def _flush():
            if not buf:
                return
            joined = "\n\n".join(buf)
            chunks.append({"text": joined, **meta_base})

        for unit in units:
            # Predict the joined size if we add this unit
            candidate = buf + [unit]
            candidate_text = "\n\n".join(candidate)
            candidate_tokens = self._tok(candidate_text)

            if candidate_tokens <= self.config.max_chunk_size:
                # Fits — accumulate
                buf = candidate
            else:
                # Doesn't fit — flush current, start new buffer
                _flush()
                buf = [unit]

        _flush()
        return chunks

    # ──────────────────────────────────────────────────────────────
    # Step 4: Merge tiny tails (Haystack split_threshold)
    # ──────────────────────────────────────────────────────────────

    def _merge_small_tails(self, chunks: List[dict]) -> List[dict]:
        """
        If a chunk is below min_chunk_size tokens and belongs to the
        same section as its predecessor, merge it into the predecessor
        (as long as the combined result stays under max_chunk_size).
        """
        if len(chunks) <= 1:
            return chunks

        merged: List[dict] = [chunks[0]]

        for i in range(1, len(chunks)):
            cur = chunks[i]
            prev = merged[-1]

            cur_tokens = self._tok(cur["text"])
            same_section = cur.get("heading_path") == prev.get("heading_path")

            if (cur_tokens < self.config.min_chunk_size
                    and same_section
                    and self._tok(prev["text"] + "\n\n" + cur["text"]) <= self.config.max_chunk_size):
                merged[-1] = {**prev, "text": prev["text"] + "\n\n" + cur["text"]}
            else:
                merged.append(cur)

        return merged

    # ──────────────────────────────────────────────────────────────
    # Step 5: Controlled overlap (never across headings)
    # ──────────────────────────────────────────────────────────────

    def _apply_controlled_overlap(self, chunks: List[dict]) -> List[dict]:
        """Sentence-level overlap; skip when heading_path differs."""
        if len(chunks) <= 1 or self.config.chunk_overlap <= 0:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            cur = chunks[i]
            prev = result[-1]

            # HARD RULE: never overlap across different sections
            if cur.get("heading_path") != prev.get("heading_path"):
                result.append(cur)
                continue

            # Pull trailing sentences from prev
            prev_sentences = self._split_sentences(prev["text"])
            overlap_parts: List[str] = []
            overlap_tokens = 0
            for sent in reversed(prev_sentences):
                st = self._tok(sent)
                if overlap_tokens + st > self.config.chunk_overlap:
                    break
                overlap_parts.insert(0, sent)
                overlap_tokens += st

            if overlap_parts:
                result.append({**cur, "text": " ".join(overlap_parts) + " " + cur["text"]})
            else:
                result.append(cur)

        return result

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _tok(self, text: str) -> int:
        """Estimate token count — delegates to base._estimate_tokens."""
        return self._estimate_tokens(text)

    def _split_sentences(self, text: str) -> List[str]:
        """Split on sentence-ending punctuation (EN + AR)."""
        parts = re.split(r"(?<=[.!?؟؛])\s+", text)
        return [p.strip() for p in parts if p.strip()]
