"""
Semantic chunking strategy — the core algorithm.

Based on research from Google, IBM, LangChain, Pinecone:
  1. Embed each sentence
  2. Compute cosine distances between consecutive pairs
  3. Detect breakpoints using threshold (percentile / std-dev / IQR / gradient)
  4. Group sentences between breakpoints into chunks
  5. Apply overlap and validation

Hard rules (structure-led chunking, RAG best practice):
  • Every ## / ### heading MUST start a new chunk — never mid-chunk.
  • Chunks are built from paragraphs / sentences — never split mid-sentence.
  • Tiny tail chunks (< min_chunk_size ≈ 80-100 tokens) are merged into
    the previous chunk *within the same heading section*.
  • Overlap defaults to 0; never crosses heading boundaries.
"""

from __future__ import annotations

import re as _re
from typing import List, Optional

import numpy as np

from chunker.config import ChunkerConfig, ThresholdType
from chunker.embeddings.base import BaseEmbeddingProvider
from chunker.strategies.base import BaseStrategy


# ────────────────────────────────────────────────────────────────
# Pre-compiled patterns
# ────────────────────────────────────────────────────────────────

# Detect markdown headings (## through ######)
_HEADING_RE = _re.compile(r"^#{2,6}\s", _re.MULTILINE)

# Inline heading that does NOT already start on its own line.
# Matches " ## " or " ### " surrounded by non-newline content,
# but NOT inside fenced code blocks.
_INLINE_HEADING_RE = _re.compile(r"(?<!\n)( #{2,6} )")

# Fenced code-block boundaries
_CODE_FENCE_RE = _re.compile(r"^```", _re.MULTILINE)


class SemanticStrategy(BaseStrategy):
    """
    Embedding-based semantic chunking with heading-led hard boundaries.

    Algorithm:
      0. Normalise headings (ensure each starts on its own line)
      1. Generate embeddings for each sentence group
      2. Compute cosine distance between adjacent groups
      3. Detect semantic breakpoints
      4. Force heading positions as additional breakpoints
      5. Build chunks from groups
      6. Merge tiny tails (same heading section only)
      7. Split oversized chunks (sentence-boundary only)
      8. Apply overlap (never across headings)
    """

    def __init__(
        self,
        config: ChunkerConfig,
        embedding_provider: BaseEmbeddingProvider,
        cache_manager=None,
        metrics_collector=None,
    ):
        super().__init__(config)
        self.embedding_provider = embedding_provider
        self.cache_manager = cache_manager
        self.metrics_collector = metrics_collector

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def chunk(self, sentences: List[str], text: str = "") -> List[str]:
        """Split sentences into semantically coherent chunks."""
        if not sentences:
            return []

        # ── Step 0: Normalise headings within each sentence ──
        sentences = self._normalise_heading_sentences(sentences)

        if len(sentences) == 1:
            return sentences

        # ── Step 1: Group sentences for embedding ──
        sentence_groups = self._create_sentence_groups(sentences)

        # ── Step 2: Generate embeddings ──
        group_texts = [" ".join(g) for g in sentence_groups]
        
        # Use cache if available
        if self.cache_manager and self.config.enable_embedding_cache:
            embeddings = []
            texts_to_embed = []
            text_indices = []
            cache_hits = 0
            cache_misses = 0
            
            # Check cache for each text
            for i, text in enumerate(group_texts):
                cached_embedding = self.cache_manager.get_embedding(
                    text, self.config.embedding_model
                )
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                    cache_hits += 1
                else:
                    # Cache miss - need to embed this text
                    embeddings.append(None)  # Placeholder
                    texts_to_embed.append(text)
                    text_indices.append(i)
                    cache_misses += 1
            
            # Embed texts that weren't in cache
            if texts_to_embed:
                import time
                start_time = time.time()
                new_embeddings = self.embedding_provider.embed(texts_to_embed)
                duration_ms = (time.time() - start_time) * 1000
                
                # Record embedding call metrics
                if self.metrics_collector:
                    self.metrics_collector.record_embedding_call(
                        model=self.config.embedding_model,
                        batch_size=len(texts_to_embed),
                        duration_ms=duration_ms,
                        cache_hit=False
                    )
                
                # Fill in the placeholders and cache the new embeddings
                for idx, text, embedding in zip(text_indices, texts_to_embed, new_embeddings):
                    embeddings[idx] = embedding
                    self.cache_manager.set_embedding(
                        text, self.config.embedding_model, embedding
                    )
            
            # Record cache hits if any
            if cache_hits > 0 and self.metrics_collector:
                # Record a single "call" representing cache hits
                self.metrics_collector.record_embedding_call(
                    model=self.config.embedding_model,
                    batch_size=cache_hits,
                    duration_ms=0.0,  # Cache hits are instant
                    cache_hit=True
                )
            
            # Convert list to numpy array
            import numpy as np
            embeddings = np.array(embeddings)
        else:
            # Cache disabled - embed directly
            import time
            start_time = time.time()
            embeddings = self.embedding_provider.embed(group_texts)
            duration_ms = (time.time() - start_time) * 1000
            
            # Record embedding call metrics
            if self.metrics_collector:
                self.metrics_collector.record_embedding_call(
                    model=self.config.embedding_model,
                    batch_size=len(group_texts),
                    duration_ms=duration_ms,
                    cache_hit=False
                )

        # ── Step 3: Compute pairwise cosine distances ──
        distances = BaseEmbeddingProvider.pairwise_cosine_distances(embeddings)

        if len(distances) == 0:
            return [" ".join(sentences)]

        # ── Step 4: Detect semantic breakpoints ──
        breakpoints = self._detect_breakpoints(distances)

        # ── Step 4b: Force heading positions as hard breakpoints ──
        breakpoints = self._inject_heading_breakpoints(
            sentence_groups, breakpoints
        )

        # ── Step 5: Build chunks from sentence groups ──
        chunks = self._build_chunks(sentence_groups, breakpoints)

        # ── Step 6: Merge tiny tail chunks (heading-aware) ──
        chunks = self._merge_small_chunks_heading_aware(chunks)

        # ── Step 7: Split oversized chunks (sentence boundary only) ──
        chunks = self._split_oversized_heading_aware(chunks)

        # ── Step 8: Apply overlap (never across headings) ──
        chunks = self._apply_overlap_heading_aware(chunks, sentences)

        return [c for c in chunks if c.strip()]

    # ──────────────────────────────────────────────────────────
    # Step 0: Heading normalisation
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_heading_sentences(sentences: List[str]) -> List[str]:
        """
        Normalise headings so each always starts on its own line.

        If a sentence contains an inline heading (e.g. "some text ## Title"),
        split it so the heading becomes a separate sentence.  Code blocks
        are left untouched.
        """
        result: List[str] = []
        for sent in sentences:
            # Skip anything that looks like it is inside a code fence
            if sent.strip().startswith("```"):
                result.append(sent)
                continue

            # If the sentence itself IS a heading, keep as-is
            if _HEADING_RE.match(sent.strip()):
                result.append(sent.strip())
                continue

            # Check for inline headings  ("... ## Title ...")
            # Split around them, emitting each fragment as its own sentence
            parts = _re.split(r"(?:^|\n| )(#{2,6}\s)", sent)
            if len(parts) <= 1:
                result.append(sent)
                continue

            # Reassemble: parts alternate between content and heading-markers
            buf = ""
            i = 0
            while i < len(parts):
                p = parts[i]
                if _re.match(r"^#{2,6}\s$", p):
                    # Flush preceding content
                    if buf.strip():
                        result.append(buf.strip())
                        buf = ""
                    # The heading marker needs the next part as its title
                    title = parts[i + 1] if i + 1 < len(parts) else ""
                    result.append((p + title).strip())
                    i += 2
                else:
                    buf += p
                    i += 1
            if buf.strip():
                result.append(buf.strip())

        return result

    # ──────────────────────────────────────────────────────────
    # Heading detection
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _is_heading(text: str) -> bool:
        """Return True if *text* looks like a markdown ## – ###### heading."""
        return bool(_HEADING_RE.match(text.strip()))

    # ──────────────────────────────────────────────────────────
    # Step 1: Sentence grouping
    # ──────────────────────────────────────────────────────────

    def _create_sentence_groups(
        self, sentences: List[str], min_group_tokens: int = 30
    ) -> List[List[str]]:
        """
        Group short sentences into embedding units (≥ min_group_tokens).

        HARD RULES:
          • A heading sentence ALWAYS starts a new group.
          • A heading is NEVER merged with non-heading sentences.
        """
        groups: List[List[str]] = []
        current_group: List[str] = []
        current_tokens = 0

        for sent in sentences:
            # ── heading ⇒ flush current, isolate heading ──
            if self._is_heading(sent):
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_tokens = 0
                groups.append([sent])  # heading lives alone
                continue

            tokens = self._estimate_tokens(sent)
            current_group.append(sent)
            current_tokens += tokens

            if current_tokens >= min_group_tokens:
                groups.append(current_group)
                current_group = []
                current_tokens = 0

        if current_group:
            if groups:
                groups[-1].extend(current_group)
            else:
                groups.append(current_group)

        return groups

    # ──────────────────────────────────────────────────────────
    # Step 4: Breakpoint detection
    # ──────────────────────────────────────────────────────────

    def _detect_breakpoints(self, distances: np.ndarray) -> List[int]:
        """
        Detect chunk boundary indices from cosine distances.

        Returns indices where a break should occur AFTER that group.
        """
        if len(distances) == 0:
            return []

        threshold = self._compute_threshold(distances)
        return [i for i, d in enumerate(distances) if d >= threshold]

    def _compute_threshold(self, distances: np.ndarray) -> float:
        """Compute the breakpoint threshold based on the configured method."""
        if self.config.threshold_type == ThresholdType.PERCENTILE:
            return float(np.percentile(distances, self.config.threshold_amount))

        elif self.config.threshold_type == ThresholdType.STANDARD_DEVIATION:
            mean = float(np.mean(distances))
            std = float(np.std(distances))
            return mean + self.config.threshold_amount * std

        elif self.config.threshold_type == ThresholdType.INTERQUARTILE:
            q1 = float(np.percentile(distances, 25))
            q3 = float(np.percentile(distances, 75))
            iqr = q3 - q1
            return q3 + self.config.threshold_amount * iqr

        elif self.config.threshold_type == ThresholdType.GRADIENT:
            gradient = np.gradient(distances)
            mean_grad = float(np.mean(np.abs(gradient)))
            return mean_grad + self.config.threshold_amount

        else:
            return float(np.percentile(distances, 90))

    # ──────────────────────────────────────────────────────────
    # Step 4b: Force heading breakpoints
    # ──────────────────────────────────────────────────────────

    def _inject_heading_breakpoints(
        self,
        sentence_groups: List[List[str]],
        breakpoints: List[int],
    ) -> List[int]:
        """
        Ensure every group containing a heading has a break BEFORE it.

        Breakpoint at index i means: split AFTER sentence_group[i].
        So to start a new chunk at group N, we add breakpoint N-1.
        """
        bp_set = set(breakpoints)

        for idx, group in enumerate(sentence_groups):
            if idx == 0:
                continue  # first group always starts a chunk
            if any(self._is_heading(s) for s in group):
                bp_set.add(idx - 1)

        return sorted(bp_set)

    # ──────────────────────────────────────────────────────────
    # Step 5: Build chunks
    # ──────────────────────────────────────────────────────────

    def _build_chunks(
        self, sentence_groups: List[List[str]], breakpoints: List[int]
    ) -> List[str]:
        """
        Assemble chunks from sentence groups and breakpoint indices.

        Breakpoint at index i means: split AFTER sentence_group[i].
        """
        if not breakpoints:
            all_sents: List[str] = []
            for g in sentence_groups:
                all_sents.extend(g)
            return [" ".join(all_sents)]

        chunks: List[str] = []
        start = 0

        for bp in breakpoints:
            chunk_sents: List[str] = []
            for g in sentence_groups[start : bp + 1]:
                chunk_sents.extend(g)
            if chunk_sents:
                chunks.append(" ".join(chunk_sents))
            start = bp + 1

        # Remainder after last breakpoint
        remaining: List[str] = []
        for g in sentence_groups[start:]:
            remaining.extend(g)
        if remaining:
            chunks.append(" ".join(remaining))

        return chunks

    # ──────────────────────────────────────────────────────────
    # Step 6: Merge tiny tails (heading-aware)
    # ──────────────────────────────────────────────────────────

    def _merge_small_chunks_heading_aware(self, chunks: List[str]) -> List[str]:
        """
        Merge chunks below min_chunk_size into the previous chunk,
        but NEVER merge across heading boundaries.

        A chunk that *starts with* a heading is always kept separate.
        """
        if not chunks:
            return chunks

        merged: List[str] = [chunks[0]]

        for i in range(1, len(chunks)):
            cur = chunks[i]
            prev = merged[-1]
            cur_tokens = self._estimate_tokens(cur)

            # Never merge a chunk that starts with a heading into prev
            cur_starts_heading = self._is_heading(cur.split("\n")[0])

            # Never merge into a prev that already hit max
            combined_tokens = self._estimate_tokens(prev + " " + cur)

            if (
                cur_tokens < self.config.min_chunk_size
                and not cur_starts_heading
                and combined_tokens <= self.config.max_chunk_size
            ):
                merged[-1] = prev + " " + cur
            else:
                merged.append(cur)

        return merged

    # ──────────────────────────────────────────────────────────
    # Step 7: Split oversized chunks (heading-aware)
    # ──────────────────────────────────────────────────────────

    def _split_on_headings(self, text: str) -> List[str]:
        """
        Pre-split text so that every heading line starts its own fragment.

        E.g. "foo bar ## Title baz" → ["foo bar", "## Title baz"]
        """
        # Split around lines that start with ##..######
        parts = _re.split(r"(?:(?<=\n)|(?<=^))(#{2,6}\s)", text, flags=_re.MULTILINE)
        fragments: List[str] = []
        i = 0
        while i < len(parts):
            p = parts[i]
            if _re.match(r"^#{2,6}\s$", p):
                # Re-attach the heading marker to the text that follows it
                title = parts[i + 1] if i + 1 < len(parts) else ""
                fragments.append(p + title)
                i += 2
            else:
                if p.strip():
                    fragments.append(p.strip())
                i += 1
        return fragments

    def _split_oversized_heading_aware(self, chunks: List[str]) -> List[str]:
        """
        Split chunks exceeding max_chunk_size at sentence boundaries.

        If a heading is encountered while splitting, it always starts
        a new sub-chunk (headings never appear mid-chunk).
        """
        result: List[str] = []
        for chunk in chunks:
            if self._estimate_tokens(chunk) <= self.config.max_chunk_size:
                result.append(chunk)
                continue

            # First, pre-split on heading boundaries
            fragments = self._split_on_headings(chunk)

            for frag in fragments:
                if self._estimate_tokens(frag) <= self.config.max_chunk_size:
                    result.append(frag)
                    continue

                # Further split this non-heading fragment at sentence boundaries
                sents = self._split_to_sentences_fast(frag)
                current = ""
                for sent in sents:
                    # If this sentence is a heading → flush and start new
                    if self._is_heading(sent):
                        if current.strip():
                            result.append(current.strip())
                        current = sent
                        continue

                    candidate = (current + " " + sent).strip() if current else sent
                    if self._estimate_tokens(candidate) > self.config.max_chunk_size:
                        if current.strip():
                            result.append(current.strip())
                        current = sent
                    else:
                        current = candidate

                if current.strip():
                    result.append(current.strip())

        return result

    # ──────────────────────────────────────────────────────────
    # Step 8: Overlap (heading-aware, default 0)
    # ──────────────────────────────────────────────────────────

    def _apply_overlap_heading_aware(
        self, chunks: List[str], sentences: List[str]
    ) -> List[str]:
        """
        Apply sentence-level overlap, but NEVER across heading boundaries.

        If chunk_overlap == 0 (default), this is a no-op.
        """
        if self.config.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped: List[str] = [chunks[0]]

        for i in range(1, len(chunks)):
            cur = chunks[i]
            prev = overlapped[-1]

            # HARD RULE: no overlap if current chunk starts with a heading
            # or if the previous chunk ends with / contains a different heading
            cur_starts_heading = self._is_heading(cur.split("\n")[0])
            if cur_starts_heading:
                overlapped.append(cur)
                continue

            # Pull trailing sentences from prev
            prev_sentences = self._split_to_sentences_fast(prev)
            overlap_parts: List[str] = []
            overlap_tokens = 0

            for sent in reversed(prev_sentences):
                # Never pull a heading sentence into overlap
                if self._is_heading(sent):
                    break
                st = self._estimate_tokens(sent)
                if overlap_tokens + st > self.config.chunk_overlap:
                    break
                overlap_parts.insert(0, sent)
                overlap_tokens += st

            if overlap_parts:
                overlapped.append(" ".join(overlap_parts) + " " + cur)
            else:
                overlapped.append(cur)

        return overlapped

    # ──────────────────────────────────────────────────────────
    # Coherence scoring (optional, expensive)
    # ──────────────────────────────────────────────────────────

    def compute_coherence_scores(self, chunks: List[str]) -> List[float]:
        """
        Compute internal semantic coherence for each chunk.

        Higher score = more semantically coherent.
        """
        scores: List[float] = []
        for chunk in chunks:
            sentences = self._split_to_sentences_fast(chunk)
            if len(sentences) <= 1:
                scores.append(1.0)
                continue

            embeddings = self.embedding_provider.embed(sentences)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normed = embeddings / norms
            sim_matrix = normed @ normed.T

            n = len(sentences)
            total = np.sum(sim_matrix) - np.trace(sim_matrix)
            avg_sim = total / (n * (n - 1)) if n > 1 else 1.0
            scores.append(float(avg_sim))

        return scores
