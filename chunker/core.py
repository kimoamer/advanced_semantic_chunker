"""
SemanticChunker — The main orchestrator.

This is the primary public API. It:
  1. Detects document language (EN/AR/mixed)
  2. Normalizes text
  3. Segments into sentences using language-appropriate pipeline
  4. Dispatches to the configured chunking strategy
  5. Enriches output with metadata
  6. Returns a list of Chunk objects ready for embedding / indexing
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Generator, List, Optional, Union

from chunker.config import (
    ChunkerConfig,
    EmbeddingProvider,
    StrategyType,
)
from chunker.embeddings.base import BaseEmbeddingProvider
from chunker.error_handler import ErrorHandler
from chunker.exceptions import (
    ChunkerException,
    EmbeddingError,
    InputValidationError,
    LanguageDetectionError,
)
from chunker.language.detector import LanguageDetector
from chunker.language.normalizer import TextNormalizer
from chunker.lazy_load import LazyLoadManager
from chunker.metrics_collector import ChunkerMetrics, MetricsCollector
from chunker.models import Chunk, ChunkMetadata, DocumentInfo, Language
from chunker.strategies.base import BaseStrategy
from chunker.structured_logger import StructuredLogger
from chunker.utils import estimate_tokens, count_sentences

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Powerful bilingual (EN/AR) semantic document chunker.

    Usage
    -----
    ```python
    from chunker import SemanticChunker, ChunkerConfig, StrategyType

    # Quick start with defaults (semantic strategy, BGE-M3 embeddings)
    chunker = SemanticChunker()
    chunks = chunker.chunk("Your document text here...")

    # Custom configuration
    config = ChunkerConfig(
        strategy=StrategyType.SEMANTIC,
        chunk_size=512,
        chunk_overlap=64,
        threshold_type=ThresholdType.PERCENTILE,
        threshold_amount=90.0,
        embedding_model="BAAI/bge-m3",
    )
    chunker = SemanticChunker(config)
    chunks = chunker.chunk(text)

    # Access chunk data
    for chunk in chunks:
        print(chunk.text)
        print(chunk.metadata.language)
        print(chunk.metadata.token_count)
    ```

    Strategies
    ----------
    - `StrategyType.STRUCTURE_AWARE` — Structure-first with heading hard boundaries (default)
    - `StrategyType.SEMANTIC`        — Embedding-based similarity splitting
    - `StrategyType.RECURSIVE`       — Hierarchical separator-based splitting
    - `StrategyType.SENTENCE`        — Complete sentence grouping
    - `StrategyType.FIXED`           — Token/character-based splitting
    - `StrategyType.HIERARCHICAL`    — Multi-level section-aware splitting
    - `StrategyType.AGENTIC`         — LLM-driven dynamic splitting (requires API key)
    """

    def __init__(
        self,
        config: Optional[ChunkerConfig] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
    ):
        self.config = config or ChunkerConfig()
        self.config.validate()

        # Language processing
        self._detector = LanguageDetector()
        self._normalizer = TextNormalizer()
        
        # Lazy load manager for expensive resources
        self._lazy_load_manager = LazyLoadManager(self.config)

        # Error handler for graceful degradation
        self._error_handler = ErrorHandler(
            enable_fallbacks=self.config.enable_fallbacks,
            log_fallbacks=True,
        )

        # Structured logger for observability
        self._structured_logger = None
        if self.config.enable_structured_logging:
            log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
            self._structured_logger = StructuredLogger(
                name="chunker",
                level=log_level,
                structured=True
            )

        # Metrics collector for monitoring
        self._metrics_collector = None
        if self.config.enable_metrics:
            self._metrics_collector = MetricsCollector(enabled=True)

        # Cache manager for embeddings and language detection
        self._cache_manager = None
        if self.config.enable_embedding_cache or self.config.enable_language_cache:
            from chunker.cache import CacheManager
            self._cache_manager = CacheManager(
                embedding_cache_size=self.config.embedding_cache_size,
                lang_detect_cache_size=self.config.language_cache_size,
                persist_to_disk=self.config.persist_cache_to_disk,
                cache_dir=self.config.cache_dir,
            )
            # Load cache from disk if persistence is enabled
            if self.config.persist_cache_to_disk:
                try:
                    self._cache_manager.load_from_disk()
                    if self.config.verbose:
                        logger.info("Loaded cache from disk")
                except (IOError, FileNotFoundError) as e:
                    if self.config.verbose:
                        logger.info(f"No existing cache found: {e}")

        # Embedding provider (lazy-loaded for non-semantic strategies)
        self._embedding_provider = embedding_provider
        self._strategy: Optional[BaseStrategy] = None

    # ── Public API ────────────────────────────────────────────

    def chunk(
        self,
        text: str,
        source_file: str = "",
        document_id: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Chunk a document into semantically coherent pieces.

        Parameters
        ----------
        text : str
            The document text to chunk.
        source_file : str
            Optional source filename for metadata.
        document_id : str
            Optional document ID. Auto-generated if not provided.
        extra_metadata : dict
            Optional extra metadata to attach to every chunk.

        Returns
        -------
        List[Chunk]
            List of chunks with rich metadata.
            
        Raises
        ------
        InputValidationError
            If text is None or not a string
        ChunkerException
            If chunking fails and cannot be recovered
        """
        # Input validation
        if text is None:
            raise InputValidationError(
                "Text input cannot be None",
                context={"source_file": source_file, "document_id": document_id}
            )
        
        if not isinstance(text, str):
            raise InputValidationError(
                f"Text input must be a string, got {type(text).__name__}",
                context={"source_file": source_file, "document_id": document_id}
            )
        
        # Handle empty strings gracefully
        if not text or not text.strip():
            return []

        # Wrap the entire chunking process with error handling
        try:
            return self._chunk_internal(text, source_file, document_id, extra_metadata)
        except (InputValidationError, ChunkerException):
            # Re-raise our own exceptions
            raise
        except Exception as e:
            # Wrap unexpected exceptions with context
            raise self._error_handler.wrap_exception(
                e,
                context="document chunking",
                component="SemanticChunker",
                operation="chunk",
                source_file=source_file,
                document_id=document_id,
            )
    
    def _chunk_internal(
        self,
        text: str,
        source_file: str = "",
        document_id: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Internal chunking implementation with error handling."""
        start_time = time.time()

        # Log chunking start
        if self._structured_logger:
            self._structured_logger.log_chunking_start(
                doc_id=document_id or source_file or "unknown",
                text_length=len(text),
                strategy=self.config.strategy
            )

        # ── Step 1: Normalize text ──
        normalized = self._normalizer.normalize(text)

        # ── Step 2: Detect language with fallback ──
        if self.config.detect_language:
            language = self._detect_language_with_fallback(normalized)
            
            # Log language detection
            if self._structured_logger:
                self._structured_logger.log_language_detected(
                    doc_id=document_id or source_file or "unknown",
                    language=language
                )
        else:
            language = Language.UNKNOWN

        if self.config.verbose:
            logger.info(f"Detected language: {language.value}")

        # ── Step 3: Build document info ──
        doc_info = DocumentInfo(
            source_file=source_file,
            language=language,
            total_chars=len(normalized),
        )
        if document_id:
            doc_info.document_id = document_id

        # ── Step 4: Segment sentences ──
        sentences = self._segment_sentences(normalized, language)
        doc_info.total_sentences = len(sentences)
        doc_info.total_tokens = sum(
            estimate_tokens(s, language.value) for s in sentences
        )

        if self.config.verbose:
            logger.info(
                f"Segmented into {len(sentences)} sentences, "
                f"~{doc_info.total_tokens} tokens"
            )

        # ── Step 5: Get strategy with fallback for embedding failures ──
        strategy = self._get_strategy_with_fallback(language)

        # ── Step 6: Chunk ──
        chunk_start_time = time.time()
        chunk_texts = strategy.chunk(sentences, normalized)
        chunk_duration_ms = (time.time() - chunk_start_time) * 1000
        
        # Log timing for chunking operation
        if self._structured_logger:
            self._structured_logger.log_timing(
                operation="strategy_chunking",
                duration_ms=chunk_duration_ms,
                strategy=strategy.name,
                sentence_count=len(sentences)
            )

        if self.config.verbose:
            logger.info(
                f"Strategy '{strategy.name}' produced {len(chunk_texts)} chunks"
            )

        # ── Step 7: Build Chunk objects with metadata ──
        chunks = self._build_chunks(
            chunk_texts=chunk_texts,
            doc_info=doc_info,
            strategy_name=strategy.name,
            original_text=normalized,
            language=language,
            extra_metadata=extra_metadata,
        )

        # ── Step 8: Compute coherence scores if requested ──
        if self.config.compute_coherence_score and hasattr(strategy, "compute_coherence_scores"):
            scores = strategy.compute_coherence_scores(chunk_texts)
            for chunk, score in zip(chunks, scores):
                chunk.metadata.coherence_score = score

        elapsed = time.time() - start_time
        elapsed_ms = elapsed * 1000
        
        # Record metrics for this document
        if self._metrics_collector:
            self._metrics_collector.record_document_processed(
                doc_id=document_id or source_file or "unknown",
                language=language,
                strategy=self.config.strategy,
                duration_ms=elapsed_ms,
                chunk_count=len(chunks),
                token_count=doc_info.total_tokens
            )
        
        # Log chunking completion with timing
        if self._structured_logger:
            self._structured_logger.log_chunking_complete(
                doc_id=document_id or source_file or "unknown",
                chunk_count=len(chunks),
                duration_ms=elapsed_ms,
                strategy=self.config.strategy
            )
        
        if self.config.verbose:
            logger.info(f"Chunking completed in {elapsed:.2f}s")

        return chunks

    def chunk_batch(
        self,
        texts: List[str],
        source_files: Optional[List[str]] = None,
    ) -> List[List[Chunk]]:
        """
        Chunk multiple documents.

        Parameters
        ----------
        texts : List[str]
            List of document texts.
        source_files : List[str], optional
            Corresponding source filenames.

        Returns
        -------
        List[List[Chunk]]
            List of chunk lists, one per document.
        """
        results = []
        for i, text in enumerate(texts):
            src = source_files[i] if source_files and i < len(source_files) else ""
            results.append(self.chunk(text, source_file=src))
        return results

    def chunk_stream(
        self,
        text: str,
        source_file: str = "",
        document_id: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator[Chunk, None, None]:
        """
        Stream chunks one at a time from a document.

        Useful for very long documents (books, transcripts) where you want
        to process or store chunks incrementally without holding the full
        result list in memory.

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

        Yields
        ------
        Chunk
            Individual chunks in document order.

        Examples
        --------
        >>> chunker = SemanticChunker()
        >>> for chunk in chunker.chunk_stream(very_long_text):
        ...     store_in_vectordb(chunk)
        """
        chunks = self.chunk(
            text,
            source_file=source_file,
            document_id=document_id,
            extra_metadata=extra_metadata,
        )
        yield from chunks

    # ── Private Methods ───────────────────────────────────────

    def _detect_language_with_fallback(self, text: str) -> Language:
        """
        Detect language with fallback to English on failure.
        
        Tries to detect language using cache (if enabled) or detector.
        Falls back to English if detection fails.
        """
        def detect_primary():
            # Try to get language from cache
            if self._cache_manager and self.config.enable_language_cache:
                language = self._cache_manager.get_language(text)
                if language is None:
                    # Cache miss - detect and cache
                    if self._metrics_collector:
                        self._metrics_collector.record_cache_access("language", hit=False)
                    language = self._detector.detect(text)
                    self._cache_manager.set_language(text, language)
                else:
                    # Cache hit
                    if self._metrics_collector:
                        self._metrics_collector.record_cache_access("language", hit=True)
                return language
            else:
                # Cache disabled - detect directly
                return self._detector.detect(text)
        
        def fallback_to_english():
            # Record error in metrics
            if self._metrics_collector:
                self._metrics_collector.record_error(
                    error_type="LanguageDetectionError",
                    component="LanguageDetector",
                    recoverable=True
                )
            
            # Log fallback activation
            if self._structured_logger:
                self._structured_logger.log_fallback(
                    component="LanguageDetector",
                    reason="language detection failure",
                    fallback_action="defaulting to English"
                )
            return Language.ENGLISH
        
        return self._error_handler.with_fallback(
            primary_fn=detect_primary,
            fallback_fn=fallback_to_english,
            error_types=(Exception,),  # Catch all language detection errors
            context="language detection",
        )
    
    def _get_strategy_with_fallback(self, language: Language) -> BaseStrategy:
        """
        Get strategy with fallback to recursive strategy on embedding failure.
        
        If the configured strategy requires embeddings and embedding provider fails,
        falls back to recursive strategy which doesn't need embeddings.
        """
        strategy_type = self.config.strategy
        
        # If strategy doesn't need embeddings, no fallback needed
        if strategy_type != StrategyType.SEMANTIC:
            return self._get_strategy(language)
        
        # Semantic strategy needs embeddings - wrap with fallback
        def get_semantic_strategy():
            return self._get_strategy(language)
        
        def fallback_to_recursive():
            logger.warning(
                "Falling back to recursive strategy due to embedding provider failure"
            )
            
            # Record error in metrics
            if self._metrics_collector:
                self._metrics_collector.record_error(
                    error_type="EmbeddingError",
                    component="SemanticChunker",
                    recoverable=True
                )
            
            # Log fallback activation
            if self._structured_logger:
                self._structured_logger.log_fallback(
                    component="SemanticChunker",
                    reason="embedding provider failure",
                    fallback_action="using recursive strategy"
                )
            
            # Temporarily switch to recursive strategy
            original_strategy = self.config.strategy
            self.config.strategy = StrategyType.RECURSIVE
            self._strategy = None  # Clear cached strategy
            strategy = self._get_strategy(language)
            self.config.strategy = original_strategy  # Restore original
            return strategy
        
        return self._error_handler.with_fallback(
            primary_fn=get_semantic_strategy,
            fallback_fn=fallback_to_recursive,
            error_types=(EmbeddingError, Exception),  # Catch embedding and general errors
            context="strategy initialization",
        )

    def _segment_sentences(
        self, text: str, language: Language
    ) -> List[str]:
        """Route to language-appropriate sentence segmenter."""
        if language == Language.ARABIC:
            ar_processor = self._lazy_load_manager.get_language_processor(Language.ARABIC)
            normalized_ar = ar_processor.normalize(text)
            return ar_processor.segment_sentences(normalized_ar)

        elif language == Language.ENGLISH:
            en_processor = self._lazy_load_manager.get_language_processor(Language.ENGLISH)
            return en_processor.segment_sentences(text)

        elif language == Language.MIXED:
            return self._segment_mixed(text)

        else:
            # Unknown — try English pipeline
            en_processor = self._lazy_load_manager.get_language_processor(Language.ENGLISH)
            return en_processor.segment_sentences(text)

    def _segment_mixed(self, text: str) -> List[str]:
        """
        Handle mixed EN/AR documents by detecting language per-line
        and routing to the appropriate segmenter.
        """
        lines = text.split("\n")
        all_sentences: List[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            lang = self._detector.detect_sentence(line)

            if lang == Language.ARABIC:
                ar_processor = self._lazy_load_manager.get_language_processor(Language.ARABIC)
                normalized = ar_processor.normalize(line)
                sents = ar_processor.segment_sentences(normalized)
            else:
                en_processor = self._lazy_load_manager.get_language_processor(Language.ENGLISH)
                sents = en_processor.segment_sentences(line)

            all_sentences.extend(sents)

        return all_sentences

    def _get_strategy(self, language: Language) -> BaseStrategy:
        """Get or create the chunking strategy."""
        if self._strategy is not None:
            return self._strategy

        strategy_type = self.config.strategy

        if strategy_type == StrategyType.STRUCTURE_AWARE:
            from chunker.strategies.structure_aware import StructureAwareStrategy
            self._strategy = StructureAwareStrategy(self.config)

        elif strategy_type == StrategyType.SEMANTIC:
            provider = self._get_embedding_provider()
            from chunker.strategies.semantic import SemanticStrategy
            self._strategy = SemanticStrategy(
                self.config, 
                provider, 
                self._cache_manager,
                self._metrics_collector
            )

        elif strategy_type == StrategyType.RECURSIVE:
            from chunker.strategies.recursive import RecursiveStrategy
            self._strategy = RecursiveStrategy(self.config, language)

        elif strategy_type == StrategyType.SENTENCE:
            from chunker.strategies.sentence import SentenceStrategy
            self._strategy = SentenceStrategy(self.config)

        elif strategy_type == StrategyType.FIXED:
            from chunker.strategies.fixed import FixedSizeStrategy
            self._strategy = FixedSizeStrategy(self.config)

        elif strategy_type == StrategyType.HIERARCHICAL:
            from chunker.strategies.hierarchical import HierarchicalStrategy
            self._strategy = HierarchicalStrategy(self.config)

        elif strategy_type == StrategyType.AGENTIC:
            from chunker.strategies.agentic import AgenticStrategy
            self._strategy = AgenticStrategy(self.config)

        else:
            raise ValueError(f"Unknown strategy: {strategy_type}")

        return self._strategy

    def _get_embedding_provider(self) -> BaseEmbeddingProvider:
        """Get or create the embedding provider."""
        if self._embedding_provider is not None:
            return self._embedding_provider

        # Use lazy load manager to get embedding provider
        self._embedding_provider = self._lazy_load_manager.get_embedding_provider()
        return self._embedding_provider

    def _build_chunks(
        self,
        chunk_texts: List[str],
        doc_info: DocumentInfo,
        strategy_name: str,
        original_text: str,
        language: Language,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Build Chunk objects with rich metadata."""
        chunks: List[Chunk] = []
        char_offset = 0

        # Get section metadata from structure-aware strategy if available
        section_meta_list = []
        if hasattr(self._strategy, 'section_metadata'):
            section_meta_list = self._strategy.section_metadata

        for i, text in enumerate(chunk_texts):
            if not text.strip():
                continue

            # Find position in original text
            # Prefer accurate offsets from structure-aware strategy
            if i < len(section_meta_list) and "chunk_start_char" in section_meta_list[i]:
                start_char = section_meta_list[i]["chunk_start_char"]
                end_char = section_meta_list[i]["chunk_end_char"]
            else:
                start_char = original_text.find(text[:50], char_offset)
                if start_char == -1:
                    start_char = char_offset
                end_char = start_char + len(text)
            char_offset = start_char + 1  # advance for next search

            # Per-chunk language detection
            chunk_lang = (
                self._detect_language_with_fallback(text)
                if self.config.detect_language
                else language
            )

            lang_code = chunk_lang.value if chunk_lang != Language.UNKNOWN else language.value
            token_count = estimate_tokens(text, lang_code)

            # Detect structural elements using lazy-loaded processors
            if chunk_lang == Language.ARABIC:
                ar_processor = self._lazy_load_manager.get_language_processor(Language.ARABIC)
                structure = ar_processor.detect_structure(text)
                script = ar_processor.detect_script(text)
            else:
                en_processor = self._lazy_load_manager.get_language_processor(Language.ENGLISH)
                structure = en_processor.detect_structure(text)
                # Get script detection from Arabic processor only if needed
                if chunk_lang == Language.ARABIC:
                    ar_processor = self._lazy_load_manager.get_language_processor(Language.ARABIC)
                    script = ar_processor.detect_script(text)
                else:
                    script = "latin"  # Default for non-Arabic text

            # Extract heading_path from structure-aware strategy metadata
            heading_path = []
            section_title = ""
            hierarchy_level = 0
            if i < len(section_meta_list):
                meta = section_meta_list[i]
                heading_path = meta.get("heading_path", [])
                section_title = meta.get("section_title", "")
                hierarchy_level = meta.get("hierarchy_level", 0)

            metadata = ChunkMetadata(
                document_id=doc_info.document_id,
                source_file=doc_info.source_file,
                chunk_index=i,
                total_chunks=len(chunk_texts),
                start_char=start_char,
                end_char=end_char,
                language=chunk_lang,
                token_count=token_count,
                char_count=len(text),
                sentence_count=count_sentences(text),
                has_overlap_before=i > 0 and self.config.chunk_overlap > 0,
                has_overlap_after=i < len(chunk_texts) - 1 and self.config.chunk_overlap > 0,
                overlap_tokens=self.config.chunk_overlap if self.config.chunk_overlap > 0 else 0,
                strategy=strategy_name,
                contains_header=structure.get("has_headers", False),
                contains_table=structure.get("has_tables", False),
                contains_code=structure.get("has_code", False),
                contains_list=structure.get("has_lists", False),
                heading_path=heading_path,
                section_title=section_title,
                hierarchy_level=hierarchy_level,
                script=script,
                custom=extra_metadata or {},
            )

            chunks.append(Chunk(text=text, metadata=metadata))

        return chunks

    # ── Convenience Methods ───────────────────────────────────

    def get_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Compute statistics about the chunking result.

        Returns dict with:
          - total_chunks, total_tokens, avg_tokens_per_chunk
          - min/max/median chunk sizes
          - language distribution
          - strategy used
        """
        if not chunks:
            return {"total_chunks": 0}

        token_counts = [c.metadata.token_count for c in chunks]
        languages = [c.metadata.language.value for c in chunks]

        lang_dist = {}
        for lang in languages:
            lang_dist[lang] = lang_dist.get(lang, 0) + 1

        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "median_tokens": sorted(token_counts)[len(token_counts) // 2],
            "language_distribution": lang_dist,
            "strategy": chunks[0].metadata.strategy if chunks else "",
            "document_id": chunks[0].metadata.document_id if chunks else "",
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns dict with cache hit/miss rates and other statistics.
        Returns empty dict if caching is disabled.
        """
        if self._cache_manager is None:
            return {}
        return self._cache_manager.get_stats()

    def get_metrics(self) -> ChunkerMetrics:
        """
        Get aggregated metrics for all chunking operations.
        
        Returns
        -------
        ChunkerMetrics
            Aggregated metrics including:
            - Documents processed, chunks created, tokens processed
            - Processing time statistics (avg, min, max)
            - Cache hit/miss rates
            - Error counts and fallback activations
            - Per-strategy and per-language breakdowns
            
        Notes
        -----
        Returns empty metrics if metrics collection is disabled.
        Metrics are cumulative since chunker initialization or last reset.
        """
        if self._metrics_collector is None:
            return ChunkerMetrics()
        return self._metrics_collector.get_summary()

    def save_cache(self) -> None:
        """
        Save cache to disk.

        Only works if persist_cache_to_disk is enabled in config.
        Raises ValueError if cache_dir is not configured.
        """
        if self._cache_manager is None:
            return
        if not self.config.persist_cache_to_disk:
            logger.warning(
                "Cache persistence is disabled. Set persist_cache_to_disk=True in config."
            )
            return
        self._cache_manager.save_to_disk()

    def clear_cache(self) -> None:
        """Clear all caches and reset statistics."""
        if self._cache_manager is not None:
            self._cache_manager.clear()

    def to_dicts(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """Serialize all chunks to dictionaries."""
        return [c.to_dict() for c in chunks]

    def to_texts(self, chunks: List[Chunk]) -> List[str]:
        """Extract just the text from chunks."""
        return [c.text for c in chunks]
