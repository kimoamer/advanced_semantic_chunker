"""
Configuration classes for the Semantic Chunker.

All tuneable parameters live here so that the chunker behaviour
can be adjusted without touching strategy code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional


class StrategyType(str, Enum):
    """Available chunking strategies."""

    STRUCTURE_AWARE = "structure_aware"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    FIXED = "fixed"
    HIERARCHICAL = "hierarchical"
    AGENTIC = "agentic"


class ThresholdType(str, Enum):
    """Methods for detecting semantic breakpoints."""

    PERCENTILE = "percentile"
    STANDARD_DEVIATION = "standard_deviation"
    INTERQUARTILE = "interquartile"
    GRADIENT = "gradient"


class EmbeddingProvider(str, Enum):
    """Supported embedding backends."""

    SENTENCE_TRANSFORMER = "sentence_transformer"
    OPENAI = "openai"
    CUSTOM = "custom"


@dataclass
class ChunkerConfig:
    """
    Master configuration for the SemanticChunker.

    Defaults are research-backed for RAG retrieval quality:
      - chunk_size 220 tokens (target 180-300 range)
      - max_chunk_size 320 tokens (hard cap under ~350)
      - chunk_overlap 0 (start at zero; add 20-60 if needed)
      - min_chunk_size 80 (merge tail fragments below this)

    Attributes
    ----------
    strategy : StrategyType
        Which chunking algorithm to use (default: structure_aware).
    chunk_size : int
        Target chunk size in tokens (default: 220).
    chunk_overlap : int
        Number of overlapping tokens between adjacent chunks (default: 0).
    min_chunk_size : int
        Minimum allowed chunk size in tokens; smaller chunks are merged (default: 80).
    max_chunk_size : int
        Maximum allowed chunk size in tokens; larger chunks are split (default: 320).
    threshold_type : ThresholdType
        Breakpoint detection method for semantic strategy (default: percentile).
    threshold_amount : float
        Threshold value — meaning depends on threshold_type:
          - percentile: 0-100 (default 90.0, i.e. 90th percentile)
          - standard_deviation: number of σ (default 1.5)
          - interquartile: multiplier of IQR (default 1.5)
          - gradient: minimum gradient change (default 0.1)
    embedding_provider : EmbeddingProvider
        Which embedding backend to use (default: sentence_transformer).
    embedding_model : str
        Model identifier for embeddings.
    similarity_metric : str
        "cosine" or "dot_product" (default: cosine).
    respect_sentence_boundaries : bool
        Never split in the middle of a sentence (default: True).
    detect_language : bool
        Automatically detect language per-chunk (default: True).
    preserve_structure : bool
        Detect and respect document structure (headers, tables, etc.) (default: True).
    headings_are_hard_boundaries : bool
        Every heading starts a new chunk — headings never appear mid-chunk (default: True).
    overlap_across_headings : bool
        Allow overlap to span across heading boundaries (default: False).
    metadata_enrichment : bool
        Attach extended metadata to each chunk (default: True).
    compute_coherence_score : bool
        Compute internal coherence score per chunk (default: False, expensive).
    enable_embedding_cache : bool
        Enable caching of embeddings to avoid recomputation (default: True).
    embedding_cache_size : int
        Maximum number of embeddings to cache (default: 10000).
    enable_language_cache : bool
        Enable caching of language detection results (default: True).
    language_cache_size : int
        Maximum number of language detection results to cache (default: 5000).
    persist_cache_to_disk : bool
        Save cache to disk for cross-session reuse (default: False).
    cache_dir : Optional[str]
        Directory for cache persistence (required if persist_cache_to_disk is True).
    batch_embedding_calls : bool
        Batch multiple embedding calls into single API/model call (default: True).
    batch_size : int
        Number of texts to process in a single batch (default: 32).
    enable_parallel_processing : bool
        Enable parallel processing for batch operations (default: False).
    num_workers : Optional[int]
        Number of worker threads for parallel processing (default: None, uses CPU count).
    enable_fallbacks : bool
        Enable graceful degradation with fallback strategies (default: True).
    retry_on_network_error : bool
        Retry operations on network errors (default: True).
    max_retries : int
        Maximum number of retry attempts for network errors (default: 3).
    retry_backoff_factor : float
        Exponential backoff factor for retries (default: 2.0).
    enable_metrics : bool
        Enable metrics collection for monitoring (default: True).
    enable_structured_logging : bool
        Enable structured JSON logging (default: True).
    log_level : str
        Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO).
    progress_callback : Optional[Callable]
        Callback function for progress reporting during batch operations.
    preload_components : List[str]
        List of components to preload explicitly (default: empty list).
    lazy_load_embeddings : bool
        Defer loading of embedding models until needed (default: True).
    lazy_load_nlp_tools : bool
        Defer loading of NLP tools until needed (default: True).
    """

    # ── Strategy ──────────────────────────────────────────────
    strategy: StrategyType = StrategyType.STRUCTURE_AWARE

    # ── Chunk sizing (research-recommended defaults) ──────────
    # Target: 180-260 tokens, max ~320, min ~80
    chunk_size: int = 220
    chunk_overlap: int = 0
    min_chunk_size: int = 80
    max_chunk_size: int = 320

    # ── Semantic thresholds ───────────────────────────────────
    threshold_type: ThresholdType = ThresholdType.PERCENTILE
    threshold_amount: float = 90.0

    # ── Embedding ─────────────────────────────────────────────
    embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMER
    embedding_model: str = "BAAI/bge-m3"
    similarity_metric: str = "cosine"

    # ── Behaviour flags ───────────────────────────────────────
    respect_sentence_boundaries: bool = True
    detect_language: bool = True
    preserve_structure: bool = True

    # ── Arabic-specific ───────────────────────────────────────
    arabic_normalize_alef: bool = True
    arabic_normalize_yeh: bool = False
    arabic_normalize_teh_marbuta: bool = False
    arabic_remove_tashkeel: bool = True
    arabic_remove_tatweel: bool = True
    arabic_normalize_punctuation: bool = True
    arabic_morphological_decomposition: bool = True

    # ── Structure-aware settings ──────────────────────────────
    headings_are_hard_boundaries: bool = True
    overlap_across_headings: bool = False

    # ── Recursive separators ──────────────────────────────────
    recursive_separators_en: List[str] = field(
        default_factory=lambda: ["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
    )
    recursive_separators_ar: List[str] = field(
        default_factory=lambda: [
            "\n\n\n",
            "\n\n",
            "\n",
            "。",       # Full-width period (sometimes used)
            ".\u200f",  # Period + RTL mark
            ". ",
            "؟ ",       # Arabic question mark
            "! ",
            "، ",       # Arabic comma
            "؛ ",       # Arabic semicolon
            " ",
        ]
    )

    # ── Metadata ──────────────────────────────────────────────
    metadata_enrichment: bool = True
    compute_coherence_score: bool = False

    # ── OpenAI / LLM (for agentic strategy) ───────────────────
    openai_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"

    # ── Logging ───────────────────────────────────────────────
    verbose: bool = False

    # ── Caching ───────────────────────────────────────────────
    enable_embedding_cache: bool = True
    embedding_cache_size: int = 10000
    enable_language_cache: bool = True
    language_cache_size: int = 5000
    persist_cache_to_disk: bool = False
    cache_dir: Optional[str] = None

    # ── Batch processing ──────────────────────────────────────
    batch_embedding_calls: bool = True
    batch_size: int = 32
    enable_parallel_processing: bool = False
    num_workers: Optional[int] = None

    # ── Error handling ────────────────────────────────────────
    enable_fallbacks: bool = True
    retry_on_network_error: bool = True
    max_retries: int = 3
    retry_backoff_factor: float = 2.0

    # ── Observability ─────────────────────────────────────────
    enable_metrics: bool = True
    enable_structured_logging: bool = True
    log_level: str = "INFO"
    progress_callback: Optional[Callable] = None

    # ── Lazy loading ──────────────────────────────────────────
    preload_components: List[str] = field(default_factory=list)
    lazy_load_embeddings: bool = True
    lazy_load_nlp_tools: bool = True

    def validate(self) -> None:
        """Raise ValueError if config is inconsistent."""
        if self.chunk_size < self.min_chunk_size:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be >= min_chunk_size ({self.min_chunk_size})"
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        if self.threshold_type == ThresholdType.PERCENTILE:
            if not (0 < self.threshold_amount <= 100):
                raise ValueError(
                    f"For percentile threshold, amount must be in (0, 100], got {self.threshold_amount}"
                )
        if self.strategy == StrategyType.AGENTIC and not self.openai_api_key:
            raise ValueError("Agentic strategy requires openai_api_key to be set")
        
        # Validate caching configuration
        if self.embedding_cache_size < 0:
            raise ValueError(
                f"embedding_cache_size must be >= 0, got {self.embedding_cache_size}"
            )
        if self.language_cache_size < 0:
            raise ValueError(
                f"language_cache_size must be >= 0, got {self.language_cache_size}"
            )
        if self.persist_cache_to_disk and not self.cache_dir:
            raise ValueError(
                "cache_dir must be specified when persist_cache_to_disk is True"
            )
        
        # Validate batch processing configuration
        if self.batch_size < 1:
            raise ValueError(
                f"batch_size must be >= 1, got {self.batch_size}"
            )
        if self.num_workers is not None and self.num_workers < 1:
            raise ValueError(
                f"num_workers must be >= 1 or None, got {self.num_workers}"
            )
        
        # Validate error handling configuration
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be >= 0, got {self.max_retries}"
            )
        if self.retry_backoff_factor <= 0:
            raise ValueError(
                f"retry_backoff_factor must be > 0, got {self.retry_backoff_factor}"
            )
        
        # Validate observability configuration
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValueError(
                f"log_level must be one of {valid_log_levels}, got {self.log_level}"
            )
