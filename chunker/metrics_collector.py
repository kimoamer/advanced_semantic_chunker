"""
Metrics collection for monitoring chunker operations.

This module provides lightweight metrics tracking for:
- Document processing statistics
- Embedding API calls and cache performance
- Processing time per strategy and language
- Error tracking and fallback activations
"""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chunker.config import StrategyType
from chunker.models import Language

__all__ = ["ChunkerMetrics", "MetricsCollector"]


@dataclass
class ChunkerMetrics:
    """Aggregated metrics for chunker operations."""
    
    # Document processing
    documents_processed: int = 0
    total_chunks_created: int = 0
    total_tokens_processed: int = 0
    
    # Timing
    total_processing_time_ms: float = 0.0
    avg_processing_time_ms: float = 0.0
    min_processing_time_ms: float = float('inf')
    max_processing_time_ms: float = 0.0
    
    # Embedding calls
    embedding_calls: int = 0
    embedding_cache_hits: int = 0
    embedding_cache_misses: int = 0
    embedding_cache_hit_rate: float = 0.0
    
    # Language detection
    language_detections: int = 0
    language_cache_hits: int = 0
    language_cache_misses: int = 0
    
    # Errors
    total_errors: int = 0
    recoverable_errors: int = 0
    unrecoverable_errors: int = 0
    fallbacks_triggered: int = 0
    
    # Per-strategy breakdown
    strategy_usage: Dict[str, int] = field(default_factory=dict)
    strategy_timing: Dict[str, float] = field(default_factory=dict)
    
    # Per-language breakdown
    language_distribution: Dict[str, int] = field(default_factory=dict)


class MetricsCollector:
    """
    Lightweight metrics collection for monitoring chunker operations.
    
    This class tracks operations without impacting performance, using
    simple counters and thread-safe operations.
    
    Usage
    -----
    ```python
    metrics = MetricsCollector(enabled=True)
    
    # Record document processing
    metrics.record_document_processed(
        doc_id="doc123",
        language=Language.ENGLISH,
        strategy=StrategyType.SEMANTIC,
        duration_ms=234.5,
        chunk_count=15,
        token_count=2048
    )
    
    # Record embedding call
    metrics.record_embedding_call(
        model="BAAI/bge-m3",
        batch_size=10,
        duration_ms=123.4,
        cache_hit=False
    )
    
    # Get summary
    summary = metrics.get_summary()
    print(f"Cache hit rate: {summary.embedding_cache_hit_rate:.2%}")
    ```
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize metrics collector.
        
        Parameters
        ----------
        enabled : bool
            Whether metrics collection is enabled. If False, all operations
            are no-ops for minimal overhead.
        """
        self.enabled = enabled
        self._lock = threading.Lock()
        
        # Document processing counters
        self._documents_processed = 0
        self._total_chunks = 0
        self._total_tokens = 0
        
        # Timing tracking
        self._processing_times: List[float] = []
        
        # Embedding counters
        self._embedding_calls = 0
        self._embedding_cache_hits = 0
        self._embedding_cache_misses = 0
        
        # Language detection counters
        self._language_detections = 0
        self._language_cache_hits = 0
        self._language_cache_misses = 0
        
        # Error counters
        self._total_errors = 0
        self._recoverable_errors = 0
        self._unrecoverable_errors = 0
        self._fallbacks_triggered = 0
        
        # Per-strategy tracking
        self._strategy_usage: Counter = Counter()
        self._strategy_timing: defaultdict = defaultdict(list)
        
        # Per-language tracking
        self._language_distribution: Counter = Counter()
    
    def record_document_processed(
        self,
        doc_id: str,
        language: Language,
        strategy: StrategyType,
        duration_ms: float,
        chunk_count: int,
        token_count: int
    ) -> None:
        """
        Record document processing metrics.
        
        Parameters
        ----------
        doc_id : str
            Document identifier
        language : Language
            Detected language
        strategy : StrategyType
            Strategy used for chunking
        duration_ms : float
            Processing duration in milliseconds
        chunk_count : int
            Number of chunks created
        token_count : int
            Total tokens processed
        """
        if not self.enabled:
            return
        
        with self._lock:
            self._documents_processed += 1
            self._total_chunks += chunk_count
            self._total_tokens += token_count
            self._processing_times.append(duration_ms)
            
            # Track per-strategy usage and timing
            strategy_name = strategy.value if isinstance(strategy, StrategyType) else str(strategy)
            self._strategy_usage[strategy_name] += 1
            self._strategy_timing[strategy_name].append(duration_ms)
            
            # Track per-language distribution
            lang_name = language.value if isinstance(language, Language) else str(language)
            self._language_distribution[lang_name] += 1
    
    def record_embedding_call(
        self,
        model: str,
        batch_size: int,
        duration_ms: float,
        cache_hit: bool
    ) -> None:
        """
        Record embedding API call metrics.
        
        Parameters
        ----------
        model : str
            Embedding model name
        batch_size : int
            Number of texts embedded in this call
        duration_ms : float
            Call duration in milliseconds
        cache_hit : bool
            Whether this was a cache hit
        """
        if not self.enabled:
            return
        
        with self._lock:
            self._embedding_calls += 1
            if cache_hit:
                self._embedding_cache_hits += 1
            else:
                self._embedding_cache_misses += 1
    
    def record_cache_access(
        self,
        cache_type: str,
        hit: bool
    ) -> None:
        """
        Record cache hit/miss.
        
        Parameters
        ----------
        cache_type : str
            Type of cache ("embedding", "language", etc.)
        hit : bool
            Whether this was a cache hit
        """
        if not self.enabled:
            return
        
        with self._lock:
            if cache_type == "language":
                self._language_detections += 1
                if hit:
                    self._language_cache_hits += 1
                else:
                    self._language_cache_misses += 1
            elif cache_type == "embedding":
                # Already tracked in record_embedding_call
                pass
    
    def record_error(
        self,
        error_type: str,
        component: str,
        recoverable: bool
    ) -> None:
        """
        Record error occurrence.
        
        Parameters
        ----------
        error_type : str
            Type of error (e.g., "EmbeddingError", "LanguageDetectionError")
        component : str
            Component where error occurred
        recoverable : bool
            Whether the error was recoverable (fallback succeeded)
        """
        if not self.enabled:
            return
        
        with self._lock:
            self._total_errors += 1
            if recoverable:
                self._recoverable_errors += 1
                self._fallbacks_triggered += 1
            else:
                self._unrecoverable_errors += 1
    
    def get_summary(self) -> ChunkerMetrics:
        """
        Get aggregated metrics summary.
        
        Returns
        -------
        ChunkerMetrics
            Aggregated metrics with calculated statistics
        """
        if not self.enabled:
            return ChunkerMetrics()
        
        with self._lock:
            # Calculate timing statistics
            total_time = sum(self._processing_times)
            avg_time = total_time / len(self._processing_times) if self._processing_times else 0.0
            min_time = min(self._processing_times) if self._processing_times else 0.0
            max_time = max(self._processing_times) if self._processing_times else 0.0
            
            # Calculate cache hit rates
            total_embedding_accesses = self._embedding_cache_hits + self._embedding_cache_misses
            embedding_hit_rate = (
                self._embedding_cache_hits / total_embedding_accesses
                if total_embedding_accesses > 0 else 0.0
            )
            
            # Calculate average timing per strategy
            strategy_avg_timing = {}
            for strategy, times in self._strategy_timing.items():
                strategy_avg_timing[strategy] = sum(times) / len(times) if times else 0.0
            
            return ChunkerMetrics(
                documents_processed=self._documents_processed,
                total_chunks_created=self._total_chunks,
                total_tokens_processed=self._total_tokens,
                total_processing_time_ms=total_time,
                avg_processing_time_ms=avg_time,
                min_processing_time_ms=min_time,
                max_processing_time_ms=max_time,
                embedding_calls=self._embedding_calls,
                embedding_cache_hits=self._embedding_cache_hits,
                embedding_cache_misses=self._embedding_cache_misses,
                embedding_cache_hit_rate=embedding_hit_rate,
                language_detections=self._language_detections,
                language_cache_hits=self._language_cache_hits,
                language_cache_misses=self._language_cache_misses,
                total_errors=self._total_errors,
                recoverable_errors=self._recoverable_errors,
                unrecoverable_errors=self._unrecoverable_errors,
                fallbacks_triggered=self._fallbacks_triggered,
                strategy_usage=dict(self._strategy_usage),
                strategy_timing=strategy_avg_timing,
                language_distribution=dict(self._language_distribution),
            )
    
    def reset(self) -> None:
        """Reset all metrics to initial state."""
        if not self.enabled:
            return
        
        with self._lock:
            self._documents_processed = 0
            self._total_chunks = 0
            self._total_tokens = 0
            self._processing_times.clear()
            self._embedding_calls = 0
            self._embedding_cache_hits = 0
            self._embedding_cache_misses = 0
            self._language_detections = 0
            self._language_cache_hits = 0
            self._language_cache_misses = 0
            self._total_errors = 0
            self._recoverable_errors = 0
            self._unrecoverable_errors = 0
            self._fallbacks_triggered = 0
            self._strategy_usage.clear()
            self._strategy_timing.clear()
            self._language_distribution.clear()
    
    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.
        
        Returns
        -------
        str
            Metrics in Prometheus exposition format
        """
        if not self.enabled:
            return ""
        
        metrics = self.get_summary()
        lines = []
        
        # Document processing metrics
        lines.append("# HELP chunker_documents_processed_total Total number of documents processed")
        lines.append("# TYPE chunker_documents_processed_total counter")
        lines.append(f"chunker_documents_processed_total {metrics.documents_processed}")
        
        lines.append("# HELP chunker_chunks_created_total Total number of chunks created")
        lines.append("# TYPE chunker_chunks_created_total counter")
        lines.append(f"chunker_chunks_created_total {metrics.total_chunks_created}")
        
        lines.append("# HELP chunker_tokens_processed_total Total number of tokens processed")
        lines.append("# TYPE chunker_tokens_processed_total counter")
        lines.append(f"chunker_tokens_processed_total {metrics.total_tokens_processed}")
        
        # Timing metrics
        lines.append("# HELP chunker_processing_time_ms Processing time in milliseconds")
        lines.append("# TYPE chunker_processing_time_ms summary")
        lines.append(f"chunker_processing_time_ms_sum {metrics.total_processing_time_ms}")
        lines.append(f"chunker_processing_time_ms_count {metrics.documents_processed}")
        
        # Cache metrics
        lines.append("# HELP chunker_embedding_cache_hits_total Total embedding cache hits")
        lines.append("# TYPE chunker_embedding_cache_hits_total counter")
        lines.append(f"chunker_embedding_cache_hits_total {metrics.embedding_cache_hits}")
        
        lines.append("# HELP chunker_embedding_cache_misses_total Total embedding cache misses")
        lines.append("# TYPE chunker_embedding_cache_misses_total counter")
        lines.append(f"chunker_embedding_cache_misses_total {metrics.embedding_cache_misses}")
        
        lines.append("# HELP chunker_embedding_cache_hit_rate Embedding cache hit rate")
        lines.append("# TYPE chunker_embedding_cache_hit_rate gauge")
        lines.append(f"chunker_embedding_cache_hit_rate {metrics.embedding_cache_hit_rate}")
        
        # Error metrics
        lines.append("# HELP chunker_errors_total Total errors encountered")
        lines.append("# TYPE chunker_errors_total counter")
        lines.append(f"chunker_errors_total {metrics.total_errors}")
        
        lines.append("# HELP chunker_fallbacks_triggered_total Total fallbacks triggered")
        lines.append("# TYPE chunker_fallbacks_triggered_total counter")
        lines.append(f"chunker_fallbacks_triggered_total {metrics.fallbacks_triggered}")
        
        # Per-strategy metrics
        lines.append("# HELP chunker_strategy_usage_total Documents processed per strategy")
        lines.append("# TYPE chunker_strategy_usage_total counter")
        for strategy, count in metrics.strategy_usage.items():
            lines.append(f'chunker_strategy_usage_total{{strategy="{strategy}"}} {count}')
        
        # Per-language metrics
        lines.append("# HELP chunker_language_distribution_total Documents per language")
        lines.append("# TYPE chunker_language_distribution_total counter")
        for language, count in metrics.language_distribution.items():
            lines.append(f'chunker_language_distribution_total{{language="{language}"}} {count}')
        
        return "\n".join(lines) + "\n"
