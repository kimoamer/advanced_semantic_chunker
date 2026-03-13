"""
Semantic Chunker — A powerful bilingual (EN/AR) semantic document chunking engine.

Supports multiple chunking strategies:
  - Structure-aware (heading-based hard boundaries — DEFAULT)
  - Semantic (embedding-based similarity splitting)
  - Recursive (hierarchical separator-based splitting)
  - Sentence-aware (complete sentence grouping)
  - Fixed-size (token/character-based splitting)
  - Hierarchical (multi-level section → paragraph → sentence)
  - Agentic (LLM-driven dynamic splitting)

Designed for RAG pipelines, search indexing, and LLM context management.
"""

from chunker.models import Chunk, ChunkMetadata, DocumentInfo
from chunker.config import ChunkerConfig, StrategyType, ThresholdType
from chunker.core import SemanticChunker
from chunker.cache import CacheManager
from chunker.batch_processor import BatchProcessor
from chunker.async_chunker import AsyncSemanticChunker
from chunker.deduplicator import ChunkDeduplicator
from chunker.error_handler import CircuitBreaker, CircuitBreakerOpenError

__version__ = "1.1.0"
__all__ = [
    # Core
    "SemanticChunker",
    "AsyncSemanticChunker",
    # Data models
    "Chunk",
    "ChunkMetadata",
    "DocumentInfo",
    # Configuration
    "ChunkerConfig",
    "StrategyType",
    "ThresholdType",
    # Utilities
    "CacheManager",
    "BatchProcessor",
    "ChunkDeduplicator",
    # Reliability
    "CircuitBreaker",
    "CircuitBreakerOpenError",
]
