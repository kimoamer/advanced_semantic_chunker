"""
Data models for the Semantic Chunker.

All chunk outputs carry rich metadata for downstream RAG pipelines:
  - source tracking (document ID, position)
  - language detection per-chunk
  - token counts & overlap markers
  - optional topic labels and coherence scores
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class Language(str, Enum):
    """Supported languages."""

    ENGLISH = "en"
    ARABIC = "ar"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class ChunkMetadata:
    """Rich metadata attached to every chunk for RAG traceability."""

    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    source_file: str = ""
    # Position tracking
    chunk_index: int = 0
    total_chunks: int = 0
    start_char: int = 0
    end_char: int = 0
    start_sentence: int = 0
    end_sentence: int = 0
    # Content analysis
    language: Language = Language.UNKNOWN
    token_count: int = 0
    char_count: int = 0
    sentence_count: int = 0
    # Overlap markers
    has_overlap_before: bool = False
    has_overlap_after: bool = False
    overlap_tokens: int = 0
    # Semantic metadata
    topic_label: str = ""
    coherence_score: float = 0.0
    # Strategy used
    strategy: str = ""
    # Structural markers
    contains_header: bool = False
    contains_table: bool = False
    contains_code: bool = False
    contains_list: bool = False
    # Hierarchy & heading path
    section_title: str = ""
    heading_path: List[str] = field(default_factory=list)
    hierarchy_level: int = 0
    parent_chunk_id: str = ""
    # Script & dialect detection
    script: str = ""  # "arabic", "latin", "mixed"
    is_dialect: bool = False
    # Timestamps
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # Custom user-defined metadata
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A single semantic chunk with text and metadata."""

    text: str
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    # Embedding (populated lazily if requested)
    embedding: Optional[List[float]] = None

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of the chunk text for deduplication."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def is_empty(self) -> bool:
        return not self.text or not self.text.strip()

    def __len__(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", "\\n")
        return (
            f"Chunk(lang={self.metadata.language.value}, "
            f"tokens={self.metadata.token_count}, "
            f'text="{preview}...")'
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize chunk to dictionary."""
        return {
            "text": self.text,
            "content_hash": self.content_hash,
            "embedding": self.embedding,
            "metadata": {
                "chunk_id": self.metadata.chunk_id,
                "document_id": self.metadata.document_id,
                "source_file": self.metadata.source_file,
                "chunk_index": self.metadata.chunk_index,
                "total_chunks": self.metadata.total_chunks,
                "start_char": self.metadata.start_char,
                "end_char": self.metadata.end_char,
                "language": self.metadata.language.value,
                "token_count": self.metadata.token_count,
                "char_count": self.metadata.char_count,
                "sentence_count": self.metadata.sentence_count,
                "strategy": self.metadata.strategy,
                "section_title": self.metadata.section_title,
                "heading_path": self.metadata.heading_path,
                "hierarchy_level": self.metadata.hierarchy_level,
                "script": self.metadata.script,
                "is_dialect": self.metadata.is_dialect,
                "coherence_score": self.metadata.coherence_score,
                "topic_label": self.metadata.topic_label,
                "has_overlap_before": self.metadata.has_overlap_before,
                "has_overlap_after": self.metadata.has_overlap_after,
                "created_at": self.metadata.created_at,
                "custom": self.metadata.custom,
            },
        }


@dataclass
class DocumentInfo:
    """Information about the source document being chunked."""

    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_file: str = ""
    title: str = ""
    language: Language = Language.UNKNOWN
    total_chars: int = 0
    total_sentences: int = 0
    total_tokens: int = 0
    detected_structure: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"DocumentInfo(id={self.document_id[:8]}..., "
            f"lang={self.language.value}, "
            f"chars={self.total_chars})"
        )
