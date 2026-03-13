# User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Chunking Strategies](#chunking-strategies)
4. [Configuration](#configuration)
5. [Caching](#caching)
6. [Error Handling & Reliability](#error-handling--reliability)
7. [Monitoring & Observability](#monitoring--observability)
8. [Advanced Features](#advanced-features)
9. [Document Readers](#document-readers)
10. [Vectorstore Integrations](#vectorstore-integrations)
11. [Async Support](#async-support)
12. [Best Practices](#best-practices)

---

## Introduction

Semantic Chunker is a production-ready bilingual (English/Arabic) document chunking library for RAG pipelines. It provides multiple chunking strategies, intelligent caching, async support, vectorstore integrations, and rich observability — out of the box.

### Key Features (v1.1.0)

- **7 Chunking Strategies** — from simple fixed-size to LLM-driven agentic
- **Bilingual** — native English and Arabic pipelines
- **Async-ready** — `AsyncSemanticChunker` for FastAPI / asyncio
- **Document Readers** — PDF, HTML, EPUB (no boilerplate)
- **Vectorstore Adapters** — Chroma, Pinecone, Qdrant
- **Streaming** — `chunk_stream()` for memory-efficient processing
- **Deduplication** — `ChunkDeduplicator` for batch pipelines
- **Circuit Breaker** — protect external API calls
- **Rate Limiting** — built-in for OpenAI provider
- **Secure Caching** — JSON+npz format (no pickle), MD5 hashing (3x faster)
- **Observability** — structured logging, metrics, Prometheus export
- **Thread-safe** — all components safe for concurrent access

---

## Core Concepts

### Chunks

A `Chunk` is a semantically coherent piece of text with rich metadata:

```python
from chunker import SemanticChunker

chunker = SemanticChunker()
chunks = chunker.chunk("Your document text here...")

for chunk in chunks:
    print(chunk.text)
    print(chunk.metadata.token_count)
    print(chunk.metadata.language.value)
    print(chunk.content_hash)    # SHA-256 for deduplication
```

### Metadata (25+ fields)

| Field | Description |
|-------|-------------|
| `chunk_id` | UUID |
| `chunk_index`, `total_chunks` | Position in document |
| `start_char`, `end_char` | Character offsets |
| `language`, `script` | Detected language |
| `token_count`, `char_count`, `sentence_count` | Size metrics |
| `contains_header`, `contains_code`, `contains_table`, `contains_list` | Structure flags |
| `section_title`, `heading_path`, `hierarchy_level` | Hierarchy info |
| `has_overlap_before`, `has_overlap_after`, `overlap_tokens` | Overlap markers |
| `coherence_score` | Optional semantic coherence |
| `strategy` | Strategy that produced this chunk |
| `source_file`, `document_id` | Provenance |
| `created_at` | ISO 8601 timestamp |
| `custom` | User-defined dict |

---

## Chunking Strategies

### 1. Structure-Aware (Default)

Respects document headings as hard boundaries:

```python
config = ChunkerConfig(
    strategy=StrategyType.STRUCTURE_AWARE,
    headings_are_hard_boundaries=True,
)
```
**Best for**: Markdown docs, README files, technical documentation.

### 2. Semantic

Embedding-based splitting at topic-shift boundaries:

```python
config = ChunkerConfig(
    strategy=StrategyType.SEMANTIC,
    embedding_model="BAAI/bge-m3",
    threshold_type=ThresholdType.PERCENTILE,
    threshold_amount=90.0,
)
```
**Best for**: Articles, books, research papers.

### 3. Recursive

Hierarchical separator-based splitting:

```python
config = ChunkerConfig(
    strategy=StrategyType.RECURSIVE,
    chunk_size=512,
    chunk_overlap=50,
)
```
**Best for**: General text, unknown structure.

### 4. Sentence

Groups complete sentences up to target size:

```python
config = ChunkerConfig(
    strategy=StrategyType.SENTENCE,
    chunk_size=256,
)
```
**Best for**: Short documents, chat logs.

### 5. Fixed

Simple token/character splitting:

```python
config = ChunkerConfig(
    strategy=StrategyType.FIXED,
    chunk_size=512,
)
```
**Best for**: Uniform chunks, benchmarking.

### 6. Hierarchical

Multi-level section → paragraph → sentence:

```python
config = ChunkerConfig(strategy=StrategyType.HIERARCHICAL)
```
**Best for**: Academic papers, long structured reports.

### 7. Agentic

LLM-driven dynamic splitting (requires OpenAI API key):

```python
config = ChunkerConfig(
    strategy=StrategyType.AGENTIC,
    openai_api_key="sk-...",
)
```
**Best for**: Maximum quality, custom splitting logic.

---

## Configuration

### Key Parameters

```python
config = ChunkerConfig(
    # Strategy
    strategy=StrategyType.STRUCTURE_AWARE,

    # Chunk sizing (research-backed defaults)
    chunk_size=220,         # target tokens
    min_chunk_size=80,      # merge smaller
    max_chunk_size=320,     # split larger
    chunk_overlap=0,        # tokens of overlap

    # Embedding
    embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
    embedding_model="BAAI/bge-m3",

    # Language
    detect_language=True,
    arabic_normalize_alef=True,
    arabic_remove_tashkeel=True,

    # Caching
    enable_embedding_cache=True,
    embedding_cache_size=10000,
    persist_cache_to_disk=True,
    cache_dir="./cache",

    # Error handling
    enable_fallbacks=True,
    retry_on_network_error=True,
    max_retries=3,
    retry_backoff_factor=2.0,

    # Observability
    enable_metrics=True,
    enable_structured_logging=True,
    log_level="INFO",
)
```

---

## Caching

The cache uses **LRU eviction**, **MD5 hashing** (3x faster than SHA-256), and stores data in **JSON + numpy `.npz`** format (safe, no pickle).

### Embedding Cache

```python
config = ChunkerConfig(
    enable_embedding_cache=True,
    embedding_cache_size=10000,
)
chunker = SemanticChunker(config)

chunks1 = chunker.chunk("Same text")    # cache miss
chunks2 = chunker.chunk("Same text")    # cache hit (faster)

stats = chunker.get_cache_stats()
print(f"Hit rate: {stats['embedding_cache_hit_rate']:.0%}")
```

### Disk Persistence

```python
config = ChunkerConfig(
    persist_cache_to_disk=True,
    cache_dir="./cache",
)
# Cache auto-loads on startup, auto-saves on shutdown
```

Files written:
- `./cache/embedding_cache.npz` — numpy arrays
- `./cache/embedding_cache_meta.json` — key order + stats
- `./cache/language_cache.json` — language detection results

---

## Error Handling & Reliability

### Automatic Fallbacks

```python
config = ChunkerConfig(
    strategy=StrategyType.SEMANTIC,
    enable_fallbacks=True,
)
# embedding fails → falls back to recursive strategy
# language detection fails → defaults to English
# NLP tools unavailable → uses regex segmentation
```

### Exponential Backoff Retry

```python
from chunker.error_handler import ErrorHandler

handler = ErrorHandler()
result = handler.with_retry(
    fn=lambda: embedding_provider.embed(texts),
    max_retries=3,
    backoff_factor=2.0,
    initial_delay=1.0,
)
# Delays: 0s → 1s → 2s → 4s
```

### Circuit Breaker

Prevents cascading failures when an external API is down:

```python
from chunker import CircuitBreaker, CircuitBreakerOpenError

cb = CircuitBreaker(
    failure_threshold=5,    # open after 5 consecutive failures
    recovery_timeout=60,    # try again after 60 seconds
)

try:
    result = cb.call(my_api_function, *args)
except CircuitBreakerOpenError:
    # Circuit is open — use fallback immediately
    result = fallback()

print(cb.state)         # "closed" | "open" | "half_open"
print(cb.failure_count) # consecutive failures
cb.reset()              # manually reset
```

### OpenAI Rate Limiting

```python
from chunker.embeddings.openai_provider import OpenAIEmbeddingProvider

provider = OpenAIEmbeddingProvider(
    model_name="text-embedding-3-small",
    requests_per_minute=500,    # token-bucket rate limiter
)
# Automatically throttles to avoid 429 errors
```

---

## Monitoring & Observability

### Metrics

```python
config = ChunkerConfig(enable_metrics=True)
chunker = SemanticChunker(config)

for doc in documents:
    chunker.chunk(doc)

m = chunker.get_metrics()
print(f"Docs processed:    {m.documents_processed}")
print(f"Total chunks:      {m.total_chunks_created}")
print(f"Avg time (ms):     {m.avg_processing_time_ms:.1f}")
print(f"Cache hit rate:    {m.embedding_cache_hit_rate:.0%}")
print(f"Errors:            {m.total_errors}")
print(f"Fallbacks:         {m.fallbacks_triggered}")
print(f"Strategy usage:    {m.strategy_usage}")
print(f"Language dist:     {m.language_distribution}")
```

### Prometheus Export

```python
prometheus_output = chunker._metrics_collector.export_prometheus()
# HELP chunker_documents_processed_total ...
# TYPE chunker_documents_processed_total counter
# chunker_documents_processed_total 42
```

### Structured Logging

See [structured_logging.md](structured_logging.md) for details.

---

## Advanced Features

### Batch Processing

```python
from chunker.batch_processor import BatchProcessor

processor = BatchProcessor(
    chunker,
    batch_size=32,
    progress_callback=lambda cur, tot: print(f"{cur}/{tot}"),
    executor_type="thread",     # or "process" for CPU-heavy work
)

# Sequential
results = processor.process_batch(texts)

# Parallel
results = processor.process_batch(texts, parallel=True, num_workers=4)

# Streaming iterator
for index, chunks in processor.process_with_progress(texts):
    print(f"Doc {index}: {len(chunks)} chunks")
```

### Streaming (chunk_stream)

```python
# Yield chunks one at a time — no memory spike for large docs
for chunk in chunker.chunk_stream(very_long_text):
    vectordb.insert(chunk)
```

### Chunk Deduplication

```python
from chunker import ChunkDeduplicator

dedup = ChunkDeduplicator()

# Within one document
unique = dedup.deduplicate(chunks)

# Across a batch (same content in doc A removed from doc B)
batch = chunker.chunk_batch(documents)
unique_batch = dedup.deduplicate_batch(batch)

# Count duplicates without removing
n = dedup.count_duplicates(chunks)
print(f"{n} duplicates found")
```

### Custom Metadata

```python
chunks = chunker.chunk(
    text,
    source_file="report.pdf",
    document_id="doc_123",
    extra_metadata={"author": "Jane Doe", "dept": "engineering"},
)

print(chunks[0].metadata.custom["author"])   # "Jane Doe"
```

### Arabic Text Processing

```python
config = ChunkerConfig(
    detect_language=True,
    arabic_normalize_alef=True,       # أ إ آ → ا
    arabic_normalize_yeh=True,        # ى → ي
    arabic_remove_tashkeel=True,      # remove diacritics
    arabic_remove_tatweel=True,       # remove ـ
    arabic_normalize_punctuation=True,
)
```

### Mixed Language Documents

```python
mixed_text = """
# Introduction
This is an English paragraph.

# المقدمة
هذه فقرة عربية.
"""

chunks = chunker.chunk(mixed_text)
for chunk in chunks:
    print(f"{chunk.metadata.language.value}: {chunk.text[:60]}")
```

---

## Document Readers

Convert files to plain text before chunking — no boilerplate:

### PDF

```python
from chunker.readers import PDFReader

reader = PDFReader()
text = reader.read("report.pdf")

# Or read page-by-page
pages = reader.read_pages("report.pdf")
```

Requires `pymupdf` (recommended) or `pdfplumber`.

### HTML

```python
from chunker.readers import HTMLReader

reader = HTMLReader()
text = reader.read("article.html")                  # from file
text = reader.read_string("<html>...</html>")        # from string
```

Strips scripts, nav, footer; preserves headings as `## Heading`.
Requires `beautifulsoup4`.

### EPUB

```python
from chunker.readers import EPUBReader

reader = EPUBReader(chapter_separator="\n\n---\n\n")
text = reader.read("book.epub")
```

Requires `ebooklib` + `beautifulsoup4`.

---

## Vectorstore Integrations

### ChromaDB

```python
import chromadb
from chunker.integrations import ChromaAdapter

client = chromadb.Client()
collection = client.get_or_create_collection("docs")

adapter = ChromaAdapter(chunker, collection)
ids = adapter.add_document("Document text...", doc_id="doc1")

results = adapter.search("query text", n_results=5)
# [{"id": ..., "text": ..., "metadata": ..., "distance": ...}]
```

### Pinecone

```python
import pinecone
from chunker.integrations import PineconeAdapter

pinecone.init(api_key="...", environment="...")
index = pinecone.Index("my-index")

adapter = PineconeAdapter(chunker, index, namespace="docs")
ids = adapter.add_document("Document text...", doc_id="doc1")

results = adapter.search(query_embedding, top_k=5)
# [{"id": ..., "score": ..., "metadata": ...}]
```

> Chunks must have embeddings (use `StrategyType.SEMANTIC`).

### Qdrant

```python
from qdrant_client import QdrantClient
from chunker.integrations import QdrantAdapter

client = QdrantClient(":memory:")
adapter = QdrantAdapter(chunker, client, "my_collection")
ids = adapter.add_document("Document text...", doc_id="doc1")

results = adapter.search(query_embedding, limit=5)
# [{"id": ..., "score": ..., "payload": ...}]
```

---

## Async Support

Use `AsyncSemanticChunker` when running inside FastAPI, aiohttp, or any asyncio application — it never blocks the event loop:

```python
from chunker import AsyncSemanticChunker, ChunkerConfig

async def main():
    config = ChunkerConfig(enable_embedding_cache=True)
    chunker = AsyncSemanticChunker(config)

    # Single document
    chunks = await chunker.chunk("Document text...")

    # Batch
    all_chunks = await chunker.chunk_batch(
        ["Doc 1...", "Doc 2..."],
        parallel=True,
        num_workers=4,
    )

    # Metrics passthrough
    metrics = chunker.get_metrics()
```

---

## Best Practices

### 1. Choose the Right Strategy

| Situation | Strategy |
|-----------|----------|
| Markdown / docs | `STRUCTURE_AWARE` |
| Long articles / books | `SEMANTIC` |
| Unknown structure | `RECURSIVE` |
| Short docs / chat | `SENTENCE` |
| Uniform chunks | `FIXED` |
| Academic papers | `HIERARCHICAL` |

### 2. RAG-Optimized Chunk Sizes

```python
config = ChunkerConfig(
    chunk_size=220,
    min_chunk_size=80,
    max_chunk_size=320,
    chunk_overlap=0,        # start with 0; add 20-50 if retrieval recall is low
)
```

### 3. Production Configuration

```python
config = ChunkerConfig(
    enable_embedding_cache=True,
    persist_cache_to_disk=True,
    cache_dir="./cache",
    enable_fallbacks=True,
    retry_on_network_error=True,
    max_retries=3,
    enable_metrics=True,
    enable_structured_logging=True,
)
```

### 4. Protect OpenAI Calls

Use both the built-in rate limiter and circuit breaker:

```python
from chunker.embeddings.openai_provider import OpenAIEmbeddingProvider
from chunker import CircuitBreaker

provider = OpenAIEmbeddingProvider(requests_per_minute=500)
cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

result = cb.call(provider.embed, texts)
```

### 5. Deduplicate Before Indexing

```python
from chunker import ChunkDeduplicator

dedup = ChunkDeduplicator()
batch = chunker.chunk_batch(docs)
batch = dedup.deduplicate_batch(batch)
```

### 6. Lazy Loading

```python
config = ChunkerConfig(
    lazy_load_embeddings=True,
    lazy_load_nlp_tools=True,
)
# Heavy models load only when first needed (~30% memory reduction)
```
