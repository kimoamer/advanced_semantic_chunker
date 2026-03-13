# Quick Start Guide

Get up and running with Semantic Chunker in 5 minutes.

## Installation

```bash
pip install semantic-chunker
```

## Basic Usage

### 1. Simple Chunking

```python
from chunker import SemanticChunker

chunker = SemanticChunker()

text = """
Machine learning is a subset of artificial intelligence.
It focuses on teaching computers to learn from data.

Deep learning is a type of machine learning. It uses neural
networks with multiple layers to process information.
"""

chunks = chunker.chunk(text)

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {chunk.text[:50]}...")
    print(f"  Tokens: {chunk.metadata.token_count}")
```

### 2. Custom Configuration

```python
from chunker import SemanticChunker, ChunkerConfig, StrategyType

config = ChunkerConfig(
    strategy=StrategyType.SEMANTIC,
    chunk_size=512,
    chunk_overlap=50,
    enable_embedding_cache=True,
)

chunker = SemanticChunker(config)
chunks = chunker.chunk(text)
```

### 3. Process Multiple Documents

```python
from chunker.batch_processor import BatchProcessor

processor = BatchProcessor(chunker)

documents = ["First document...", "Second document...", "Third document..."]

# Sequential
results = processor.process_batch(documents)

# Parallel thread-based (good for I/O-bound work)
results = processor.process_batch(documents, parallel=True, num_workers=4)

# Parallel process-based (better for CPU-heavy workloads)
processor_cpu = BatchProcessor(chunker, executor_type="process")
results = processor_cpu.process_batch(documents, parallel=True, num_workers=4)

print(f"Processed {len(results)} documents")
```

### 4. Stream Chunks (Memory-Efficient)

For very large documents — yield one chunk at a time without loading everything in memory:

```python
for chunk in chunker.chunk_stream(very_long_text):
    store_in_vectordb(chunk)
```

### 5. Async Support (FastAPI / asyncio)

```python
import asyncio
from chunker import AsyncSemanticChunker

async def process():
    chunker = AsyncSemanticChunker()
    chunks = await chunker.chunk("Document text here...")
    return chunks

asyncio.run(process())

# FastAPI example:
# @app.post("/chunk")
# async def chunk_document(text: str):
#     chunker = AsyncSemanticChunker()
#     return await chunker.chunk(text)
```

### 6. Access Rich Metadata

```python
chunks = chunker.chunk(text)

for chunk in chunks:
    print(f"Text:       {chunk.text[:80]}")
    print(f"Language:   {chunk.metadata.language.value}")
    print(f"Tokens:     {chunk.metadata.token_count}")
    print(f"Strategy:   {chunk.metadata.strategy}")
    print(f"Has code:   {chunk.metadata.contains_code}")
    print(f"Section:    {chunk.metadata.section_title}")
```

### 7. Enable Caching

```python
config = ChunkerConfig(
    enable_embedding_cache=True,
    embedding_cache_size=10000,
    persist_cache_to_disk=True,   # Survives restarts
    cache_dir="./cache",
)

chunker = SemanticChunker(config)
chunks = chunker.chunk(text)    # First call — cache miss
chunks = chunker.chunk(text)    # Second call — cache hit (faster)
```

### 8. Monitor Performance

```python
config = ChunkerConfig(enable_metrics=True)
chunker = SemanticChunker(config)

for doc in documents:
    chunker.chunk(doc)

metrics = chunker.get_metrics()
print(f"Documents:      {metrics.documents_processed}")
print(f"Chunks:         {metrics.total_chunks_created}")
print(f"Avg time:       {metrics.avg_processing_time_ms:.1f} ms")
print(f"Cache hit rate: {metrics.embedding_cache_hit_rate:.0%}")
```

### 9. Handle Arabic Text

```python
config = ChunkerConfig(
    detect_language=True,
    arabic_normalize_alef=True,
    arabic_remove_tashkeel=True,
)

chunker = SemanticChunker(config)
arabic_text = "الذكاء الاصطناعي هو فرع من علوم الحاسوب."
chunks = chunker.chunk(arabic_text)

for chunk in chunks:
    print(f"Language: {chunk.metadata.language.value}")
```

### 10. Deduplicate Chunks

Useful in batch pipelines where the same paragraph appears in multiple documents:

```python
from chunker import ChunkDeduplicator

dedup = ChunkDeduplicator()

batch_results = chunker.chunk_batch(documents)
unique_results = dedup.deduplicate_batch(batch_results)
```

### 11. Read Documents from Files

```python
from chunker.readers import PDFReader, HTMLReader, EPUBReader

# PDF
pdf_reader = PDFReader()
text = pdf_reader.read("report.pdf")
chunks = chunker.chunk(text, source_file="report.pdf")

# HTML
html_reader = HTMLReader()
text = html_reader.read("article.html")

# EPUB (e-books)
epub_reader = EPUBReader()
text = epub_reader.read("book.epub")
```

### 12. Index into a Vectorstore

```python
import chromadb
from chunker.integrations import ChromaAdapter

client = chromadb.Client()
collection = client.get_or_create_collection("my_docs")

adapter = ChromaAdapter(chunker, collection)
adapter.add_document("Long document text...", doc_id="doc1")

results = adapter.search("What is machine learning?", n_results=5)
for r in results:
    print(r["text"][:100])
```

Also available: `PineconeAdapter`, `QdrantAdapter`.

### 13. Protect External APIs with Circuit Breaker

```python
from chunker import CircuitBreaker, CircuitBreakerOpenError

cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

try:
    result = cb.call(my_embedding_provider.embed, texts)
except CircuitBreakerOpenError:
    result = fallback_strategy(texts)

print(f"Circuit state: {cb.state}")   # "closed" / "open" / "half_open"
```

### 14. Error Handling

```python
from chunker.exceptions import ChunkerException

config = ChunkerConfig(enable_fallbacks=True)
chunker = SemanticChunker(config)

try:
    chunks = chunker.chunk(text)
except ChunkerException as e:
    print(f"Error: {e}")
    print(f"Component: {e.context.get('component')}")
```

---

## Chunking Strategies Comparison

| Strategy | Best For | Speed | Quality |
|----------|----------|-------|---------|
| Structure-Aware | Markdown, docs | Fast | High |
| Semantic | Articles, books | Slow | Highest |
| Recursive | General text | Medium | Medium |
| Sentence | Short docs | Fast | Medium |
| Fixed | Uniform chunks | Fastest | Low |
| Hierarchical | Academic papers | Medium | High |
| Agentic | Custom splitting | Slowest | Best |

## Configuration Presets

```python
# Fast processing (no embeddings)
config = ChunkerConfig(
    strategy=StrategyType.SENTENCE,
    lazy_load_embeddings=True,
)

# Balanced — good for most RAG use cases
config = ChunkerConfig(
    strategy=StrategyType.STRUCTURE_AWARE,
    chunk_size=220,
    enable_embedding_cache=True,
)

# Maximum quality
config = ChunkerConfig(
    strategy=StrategyType.SEMANTIC,
    embedding_model="BAAI/bge-m3",
    threshold_amount=95.0,
    compute_coherence_score=True,
)

# Arabic optimized
config = ChunkerConfig(
    detect_language=True,
    arabic_normalize_alef=True,
    arabic_normalize_yeh=True,
    arabic_remove_tashkeel=True,
)

# High-throughput production
config = ChunkerConfig(
    enable_embedding_cache=True,
    persist_cache_to_disk=True,
    cache_dir="./cache",
    enable_metrics=True,
    enable_structured_logging=True,
    retry_on_network_error=True,
    max_retries=3,
)
```

## Next Steps

- [User Guide](user_guide.md) — detailed documentation for all features
- [Structured Logging](structured_logging.md) — production observability
- [Installation](installation.md) — optional dependencies and troubleshooting
