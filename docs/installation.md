# Installation Guide

## Requirements

- Python 3.9 or higher
- pip package manager

## Installation

### Basic Installation

```bash
pip install semantic-chunker
```

Or from source:

```bash
git clone https://github.com/yourusername/semantic-chunker.git
cd semantic-chunker
pip install -e .
```

### Core Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| `numpy` | Embedding computations | ✅ Always |
| `sentence-transformers` | Default embedding backend (BGE-M3) | Recommended |
| `nltk` | English sentence segmentation | Recommended |

### Optional Dependencies

Install what you need:

```bash
# OpenAI embeddings
pip install openai

# Arabic NLP
pip install stanza camel-tools

# Document format readers
pip install pymupdf                    # PDF (fast, recommended)
pip install pdfplumber                 # PDF (alternative)
pip install "beautifulsoup4[lxml]"     # HTML
pip install ebooklib                   # EPUB

# Vectorstore integrations
pip install chromadb                   # ChromaDB
pip install pinecone-client            # Pinecone
pip install qdrant-client              # Qdrant

# Development & testing
pip install pytest hypothesis
```

### Install Everything

```bash
pip install semantic-chunker[all]
# or from source:
pip install -e ".[all]"
```

## Verifying Installation

```python
from chunker import SemanticChunker, __version__

print(f"Semantic Chunker version: {__version__}")

chunker = SemanticChunker()
chunks = chunker.chunk("Hello world. This is a test.")
print(f"Successfully created {len(chunks)} chunk(s)")
```

## Troubleshooting

**ModuleNotFoundError for optional dependencies**
```bash
pip install sentence-transformers   # embeddings
pip install camel-tools             # Arabic support
pip install pymupdf                 # PDF reader
```

**NLTK data not found**
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

**Stanza models not found**
```python
import stanza
stanza.download('en')
stanza.download('ar')
```

**Out of memory errors**
```python
config = ChunkerConfig(
    embedding_cache_size=1000,
    enable_embedding_cache=False,
    lazy_load_embeddings=True,
)
```

**Legacy pickle cache files (upgrade from v1.0.0)**

Delete the old `.pkl` files and let the cache rebuild automatically:
```bash
rm ./cache/embedding_cache.pkl
rm ./cache/language_cache.pkl
```
Starting from v1.1.0 the cache uses `.npz` + `.json` which is
safer and more portable.

## Next Steps

- [Quick Start](quick_start.md) — up and running in 5 minutes
- [User Guide](user_guide.md) — complete feature reference
- [Structured Logging](structured_logging.md) — production observability
