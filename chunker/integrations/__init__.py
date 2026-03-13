"""
Vectorstore integration adapters for the Semantic Chunker.

Each adapter wraps a vectorstore client and provides a simple
``add_document`` / ``add_chunks`` / ``search`` interface so you can
go from raw text to indexed, searchable vectors in a few lines.

Available adapters
------------------
- :class:`ChromaAdapter`   — ChromaDB (requires ``chromadb``)
- :class:`PineconeAdapter` — Pinecone (requires ``pinecone-client``)
- :class:`QdrantAdapter`   — Qdrant (requires ``qdrant-client``)
"""

from chunker.integrations.chroma import ChromaAdapter
from chunker.integrations.pinecone import PineconeAdapter
from chunker.integrations.qdrant import QdrantAdapter

__all__ = ["ChromaAdapter", "PineconeAdapter", "QdrantAdapter"]
