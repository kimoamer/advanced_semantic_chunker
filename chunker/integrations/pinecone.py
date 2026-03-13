"""
Pinecone integration adapter.

Requires::

    pip install pinecone-client
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from chunker.core import SemanticChunker
    from chunker.models import Chunk

__all__ = ["PineconeAdapter"]


class PineconeAdapter:
    """
    Adapter that upserts chunks into a Pinecone index.

    Chunks *must* have embeddings attached (either via the semantic
    strategy or by passing a pre-loaded embedding provider to the
    chunker).  Pinecone is a vector-first store — it requires a numeric
    embedding for every vector.

    Parameters
    ----------
    chunker : SemanticChunker
        Chunker instance used to split documents.
    index : Any
        A ``pinecone.Index`` object (already initialised).
    namespace : str
        Pinecone namespace (default: ``""``).

    Examples
    --------
    >>> import pinecone
    >>> from chunker import SemanticChunker, ChunkerConfig, StrategyType
    >>> from chunker.integrations import PineconeAdapter
    >>>
    >>> pinecone.init(api_key="...", environment="...")
    >>> index = pinecone.Index("my-index")
    >>>
    >>> config = ChunkerConfig(strategy=StrategyType.SEMANTIC)
    >>> chunker = SemanticChunker(config)
    >>> adapter = PineconeAdapter(chunker, index)
    >>>
    >>> adapter.add_document("Long document...", doc_id="doc1")
    """

    def __init__(
        self,
        chunker: "SemanticChunker",
        index: Any,
        namespace: str = "",
    ) -> None:
        self.chunker = chunker
        self.index = index
        self.namespace = namespace

    # ── Public API ────────────────────────────────────────────────────────

    def add_document(
        self,
        text: str,
        doc_id: str = "",
        source_file: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 100,
    ) -> List[str]:
        """
        Chunk a document and upsert all chunks into Pinecone.

        Parameters
        ----------
        text : str
            Document text.
        doc_id : str
            Document identifier.
        source_file : str
            Optional source filename.
        extra_metadata : dict, optional
            Additional metadata fields.
        batch_size : int
            Pinecone upsert batch size (default: 100).

        Returns
        -------
        List[str]
            IDs of the upserted vectors.
        """
        chunks = self.chunker.chunk(
            text,
            source_file=source_file,
            document_id=doc_id,
            extra_metadata=extra_metadata,
        )
        return self.add_chunks(chunks, batch_size=batch_size)

    def add_chunks(
        self,
        chunks: "List[Chunk]",
        batch_size: int = 100,
    ) -> List[str]:
        """
        Upsert pre-computed chunks into Pinecone.

        Parameters
        ----------
        chunks : List[Chunk]
            Chunks with ``embedding`` attribute populated.
        batch_size : int
            Pinecone upsert batch size (default: 100).

        Returns
        -------
        List[str]
            IDs of upserted vectors.

        Raises
        ------
        ValueError
            If any chunk is missing its embedding.
        """
        if not chunks:
            return []

        vectors = []
        ids = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(
                    f"Chunk '{chunk.metadata.chunk_id}' has no embedding. "
                    "Use StrategyType.SEMANTIC or attach embeddings before upserting."
                )
            vec_id = chunk.metadata.chunk_id
            ids.append(vec_id)
            vectors.append({
                "id": vec_id,
                "values": chunk.embedding.tolist(),
                "metadata": self._serialize_metadata(chunk),
            })

        # Upsert in batches to stay within Pinecone limits
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(
                vectors=vectors[i : i + batch_size],
                namespace=self.namespace,
            )

        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query Pinecone for the nearest neighbours of *query_embedding*.

        Parameters
        ----------
        query_embedding : List[float]
            Query vector (same dimension as the indexed embeddings).
        top_k : int
            Number of results to return (default: 5).
        filter : dict, optional
            Pinecone metadata filter.

        Returns
        -------
        List[dict]
            Each result has keys ``id``, ``score``, ``metadata``.
        """
        kwargs: Dict[str, Any] = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": self.namespace,
        }
        if filter:
            kwargs["filter"] = filter

        response = self.index.query(**kwargs)
        return [
            {"id": m["id"], "score": m["score"], "metadata": m.get("metadata", {})}
            for m in response.get("matches", [])
        ]

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _serialize_metadata(chunk: "Chunk") -> Dict[str, Any]:
        m = chunk.metadata
        return {
            "text": chunk.text,
            "chunk_index": m.chunk_index,
            "language": m.language.value if hasattr(m.language, "value") else str(m.language),
            "token_count": m.token_count,
            "source_file": m.source_file or "",
            "document_id": m.document_id or "",
            "strategy": m.strategy or "",
        }
