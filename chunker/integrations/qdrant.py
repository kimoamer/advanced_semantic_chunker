"""
Qdrant integration adapter.

Requires::

    pip install qdrant-client
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from chunker.core import SemanticChunker
    from chunker.models import Chunk

__all__ = ["QdrantAdapter"]


class QdrantAdapter:
    """
    Adapter that upserts chunks into a Qdrant collection.

    Chunks must carry pre-computed embeddings.

    Parameters
    ----------
    chunker : SemanticChunker
        Chunker instance used to split documents.
    client : Any
        A ``qdrant_client.QdrantClient`` instance.
    collection_name : str
        Name of the Qdrant collection to write to.

    Examples
    --------
    >>> from qdrant_client import QdrantClient
    >>> from chunker import SemanticChunker, ChunkerConfig, StrategyType
    >>> from chunker.integrations import QdrantAdapter
    >>>
    >>> client = QdrantClient(":memory:")
    >>> config = ChunkerConfig(strategy=StrategyType.SEMANTIC)
    >>> chunker = SemanticChunker(config)
    >>> adapter = QdrantAdapter(chunker, client, "my_collection")
    >>>
    >>> adapter.add_document("Long document...", doc_id="doc1")
    """

    def __init__(
        self,
        chunker: "SemanticChunker",
        client: Any,
        collection_name: str,
    ) -> None:
        self.chunker = chunker
        self.client = client
        self.collection_name = collection_name

    # ── Public API ────────────────────────────────────────────────────────

    def add_document(
        self,
        text: str,
        doc_id: str = "",
        source_file: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 64,
    ) -> List[str]:
        """
        Chunk a document and upsert all chunks into Qdrant.

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
            Qdrant upsert batch size (default: 64).

        Returns
        -------
        List[str]
            IDs of the upserted points.
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
        batch_size: int = 64,
    ) -> List[str]:
        """
        Upsert pre-computed chunks into Qdrant.

        Parameters
        ----------
        chunks : List[Chunk]
            Chunks with ``embedding`` populated.
        batch_size : int
            Upsert batch size (default: 64).

        Returns
        -------
        List[str]
            IDs of upserted points.

        Raises
        ------
        ValueError
            If any chunk is missing its embedding.
        ImportError
            If ``qdrant-client`` is not installed.
        """
        try:
            from qdrant_client.models import PointStruct  # type: ignore
        except ImportError:
            raise ImportError(
                "qdrant-client is required for QdrantAdapter. "
                "Install it with: pip install qdrant-client"
            )

        if not chunks:
            return []

        points = []
        ids = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(
                    f"Chunk '{chunk.metadata.chunk_id}' has no embedding."
                )
            point_id = chunk.metadata.chunk_id
            ids.append(point_id)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=chunk.embedding.tolist(),
                    payload=self._serialize_metadata(chunk),
                )
            )

        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
            )

        return ids

    def search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        query_filter: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search Qdrant for nearest neighbours.

        Parameters
        ----------
        query_embedding : List[float]
            Query vector.
        limit : int
            Number of results to return (default: 5).
        query_filter : qdrant_client.models.Filter, optional
            Qdrant filter object.

        Returns
        -------
        List[dict]
            Each result has keys ``id``, ``score``, ``payload``.
        """
        kwargs: Dict[str, Any] = {
            "collection_name": self.collection_name,
            "query_vector": query_embedding,
            "limit": limit,
        }
        if query_filter is not None:
            kwargs["query_filter"] = query_filter

        hits = self.client.search(**kwargs)
        return [
            {"id": hit.id, "score": hit.score, "payload": hit.payload}
            for hit in hits
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
            "start_char": m.start_char,
            "end_char": m.end_char,
        }
