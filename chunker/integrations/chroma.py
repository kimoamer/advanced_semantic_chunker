"""
ChromaDB integration adapter.

Requires::

    pip install chromadb
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from chunker.core import SemanticChunker
    from chunker.models import Chunk

__all__ = ["ChromaAdapter"]


class ChromaAdapter:
    """
    Adapter that indexes chunks directly into a ChromaDB collection.

    Parameters
    ----------
    chunker : SemanticChunker
        Chunker instance used to split documents.
    collection : Any
        A ``chromadb.Collection`` object (already created/retrieved).
    include_embeddings : bool
        If True and chunks carry pre-computed embeddings, those are
        forwarded to Chroma instead of letting Chroma recompute them
        (default: False).

    Examples
    --------
    >>> import chromadb
    >>> from chunker import SemanticChunker
    >>> from chunker.integrations import ChromaAdapter
    >>>
    >>> client = chromadb.Client()
    >>> collection = client.get_or_create_collection("my_docs")
    >>>
    >>> chunker = SemanticChunker()
    >>> adapter = ChromaAdapter(chunker, collection)
    >>>
    >>> adapter.add_document("Long document text...", doc_id="doc1")
    >>> results = adapter.search("What is the main topic?", n_results=5)
    """

    def __init__(
        self,
        chunker: "SemanticChunker",
        collection: Any,
        include_embeddings: bool = False,
    ) -> None:
        self.chunker = chunker
        self.collection = collection
        self.include_embeddings = include_embeddings

    # ── Public API ────────────────────────────────────────────────────────

    def add_document(
        self,
        text: str,
        doc_id: str = "",
        source_file: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Chunk a document and add all chunks to the collection.

        Parameters
        ----------
        text : str
            Document text.
        doc_id : str
            Document identifier (used as chunk ID prefix).
        source_file : str
            Optional source filename stored in metadata.
        extra_metadata : dict, optional
            Extra fields merged into each chunk's metadata.

        Returns
        -------
        List[str]
            IDs of the added chunks.
        """
        chunks = self.chunker.chunk(
            text,
            source_file=source_file,
            document_id=doc_id,
            extra_metadata=extra_metadata,
        )
        return self.add_chunks(chunks)

    def add_chunks(self, chunks: "List[Chunk]") -> List[str]:
        """
        Add pre-computed chunks to the ChromaDB collection.

        Parameters
        ----------
        chunks : List[Chunk]
            Chunks to index.

        Returns
        -------
        List[str]
            IDs of the added chunks.
        """
        if not chunks:
            return []

        ids = [c.metadata.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [self._serialize_metadata(c) for c in chunks]

        kwargs: Dict[str, Any] = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }

        if self.include_embeddings:
            embeddings = [
                c.embedding.tolist() if c.embedding is not None else None
                for c in chunks
            ]
            if any(e is not None for e in embeddings):
                kwargs["embeddings"] = embeddings

        self.collection.add(**kwargs)
        return ids

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the collection for chunks similar to *query*.

        Parameters
        ----------
        query : str
            Natural language query.
        n_results : int
            Number of results to return (default: 5).
        where : dict, optional
            Chroma metadata filter.

        Returns
        -------
        List[dict]
            Each result dict has keys ``id``, ``text``, ``metadata``,
            ``distance``.
        """
        kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where

        response = self.collection.query(**kwargs)

        results = []
        for i, doc_id in enumerate(response["ids"][0]):
            results.append({
                "id": doc_id,
                "text": response["documents"][0][i],
                "metadata": response["metadatas"][0][i],
                "distance": response["distances"][0][i],
            })
        return results

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _serialize_metadata(chunk: "Chunk") -> Dict[str, Any]:
        """Convert ChunkMetadata to a flat dict of scalar values for Chroma."""
        m = chunk.metadata
        return {
            "chunk_index": m.chunk_index,
            "total_chunks": m.total_chunks,
            "language": m.language.value if hasattr(m.language, "value") else str(m.language),
            "token_count": m.token_count,
            "char_count": m.char_count,
            "source_file": m.source_file or "",
            "document_id": m.document_id or "",
            "strategy": m.strategy or "",
            "start_char": m.start_char,
            "end_char": m.end_char,
        }
