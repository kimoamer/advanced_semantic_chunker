"""
Cache Manager for the Semantic Chunker.

Provides unified caching for embeddings and language detection results
with LRU eviction and optional disk persistence.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np

from chunker.models import Language

__all__ = ["CacheManager"]


class CacheManager:
    """
    Unified caching for embeddings and language detection.
    
    Uses LRU (Least Recently Used) eviction policy when cache size exceeds
    configured limits. Thread-safe for concurrent access.
    
    Parameters
    ----------
    embedding_cache_size : int
        Maximum number of embeddings to cache (default: 10000).
    lang_detect_cache_size : int
        Maximum number of language detection results to cache (default: 5000).
    persist_to_disk : bool
        Whether to persist caches to disk (default: False).
    cache_dir : Optional[str]
        Directory for cache persistence (default: None).
        
    Examples
    --------
    >>> cache = CacheManager(embedding_cache_size=1000)
    >>> embedding = np.array([0.1, 0.2, 0.3])
    >>> cache.set_embedding("Hello world", "model-name", embedding)
    >>> cached = cache.get_embedding("Hello world", "model-name")
    >>> assert np.array_equal(cached, embedding)
    """
    
    def __init__(
        self,
        embedding_cache_size: int = 10000,
        lang_detect_cache_size: int = 5000,
        persist_to_disk: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """Initialize cache manager with configurable sizes."""
        self.embedding_cache_size = embedding_cache_size
        self.lang_detect_cache_size = lang_detect_cache_size
        self.persist_to_disk = persist_to_disk
        self.cache_dir = cache_dir
        
        # LRU caches using OrderedDict
        self._embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._language_cache: OrderedDict[str, Language] = OrderedDict()
        
        # Thread locks for concurrent access
        self._embedding_lock = threading.Lock()
        self._language_lock = threading.Lock()
        
        # Statistics tracking
        self._embedding_hits = 0
        self._embedding_misses = 0
        self._embedding_evictions = 0
        self._language_hits = 0
        self._language_misses = 0
        self._language_evictions = 0
    
    def _compute_cache_key(self, text: str, model: str) -> str:
        """
        Compute MD5 hash for cache key.

        Uses MD5 (non-cryptographic, ~3x faster than SHA-256) since this
        is a performance cache, not a security context.

        Parameters
        ----------
        text : str
            The text content to hash.
        model : str
            The model name to include in the hash.

        Returns
        -------
        str
            MD5 hash as hexadecimal string.
        """
        content = f"{model}:{text}"
        return hashlib.md5(content.encode("utf-8"), usedforsecurity=False).hexdigest()
    
    def get_embedding(self, text: str, model: str) -> Optional[np.ndarray]:
        """
        Get cached embedding by content hash.
        
        Parameters
        ----------
        text : str
            The text whose embedding to retrieve.
        model : str
            The embedding model name.
            
        Returns
        -------
        Optional[np.ndarray]
            Cached embedding array if found, None otherwise.
        """
        cache_key = self._compute_cache_key(text, model)
        
        with self._embedding_lock:
            if cache_key in self._embedding_cache:
                # Move to end (most recently used)
                self._embedding_cache.move_to_end(cache_key)
                self._embedding_hits += 1
                return self._embedding_cache[cache_key]
            else:
                self._embedding_misses += 1
                return None
    
    def set_embedding(self, text: str, model: str, embedding: np.ndarray) -> None:
        """
        Cache embedding with LRU eviction.
        
        If cache is full, removes the least recently used entry before
        adding the new one.
        
        Parameters
        ----------
        text : str
            The text content.
        model : str
            The embedding model name.
        embedding : np.ndarray
            The embedding vector to cache.
        """
        cache_key = self._compute_cache_key(text, model)
        
        with self._embedding_lock:
            # If key exists, move to end
            if cache_key in self._embedding_cache:
                self._embedding_cache.move_to_end(cache_key)
            
            # Add new entry
            self._embedding_cache[cache_key] = embedding
            
            # Evict LRU if over size limit
            if len(self._embedding_cache) > self.embedding_cache_size:
                # Remove first item (least recently used)
                self._embedding_cache.popitem(last=False)
                self._embedding_evictions += 1
    
    def get_language(self, text: str) -> Optional[Language]:
        """
        Get cached language detection result.
        
        Parameters
        ----------
        text : str
            The text whose language to retrieve.
            
        Returns
        -------
        Optional[Language]
            Cached language if found, None otherwise.
        """
        # Use text hash as key (no model needed for language detection)
        cache_key = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()

        with self._language_lock:
            if cache_key in self._language_cache:
                # Move to end (most recently used)
                self._language_cache.move_to_end(cache_key)
                self._language_hits += 1
                return self._language_cache[cache_key]
            else:
                self._language_misses += 1
                return None

    def set_language(self, text: str, language: Language) -> None:
        """
        Cache language detection result.

        If cache is full, removes the least recently used entry before
        adding the new one.

        Parameters
        ----------
        text : str
            The text content.
        language : Language
            The detected language.
        """
        cache_key = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()
        
        with self._language_lock:
            # If key exists, move to end
            if cache_key in self._language_cache:
                self._language_cache.move_to_end(cache_key)
            
            # Add new entry
            self._language_cache[cache_key] = language
            
            # Evict LRU if over size limit
            if len(self._language_cache) > self.lang_detect_cache_size:
                # Remove first item (least recently used)
                self._language_cache.popitem(last=False)
                self._language_evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache hit/miss statistics.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing cache statistics including:
            - embedding_hits, embedding_misses, embedding_hit_rate, embedding_evictions
            - language_hits, language_misses, language_hit_rate, language_evictions
            - embedding_cache_size, language_cache_size
            - total_accesses (hits + misses)
        """
        with self._embedding_lock:
            embedding_total = self._embedding_hits + self._embedding_misses
            embedding_hit_rate = (
                self._embedding_hits / embedding_total if embedding_total > 0 else 0.0
            )
            embedding_size = len(self._embedding_cache)
            embedding_evictions = self._embedding_evictions
        
        with self._language_lock:
            language_total = self._language_hits + self._language_misses
            language_hit_rate = (
                self._language_hits / language_total if language_total > 0 else 0.0
            )
            language_size = len(self._language_cache)
            language_evictions = self._language_evictions
        
        return {
            "embedding_hits": self._embedding_hits,
            "embedding_misses": self._embedding_misses,
            "embedding_hit_rate": embedding_hit_rate,
            "embedding_evictions": embedding_evictions,
            "embedding_total_accesses": embedding_total,
            "embedding_cache_size": embedding_size,
            "embedding_cache_max_size": self.embedding_cache_size,
            "language_hits": self._language_hits,
            "language_misses": self._language_misses,
            "language_hit_rate": language_hit_rate,
            "language_evictions": language_evictions,
            "language_total_accesses": language_total,
            "language_cache_size": language_size,
            "language_cache_max_size": self.lang_detect_cache_size,
        }
    
    def clear(self) -> None:
        """Clear all caches and reset statistics."""
        with self._embedding_lock:
            self._embedding_cache.clear()
            self._embedding_hits = 0
            self._embedding_misses = 0
            self._embedding_evictions = 0
        
        with self._language_lock:
            self._language_cache.clear()
            self._language_hits = 0
            self._language_misses = 0
            self._language_evictions = 0
    
    def save_to_disk(self) -> None:
        """
        Persist caches to disk using JSON + numpy .npz format.

        Saves embedding cache as ``embedding_cache.npz`` (numpy arrays)
        with a companion ``embedding_cache_keys.json`` for key ordering,
        and language cache as ``language_cache.json``.

        This replaces the previous pickle-based serialisation which was a
        security risk when loading files from untrusted sources.

        Raises
        ------
        ValueError
            If cache_dir is not configured.
        IOError
            If file writing fails, includes the file path in the error message.

        Examples
        --------
        >>> cache = CacheManager(persist_to_disk=True, cache_dir="./cache")
        >>> cache.set_embedding("text", "model", np.array([0.1, 0.2]))
        >>> cache.save_to_disk()
        """
        import io
        import json
        import os

        if self.cache_dir is None:
            raise ValueError(
                "cache_dir must be configured to use disk persistence. "
                "Set cache_dir parameter when creating CacheManager."
            )

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError as e:
            raise IOError(
                f"Failed to create cache directory '{self.cache_dir}': {e}"
            ) from e

        # ── Save embedding cache ──────────────────────────────────────────
        emb_npz_path = os.path.join(self.cache_dir, "embedding_cache.npz")
        emb_meta_path = os.path.join(self.cache_dir, "embedding_cache_meta.json")

        try:
            with self._embedding_lock:
                keys = list(self._embedding_cache.keys())
                arrays = {f"arr_{i}": v for i, v in enumerate(self._embedding_cache.values())}
                meta = {
                    "keys": keys,
                    "hits": self._embedding_hits,
                    "misses": self._embedding_misses,
                    "evictions": self._embedding_evictions,
                }

            buf = io.BytesIO()
            np.savez_compressed(buf, **arrays)
            with open(emb_npz_path, "wb") as f:
                f.write(buf.getvalue())
            with open(emb_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
        except (OSError, IOError) as e:
            raise IOError(
                f"Failed to save embedding cache to '{emb_npz_path}': {e}"
            ) from e

        # ── Save language cache ───────────────────────────────────────────
        lang_path = os.path.join(self.cache_dir, "language_cache.json")

        try:
            with self._language_lock:
                lang_data = {
                    "cache": {k: v.value for k, v in self._language_cache.items()},
                    "hits": self._language_hits,
                    "misses": self._language_misses,
                    "evictions": self._language_evictions,
                }
            with open(lang_path, "w", encoding="utf-8") as f:
                json.dump(lang_data, f)
        except (OSError, IOError) as e:
            raise IOError(
                f"Failed to save language cache to '{lang_path}': {e}"
            ) from e

    def load_from_disk(self) -> None:
        """
        Load caches from disk (JSON + numpy .npz format).

        Loads the embedding cache from ``embedding_cache.npz`` /
        ``embedding_cache_meta.json`` and the language cache from
        ``language_cache.json``.  If the files don't exist the caches
        start empty.  Falls back gracefully if only one cache file is
        present.

        Raises
        ------
        ValueError
            If cache_dir is not configured.
        IOError
            If file reading fails, includes the file path in the error message.

        Examples
        --------
        >>> cache = CacheManager(persist_to_disk=True, cache_dir="./cache")
        >>> cache.load_from_disk()
        """
        import json
        import os

        if self.cache_dir is None:
            raise ValueError(
                "cache_dir must be configured to use disk persistence. "
                "Set cache_dir parameter when creating CacheManager."
            )

        # ── Load embedding cache ──────────────────────────────────────────
        emb_npz_path = os.path.join(self.cache_dir, "embedding_cache.npz")
        emb_meta_path = os.path.join(self.cache_dir, "embedding_cache_meta.json")

        if os.path.exists(emb_npz_path) and os.path.exists(emb_meta_path):
            try:
                npz = np.load(emb_npz_path, allow_pickle=False)
                with open(emb_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                keys = meta["keys"]
                restored: OrderedDict = OrderedDict()
                for i, key in enumerate(keys):
                    arr_key = f"arr_{i}"
                    if arr_key in npz:
                        restored[key] = npz[arr_key]

                with self._embedding_lock:
                    self._embedding_cache = restored
                    self._embedding_hits = meta.get("hits", 0)
                    self._embedding_misses = meta.get("misses", 0)
                    self._embedding_evictions = meta.get("evictions", 0)
            except (OSError, IOError, ValueError) as e:
                raise IOError(
                    f"Failed to load embedding cache from '{emb_npz_path}': {e}"
                ) from e
        else:
            # Legacy: try old pickle file for backwards compatibility
            legacy_pkl = os.path.join(self.cache_dir, "embedding_cache.pkl")
            if os.path.exists(legacy_pkl):
                raise IOError(
                    f"Legacy pickle cache found at '{legacy_pkl}'. "
                    "Pickle files are no longer supported for security reasons. "
                    "Delete the file and let the cache rebuild."
                )

        # ── Load language cache ───────────────────────────────────────────
        lang_path = os.path.join(self.cache_dir, "language_cache.json")

        if os.path.exists(lang_path):
            try:
                with open(lang_path, "r", encoding="utf-8") as f:
                    lang_data = json.load(f)

                from chunker.models import Language as _Language
                restored_lang: OrderedDict = OrderedDict(
                    (k, _Language(v)) for k, v in lang_data["cache"].items()
                )
                with self._language_lock:
                    self._language_cache = restored_lang
                    self._language_hits = lang_data.get("hits", 0)
                    self._language_misses = lang_data.get("misses", 0)
                    self._language_evictions = lang_data.get("evictions", 0)
            except (OSError, IOError, ValueError) as e:
                raise IOError(
                    f"Failed to load language cache from '{lang_path}': {e}"
                ) from e
        else:
            # Legacy: warn about old pickle file
            legacy_pkl = os.path.join(self.cache_dir, "language_cache.pkl")
            if os.path.exists(legacy_pkl):
                raise IOError(
                    f"Legacy pickle cache found at '{legacy_pkl}'. "
                    "Pickle files are no longer supported for security reasons. "
                    "Delete the file and let the cache rebuild."
                )
