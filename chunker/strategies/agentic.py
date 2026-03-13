"""
Agentic chunking strategy — LLM-driven dynamic splitting.

Based on IBM watsonx's agentic chunking research:
  - Uses an LLM to analyze document structure
  - Dynamically decides where to place chunk boundaries
  - Enriches chunks with topic metadata
  - Most computationally expensive but highest quality

Requires OpenAI API key (or compatible API).
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from chunker.config import ChunkerConfig
from chunker.strategies.base import BaseStrategy


_AGENTIC_SYSTEM_PROMPT = """You are an expert document analyst. Your task is to split a document into semantically coherent chunks for a RAG (Retrieval-Augmented Generation) system.

Rules:
1. Each chunk should cover ONE coherent topic or idea
2. Never split in the middle of a sentence
3. Each chunk should be self-contained and understandable without surrounding context
4. Respect document structure (headers, lists, tables, code blocks)
5. Target chunk size: {chunk_size} tokens (roughly {char_target} characters)
6. Minimum chunk size: {min_chunk_size} tokens
7. Maximum chunk size: {max_chunk_size} tokens
8. The document may contain English, Arabic, or mixed text — handle both correctly
9. For Arabic text, respect Arabic sentence boundaries (؟ ، ؛)

Return a JSON array where each element is an object with:
  - "text": the chunk text (exact text from the document, no modifications)
  - "topic": a short topic label for this chunk (in the same language as the content)

IMPORTANT: Return ONLY the JSON array, no other text."""

_AGENTIC_USER_PROMPT = """Split the following document into semantic chunks:

---
{text}
---

Return the JSON array of chunks:"""


class AgenticStrategy(BaseStrategy):
    """
    LLM-driven agentic chunker.

    Uses an LLM (OpenAI-compatible) to intelligently decide chunk
    boundaries based on semantic content analysis.

    Falls back to sentence-based strategy if the LLM call fails.
    """

    def __init__(self, config: ChunkerConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazy-load the OpenAI client."""
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for AgenticStrategy. "
                "Install it with: pip install openai"
            )

        api_key = self.config.openai_api_key
        if not api_key:
            import os
            api_key = os.environ.get("OPENAI_API_KEY", "")

        if not api_key:
            raise ValueError(
                "OpenAI API key is required for agentic chunking. "
                "Set config.openai_api_key or OPENAI_API_KEY env var."
            )

        self._client = OpenAI(api_key=api_key)
        return self._client

    def chunk(self, sentences: List[str], text: str = "") -> List[str]:
        """Use LLM to intelligently chunk the document."""
        source = text if text else " ".join(sentences)

        if not source.strip():
            return []

        # For very short text, no need for LLM
        if self._estimate_tokens(source) <= self.config.chunk_size:
            return [source.strip()]

        try:
            return self._llm_chunk(source)
        except Exception as e:
            if self.config.verbose:
                print(f"[AgenticStrategy] LLM chunking failed: {e}")
                print("[AgenticStrategy] Falling back to sentence-based chunking")
            return self._fallback_chunk(sentences, source)

    def _llm_chunk(self, text: str) -> List[str]:
        """Call the LLM to perform chunking."""
        client = self._get_client()

        system_prompt = _AGENTIC_SYSTEM_PROMPT.format(
            chunk_size=self.config.chunk_size,
            char_target=self.config.chunk_size * 4,
            min_chunk_size=self.config.min_chunk_size,
            max_chunk_size=self.config.max_chunk_size,
        )

        # For very long documents, process in windows
        max_input_tokens = 12000  # Leave room for response
        if self._estimate_tokens(text) > max_input_tokens:
            return self._chunked_llm_processing(text, system_prompt)

        user_prompt = _AGENTIC_USER_PROMPT.format(text=text)

        response = client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=4096,
        )

        result = response.choices[0].message.content.strip()
        return self._parse_llm_response(result)

    def _chunked_llm_processing(
        self, text: str, system_prompt: str
    ) -> List[str]:
        """Process very long documents by splitting into windows first."""
        # First pass: split into manageable sections
        sections = re.split(r"\n{2,}", text)
        windows: List[str] = []
        current = ""

        max_window_chars = 12000 * 4  # ~12000 tokens

        for section in sections:
            if len(current) + len(section) > max_window_chars:
                if current:
                    windows.append(current)
                current = section
            else:
                current = current + "\n\n" + section if current else section

        if current:
            windows.append(current)

        # Process each window with the LLM
        all_chunks: List[str] = []
        for window in windows:
            user_prompt = _AGENTIC_USER_PROMPT.format(text=window)
            try:
                response = self._get_client().chat.completions.create(
                    model=self.config.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=4096,
                )
                result = response.choices[0].message.content.strip()
                window_chunks = self._parse_llm_response(result)
                all_chunks.extend(window_chunks)
            except Exception:
                # Fallback for this window
                all_chunks.append(window.strip())

        return all_chunks

    def _parse_llm_response(self, response: str) -> List[str]:
        """Parse LLM JSON response into chunk list."""
        # Strip markdown code fences if present
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r"^```(?:json)?\s*\n?", "", response)
            response = re.sub(r"\n?```\s*$", "", response)

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON array in the response
            match = re.search(r"\[[\s\S]*\]", response)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return [response]
            else:
                return [response]

        if isinstance(data, list):
            chunks = []
            for item in data:
                if isinstance(item, dict) and "text" in item:
                    chunks.append(item["text"])
                elif isinstance(item, str):
                    chunks.append(item)
            return chunks if chunks else [response]

        return [response]

    def _fallback_chunk(self, sentences: List[str], text: str) -> List[str]:
        """Fallback to sentence-based chunking when LLM fails."""
        from chunker.strategies.sentence import SentenceStrategy

        fallback = SentenceStrategy(self.config)
        return fallback.chunk(sentences, text)

    @property
    def topic_labels(self) -> List[str]:
        """
        Return topic labels from the last agentic chunking call.

        Note: only available if the LLM response was parsed successfully.
        """
        return getattr(self, "_last_topics", [])
