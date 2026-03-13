"""
English text processing pipeline.

Handles sentence segmentation, structural detection (headers, tables, code),
and token counting for English text.
"""

from __future__ import annotations

import re
from typing import List, Optional


# ── Regex-based sentence splitter (no external deps) ─────────────────
_SENTENCE_END = re.compile(
    r"""
    (?<=[.!?])          # Look-behind for sentence-ending punctuation
    [\"\'\)\]]*         # Optional closing quotes / brackets
    \s+                 # Whitespace boundary
    (?=[A-Z\"\'\(\[])   # Look-ahead for next sentence start
    """,
    re.VERBOSE,
)

# Abbreviation patterns to avoid false sentence boundaries
_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.",
    "vs.", "etc.", "inc.", "ltd.", "co.", "corp.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.",
    "sep.", "oct.", "nov.", "dec.",
    "fig.", "eq.", "ref.", "vol.", "no.", "pp.", "ed.",
    "e.g.", "i.e.", "a.m.", "p.m.", "u.s.", "u.k.",
}

# Structural patterns
_HEADER_PATTERN = re.compile(r"^#+\s+.+$|^.+\n[=\-]{3,}$", re.MULTILINE)
_TABLE_PATTERN = re.compile(r"^\|.*\|$", re.MULTILINE)
_CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~", re.MULTILINE)
_LIST_PATTERN = re.compile(r"^\s*[-*+]\s+.+$|^\s*\d+\.\s+.+$", re.MULTILINE)


class EnglishProcessor:
    """
    English text processor for sentence segmentation and structure detection.

    Uses an enhanced regex splitter with abbreviation awareness.
    Optionally uses NLTK/spaCy if available for higher accuracy.
    """

    def __init__(self, use_nltk: bool = True, use_spacy: bool = False):
        self._nltk_tokenizer = None
        self._spacy_nlp = None

        if use_spacy:
            self._load_spacy()
        elif use_nltk:
            self._load_nltk()

    def _load_nltk(self) -> None:
        """Try to load NLTK sentence tokenizer."""
        try:
            import nltk
            try:
                self._nltk_tokenizer = nltk.data.load("tokenizers/punkt_tab/english/english.pickle")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
                self._nltk_tokenizer = nltk.data.load("tokenizers/punkt_tab/english/english.pickle")
        except ImportError:
            pass

    def _load_spacy(self) -> None:
        """Try to load spaCy English model."""
        try:
            import spacy
            try:
                self._spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                pass
        except ImportError:
            pass

    def segment_sentences(self, text: str) -> List[str]:
        """
        Split English text into sentences.

        Priority: spaCy > NLTK > regex fallback.
        """
        if not text or not text.strip():
            return []

        if self._spacy_nlp:
            return self._segment_spacy(text)
        elif self._nltk_tokenizer:
            return self._segment_nltk(text)
        else:
            return self._segment_regex(text)

    def _segment_spacy(self, text: str) -> List[str]:
        doc = self._spacy_nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _segment_nltk(self, text: str) -> List[str]:
        sentences = self._nltk_tokenizer.tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

    def _segment_regex(self, text: str) -> List[str]:
        """Regex-based sentence splitting with abbreviation awareness."""
        # First, protect abbreviations by temporarily replacing their periods
        protected = text
        for abbr in _ABBREVIATIONS:
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            protected = pattern.sub(abbr.replace(".", "\x00"), protected)

        # Split on sentence boundaries
        parts = _SENTENCE_END.split(protected)

        # Restore abbreviation periods
        sentences = []
        for part in parts:
            restored = part.replace("\x00", ".").strip()
            if restored:
                sentences.append(restored)

        return sentences

    def detect_structure(self, text: str) -> dict:
        """
        Detect structural elements in the text.

        Returns dict with booleans: has_headers, has_tables, has_code, has_lists.
        """
        return {
            "has_headers": bool(_HEADER_PATTERN.search(text)),
            "has_tables": bool(_TABLE_PATTERN.search(text)),
            "has_code": bool(_CODE_BLOCK_PATTERN.search(text)),
            "has_lists": bool(_LIST_PATTERN.search(text)),
        }

    def extract_sections(self, text: str) -> List[dict]:
        """
        Extract document sections based on headers.

        Returns list of {"title": str, "level": int, "content": str} dicts.
        """
        lines = text.split("\n")
        sections = []
        current_section = {"title": "", "level": 0, "content": ""}

        for line in lines:
            # Markdown header
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                if current_section["content"].strip():
                    sections.append(current_section)
                current_section = {
                    "title": header_match.group(2).strip(),
                    "level": len(header_match.group(1)),
                    "content": "",
                }
            else:
                current_section["content"] += line + "\n"

        if current_section["content"].strip():
            sections.append(current_section)

        return sections

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Rough token count estimation (≈ 1 token per 4 chars for English).

        For accurate counts, use a proper tokenizer (tiktoken, etc.).
        """
        return max(1, len(text) // 4)
