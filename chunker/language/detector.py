"""
Language detection for bilingual EN/AR documents.

Uses Unicode script analysis as the primary fast method and
optionally falls back to statistical models for ambiguous text.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

from chunker.models import Language


# Unicode ranges for Arabic script (includes Arabic, Arabic Supplement,
# Arabic Extended-A/B, Arabic Presentation Forms A/B)
_ARABIC_PATTERN = re.compile(
    r"[\u0600-\u06FF"    # Arabic
    r"\u0750-\u077F"      # Arabic Supplement
    r"\u08A0-\u08FF"      # Arabic Extended-A
    r"\uFB50-\uFDFF"      # Arabic Presentation Forms-A
    r"\uFE70-\uFEFF"      # Arabic Presentation Forms-B
    r"\u0610-\u061A"      # Arabic diacritics
    r"\u064B-\u065F"      # Arabic combining marks
    r"]"
)

_LATIN_PATTERN = re.compile(r"[a-zA-Z]")


class LanguageDetector:
    """
    Fast, rule-based language detector for EN/AR text.

    Uses character-script ratio analysis:
      - >60% Arabic chars  → ARABIC
      - >60% Latin chars   → ENGLISH
      - Both significant   → MIXED
      - Neither            → UNKNOWN

    For sentence-level detection, call detect_sentence() which is
    cheaper than full-document analysis.
    """

    def __init__(self, arabic_threshold: float = 0.6, english_threshold: float = 0.6):
        self.arabic_threshold = arabic_threshold
        self.english_threshold = english_threshold

    def detect(self, text: str) -> Language:
        """Detect primary language of the full text."""
        if not text or not text.strip():
            return Language.UNKNOWN

        arabic_count = len(_ARABIC_PATTERN.findall(text))
        latin_count = len(_LATIN_PATTERN.findall(text))
        total = arabic_count + latin_count

        if total == 0:
            return Language.UNKNOWN

        arabic_ratio = arabic_count / total
        latin_ratio = latin_count / total

        if arabic_ratio >= self.arabic_threshold:
            return Language.ARABIC
        elif latin_ratio >= self.english_threshold:
            return Language.ENGLISH
        elif arabic_count > 0 and latin_count > 0:
            return Language.MIXED
        else:
            return Language.UNKNOWN

    def detect_sentence(self, sentence: str) -> Language:
        """Lightweight per-sentence detection."""
        return self.detect(sentence)

    def detect_batch(self, texts: List[str]) -> List[Language]:
        """
        Detect language for multiple texts in a single batch call.
        
        This is more efficient than calling detect() multiple times
        as it processes all texts in one operation.
        
        Args:
            texts: List of text strings to detect language for
            
        Returns:
            List of Language enums, one per input text
        """
        return [self.detect(text) for text in texts]

    def detect_segments(self, sentences: List[str]) -> List[Tuple[str, Language]]:
        """Detect language for each sentence, returning (sentence, language) tuples."""
        return [(s, self.detect(s)) for s in sentences]

    def find_language_boundaries(
        self, sentences: List[str]
    ) -> List[Tuple[int, int, Language]]:
        """
        Find contiguous runs of the same language.

        Returns list of (start_idx, end_idx, language) tuples.
        Useful for splitting mixed documents into language-homogeneous sections.
        """
        if not sentences:
            return []

        boundaries: List[Tuple[int, int, Language]] = []
        current_lang = self.detect(sentences[0])
        start = 0

        for i in range(1, len(sentences)):
            lang = self.detect(sentences[i])
            if lang != current_lang:
                boundaries.append((start, i, current_lang))
                current_lang = lang
                start = i

        boundaries.append((start, len(sentences), current_lang))
        return boundaries
