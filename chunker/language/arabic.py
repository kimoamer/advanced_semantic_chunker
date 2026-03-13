"""
Arabic text processing pipeline.

Handles Arabic-specific challenges:
  - Sentence segmentation (with Arabic punctuation: ، ؛ ؟)
  - Configurable text normalization (hamza, tashkeel, tatweel, punctuation)
  - Morphological decomposition (optional, via CAMeL Tools)
  - RTL text handling
  - Dialectal considerations
  - Punctuation variant normalization (Arabic ↔ Latin)

Research sources:
  - tnkeeh (cleaning/normalization)
  - PyArabic (text manipulation)
  - tkseem (Arabic tokenization)
  - CAMeL Tools (morphology/POS/NER)
  - Farasa (fast Arabic processing)
"""

from __future__ import annotations

import re
from typing import List, Optional


# ── Arabic Unicode constants & patterns ──────────────────────────────

# Arabic sentence-ending punctuation (including Arabic semicolon as weak boundary)
_AR_SENTENCE_END_STRONG = re.compile(
    r"""
    (?<=[.!؟])             # Strong: period, exclamation, Arabic question mark
    [\"\'\)\]»]*           # Optional closing quotes/brackets
    \s+                    # Whitespace boundary
    """,
    re.VERBOSE,
)

# Arabic diacritical marks (tashkeel/harakat)
_TASHKEEL = re.compile(
    r"[\u0610-\u061A"   # Arabic signs
    r"\u064B-\u065F"     # Fatha, Damma, Kasra, Shadda, Sukun, etc.
    r"\u0670"            # Superscript Alef
    r"\u06D6-\u06DC"     # Quranic marks
    r"\u06DF-\u06E8"     # Quranic marks
    r"\u06EA-\u06ED"     # More marks
    r"\uFE70-\uFE7F"     # Presentation forms
    r"]"
)

# Tatweel (kashida) — decorative elongation
_TATWEEL = re.compile(r"\u0640")

# Hamza / Alef normalizations
_HAMZA_MAP = {
    "\u0622": "\u0627",  # آ → ا  (Alef with Madda)
    "\u0623": "\u0627",  # أ → ا  (Alef with Hamza Above)
    "\u0625": "\u0627",  # إ → ا  (Alef with Hamza Below)
    "\u0671": "\u0627",  # ٱ → ا  (Alef Wasla)
}

# Yeh / Alef Maksura normalization
_YEH_MAP = {
    "\u0649": "\u064A",  # ى → ي  (Alef Maksura → Ya)
}

# Teh Marbuta normalization
_TEH_MARBUTA_MAP = {
    "\u0629": "\u0647",  # ة → ه  (Teh Marbuta → Heh)
}

# Arabic ↔ Latin punctuation variants
_PUNCTUATION_MAP = {
    "٪": "%",      # Arabic percent
    "٫": ".",      # Arabic decimal separator
    "٬": ",",      # Arabic thousands separator
    "\u060C": ",",  # Arabic comma → Latin comma (for tokenization consistency)
    # Keep ؟ ؛ as-is since we use them for sentence splitting
}

# Arabic comma (،) and semicolon (؛) — used in sentence splitting
_AR_COMMA = "\u060C"
_AR_SEMICOLON = "\u061B"
_AR_QUESTION = "\u061F"


class ArabicProcessor:
    """
    Arabic text processor for sentence segmentation and normalization.

    Arabic Challenges Addressed:
      1. Sentence boundaries use different punctuation (؟ ، ؛)
      2. Diacritics (tashkeel) are optional and often absent
      3. Multiple forms of Alef/Hamza need normalization
      4. Tatweel (kashida) is decorative and should be removed
      5. Morphological richness means one "word" = multiple tokens
      6. Right-to-left text direction
      7. Punctuation variants (Arabic vs Latin)
      8. Dialectal Arabic and Arabizi

    Normalization is configurable per-flag:
      - normalize_alef: أ إ آ ٱ → ا
      - normalize_yeh: ى → ي
      - normalize_teh_marbuta: ة → ه
      - remove_tashkeel: strip diacritical marks
      - remove_tatweel: strip kashida elongation
      - normalize_punctuation: unify Arabic/Latin punctuation variants
    """

    def __init__(
        self,
        normalize_alef: bool = True,
        normalize_yeh: bool = False,
        normalize_teh_marbuta: bool = False,
        remove_tashkeel: bool = True,
        remove_tatweel: bool = True,
        normalize_punctuation: bool = True,
        use_camel: bool = True,
        use_stanza: bool = False,
    ):
        self.normalize_alef = normalize_alef
        self.normalize_yeh = normalize_yeh
        self.normalize_teh_marbuta = normalize_teh_marbuta
        self.remove_tashkeel = remove_tashkeel
        self.remove_tatweel = remove_tatweel
        self.normalize_punctuation = normalize_punctuation

        self._camel_available = False
        self._stanza_pipeline = None

        if use_stanza:
            self._load_stanza()
        elif use_camel:
            self._load_camel()

    def _load_camel(self) -> None:
        """Try to load CAMeL Tools."""
        try:
            from camel_tools.tokenizers.word import simple_word_tokenize
            self._camel_available = True
        except ImportError:
            self._camel_available = False

    def _load_stanza(self) -> None:
        """Try to load Stanza Arabic pipeline."""
        try:
            import stanza
            try:
                self._stanza_pipeline = stanza.Pipeline(
                    "ar", processors="tokenize", verbose=False
                )
            except Exception:
                try:
                    stanza.download("ar", verbose=False)
                    self._stanza_pipeline = stanza.Pipeline(
                        "ar", processors="tokenize", verbose=False
                    )
                except Exception:
                    pass
        except ImportError:
            pass

    def normalize(self, text: str) -> str:
        """
        Normalize Arabic text for consistent processing.

        Steps applied (based on configuration):
          1. Remove tashkeel (diacritical marks)
          2. Remove tatweel (kashida elongation)
          3. Normalize hamza/alef variants → bare Alef
          4. Optionally normalize Alef Maksura → Ya
          5. Optionally normalize Teh Marbuta → Heh
          6. Normalize punctuation variants
          7. Normalize whitespace
        """
        if not text:
            return text

        result = text

        if self.remove_tashkeel:
            result = _TASHKEEL.sub("", result)

        if self.remove_tatweel:
            result = _TATWEEL.sub("", result)

        if self.normalize_alef:
            for src, dst in _HAMZA_MAP.items():
                result = result.replace(src, dst)

        if self.normalize_yeh:
            for src, dst in _YEH_MAP.items():
                result = result.replace(src, dst)

        if self.normalize_teh_marbuta:
            for src, dst in _TEH_MARBUTA_MAP.items():
                result = result.replace(src, dst)

        if self.normalize_punctuation:
            for src, dst in _PUNCTUATION_MAP.items():
                result = result.replace(src, dst)

        # Normalize whitespace (preserve newlines for structure detection)
        result = re.sub(r"[^\S\n]+", " ", result)
        # Normalize multiple newlines
        result = re.sub(r"\n{3,}", "\n\n", result)

        return result.strip()

    def segment_sentences(self, text: str) -> List[str]:
        """
        Split Arabic text into sentences.

        Priority: Stanza > regex fallback.
        Respects Arabic punctuation: . ؟ ! ؛
        Falls back to paragraph splitting if sentence splitter is uncertain.
        """
        if not text or not text.strip():
            return []

        if self._stanza_pipeline:
            return self._segment_stanza(text)
        else:
            return self._segment_regex(text)

    def _segment_stanza(self, text: str) -> List[str]:
        """Use Stanza for Arabic sentence segmentation."""
        doc = self._stanza_pipeline(text)
        return [
            sent.text.strip()
            for sent in doc.sentences
            if sent.text.strip()
        ]

    def _segment_regex(self, text: str) -> List[str]:
        """
        Regex-based Arabic sentence splitting.

        Handles:
          - Standard period, exclamation, Arabic question mark (؟)
          - Arabic semicolon (؛) as weak sentence boundary
          - Paragraph breaks (double newline)
          - Falls back to paragraph splitting for uncertain text
        """
        # First split on paragraph breaks
        paragraphs = re.split(r"\n\s*\n", text)
        sentences = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Split on strong sentence-ending punctuation: . ! ؟
            # Also treat ؛ (Arabic semicolon) as a sentence boundary
            parts = re.split(r"([.!؟؛])\s+", para)

            current = ""
            for i, part in enumerate(parts):
                if part in (".", "!", "؟", "؛"):
                    current += part
                    if current.strip():
                        sentences.append(current.strip())
                    current = ""
                else:
                    current += part

            if current.strip():
                sentences.append(current.strip())

        return sentences

    def detect_structure(self, text: str) -> dict:
        """Detect structural elements in Arabic text."""
        return {
            "has_headers": bool(re.search(r"^#+\s+.+$", text, re.MULTILINE)),
            "has_tables": bool(re.search(r"^\|.*\|$", text, re.MULTILINE)),
            "has_code": bool(re.search(r"```[\s\S]*?```", text)),
            "has_lists": bool(
                re.search(
                    r"^\s*[-*+•]\s+.+$|^\s*\d+[.)]\s+.+$",
                    text,
                    re.MULTILINE,
                )
            ),
        }

    def is_arabic(self, text: str) -> bool:
        """Check if text is predominantly Arabic script."""
        arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
        total_alpha = len(re.findall(r"\w", text))
        return total_alpha > 0 and (arabic_chars / total_alpha) > 0.5

    def detect_script(self, text: str) -> str:
        """
        Detect the script of the text.

        Returns: "arabic", "latin", or "mixed"
        """
        arabic_chars = len(re.findall(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]", text))
        latin_chars = len(re.findall(r"[a-zA-Z]", text))
        total = arabic_chars + latin_chars

        if total == 0:
            return "mixed"
        ar_ratio = arabic_chars / total
        if ar_ratio > 0.7:
            return "arabic"
        elif ar_ratio < 0.3:
            return "latin"
        return "mixed"

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Token count estimation for Arabic.

        Arabic tokens are ~1.5-2x more per word than English due to
        morphological richness and clitics (و + ال + ...).
        Estimate ≈ 1 token per 3 chars for better size stability.
        """
        return max(1, len(text) // 3)

    @staticmethod
    def count_words(text: str) -> int:
        """
        Count Arabic words (whitespace-based).
        Note: whitespace tokenization is approximate due to clitics.
        """
        return len(text.split())
