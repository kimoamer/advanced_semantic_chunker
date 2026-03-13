"""
Text normalizer that dispatches to language-specific normalization.

Handles common pre-processing steps for both EN and AR text:
  - Unicode normalization (NFC/NFKC)
  - Whitespace cleanup
  - Control character removal
  - Language-specific normalization delegation
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from chunker.models import Language


# Control characters (excluding newline, tab)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Multiple whitespace → single space (preserving newlines)
_MULTI_SPACE = re.compile(r"[^\S\n]+")

# Multiple newlines → max 2
_MULTI_NEWLINE = re.compile(r"\n{3,}")


class TextNormalizer:
    """
    Universal text normalizer for pre-chunking processing.

    Applied before any chunking strategy to ensure consistent input.
    """

    def __init__(
        self,
        unicode_form: str = "NFC",
        remove_control_chars: bool = True,
        normalize_whitespace: bool = True,
        max_consecutive_newlines: int = 2,
    ):
        self.unicode_form = unicode_form
        self.remove_control_chars = remove_control_chars
        self.normalize_whitespace = normalize_whitespace
        self.max_consecutive_newlines = max_consecutive_newlines

    def normalize(self, text: str, language: Optional[Language] = None) -> str:
        """
        Apply normalizations suitable for the detected language.

        Steps:
          1. Unicode normalization (NFC)
          2. Remove control characters
          3. Normalize whitespace
          4. Cap consecutive newlines
        """
        if not text:
            return text

        result = text

        # Unicode normalization
        result = unicodedata.normalize(self.unicode_form, result)

        # Remove control characters
        if self.remove_control_chars:
            result = _CONTROL_CHARS.sub("", result)

        # Normalize whitespace
        if self.normalize_whitespace:
            result = _MULTI_SPACE.sub(" ", result)
            max_nl = "\n" * self.max_consecutive_newlines
            result = re.sub(
                r"\n{" + str(self.max_consecutive_newlines + 1) + r",}",
                max_nl,
                result,
            )

        return result.strip()
