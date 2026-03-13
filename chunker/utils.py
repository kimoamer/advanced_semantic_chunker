"""
Utility functions for the Semantic Chunker.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional


def estimate_tokens(text: str, language: str = "en") -> int:
    """
    Estimate token count for text.

    Uses tighter heuristics to prevent exceeding LLM context limits:
      - English: ~1 token per 3.0 chars
      - Arabic: ~1 token per 2.5 chars
    """
    if not text:
        return 0
    chars_per_token = 2.5 if language == "ar" else 3.0
    return max(1, int(len(text) / chars_per_token))


def compute_hash(text: str) -> str:
    """Compute SHA-256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""
    # Normalize spaces (not newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Cap consecutive newlines at 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_sentences(text: str) -> int:
    """Quick sentence count estimate."""
    if not text:
        return 0
    # Count sentence-ending punctuation
    endings = len(re.findall(r"[.!?؟]", text))
    return max(1, endings)


def truncate_text(text: str, max_chars: int = 200) -> str:
    """Truncate text with ellipsis for display purposes."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def find_overlap_boundary(
    prev_text: str, overlap_tokens: int, language: str = "en"
) -> str:
    """
    Extract the trailing portion of prev_text that constitutes
    the overlap for the next chunk. Respects sentence boundaries.
    """
    if overlap_tokens <= 0:
        return ""

    chars_per_token = 3 if language == "ar" else 4
    overlap_chars = overlap_tokens * chars_per_token

    if len(prev_text) <= overlap_chars:
        return prev_text

    # Take from the end, then find the nearest sentence boundary
    candidate = prev_text[-overlap_chars:]
    # Find first sentence start in the overlap region
    match = re.search(r"(?<=[.!?؟])\s+", candidate)
    if match:
        return candidate[match.end() :].strip()

    return candidate.strip()


def is_structural_element(line: str) -> Optional[str]:
    """
    Check if a line is a structural element.

    Returns element type or None:
      - "header" for Markdown headers
      - "table_row" for table rows
      - "list_item" for list items
      - "code_fence" for code block delimiters
      - "hr" for horizontal rules
    """
    stripped = line.strip()

    if re.match(r"^#{1,6}\s+", stripped):
        return "header"
    if re.match(r"^\|.*\|$", stripped):
        return "table_row"
    if re.match(r"^\s*[-*+•]\s+", stripped) or re.match(r"^\s*\d+[.)]\s+", stripped):
        return "list_item"
    if re.match(r"^```", stripped) or re.match(r"^~~~", stripped):
        return "code_fence"
    if re.match(r"^[-*_]{3,}$", stripped):
        return "hr"

    return None


def split_keeping_structure(text: str) -> List[dict]:
    """
    Split text into blocks preserving structural elements.

    Returns list of {"type": str, "content": str} dicts where type is
    one of: "text", "header", "table", "code", "list".
    """
    lines = text.split("\n")
    blocks: List[dict] = []
    current_type = "text"
    current_lines: List[str] = []
    in_code_block = False

    for line in lines:
        element = is_structural_element(line)

        if element == "code_fence":
            if in_code_block:
                # End of code block
                current_lines.append(line)
                blocks.append({
                    "type": "code",
                    "content": "\n".join(current_lines),
                })
                current_lines = []
                in_code_block = False
                current_type = "text"
            else:
                # Start of code block
                if current_lines:
                    blocks.append({
                        "type": current_type,
                        "content": "\n".join(current_lines),
                    })
                current_lines = [line]
                in_code_block = True
                current_type = "code"
            continue

        if in_code_block:
            current_lines.append(line)
            continue

        if element == "header":
            if current_lines:
                blocks.append({
                    "type": current_type,
                    "content": "\n".join(current_lines),
                })
            blocks.append({"type": "header", "content": line})
            current_lines = []
            current_type = "text"
        elif element == "table_row":
            if current_type != "table" and current_lines:
                blocks.append({
                    "type": current_type,
                    "content": "\n".join(current_lines),
                })
                current_lines = []
            current_type = "table"
            current_lines.append(line)
        elif element == "list_item":
            if current_type != "list" and current_lines:
                blocks.append({
                    "type": current_type,
                    "content": "\n".join(current_lines),
                })
                current_lines = []
            current_type = "list"
            current_lines.append(line)
        else:
            if current_type not in ("text",) and current_lines:
                blocks.append({
                    "type": current_type,
                    "content": "\n".join(current_lines),
                })
                current_lines = []
                current_type = "text"
            current_lines.append(line)

    if current_lines:
        blocks.append({
            "type": current_type,
            "content": "\n".join(current_lines),
        })

    return [b for b in blocks if b["content"].strip()]
