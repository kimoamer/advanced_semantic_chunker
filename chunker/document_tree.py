"""
Document tree parser — Step 0 of the structure-aware pipeline.

Parses raw text (Markdown or plain) into a tree of typed nodes:
  - HeadingNode  (title, level 1-6)
  - ParagraphNode
  - ListNode     (items)
  - TableNode    (rows/cells)
  - CodeNode     (language, content)

This structural tree is then consumed by StructureAwareStrategy
to enforce heading-based hard boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class NodeType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"


@dataclass
class DocumentNode:
    """A single structural node in the document tree."""

    node_type: NodeType
    content: str
    # For headings
    heading_level: int = 0
    heading_title: str = ""
    # For code blocks
    code_language: str = ""
    # For lists
    list_items: List[str] = field(default_factory=list)
    # For tables
    table_rows: List[List[str]] = field(default_factory=list)
    # Position in original text
    start_char: int = 0
    end_char: int = 0

    @property
    def is_heading(self) -> bool:
        return self.node_type == NodeType.HEADING

    def __repr__(self) -> str:
        if self.is_heading:
            return f"DocumentNode(HEADING L{self.heading_level}: {self.heading_title})"
        preview = self.content[:60].replace("\n", "\\n")
        return f'DocumentNode({self.node_type.value}: "{preview}...")'


@dataclass
class DocumentSection:
    """
    A section of the document defined by its heading and content nodes.

    heading_path tracks the full breadcrumb, e.g.:
      ["Introduction", "Background"]  for a ## heading under # Introduction
    """

    heading_text: str = ""
    heading_title: str = ""
    heading_level: int = 0
    heading_path: List[str] = field(default_factory=list)
    nodes: List[DocumentNode] = field(default_factory=list)
    start_char: int = 0
    end_char: int = 0

    @property
    def text(self) -> str:
        """Concatenate all node content in this section."""
        parts = []
        for node in self.nodes:
            if node.content.strip():
                parts.append(node.content.strip())
        return "\n\n".join(parts)

    @property
    def is_empty(self) -> bool:
        return not any(n.content.strip() for n in self.nodes)


# ── Regex patterns ──────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"^(```|~~~)", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\|.+\|$")
_TABLE_SEP_RE = re.compile(r"^\|[\s:|-]+\|$")
_LIST_ITEM_RE = re.compile(r"^(\s*)([-*+•]|\d+[.)])\s+(.+)$")


def parse_document_tree(text: str) -> List[DocumentNode]:
    """
    Parse text into a flat list of typed document nodes.

    Handles:
      - Markdown headings (# to ######)
      - Code blocks (``` ... ```)
      - Tables (| ... | rows)
      - Lists (- / * / + / 1. items)
      - Paragraphs (everything else, split on double newlines)
    """
    if not text or not text.strip():
        return []

    nodes: List[DocumentNode] = []
    lines = text.split("\n")
    i = 0
    char_offset = 0

    while i < len(lines):
        line = lines[i]
        line_start_char = char_offset

        # ── Code block ──
        if _CODE_FENCE_RE.match(line.strip()):
            fence = line.strip()[:3]
            code_lang_match = re.match(r"^(?:```|~~~)(\w*)", line.strip())
            code_lang = code_lang_match.group(1) if code_lang_match else ""
            code_lines = [line]
            i += 1
            char_offset += len(line) + 1

            while i < len(lines):
                code_lines.append(lines[i])
                if lines[i].strip().startswith(fence) and i > 0:
                    char_offset += len(lines[i]) + 1
                    i += 1
                    break
                char_offset += len(lines[i]) + 1
                i += 1

            content = "\n".join(code_lines)
            nodes.append(DocumentNode(
                node_type=NodeType.CODE,
                content=content,
                code_language=code_lang,
                start_char=line_start_char,
                end_char=char_offset,
            ))
            continue

        # ── Heading ──
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            nodes.append(DocumentNode(
                node_type=NodeType.HEADING,
                content=line,
                heading_level=level,
                heading_title=title,
                start_char=line_start_char,
                end_char=line_start_char + len(line),
            ))
            char_offset += len(line) + 1
            i += 1
            continue

        # ── Table ──
        if _TABLE_ROW_RE.match(line.strip()):
            table_lines = []
            table_rows = []
            while i < len(lines) and (_TABLE_ROW_RE.match(lines[i].strip()) or _TABLE_SEP_RE.match(lines[i].strip())):
                table_lines.append(lines[i])
                if not _TABLE_SEP_RE.match(lines[i].strip()):
                    cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                    table_rows.append(cells)
                char_offset += len(lines[i]) + 1
                i += 1

            content = "\n".join(table_lines)
            nodes.append(DocumentNode(
                node_type=NodeType.TABLE,
                content=content,
                table_rows=table_rows,
                start_char=line_start_char,
                end_char=char_offset,
            ))
            continue

        # ── List ──
        if _LIST_ITEM_RE.match(line):
            list_lines = []
            list_items = []
            while i < len(lines) and (_LIST_ITEM_RE.match(lines[i]) or (lines[i].strip() and lines[i].startswith("  "))):
                list_lines.append(lines[i])
                item_match = _LIST_ITEM_RE.match(lines[i])
                if item_match:
                    list_items.append(item_match.group(3).strip())
                char_offset += len(lines[i]) + 1
                i += 1

            content = "\n".join(list_lines)
            nodes.append(DocumentNode(
                node_type=NodeType.LIST,
                content=content,
                list_items=list_items,
                start_char=line_start_char,
                end_char=char_offset,
            ))
            continue

        # ── Paragraph (collect until blank line or structural element) ──
        para_lines = []
        while i < len(lines):
            current = lines[i]
            # Stop at blank lines
            if not current.strip():
                char_offset += len(current) + 1
                i += 1
                break
            # Stop at structural elements
            if (re.match(r"^#{1,6}\s+", current.strip()) or
                _CODE_FENCE_RE.match(current.strip()) or
                _TABLE_ROW_RE.match(current.strip()) or
                _LIST_ITEM_RE.match(current)):
                break
            para_lines.append(current)
            char_offset += len(current) + 1
            i += 1

        if para_lines:
            content = "\n".join(para_lines)
            nodes.append(DocumentNode(
                node_type=NodeType.PARAGRAPH,
                content=content,
                start_char=line_start_char,
                end_char=char_offset,
            ))
            continue

        # Skip blank lines
        if not line.strip():
            char_offset += len(line) + 1
            i += 1

    return nodes


def build_sections(nodes: List[DocumentNode]) -> List[DocumentSection]:
    """
    Group document nodes into sections based on headings.

    Each heading starts a new section. The heading_path is maintained
    as a breadcrumb trail, e.g. for:
      # Top         → heading_path = ["Top"]
      ## Sub        → heading_path = ["Top", "Sub"]
      ### Detail    → heading_path = ["Top", "Sub", "Detail"]
      ## Other      → heading_path = ["Top", "Other"]

    Content before the first heading goes into a section with
    heading_title="" and heading_level=0.
    """
    if not nodes:
        return []

    sections: List[DocumentSection] = []
    # Track heading stack for breadcrumb path
    heading_stack: List[Tuple[int, str]] = []  # (level, title)

    current_section = DocumentSection()

    for node in nodes:
        if node.is_heading:
            # Flush previous section
            if current_section.nodes or current_section.heading_title:
                sections.append(current_section)

            # Update heading stack — pop everything at same or lower level
            while heading_stack and heading_stack[-1][0] >= node.heading_level:
                heading_stack.pop()
            heading_stack.append((node.heading_level, node.heading_title))

            # Build heading path
            heading_path = [title for _, title in heading_stack]

            current_section = DocumentSection(
                heading_text=node.content.strip(),
                heading_title=node.heading_title,
                heading_level=node.heading_level,
                heading_path=list(heading_path),
                start_char=node.start_char,
                end_char=node.end_char,  # default to heading end; extended by content nodes
            )
        else:
            current_section.nodes.append(node)
            current_section.end_char = node.end_char

    # Flush last section
    if current_section.nodes or current_section.heading_title:
        sections.append(current_section)

    return sections
