"""
Document format readers for the Semantic Chunker.

Converts various file formats to plain text so they can be passed to
:class:`~chunker.core.SemanticChunker`.

Available readers
-----------------
- :class:`PDFReader`  — requires ``pymupdf`` or ``pdfplumber``
- :class:`HTMLReader` — requires ``beautifulsoup4``
- :class:`EPUBReader` — requires ``ebooklib``

All readers expose a single ``read(path) -> str`` method.
"""

from chunker.readers.pdf_reader import PDFReader
from chunker.readers.html_reader import HTMLReader
from chunker.readers.epub_reader import EPUBReader

__all__ = ["PDFReader", "HTMLReader", "EPUBReader"]
