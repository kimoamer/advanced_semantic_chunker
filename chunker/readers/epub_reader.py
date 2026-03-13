"""
EPUB reader — extracts plain text from EPUB e-books.

Requires ``ebooklib`` and ``beautifulsoup4``::

    pip install ebooklib beautifulsoup4
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["EPUBReader"]


class EPUBReader:
    """
    Extract plain text from EPUB files.

    Iterates over every EPUB document item, strips HTML tags using
    :class:`~chunker.readers.html_reader.HTMLReader`, and joins chapters
    with a configurable separator.

    Parameters
    ----------
    chapter_separator : str
        String inserted between chapters (default: ``"\\n\\n---\\n\\n"``).

    Examples
    --------
    >>> from chunker.readers import EPUBReader
    >>> reader = EPUBReader()
    >>> text = reader.read("book.epub")
    >>> chunks = chunker.chunk(text, source_file="book.epub")
    """

    def __init__(self, chapter_separator: str = "\n\n---\n\n") -> None:
        self.chapter_separator = chapter_separator

    def read(self, path: str) -> str:
        """
        Extract all text from an EPUB file.

        Parameters
        ----------
        path : str
            Path to the EPUB file.

        Returns
        -------
        str
            Full book text with chapters separated by ``chapter_separator``.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ImportError
            If ``ebooklib`` or ``beautifulsoup4`` is not installed.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"EPUB file not found: {path}")

        try:
            import ebooklib  # type: ignore
            from ebooklib import epub  # type: ignore
        except ImportError:
            raise ImportError(
                "ebooklib is required for EPUBReader. "
                "Install it with: pip install ebooklib beautifulsoup4"
            )

        from chunker.readers.html_reader import HTMLReader
        html_reader = HTMLReader()

        book = epub.read_epub(str(file_path))
        chapters = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            html_content = item.get_content().decode("utf-8", errors="replace")
            text = html_reader.read_string(html_content)
            if text.strip():
                chapters.append(text)

        return self.chapter_separator.join(chapters)
