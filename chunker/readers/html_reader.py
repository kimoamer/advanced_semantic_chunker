"""
HTML reader — extracts plain text from HTML files or strings.

Requires ``beautifulsoup4``::

    pip install beautifulsoup4 lxml
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

__all__ = ["HTMLReader"]


class HTMLReader:
    """
    Extract plain text from HTML files or raw HTML strings.

    Strips tags, scripts, styles, and navigation boilerplate; preserves
    paragraph and heading structure as plain text.

    Parameters
    ----------
    parser : str
        BeautifulSoup parser: ``"lxml"`` (fast), ``"html.parser"`` (stdlib).
        Defaults to ``"lxml"`` with ``"html.parser"`` as fallback.
    heading_separator : str
        Separator inserted after each heading (default: ``"\\n"``).

    Examples
    --------
    >>> from chunker.readers import HTMLReader
    >>> reader = HTMLReader()
    >>> text = reader.read("article.html")
    >>> text_from_string = reader.read_string("<html>...</html>")
    """

    # Tags whose contents are always discarded
    _NOISE_TAGS = {"script", "style", "nav", "footer", "header", "aside", "form"}

    def __init__(
        self,
        parser: str = "lxml",
        heading_separator: str = "\n",
    ) -> None:
        self.parser = parser
        self.heading_separator = heading_separator

    # ── Public API ────────────────────────────────────────────────────────

    def read(self, path: str) -> str:
        """
        Read and extract text from an HTML file on disk.

        Parameters
        ----------
        path : str
            Path to the HTML file.

        Returns
        -------
        str
            Clean plain text.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ImportError
            If ``beautifulsoup4`` is not installed.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"HTML file not found: {path}")
        html = file_path.read_text(encoding="utf-8", errors="replace")
        return self.read_string(html)

    def read_string(self, html: str) -> str:
        """
        Extract plain text from an HTML string.

        Parameters
        ----------
        html : str
            Raw HTML content.

        Returns
        -------
        str
            Clean plain text.
        """
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except ImportError:
            raise ImportError(
                "beautifulsoup4 is required for HTMLReader. "
                "Install it with: pip install beautifulsoup4 lxml"
            )

        parser = self.parser
        try:
            soup = BeautifulSoup(html, parser)
        except Exception:
            soup = BeautifulSoup(html, "html.parser")

        # Remove noise tags entirely
        for tag in soup(list(self._NOISE_TAGS)):
            tag.decompose()

        # Build text with heading markers preserved
        parts = []
        for element in soup.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "blockquote"]
        ):
            text = element.get_text(separator=" ", strip=True)
            if not text:
                continue
            tag_name = element.name
            if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(tag_name[1])
                prefix = "#" * level
                parts.append(f"{prefix} {text}{self.heading_separator}")
            else:
                parts.append(text)

        return "\n\n".join(parts)
