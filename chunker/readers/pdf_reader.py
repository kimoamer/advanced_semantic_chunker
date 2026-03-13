"""
PDF reader — extracts plain text from PDF files.

Tries ``pymupdf`` (fitz) first for speed; falls back to ``pdfplumber``
if pymupdf is not installed.

Install one of::

    pip install pymupdf          # fast, recommended
    pip install pdfplumber        # alternative
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

__all__ = ["PDFReader"]


class PDFReader:
    """
    Extract text from PDF files.

    Parameters
    ----------
    page_separator : str
        String inserted between pages (default: ``"\\n\\n"``).
    backend : str, optional
        Force a specific backend: ``"pymupdf"`` or ``"pdfplumber"``.
        Auto-detected if not specified.

    Examples
    --------
    >>> from chunker.readers import PDFReader
    >>> reader = PDFReader()
    >>> text = reader.read("report.pdf")
    >>> chunks = chunker.chunk(text, source_file="report.pdf")
    """

    def __init__(
        self,
        page_separator: str = "\n\n",
        backend: Optional[str] = None,
    ) -> None:
        self.page_separator = page_separator
        self._backend = backend or self._detect_backend()

    # ── Public API ────────────────────────────────────────────────────────

    def read(self, path: str) -> str:
        """
        Extract all text from a PDF file.

        Parameters
        ----------
        path : str
            Path to the PDF file.

        Returns
        -------
        str
            Extracted plain text with pages separated by ``page_separator``.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ImportError
            If neither ``pymupdf`` nor ``pdfplumber`` is installed.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if self._backend == "pymupdf":
            return self._read_pymupdf(file_path)
        elif self._backend == "pdfplumber":
            return self._read_pdfplumber(file_path)
        else:
            raise ImportError(
                "No PDF backend available. Install one of:\n"
                "  pip install pymupdf\n"
                "  pip install pdfplumber"
            )

    def read_pages(self, path: str) -> List[str]:
        """
        Extract text per page as a list of strings.

        Parameters
        ----------
        path : str
            Path to the PDF file.

        Returns
        -------
        List[str]
            One string per page.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if self._backend == "pymupdf":
            return self._pages_pymupdf(file_path)
        elif self._backend == "pdfplumber":
            return self._pages_pdfplumber(file_path)
        return []

    # ── Backends ──────────────────────────────────────────────────────────

    @staticmethod
    def _detect_backend() -> Optional[str]:
        try:
            import fitz  # noqa: F401
            return "pymupdf"
        except ImportError:
            pass
        try:
            import pdfplumber  # noqa: F401
            return "pdfplumber"
        except ImportError:
            pass
        return None

    def _read_pymupdf(self, path: Path) -> str:
        import fitz  # type: ignore
        pages = []
        with fitz.open(str(path)) as doc:
            for page in doc:
                pages.append(page.get_text())
        return self.page_separator.join(p for p in pages if p.strip())

    def _pages_pymupdf(self, path: Path) -> List[str]:
        import fitz  # type: ignore
        pages = []
        with fitz.open(str(path)) as doc:
            for page in doc:
                pages.append(page.get_text())
        return pages

    def _read_pdfplumber(self, path: Path) -> str:
        import pdfplumber  # type: ignore
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)
        return self.page_separator.join(p for p in pages if p.strip())

    def _pages_pdfplumber(self, path: Path) -> List[str]:
        import pdfplumber  # type: ignore
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return pages
