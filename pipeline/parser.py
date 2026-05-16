from __future__ import annotations

from pathlib import Path
from docling.document_converter import DocumentConverter # type: ignore

_converter = DocumentConverter()

def parse_document(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Docling preserves headings, paragraphs, tables, lists
    result = _converter.convert(str(file_path))
    
    # Export to markdown to easily split by headings
    text = result.document.export_to_markdown()
    text = text.strip()

    if not text:
        raise ValueError(f"No text extracted from file: {file_path}")

    return text

def parse_pdf(path: str) -> str:
    # Backward-compatible wrapper.
    return parse_document(path)
