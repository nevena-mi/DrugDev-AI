"""Document ingestion for PDF sources."""

from __future__ import annotations

import logging
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:  # pragma: no cover - exercised when LangChain is available
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - local workspace fallback
    @dataclass
    class Document:
        """Minimal compatibility document used when LangChain is unavailable."""

        page_content: str
        metadata: dict[str, Any] = field(default_factory=dict)


try:  # pragma: no cover - exercised when LangChain is available
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:  # pragma: no cover - local workspace fallback
    from pypdf import PdfReader

    class PyPDFLoader:
        """Fallback PDF loader that mirrors the LangChain interface."""

        def __init__(self, file_path: str | Path) -> None:
            self.file_path = Path(file_path)

        def load(self) -> list[Document]:
            reader = PdfReader(str(self.file_path))
            documents: list[Document] = []
            for page_number, page in enumerate(reader.pages, start=1):
                documents.append(
                    Document(
                        page_content=page.extract_text() or "",
                        metadata={
                            "source": str(self.file_path),
                            "page": page_number,
                        },
                    )
                )
            return documents


try:  # pragma: no cover - exercised when LangChain is available
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - local workspace fallback
    class RecursiveCharacterTextSplitter:
        """Fallback splitter compatible with the LangChain API used here."""

        def __init__(
            self,
            *,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            separators: list[str] | None = None,
        ) -> None:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators or ["\n\n", "\n", " ", ""]

        def split_documents(self, documents: Iterable[Document]) -> list[Document]:
            chunks: list[Document] = []
            for document in documents:
                chunks.extend(self._split_document(document))
            return chunks

        def _split_document(self, document: Document) -> list[Document]:
            text = document.page_content
            if not text:
                return [Document(page_content="", metadata=dict(document.metadata))]

            step = max(self.chunk_size - self.chunk_overlap, 1)
            chunks: list[Document] = []
            for start in range(0, len(text), step):
                chunk_text = text[start : start + self.chunk_size]
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=dict(document.metadata),
                    )
                )
                if start + self.chunk_size >= len(text):
                    break
            return chunks


logger = logging.getLogger(__name__)
DOCUMENTS_METADATA_PATH = Path(__file__).resolve().parents[1] / "documents.yaml"

try:  # pragma: no cover - exercised when PyYAML is available
    import yaml
except ImportError:  # pragma: no cover - local workspace fallback
    yaml = None


def _default_document_metadata(relative_file_path: str) -> dict[str, str]:
    """Return fallback metadata derived from the file path."""

    path = Path(relative_file_path)
    return {
        "title": path.stem,
        "organization": path.parts[0] if path.parts else "",
    }


def _normalise_metadata_entry(entry: Any, relative_file_path: str) -> dict[str, str]:
    """Return a normalised metadata entry with safe fallbacks."""

    defaults = _default_document_metadata(relative_file_path)
    if not isinstance(entry, dict):
        return defaults

    title = entry.get("title") or defaults["title"]
    organization = entry.get("organization") or defaults["organization"]
    return {
        "title": str(title),
        "organization": str(organization),
    }


def _parse_simple_documents_metadata(text: str) -> dict[str, dict[str, str]]:
    """Parse a restricted documents.yaml structure without PyYAML."""

    metadata: dict[str, dict[str, str]] = {}
    current_key: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if not line.startswith(" ") and stripped.endswith(":"):
            current_key = stripped[:-1].strip()
            metadata[current_key] = {}
            continue

        if current_key is None:
            continue

        if ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        metadata[current_key][key.strip()] = value.strip().strip("'\"")

    return {
        key: _normalise_metadata_entry(value, key)
        for key, value in metadata.items()
    }


@lru_cache(maxsize=None)
def load_documents_metadata(metadata_path: Path = DOCUMENTS_METADATA_PATH) -> dict[str, dict[str, str]]:
    """Load document metadata keyed by relative PDF path."""

    if not metadata_path.exists():
        logger.info("Documents metadata file not found: %s", metadata_path)
        return {}

    raw_text = metadata_path.read_text(encoding="utf-8")
    if yaml is not None:
        loaded = yaml.safe_load(raw_text) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid documents metadata structure: {metadata_path}")
        return {
            str(relative_file_path): _normalise_metadata_entry(entry, str(relative_file_path))
            for relative_file_path, entry in loaded.items()
        }

    logger.warning(
        "PyYAML is not installed; using a minimal documents metadata parser for %s",
        metadata_path,
    )
    return _parse_simple_documents_metadata(raw_text)


def discover_pdfs(source_root: Path) -> list[Path]:
    """Return all PDF files under ``source_root`` sorted by relative path."""

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source root is not a directory: {source_root}")

    pdfs = sorted(
        (
            path
            for path in source_root.rglob("*.pdf")
            if path.is_file()
        ),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    logger.info("Discovered %d PDF files under %s", len(pdfs), source_root)
    return pdfs


def ingest_documents(
    source_root: Path,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Load, split and annotate PDFs from ``source_root``."""

    documents: list[Document] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    document_metadata = load_documents_metadata(DOCUMENTS_METADATA_PATH)

    for pdf_path in discover_pdfs(source_root):
        logger.info("Loading PDF: %s", pdf_path)
        page_documents = PyPDFLoader(str(pdf_path)).load()
        logger.debug("Loaded %d pages from %s", len(page_documents), pdf_path)

        chunk_documents = splitter.split_documents(page_documents)
        relative_path = pdf_path.relative_to(source_root)
        relative_file_path = relative_path.as_posix()
        metadata_entry = document_metadata.get(relative_file_path, {})
        fallback_metadata = _default_document_metadata(relative_file_path)
        document_title = metadata_entry.get("title") or fallback_metadata["title"]
        source_organization = (
            metadata_entry.get("organization") or fallback_metadata["organization"]
        )

        logger.debug(
            "Split %s into %d chunks",
            pdf_path.name,
            len(chunk_documents),
        )
        for chunk_index, document in enumerate(chunk_documents):
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "filename": pdf_path.name,
                    "relative_file_path": relative_file_path,
                    "source_organization": source_organization,
                    "document_title": document_title,
                    "chunk_id": f"{relative_file_path}::chunk-{chunk_index}",
                }
            )
            documents.append(
                Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

    logger.info("Prepared %d document chunks for embedding", len(documents))
    return documents


load_documents = ingest_documents
