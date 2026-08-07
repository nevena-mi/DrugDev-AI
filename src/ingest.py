"""Document ingestion for PDF sources."""

from __future__ import annotations

import logging
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

    for pdf_path in discover_pdfs(source_root):
        logger.info("Loading PDF: %s", pdf_path)
        page_documents = PyPDFLoader(str(pdf_path)).load()
        logger.debug("Loaded %d pages from %s", len(page_documents), pdf_path)

        chunk_documents = splitter.split_documents(page_documents)
        relative_path = pdf_path.relative_to(source_root)
        source_organization = relative_path.parts[0] if relative_path.parts else ""

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
                    "relative_file_path": relative_path.as_posix(),
                    "source_organization": source_organization,
                    "document_title": pdf_path.stem,
                    "chunk_id": f"{relative_path.as_posix()}::chunk-{chunk_index}",
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
