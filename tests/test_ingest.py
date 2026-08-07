'''
Verifies recursive discovery.
Verifies PDFs load and chunk.
Verifies metadata is assigned correctly.
Verifies an empty source tree returns [].
'''



from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

from src.ingest import discover_pdfs, ingest_documents


def _copy_pdf(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    shutil.copy2(source, destination)
    return destination


def test_discover_pdfs_recursively(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    _copy_pdf(
        Path("sources/ema/ema_marketing_authorisation.pdf"),
        source_root / "ema" / "nested",
    )
    _copy_pdf(
        Path("sources/ich/ich_e6_r3.pdf"),
        source_root / "ich",
    )

    pdfs = discover_pdfs(source_root)

    assert [path.relative_to(source_root).as_posix() for path in pdfs] == [
        "ema/nested/ema_marketing_authorisation.pdf",
        "ich/ich_e6_r3.pdf",
    ]


def test_ingest_documents_loads_chunks_and_metadata(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    _copy_pdf(
        Path("sources/ema/ema_marketing_authorisation.pdf"),
        source_root / "ema" / "nested",
    )
    _copy_pdf(
        Path("sources/ich/ich_e6_r3.pdf"),
        source_root / "ich",
    )

    documents = ingest_documents(source_root, chunk_size=250, chunk_overlap=50)

    assert documents
    assert all(document.page_content.strip() for document in documents)

    by_relative_path: dict[str, list] = defaultdict(list)
    for document in documents:
        by_relative_path[document.metadata["relative_file_path"]].append(document)

    assert set(by_relative_path) == {
        "ema/nested/ema_marketing_authorisation.pdf",
        "ich/ich_e6_r3.pdf",
    }

    for relative_file_path, chunk_documents in by_relative_path.items():
        expected_filename = Path(relative_file_path).name
        expected_title = Path(relative_file_path).stem
        expected_organization = Path(relative_file_path).parts[0]

        assert [doc.metadata["chunk_id"] for doc in chunk_documents] == [
            f"{relative_file_path}::chunk-{index}"
            for index in range(len(chunk_documents))
        ]
        assert all(doc.metadata["filename"] == expected_filename for doc in chunk_documents)
        assert all(
            doc.metadata["relative_file_path"] == relative_file_path
            for doc in chunk_documents
        )
        assert all(
            doc.metadata["source_organization"] == expected_organization
            for doc in chunk_documents
        )
        assert all(
            doc.metadata["document_title"] == expected_title
            for doc in chunk_documents
        )

    assert any(len(chunk_documents) > 1 for chunk_documents in by_relative_path.values())


def test_ingest_documents_returns_empty_list_for_empty_tree(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir(exist_ok=True)

    assert ingest_documents(source_root) == []
