from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

import src.ingest as ingest_module
from src.ingest import discover_pdfs, ingest_documents


def _copy_pdf(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    shutil.copy2(source, destination)
    return destination


def _write_documents_yaml(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ich/ich_e6_r3.pdf:",
                '  title: "ICH E6(R3) Guideline for Good Clinical Practice"',
                '  organization: "ICH"',
            ]
        ),
        encoding="utf-8",
    )


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


def test_ingest_documents_uses_configured_metadata_and_fallbacks(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sources"
    _copy_pdf(
        Path("sources/ich/ich_e6_r3.pdf"),
        source_root / "ich",
    )
    _copy_pdf(
        Path("sources/fda/fda_drug_development_process.pdf"),
        source_root / "fda",
    )

    metadata_path = tmp_path / "documents.yaml"
    _write_documents_yaml(metadata_path)

    original_metadata_path = ingest_module.DOCUMENTS_METADATA_PATH
    try:
        ingest_module.DOCUMENTS_METADATA_PATH = metadata_path
        documents = ingest_documents(source_root, chunk_size=250, chunk_overlap=50)
    finally:
        ingest_module.DOCUMENTS_METADATA_PATH = original_metadata_path

    assert documents
    assert all(document.page_content.strip() for document in documents)
    assert all("source" in document.metadata for document in documents)
    assert all("page" in document.metadata for document in documents)

    by_relative_path: dict[str, list] = defaultdict(list)
    for document in documents:
        by_relative_path[document.metadata["relative_file_path"]].append(document)

    assert set(by_relative_path) == {
        "ich/ich_e6_r3.pdf",
        "fda/fda_drug_development_process.pdf",
    }

    ich_chunks = by_relative_path["ich/ich_e6_r3.pdf"]
    fda_chunks = by_relative_path["fda/fda_drug_development_process.pdf"]

    assert all(
        doc.metadata["document_title"] == "ICH E6(R3) Guideline for Good Clinical Practice"
        for doc in ich_chunks
    )
    assert all(doc.metadata["source_organization"] == "ICH" for doc in ich_chunks)

    assert all(doc.metadata["document_title"] == "fda_drug_development_process" for doc in fda_chunks)
    assert all(doc.metadata["source_organization"] == "fda" for doc in fda_chunks)

    for relative_file_path, chunk_documents in by_relative_path.items():
        assert [doc.metadata["chunk_id"] for doc in chunk_documents] == [
            f"{relative_file_path}::chunk-{index}"
            for index in range(len(chunk_documents))
        ]
        assert all(
            doc.metadata["filename"] == Path(relative_file_path).name
            for doc in chunk_documents
        )
        assert all(
            doc.metadata["relative_file_path"] == relative_file_path
            for doc in chunk_documents
        )

    assert any(len(chunk_documents) > 1 for chunk_documents in by_relative_path.values())


def test_ingest_documents_returns_empty_list_for_empty_tree(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir(exist_ok=True)

    assert ingest_documents(source_root) == []
