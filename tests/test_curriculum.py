from __future__ import annotations

from pathlib import Path

import pytest

from src.curriculum import (
    CURRICULUM_PATH,
    CurriculumValidationError,
    get_module,
    get_module_documents,
    list_modules,
    missing_document_paths,
    next_modules,
    prerequisites_satisfied,
    load_curriculum,
)


def test_curriculum_loads_and_module_ids_are_unique() -> None:
    modules = list_modules()

    assert modules
    module_ids = [module.id for module in modules]
    assert len(module_ids) == len(set(module_ids))
    assert module_ids[0] == "foundations"


def test_prerequisites_resolve_and_next_modules_progression() -> None:
    assert prerequisites_satisfied("foundations", [])
    assert not prerequisites_satisfied("regulatory", [])
    assert prerequisites_satisfied("regulatory", ["foundations"])
    assert [module.id for module in next_modules(["foundations"])] == ["regulatory"]
    assert [module.id for module in next_modules(["foundations", "regulatory"])] == [
        "clinical_trials"
    ]


def test_document_mappings_resolve_to_source_files() -> None:
    clinical_trials = get_module("clinical_trials")
    emerging_topics = get_module("emerging_topics")

    assert clinical_trials is not None
    assert emerging_topics is not None
    assert "wma/declaration_of_helsinki.pdf" in clinical_trials.documents
    assert "who/who_ai_ethics.pdf" in emerging_topics.documents
    assert missing_document_paths() == []

    for document_path in get_module_documents("clinical_trials"):
        assert (Path("sources") / document_path).is_file()


def test_invalid_prerequisites_are_detected(tmp_path: Path) -> None:
    bad_path = tmp_path / "curriculum.yaml"
    bad_text = CURRICULUM_PATH.read_text(encoding="utf-8").replace(
        "      - foundations",
        "      - missing_prereq",
        1,
    )
    bad_path.write_text(bad_text, encoding="utf-8")

    with pytest.raises(CurriculumValidationError, match="unknown prerequisites"):
        load_curriculum(bad_path)

