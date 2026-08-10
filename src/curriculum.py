"""Curriculum loading and validation for Learn mode."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import yaml


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRICULUM_PATH = PROJECT_ROOT / "curriculum.yaml"
SOURCES_ROOT = PROJECT_ROOT / "sources"


class CurriculumValidationError(RuntimeError):
    """Raised when the curriculum file is malformed or inconsistent."""


@dataclass(slots=True)
class CurriculumModule:
    """A single curriculum module loaded from YAML."""

    id: str
    title: str
    description: str
    difficulty: str
    duration: str
    prerequisites: list[str]
    objectives: list[str]
    documents: list[str]
    quiz: bool


def _coerce_string_list(value: Any, *, field_name: str, module_id: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise CurriculumValidationError(
            f"Module {module_id!r} field {field_name!r} must be a list of strings"
        )
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise CurriculumValidationError(
                f"Module {module_id!r} field {field_name!r} must contain only strings"
            )
        result.append(item.strip())
    return result


def _build_module(module_data: dict[str, Any]) -> CurriculumModule:
    module_id = module_data.get("id")
    if not isinstance(module_id, str) or not module_id.strip():
        raise CurriculumValidationError("Each curriculum module must define a non-empty id")

    required_string_fields = ["title", "description", "difficulty", "duration"]
    values: dict[str, str] = {}
    for field_name in required_string_fields:
        value = module_data.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise CurriculumValidationError(
                f"Module {module_id!r} field {field_name!r} must be a non-empty string"
            )
        values[field_name] = value.strip()

    prerequisites = _coerce_string_list(
        module_data.get("prerequisites"),
        field_name="prerequisites",
        module_id=module_id,
    )
    objectives = _coerce_string_list(
        module_data.get("objectives"),
        field_name="objectives",
        module_id=module_id,
    )
    documents = _coerce_string_list(
        module_data.get("documents"),
        field_name="documents",
        module_id=module_id,
    )

    quiz_value = module_data.get("quiz", False)
    if not isinstance(quiz_value, bool):
        raise CurriculumValidationError(
            f"Module {module_id!r} field 'quiz' must be a boolean if provided"
        )

    return CurriculumModule(
        id=module_id.strip(),
        title=values["title"],
        description=values["description"],
        difficulty=values["difficulty"],
        duration=values["duration"],
        prerequisites=prerequisites,
        objectives=objectives,
        documents=documents,
        quiz=quiz_value,
    )


def _topologically_sort_modules(modules: Sequence[CurriculumModule]) -> list[CurriculumModule]:
    module_by_id = {module.id: module for module in modules}
    original_order = {module.id: index for index, module in enumerate(modules)}
    indegree = {module.id: len(module.prerequisites) for module in modules}
    adjacency: dict[str, list[str]] = {module.id: [] for module in modules}

    for module in modules:
        for prerequisite in module.prerequisites:
            adjacency.setdefault(prerequisite, []).append(module.id)

    queue = deque(
        sorted(
            (module_id for module_id, degree in indegree.items() if degree == 0),
            key=lambda module_id: original_order[module_id],
        )
    )

    ordered_modules: list[CurriculumModule] = []
    while queue:
        module_id = queue.popleft()
        ordered_modules.append(module_by_id[module_id])
        for dependent_id in sorted(adjacency.get(module_id, []), key=lambda item: original_order[item]):
            indegree[dependent_id] -= 1
            if indegree[dependent_id] == 0:
                queue.append(dependent_id)

    if len(ordered_modules) != len(modules):
        raise CurriculumValidationError("Curriculum contains a cycle or unresolved dependency")

    return ordered_modules


@lru_cache(maxsize=None)
def load_curriculum(curriculum_path: Path = CURRICULUM_PATH) -> tuple[CurriculumModule, ...]:
    """Load and validate the curriculum YAML file."""

    if not curriculum_path.exists():
        raise CurriculumValidationError(f"Curriculum file not found: {curriculum_path}")

    try:
        raw_data = yaml.safe_load(curriculum_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - malformed YAML is guarded by tests
        raise CurriculumValidationError(f"Invalid curriculum YAML: {curriculum_path}") from exc

    if not isinstance(raw_data, dict):
        raise CurriculumValidationError("Curriculum root must be a mapping")

    curriculum_entries = raw_data.get("curriculum")
    if not isinstance(curriculum_entries, list):
        raise CurriculumValidationError("Curriculum root must contain a 'curriculum' list")

    modules = [_build_module(entry) for entry in curriculum_entries]
    module_ids = [module.id for module in modules]
    if len(module_ids) != len(set(module_ids)):
        raise CurriculumValidationError("Curriculum module ids must be unique")

    module_id_set = set(module_ids)
    for module in modules:
        missing_prerequisites = [prerequisite for prerequisite in module.prerequisites if prerequisite not in module_id_set]
        if missing_prerequisites:
            raise CurriculumValidationError(
                f"Module {module.id!r} has unknown prerequisites: {', '.join(missing_prerequisites)}"
            )

    ordered_modules = _topologically_sort_modules(modules)
    logger.info("Loaded %d curriculum modules from %s", len(ordered_modules), curriculum_path)
    return tuple(ordered_modules)


def list_modules(curriculum_path: Path = CURRICULUM_PATH) -> list[CurriculumModule]:
    """Return the curriculum modules in valid progression order."""

    return list(load_curriculum(curriculum_path))


def get_module(module_id: str, curriculum_path: Path = CURRICULUM_PATH) -> CurriculumModule | None:
    """Return a curriculum module by id if it exists."""

    return next((module for module in list_modules(curriculum_path) if module.id == module_id), None)


def get_module_documents(module_id: str, curriculum_path: Path = CURRICULUM_PATH) -> list[str]:
    """Return the document paths assigned to a module."""

    module = get_module(module_id, curriculum_path)
    if module is None:
        raise CurriculumValidationError(f"Unknown curriculum module: {module_id}")
    return list(module.documents)


def prerequisites_satisfied(
    module_id: str,
    completed_module_ids: Sequence[str],
    curriculum_path: Path = CURRICULUM_PATH,
) -> bool:
    """Check whether the prerequisites for a module are satisfied."""

    module = get_module(module_id, curriculum_path)
    if module is None:
        raise CurriculumValidationError(f"Unknown curriculum module: {module_id}")

    completed = set(completed_module_ids)
    return all(prerequisite in completed for prerequisite in module.prerequisites)


def next_modules(
    completed_module_ids: Sequence[str],
    curriculum_path: Path = CURRICULUM_PATH,
) -> list[CurriculumModule]:
    """Return curriculum modules whose prerequisites are satisfied."""

    completed = set(completed_module_ids)
    return [
        module
        for module in list_modules(curriculum_path)
        if module.id not in completed and prerequisites_satisfied(module.id, completed, curriculum_path)
    ]


def resolve_learning_progression(curriculum_path: Path = CURRICULUM_PATH) -> list[CurriculumModule]:
    """Return the curriculum in valid learning order."""

    return list_modules(curriculum_path)


def earliest_entry_module(curriculum_path: Path = CURRICULUM_PATH) -> CurriculumModule:
    """Return the first module that can serve as a starting entry point."""

    modules = list_modules(curriculum_path)
    for module in modules:
        if not module.prerequisites:
            return module
    raise CurriculumValidationError("Curriculum does not contain an entry module")


def missing_document_paths(
    source_root: Path = SOURCES_ROOT,
    curriculum_path: Path = CURRICULUM_PATH,
) -> list[str]:
    """Return curriculum document paths that do not exist under sources/."""

    missing: list[str] = []
    for module in list_modules(curriculum_path):
        for document_path in module.documents:
            if not (source_root / document_path).is_file():
                missing.append(document_path)
    return sorted(set(missing))

