"""Retrieval evaluation helpers for the frozen baseline and optional reranking."""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence

import yaml

from src.rerank import RerankedChunk, rerank_chunks
from src.retrieve import RetrievedChunk, retrieve_chunks


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
EVALUATION_QUERIES_PATH = EVALUATION_DIR / "retrieval_queries.yaml"
BASELINE_REPORT_PATH = EVALUATION_DIR / "retrieval_baseline.md"
RERANKED_REPORT_PATH = EVALUATION_DIR / "retrieval_reranked.md"
SOURCES_ROOT = PROJECT_ROOT / "sources"
BASELINE_TOP_K = 5
RERANK_CANDIDATE_TOP_K = 15
RERANK_FINAL_TOP_K = 5


@dataclass(slots=True)
class RetrievalQuery:
    """A single curated evaluation query."""

    id: str
    question: str
    expected_primary: list[str]


@dataclass(slots=True)
class RetrievalQueryResult:
    """Per-query baseline retrieval output."""

    id: str
    question: str
    expected_primary: list[str]
    rank: int | None
    top_5_titles: list[str]
    top_5_scores: list[float]
    hit_at_1: int
    hit_at_3: int
    hit_at_5: int
    reciprocal_rank: float


@dataclass(slots=True)
class RerankedQueryResult:
    """Per-query reranked retrieval output."""

    id: str
    question: str
    expected_primary: list[str]
    rank: int | None
    candidate_rank: int | None
    top_5_titles: list[str]
    top_5_pinecone_scores: list[float]
    top_5_cohere_scores: list[float]
    hit_at_1: int
    hit_at_3: int
    hit_at_5: int
    reciprocal_rank: float


@dataclass(slots=True)
class RetrievalBaselineSummary:
    """Aggregate retrieval metrics across the evaluation set."""

    corpus_size: int | None
    query_count: int
    mean_hit_at_1: float
    mean_hit_at_3: float
    mean_hit_at_5: float
    mrr: float


@dataclass(slots=True)
class RetrievalBaselineReport:
    """Complete retrieval baseline output."""

    summary: RetrievalBaselineSummary
    results: list[RetrievalQueryResult]
    observations: list[str]


@dataclass(slots=True)
class RetrievalComparisonRow:
    """A single comparison metric row for baseline vs reranked evaluation."""

    metric: str
    baseline: float
    reranked: float
    change: float


@dataclass(slots=True)
class RerankedEvaluationReport:
    """Complete reranked evaluation output."""

    summary: RetrievalBaselineSummary
    baseline_summary: RetrievalBaselineSummary
    results: list[RerankedQueryResult]
    comparison_rows: list[RetrievalComparisonRow]
    observations: list[str]
    decision_analysis: list[str]


def _normalise_text(value: str) -> str:
    """Return a whitespace-normalised comparison string."""

    return " ".join(value.split()).casefold()


def _load_yaml(path: Path) -> Any:
    """Load a YAML document from ``path``."""

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_retrieval_queries(path: Path = EVALUATION_QUERIES_PATH) -> list[RetrievalQuery]:
    """Load the curated retrieval evaluation queries."""

    if not path.exists():
        raise FileNotFoundError(f"Retrieval evaluation queries file not found: {path}")

    loaded = _load_yaml(path)
    if not isinstance(loaded, list):
        raise ValueError(f"Retrieval query file must contain a list of queries: {path}")

    queries: list[RetrievalQuery] = []
    for index, item in enumerate(loaded, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid query entry at position {index} in {path}")

        query_id = item.get("id")
        question = item.get("question")
        expected_primary = item.get("expected_primary")

        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"Invalid or missing query id at position {index} in {path}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Invalid or missing question for query {query_id!r}")

        if isinstance(expected_primary, str):
            expected_sources = [expected_primary]
        elif isinstance(expected_primary, list):
            expected_sources = [str(source) for source in expected_primary if str(source).strip()]
        else:
            raise ValueError(
                f"Invalid expected_primary for query {query_id!r}; expected a string or list"
            )

        if not expected_sources:
            raise ValueError(f"Query {query_id!r} must define at least one expected source")

        queries.append(
            RetrievalQuery(
                id=query_id.strip(),
                question=question.strip(),
                expected_primary=expected_sources,
            )
        )

    logger.info("Loaded %d retrieval evaluation queries from %s", len(queries), path)
    return queries


def _chunk_title(chunk: RetrievedChunk | RerankedChunk) -> str:
    metadata = chunk.metadata or {}
    return str(metadata.get("document_title") or metadata.get("filename") or "").strip()


def _resolve_rank(
    retrieved_chunks: Sequence[RetrievedChunk | RerankedChunk],
    expected_primary: Sequence[str],
) -> tuple[int | None, str | None]:
    """Return the highest-ranked matching source and its title."""

    expected = {_normalise_text(title) for title in expected_primary}
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        title = _chunk_title(chunk)
        if title and _normalise_text(title) in expected:
            return rank, title
    return None, None


def evaluate_query(
    query: RetrievalQuery,
    *,
    retrieve_fn: Callable[..., list[RetrievedChunk]] = retrieve_chunks,
    top_k: int = BASELINE_TOP_K,
) -> RetrievalQueryResult:
    """Evaluate a single baseline retrieval query."""

    retrieved_chunks = retrieve_fn(query.question, top_k=top_k)
    top_titles = [_chunk_title(chunk) for chunk in retrieved_chunks[:top_k]]
    top_scores = [float(chunk.score) for chunk in retrieved_chunks[:top_k]]

    rank, _ = _resolve_rank(retrieved_chunks[:top_k], query.expected_primary)
    hit_at_1 = 1 if rank == 1 else 0
    hit_at_3 = 1 if rank is not None and rank <= 3 else 0
    hit_at_5 = 1 if rank is not None and rank <= 5 else 0
    reciprocal_rank = 1.0 / rank if rank is not None else 0.0

    return RetrievalQueryResult(
        id=query.id,
        question=query.question,
        expected_primary=list(query.expected_primary),
        rank=rank,
        top_5_titles=top_titles,
        top_5_scores=top_scores,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        hit_at_5=hit_at_5,
        reciprocal_rank=reciprocal_rank,
    )


def evaluate_query_reranked(
    query: RetrievalQuery,
    *,
    retrieve_fn: Callable[..., list[RetrievedChunk]] = retrieve_chunks,
    rerank_fn: Callable[..., list[RerankedChunk]] = rerank_chunks,
    candidate_top_k: int = RERANK_CANDIDATE_TOP_K,
    final_top_k: int = RERANK_FINAL_TOP_K,
) -> RerankedQueryResult:
    """Evaluate a single query after reranking the broader candidate set."""

    candidate_chunks = retrieve_fn(query.question, top_k=candidate_top_k)
    candidate_rank, _ = _resolve_rank(candidate_chunks, query.expected_primary)

    reranked_chunks = rerank_fn(
        query.question,
        candidate_chunks,
        top_n=len(candidate_chunks),
    )
    final_chunks = reranked_chunks[:final_top_k]

    top_titles = [_chunk_title(chunk) for chunk in final_chunks]
    top_pinecone_scores = [float(chunk.pinecone_score) for chunk in final_chunks]
    top_cohere_scores = [float(chunk.cohere_score) for chunk in final_chunks]

    rank, _ = _resolve_rank(final_chunks, query.expected_primary)
    hit_at_1 = 1 if rank == 1 else 0
    hit_at_3 = 1 if rank is not None and rank <= 3 else 0
    hit_at_5 = 1 if rank is not None and rank <= 5 else 0
    reciprocal_rank = 1.0 / rank if rank is not None else 0.0

    return RerankedQueryResult(
        id=query.id,
        question=query.question,
        expected_primary=list(query.expected_primary),
        rank=rank,
        candidate_rank=candidate_rank,
        top_5_titles=top_titles,
        top_5_pinecone_scores=top_pinecone_scores,
        top_5_cohere_scores=top_cohere_scores,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        hit_at_5=hit_at_5,
        reciprocal_rank=reciprocal_rank,
    )


def _compute_summary(
    results: Sequence[RetrievalQueryResult | RerankedQueryResult],
    *,
    corpus_size: int | None,
) -> RetrievalBaselineSummary:
    """Compute aggregate retrieval metrics."""

    query_count = len(results)
    if query_count == 0:
        return RetrievalBaselineSummary(
            corpus_size=corpus_size,
            query_count=0,
            mean_hit_at_1=0.0,
            mean_hit_at_3=0.0,
            mean_hit_at_5=0.0,
            mrr=0.0,
        )

    return RetrievalBaselineSummary(
        corpus_size=corpus_size,
        query_count=query_count,
        mean_hit_at_1=mean(result.hit_at_1 for result in results),
        mean_hit_at_3=mean(result.hit_at_3 for result in results),
        mean_hit_at_5=mean(result.hit_at_5 for result in results),
        mrr=mean(result.reciprocal_rank for result in results),
    )


def _build_observations(results: Sequence[RetrievalQueryResult]) -> list[str]:
    """Return concise objective observations about ranking failures."""

    observations: list[str] = []
    not_found = [result.id for result in results if result.rank is None]
    late_hits = [f"{result.id} (rank {result.rank})" for result in results if result.rank not in (None, 1)]

    if not_found:
        observations.append(
            "Queries not found in the top 5: " + ", ".join(not_found)
        )
    if late_hits:
        observations.append(
            "Queries where the expected source was not ranked first: " + ", ".join(late_hits)
        )
    if not observations:
        observations.append("No top-5 misses observed in this evaluation set.")
    return observations


def _build_reranked_observations(results: Sequence[RerankedQueryResult]) -> list[str]:
    """Return objective observations for the reranked pass."""

    observations: list[str] = []
    not_found = [result.id for result in results if result.rank is None]
    recovered_but_late = [
        f"{result.id} (candidate rank {result.candidate_rank}, final rank {result.rank})"
        for result in results
        if result.candidate_rank is not None and result.rank is not None and result.rank > 1
    ]
    unrecovered = [
        f"{result.id} (not present in original top 15)"
        for result in results
        if result.candidate_rank is None
    ]

    if not_found:
        observations.append(
            "Queries not found in the final top 5: " + ", ".join(not_found)
        )
    if recovered_but_late:
        observations.append(
            "Expected sources still not ranked first after reranking: "
            + ", ".join(recovered_but_late)
        )
    if unrecovered:
        observations.append(
            "Queries that reranking cannot recover from the original top 15: "
            + ", ".join(unrecovered)
        )
    if not observations:
        observations.append("No top-5 misses observed in the reranked evaluation set.")
    return observations


def _read_metric_from_line(pattern: str, text: str) -> float:
    """Extract a numeric metric from a markdown report."""

    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Unable to parse metric using pattern: {pattern}")
    return float(match.group(1))


def load_frozen_baseline_summary(path: Path = BASELINE_REPORT_PATH) -> RetrievalBaselineSummary:
    """Load the frozen baseline metrics from the saved markdown report."""

    if not path.exists():
        raise FileNotFoundError(f"Baseline report not found: {path}")

    text = path.read_text(encoding="utf-8")
    corpus_size_match = re.search(r"- Corpus size: (\d+) PDFs", text)
    query_count_match = re.search(r"- Number of evaluation queries: (\d+)", text)
    if corpus_size_match is None or query_count_match is None:
        raise ValueError(f"Baseline report does not contain corpus/query counts: {path}")

    return RetrievalBaselineSummary(
        corpus_size=int(corpus_size_match.group(1)),
        query_count=int(query_count_match.group(1)),
        mean_hit_at_1=_read_metric_from_line(r"- Mean Hit@1: ([0-9.]+)", text),
        mean_hit_at_3=_read_metric_from_line(r"- Mean Hit@3: ([0-9.]+)", text),
        mean_hit_at_5=_read_metric_from_line(r"- Mean Hit@5: ([0-9.]+)", text),
        mrr=_read_metric_from_line(r"- Mean Reciprocal Rank \(MRR\): ([0-9.]+)", text),
    )


def build_markdown_report(report: RetrievalBaselineReport) -> str:
    """Render the retrieval baseline report as Markdown."""

    lines: list[str] = [
        "# Retrieval Baseline Report",
        "",
        "## Summary",
        "",
    ]

    if report.summary.corpus_size is not None:
        lines.append(f"- Corpus size: {report.summary.corpus_size} PDFs")
    else:
        lines.append("- Corpus size: unavailable")
    lines.extend(
        [
            f"- Number of evaluation queries: {report.summary.query_count}",
            f"- Mean Hit@1: {report.summary.mean_hit_at_1:.4f}",
            f"- Mean Hit@3: {report.summary.mean_hit_at_3:.4f}",
            f"- Mean Hit@5: {report.summary.mean_hit_at_5:.4f}",
            f"- Mean Reciprocal Rank (MRR): {report.summary.mrr:.4f}",
            "",
            "## Per-Query Results",
            "",
        ]
    )

    for result in report.results:
        rank_display = result.rank if result.rank is not None else "not found"
        lines.extend(
            [
                f"### `{result.id}`",
                f"- Question: {result.question}",
                f"- Expected primary source: {', '.join(result.expected_primary)}",
                f"- Rank in top 5: {rank_display}",
                f"- Hit@1: {result.hit_at_1}",
                f"- Hit@3: {result.hit_at_3}",
                f"- Hit@5: {result.hit_at_5}",
                f"- Reciprocal rank: {result.reciprocal_rank:.4f}",
                "- Top-5 retrieved document titles and scores:",
            ]
        )

        if result.top_5_titles:
            for index, (title, score) in enumerate(
                zip(result.top_5_titles, result.top_5_scores, strict=True),
                start=1,
            ):
                lines.append(f"  {index}. {title} — {score:.4f}")
        else:
            lines.append("  - No retrieval results.")
        lines.append("")

    lines.extend(
        [
            "## Observations",
            "",
        ]
    )
    for observation in report.observations:
        lines.append(f"- {observation}")
    lines.append("")
    return "\n".join(lines)


def build_reranked_markdown_report(report: RerankedEvaluationReport) -> str:
    """Render the reranked retrieval report as Markdown."""

    lines: list[str] = [
        "# Retrieval Reranked Report",
        "",
        "## Summary",
        "",
        f"- Corpus size: {report.summary.corpus_size} PDFs"
        if report.summary.corpus_size is not None
        else "- Corpus size: unavailable",
        f"- Number of evaluation queries: {report.summary.query_count}",
        f"- Mean Hit@1: {report.summary.mean_hit_at_1:.4f}",
        f"- Mean Hit@3: {report.summary.mean_hit_at_3:.4f}",
        f"- Mean Hit@5: {report.summary.mean_hit_at_5:.4f}",
        f"- Mean Reciprocal Rank (MRR): {report.summary.mrr:.4f}",
        "",
        "## Comparison Table",
        "",
        "| Metric | Baseline | Reranked | Change |",
        "| --- | ---: | ---: | ---: |",
    ]

    for row in report.comparison_rows:
        lines.append(
            f"| {row.metric} | {row.baseline:.4f} | {row.reranked:.4f} | {row.change:+.4f} |"
        )

    lines.extend(["", "## Per-Query Results", ""])
    for result in report.results:
        rank_display = result.rank if result.rank is not None else "not found"
        candidate_rank_display = (
            result.candidate_rank if result.candidate_rank is not None else "not found"
        )
        lines.extend(
            [
                f"### `{result.id}`",
                f"- Question: {result.question}",
                f"- Expected primary source: {', '.join(result.expected_primary)}",
                f"- Final rank in top 5: {rank_display}",
                f"- Rank in original top 15: {candidate_rank_display}",
                f"- Hit@1: {result.hit_at_1}",
                f"- Hit@3: {result.hit_at_3}",
                f"- Hit@5: {result.hit_at_5}",
                f"- Reciprocal rank: {result.reciprocal_rank:.4f}",
                "- Top-5 reranked titles and scores:",
            ]
        )

        if result.top_5_titles:
            for index, (title, pinecone_score, cohere_score) in enumerate(
                zip(
                    result.top_5_titles,
                    result.top_5_pinecone_scores,
                    result.top_5_cohere_scores,
                    strict=True,
                ),
                start=1,
            ):
                lines.append(
                    f"  {index}. {title} — Pinecone: {pinecone_score:.4f} | Cohere: {cohere_score:.4f}"
                )
        else:
            lines.append("  - No reranked results.")
        lines.append("")

    lines.extend(["## Observations", ""])
    for observation in report.observations:
        lines.append(f"- {observation}")

    lines.extend(["", "## Decision Analysis", ""])
    for item in report.decision_analysis:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _build_comparison_rows(
    baseline_summary: RetrievalBaselineSummary,
    reranked_summary: RetrievalBaselineSummary,
) -> list[RetrievalComparisonRow]:
    """Build comparison rows between the frozen baseline and reranked results."""

    return [
        RetrievalComparisonRow(
            metric="Hit@1",
            baseline=baseline_summary.mean_hit_at_1,
            reranked=reranked_summary.mean_hit_at_1,
            change=reranked_summary.mean_hit_at_1 - baseline_summary.mean_hit_at_1,
        ),
        RetrievalComparisonRow(
            metric="Hit@3",
            baseline=baseline_summary.mean_hit_at_3,
            reranked=reranked_summary.mean_hit_at_3,
            change=reranked_summary.mean_hit_at_3 - baseline_summary.mean_hit_at_3,
        ),
        RetrievalComparisonRow(
            metric="Hit@5",
            baseline=baseline_summary.mean_hit_at_5,
            reranked=reranked_summary.mean_hit_at_5,
            change=reranked_summary.mean_hit_at_5 - baseline_summary.mean_hit_at_5,
        ),
        RetrievalComparisonRow(
            metric="MRR",
            baseline=baseline_summary.mrr,
            reranked=reranked_summary.mrr,
            change=reranked_summary.mrr - baseline_summary.mrr,
        ),
    ]


def _build_decision_analysis(
    baseline_summary: RetrievalBaselineSummary,
    reranked_summary: RetrievalBaselineSummary,
    results: Sequence[RerankedQueryResult],
) -> list[str]:
    """Summarise whether reranking is justified using objective criteria."""

    gcp = next((result for result in results if result.id == "gcp_definition"), None)
    q9 = next((result for result in results if result.id == "quality_risk_management"), None)
    eu_ai = next((result for result in results if result.id == "eu_ai_act"), None)

    hit_at_1_improved = reranked_summary.mean_hit_at_1 > baseline_summary.mean_hit_at_1
    mrr_improved = reranked_summary.mrr > baseline_summary.mrr
    hit_at_3_ok = reranked_summary.mean_hit_at_3 >= baseline_summary.mean_hit_at_3
    hit_at_5_ok = reranked_summary.mean_hit_at_5 >= baseline_summary.mean_hit_at_5
    gcp_above_gclp = gcp is not None and gcp.rank == 1
    q9_above_q10 = q9 is not None and q9.rank == 1
    eu_ai_recovered = eu_ai is not None and eu_ai.candidate_rank is not None

    recommendation = (
        "Reranking is justified for production consideration"
        if hit_at_1_improved and mrr_improved and hit_at_3_ok and hit_at_5_ok
        else "Reranking is not yet justified for production enablement"
    )

    lines = [
        recommendation + ".",
        f"Hit@1 improved: {'yes' if hit_at_1_improved else 'no'}.",
        f"MRR improved: {'yes' if mrr_improved else 'no'}.",
        f"Hit@3 stayed equal or improved: {'yes' if hit_at_3_ok else 'no'}.",
        f"Hit@5 stayed equal or improved: {'yes' if hit_at_5_ok else 'no'}.",
        f"GCP moved above GCLP: {'yes' if gcp_above_gclp else 'no'}.",
        f"ICH Q9 moved above ICH Q10: {'yes' if q9_above_q10 else 'no'}.",
    ]

    if eu_ai is None:
        lines.append("EU AI Act query was not evaluated.")
    elif eu_ai.candidate_rank is None:
        lines.append(
            "EU AI Act was not present in the original top 15 candidates, so reranking cannot recover that miss."
        )
    elif eu_ai_recovered:
        lines.append(
            f"EU AI Act was present in the original top 15 candidates at rank {eu_ai.candidate_rank}."
        )
    else:
        lines.append(
            f"EU AI Act was present in the original top 15 candidates at rank {eu_ai.candidate_rank}, but reranking did not promote it into the final top 5."
        )

    lines.append(
        "Added Cohere latency and API usage: one Cohere rerank call per evaluation query on top of the existing Pinecone retrieval call."
    )
    return lines


def run_baseline_evaluation(
    *,
    queries_path: Path = EVALUATION_QUERIES_PATH,
    report_path: Path = BASELINE_REPORT_PATH,
    retrieve_fn: Callable[..., list[RetrievedChunk]] = retrieve_chunks,
) -> RetrievalBaselineReport:
    """Run the retrieval baseline evaluation and persist the Markdown report."""

    queries = load_retrieval_queries(queries_path)
    corpus_size = (
        len([path for path in SOURCES_ROOT.rglob("*.pdf") if path.is_file()])
        if SOURCES_ROOT.exists()
        else None
    )
    results = [
        evaluate_query(query, retrieve_fn=retrieve_fn, top_k=BASELINE_TOP_K)
        for query in queries
    ]
    summary = _compute_summary(results, corpus_size=corpus_size)
    observations = _build_observations(results)
    report = RetrievalBaselineReport(summary=summary, results=results, observations=observations)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_markdown_report(report), encoding="utf-8")
    logger.info("Wrote retrieval baseline report to %s", report_path)
    return report


def run_reranked_evaluation(
    *,
    queries_path: Path = EVALUATION_QUERIES_PATH,
    baseline_report_path: Path = BASELINE_REPORT_PATH,
    report_path: Path = RERANKED_REPORT_PATH,
    retrieve_fn: Callable[..., list[RetrievedChunk]] = retrieve_chunks,
    rerank_fn: Callable[..., list[RerankedChunk]] = rerank_chunks,
) -> RerankedEvaluationReport:
    """Run the reranked retrieval evaluation and persist the Markdown report."""

    queries = load_retrieval_queries(queries_path)
    baseline_summary = load_frozen_baseline_summary(baseline_report_path)
    results = [
        evaluate_query_reranked(
            query,
            retrieve_fn=retrieve_fn,
            rerank_fn=rerank_fn,
            candidate_top_k=RERANK_CANDIDATE_TOP_K,
            final_top_k=RERANK_FINAL_TOP_K,
        )
        for query in queries
    ]
    summary = _compute_summary(results, corpus_size=baseline_summary.corpus_size)
    comparison_rows = _build_comparison_rows(baseline_summary, summary)
    observations = _build_reranked_observations(results)
    decision_analysis = _build_decision_analysis(baseline_summary, summary, results)
    report = RerankedEvaluationReport(
        summary=summary,
        baseline_summary=baseline_summary,
        results=results,
        comparison_rows=comparison_rows,
        observations=observations,
        decision_analysis=decision_analysis,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_reranked_markdown_report(report), encoding="utf-8")
    logger.info("Wrote reranked retrieval report to %s", report_path)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the evaluation entry point."""

    parser = argparse.ArgumentParser(description="Run retrieval evaluation.")
    parser.add_argument(
        "--mode",
        choices=("baseline", "reranked"),
        default="baseline",
        help="Evaluation mode to run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for retrieval evaluation."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode == "baseline":
        report = run_baseline_evaluation()
        print(build_markdown_report(report))
    else:
        report = run_reranked_evaluation()
        print(build_reranked_markdown_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    raise SystemExit(main())
