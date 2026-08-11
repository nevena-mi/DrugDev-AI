from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.evaluate_retrieval import (
    RerankedEvaluationReport,
    RetrievalBaselineSummary,
    RetrievalQuery,
    RetrievalQueryResult,
    RerankedQueryResult,
    build_markdown_report,
    build_reranked_markdown_report,
    evaluate_query,
    evaluate_query_reranked,
    load_frozen_baseline_summary,
    load_retrieval_queries,
    run_baseline_evaluation,
    run_reranked_evaluation,
    _compute_summary,
)
from src.rerank import RerankedChunk
from src.retrieve import RetrievedChunk


def _build_reranked_chunk(
    *,
    chunk: RetrievedChunk,
    cohere_score: float,
    original_index: int,
    reranked_rank: int,
) -> RerankedChunk:
    return RerankedChunk(
        id=chunk.id,
        text=chunk.text,
        metadata=dict(chunk.metadata),
        pinecone_score=chunk.score,
        cohere_score=cohere_score,
        original_index=original_index,
        reranked_rank=reranked_rank,
    )


def _make_candidates() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(id=f"chunk-{index}", score=1.0 - index * 0.01, text=f"text {index}", metadata={"document_title": f"Doc {index}"})
        for index in range(15)
    ]


def test_load_retrieval_queries_reads_curated_yaml() -> None:
    queries = load_retrieval_queries()

    assert [query.id for query in queries] == [
        "gcp_definition",
        "quality_risk_management",
        "pharmacovigilance_planning",
        "ind",
        "eu_ai_act",
    ]
    assert queries[0].expected_primary == [
        "ICH E6(R3) Guideline for Good Clinical Practice",
    ]


def test_evaluate_query_uses_highest_ranked_matching_expected_source() -> None:
    query = RetrievalQuery(
        id="multi_expected",
        question="Which source applies?",
        expected_primary=["Doc A", "Doc B"],
    )
    retrieved = [
        RetrievedChunk(id="1", score=0.95, text="", metadata={"document_title": "Other"}),
        RetrievedChunk(id="2", score=0.90, text="", metadata={"document_title": "Doc B"}),
        RetrievedChunk(id="3", score=0.85, text="", metadata={"document_title": "Doc A"}),
    ]

    result = evaluate_query(query, retrieve_fn=Mock(return_value=retrieved))

    assert result.rank == 2
    assert result.hit_at_1 == 0
    assert result.hit_at_3 == 1
    assert result.hit_at_5 == 1
    assert result.reciprocal_rank == 0.5


def test_evaluate_query_handles_not_found() -> None:
    query = RetrievalQuery(
        id="missing",
        question="Missing source?",
        expected_primary=["Doc A", "Doc B"],
    )

    result = evaluate_query(query, retrieve_fn=Mock(return_value=[]))

    assert result.rank is None
    assert result.hit_at_1 == 0
    assert result.hit_at_3 == 0
    assert result.hit_at_5 == 0
    assert result.reciprocal_rank == 0.0
    assert result.top_5_titles == []
    assert result.top_5_scores == []


def test_reranked_query_uses_only_final_top_five() -> None:
    query = RetrievalQuery(
        id="reranked",
        question="What is the regulation?",
        expected_primary=["Doc 7"],
    )
    candidates = _make_candidates()
    reranked = [
        _build_reranked_chunk(chunk=candidates[index], cohere_score=1.0 - index * 0.02, original_index=index, reranked_rank=rank)
        for rank, index in enumerate([0, 1, 2, 3, 4, 5, 7, 6, 8, 9, 10, 11, 12, 13, 14], start=1)
    ]

    result = evaluate_query_reranked(
        query,
        retrieve_fn=Mock(return_value=candidates),
        rerank_fn=Mock(return_value=reranked),
    )

    assert result.candidate_rank == 8
    assert result.rank is None
    assert result.hit_at_1 == 0
    assert result.hit_at_3 == 0
    assert result.hit_at_5 == 0
    assert result.top_5_titles == ["Doc 0", "Doc 1", "Doc 2", "Doc 3", "Doc 4"]
    assert result.top_5_pinecone_scores == [candidate.score for candidate in candidates[:5]]
    assert result.top_5_cohere_scores == [1.0, 0.98, 0.96, 0.94, 0.92]


def test_reranked_query_uses_reordered_cands_and_preserves_scores() -> None:
    query = RetrievalQuery(
        id="reranked_hit",
        question="What is the regulation?",
        expected_primary=["Doc 2"],
    )
    candidates = _make_candidates()
    reranked = [
        _build_reranked_chunk(chunk=candidates[index], cohere_score=score, original_index=index, reranked_rank=rank)
        for rank, (index, score) in enumerate([(2, 0.99), (0, 0.88), (1, 0.77), (3, 0.66), (4, 0.55)], start=1)
    ]

    result = evaluate_query_reranked(
        query,
        retrieve_fn=Mock(return_value=candidates),
        rerank_fn=Mock(return_value=reranked),
        candidate_top_k=15,
        final_top_k=5,
    )

    assert result.candidate_rank == 3
    assert result.rank == 1
    assert result.hit_at_1 == 1
    assert result.hit_at_3 == 1
    assert result.hit_at_5 == 1
    assert result.top_5_titles == ["Doc 2", "Doc 0", "Doc 1", "Doc 3", "Doc 4"]
    assert result.top_5_pinecone_scores == [candidates[2].score, candidates[0].score, candidates[1].score, candidates[3].score, candidates[4].score]
    assert result.top_5_cohere_scores == [0.99, 0.88, 0.77, 0.66, 0.55]


def test_summary_metrics_and_report_rendering() -> None:
    results = [
        evaluate_query(
            RetrievalQuery(
                id="hit1",
                question="Q1",
                expected_primary=["Doc A"],
            ),
            retrieve_fn=Mock(
                return_value=[
                    RetrievedChunk(id="a", score=0.99, text="", metadata={"document_title": "Doc A"})
                ]
            ),
        ),
        evaluate_query(
            RetrievalQuery(
                id="miss",
                question="Q2",
                expected_primary=["Doc B"],
            ),
            retrieve_fn=Mock(
                return_value=[
                    RetrievedChunk(id="c", score=0.80, text="", metadata={"document_title": "Doc C"})
                ]
            ),
        ),
    ]
    summary = _compute_summary(results, corpus_size=12)
    report = build_markdown_report(
        SimpleNamespace(summary=summary, results=results, observations=["Queries not found in the top 5: miss"])
    )

    assert summary.mean_hit_at_1 == 0.5
    assert summary.mean_hit_at_3 == 0.5
    assert summary.mean_hit_at_5 == 0.5
    assert summary.mrr == 0.5
    assert "Retrieval Baseline Report" in report
    assert "Corpus size: 12 PDFs" in report
    assert "Queries not found in the top 5: miss" in report
    assert "hit1" in report
    assert "miss" in report


def test_reranked_report_rendering_includes_comparison_table() -> None:
    baseline_summary = RetrievalBaselineSummary(
        corpus_size=37,
        query_count=5,
        mean_hit_at_1=0.4,
        mean_hit_at_3=0.8,
        mean_hit_at_5=0.8,
        mrr=0.6,
    )
    reranked_summary = RetrievalBaselineSummary(
        corpus_size=37,
        query_count=5,
        mean_hit_at_1=0.6,
        mean_hit_at_3=0.8,
        mean_hit_at_5=0.8,
        mrr=0.7,
    )
    results = [
        RerankedQueryResult(
            id="gcp_definition",
            question="What is Good Clinical Practice?",
            expected_primary=["ICH E6(R3) Guideline for Good Clinical Practice"],
            rank=1,
            candidate_rank=2,
            top_5_titles=["ICH E6(R3) Guideline for Good Clinical Practice"],
            top_5_pinecone_scores=[0.91],
            top_5_cohere_scores=[0.99],
            hit_at_1=1,
            hit_at_3=1,
            hit_at_5=1,
            reciprocal_rank=1.0,
        )
    ]
    report = RerankedEvaluationReport(
        summary=reranked_summary,
        baseline_summary=baseline_summary,
        results=results,
        comparison_rows=[
            SimpleNamespace(metric="Hit@1", baseline=0.4, reranked=0.6, change=0.2),
            SimpleNamespace(metric="MRR", baseline=0.6, reranked=0.7, change=0.1),
        ],
        observations=["Queries not found in the final top 5: eu_ai_act"],
        decision_analysis=["Reranking is not yet justified for production enablement."],
    )

    text = build_reranked_markdown_report(report)

    assert "Retrieval Reranked Report" in text
    assert "| Metric | Baseline | Reranked | Change |" in text
    assert "Cohere: 0.9900" in text
    assert "Decision Analysis" in text


def test_load_frozen_baseline_summary_reads_frozen_report(tmp_path) -> None:
    report_path = tmp_path / "retrieval_baseline.md"
    report_path.write_text(
        "# Retrieval Baseline Report\n\n## Summary\n\n- Corpus size: 37 PDFs\n- Number of evaluation queries: 5\n- Mean Hit@1: 0.4000\n- Mean Hit@3: 0.8000\n- Mean Hit@5: 0.8000\n- Mean Reciprocal Rank (MRR): 0.6000\n",
        encoding="utf-8",
    )

    summary = load_frozen_baseline_summary(report_path)

    assert summary.corpus_size == 37
    assert summary.query_count == 5
    assert summary.mean_hit_at_1 == 0.4
    assert summary.mean_hit_at_3 == 0.8
    assert summary.mean_hit_at_5 == 0.8
    assert summary.mrr == 0.6


def test_run_baseline_evaluation_writes_report(tmp_path) -> None:
    queries_path = tmp_path / "queries.yaml"
    queries_path.write_text(
        "- id: sample\n  question: \"Q?\"\n  expected_primary:\n    - \"Doc A\"\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "retrieval_baseline.md"

    fake_retrieve = Mock(
        return_value=[
            RetrievedChunk(id="x", score=0.77, text="", metadata={"document_title": "Doc A"})
        ]
    )

    report = run_baseline_evaluation(
        queries_path=queries_path,
        report_path=report_path,
        retrieve_fn=fake_retrieve,
    )

    assert report.summary.query_count == 1
    assert report.results[0].rank == 1
    assert report_path.exists()
    written = report_path.read_text(encoding="utf-8")
    assert "Retrieval Baseline Report" in written
    assert "Doc A" in written


def test_run_reranked_evaluation_writes_report(tmp_path) -> None:
    queries_path = tmp_path / "queries.yaml"
    queries_path.write_text(
        "- id: sample\n  question: \"Q?\"\n  expected_primary:\n    - \"Doc A\"\n",
        encoding="utf-8",
    )
    baseline_report_path = tmp_path / "retrieval_baseline.md"
    baseline_report_path.write_text(
        "# Retrieval Baseline Report\n\n## Summary\n\n- Corpus size: 37 PDFs\n- Number of evaluation queries: 1\n- Mean Hit@1: 0.0000\n- Mean Hit@3: 0.0000\n- Mean Hit@5: 0.0000\n- Mean Reciprocal Rank (MRR): 0.0000\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "retrieval_reranked.md"

    candidates = [
        RetrievedChunk(id="x", score=0.77, text="Doc A text", metadata={"document_title": "Doc A"}),
        RetrievedChunk(id="y", score=0.70, text="Doc B text", metadata={"document_title": "Doc B"}),
    ]
    reranked = [
        _build_reranked_chunk(chunk=candidates[0], cohere_score=0.99, original_index=0, reranked_rank=1),
        _build_reranked_chunk(chunk=candidates[1], cohere_score=0.88, original_index=1, reranked_rank=2),
    ]

    report = run_reranked_evaluation(
        queries_path=queries_path,
        baseline_report_path=baseline_report_path,
        report_path=report_path,
        retrieve_fn=Mock(return_value=candidates),
        rerank_fn=Mock(return_value=reranked),
    )

    assert report.summary.query_count == 1
    assert report.results[0].rank == 1
    assert report.results[0].candidate_rank == 1
    assert report_path.exists()
    written = report_path.read_text(encoding="utf-8")
    assert "Retrieval Reranked Report" in written
    assert "Comparison Table" in written
