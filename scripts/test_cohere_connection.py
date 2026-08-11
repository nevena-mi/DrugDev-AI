"""Standalone validation script for Cohere reranking connectivity."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"


def _bootstrap_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def main() -> int:
    """Run a small Cohere reranking smoke test."""

    _bootstrap_path()
    load_dotenv(ENV_PATH)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        import cohere
    except Exception as exc:  # pragma: no cover - environment failure
        print(f"Failed to import Cohere SDK: {type(exc).__name__}: {exc}")
        return 1

    load_dotenv(ENV_PATH)
    loaded_key = os.getenv("COHERE_API_KEY")

    print(f"Cohere SDK version: {cohere.__version__}")
    print(f"COHERE_API_KEY loaded: {'yes' if loaded_key else 'no'}")

    if not loaded_key:
        print("Missing COHERE_API_KEY in .env")
        return 1

    documents = [
        "Good Clinical Practice sets ethical and scientific quality standards for clinical trials.",
        "Pharmacovigilance planning describes the activities needed to detect and manage safety risks.",
        "The IND application is a request to begin human trials of a new investigational drug.",
        "AI systems in healthcare must be managed with attention to governance and safety.",
        "Quality risk management identifies, evaluates, and controls risks to product quality.",
    ]
    query = "What is Good Clinical Practice?"

    try:
        client = cohere.ClientV2(api_key=loaded_key)
        response = client.rerank(
            model="rerank-v4.0-pro",
            query=query,
            documents=documents,
            top_n=5,
        )
    except (
        cohere.UnauthorizedError,
        cohere.ForbiddenError,
        cohere.BadRequestError,
        cohere.UnprocessableEntityError,
        cohere.TooManyRequestsError,
        cohere.InternalServerError,
        cohere.ServiceUnavailableError,
        cohere.GatewayTimeoutError,
    ) as exc:
        print(f"Cohere rerank failed: {type(exc).__name__}: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"Cohere rerank failed: {type(exc).__name__}: {exc}")
        return 1

    print("Rerank completed successfully.")
    for result in response.results:
        document_text = documents[result.index]
        print(f"index={result.index}")
        print(f"relevance_score={result.relevance_score}")
        print(f"document={document_text}")

    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    raise SystemExit(main())
