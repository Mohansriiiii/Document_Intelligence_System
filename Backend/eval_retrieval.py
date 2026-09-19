"""
Retrieval evaluation script for the RAG pipeline.

Computes, for both PLAIN vector search and RE-RANKED retrieval:
  - Recall@k       : did the expected source appear anywhere in the top-k?
  - Precision@k    : what fraction of the top-k chunks came from the expected source?
  - MRR            : how high up (reciprocal rank) was the first correct-source chunk?

Also reports a per-stage latency breakdown (embedding+search vs re-ranking vs LLM generation).

USAGE
-----
1. Make sure your backend's Chroma DB already has documents indexed (via /upload,
   or by running this in the same environment as main.py after documents are loaded).
2. Fill in eval/test_set.json with real questions + expected source filenames.
3. Run:  python eval_retrieval.py
4. Results print to console and are saved to eval_results.json

This script imports directly from main.py, so run it from your backend/ folder
(or adjust the import path below) so it reuses the same embeddings, vectorstore,
reranker, and LLM config as your live app.
"""

import json
import time
import statistics
from pathlib import Path

# Import the already-configured objects from your backend instead of
# re-instantiating them, so the eval uses the exact same models/config.
from main import vectorstore, reranker, llm, rerank_results  # noqa: E402

TEST_SET_PATH = Path(__file__).parent / "test_set.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"

CANDIDATE_K = 12   # matches main.py's vectorstore.similarity_search(k=12)
TOP_N = 4          # matches main.py's rerank_results(top_n=4)


def load_test_set():
    data = json.loads(TEST_SET_PATH.read_text())
    questions = data.get("questions", [])
    if not questions:
        raise ValueError(
            f"No questions found in {TEST_SET_PATH}. "
            "Fill in test_set.json with real questions before running."
        )
    return questions


def reciprocal_rank(docs, expected_source: str) -> float:
    """1 / rank of the first chunk whose source matches expected_source. 0 if none found."""
    for i, doc in enumerate(docs):
        if doc.metadata.get("source") == expected_source:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(docs, expected_source: str) -> int:
    """1 if expected_source appears anywhere in docs, else 0."""
    return int(any(d.metadata.get("source") == expected_source for d in docs))


def precision_at_k(docs, expected_source: str) -> float:
    """Fraction of docs whose source matches expected_source."""
    if not docs:
        return 0.0
    matches = sum(1 for d in docs if d.metadata.get("source") == expected_source)
    return matches / len(docs)


def evaluate_question(item: dict) -> dict:
    question = item["question"]
    expected_source = item["expected_source"]

    # --- Stage timing + plain vector search (baseline, no re-ranking) ---
    t0 = time.perf_counter()
    candidates = vectorstore.similarity_search(question, k=CANDIDATE_K)
    t1 = time.perf_counter()
    search_latency = t1 - t0

    baseline_topn = candidates[:TOP_N]  # naive "just take the first N" baseline

    # --- Re-ranking stage ---
    t2 = time.perf_counter()
    reranked_topn = rerank_results(question, candidates, top_n=TOP_N)
    t3 = time.perf_counter()
    rerank_latency = t3 - t2

    return {
        "question": question,
        "expected_source": expected_source,
        "baseline": {
            "recall_at_n": recall_at_k(baseline_topn, expected_source),
            "precision_at_n": precision_at_k(baseline_topn, expected_source),
            "mrr": reciprocal_rank(baseline_topn, expected_source),
        },
        "reranked": {
            "recall_at_n": recall_at_k(reranked_topn, expected_source),
            "precision_at_n": precision_at_k(reranked_topn, expected_source),
            "mrr": reciprocal_rank(reranked_topn, expected_source),
        },
        "latency_sec": {
            "vector_search": round(search_latency, 4),
            "reranking": round(rerank_latency, 4),
        },
    }


def aggregate(results: list[dict]) -> dict:
    def avg(key_path):
        vals = []
        for r in results:
            d = r
            for k in key_path:
                d = d[k]
            vals.append(d)
        return round(statistics.mean(vals), 4) if vals else 0.0

    return {
        "n_questions": len(results),
        "baseline": {
            "recall_at_n": avg(["baseline", "recall_at_n"]),
            "precision_at_n": avg(["baseline", "precision_at_n"]),
            "mrr": avg(["baseline", "mrr"]),
        },
        "reranked": {
            "recall_at_n": avg(["reranked", "recall_at_n"]),
            "precision_at_n": avg(["reranked", "precision_at_n"]),
            "mrr": avg(["reranked", "mrr"]),
        },
        "avg_latency_sec": {
            "vector_search": avg(["latency_sec", "vector_search"]),
            "reranking": avg(["latency_sec", "reranking"]),
        },
    }


def print_report(summary: dict):
    def pct(x):
        return f"{x * 100:.1f}%"

    b, r = summary["baseline"], summary["reranked"]
    print("\n" + "=" * 60)
    print(f"RETRIEVAL EVAL — {summary['n_questions']} questions, "
          f"candidates k={CANDIDATE_K}, top_n={TOP_N}")
    print("=" * 60)
    print(f"{'Metric':<16}{'Baseline (vector only)':<26}{'Re-ranked':<16}{'Lift'}")
    print("-" * 60)

    def row(name, key, is_pct=True):
        bv, rv = b[key], r[key]
        fmt = pct if is_pct else (lambda x: f"{x:.4f}")
        lift = f"+{pct(rv - bv)}" if is_pct else f"+{rv - bv:.4f}"
        print(f"{name:<16}{fmt(bv):<26}{fmt(rv):<16}{lift}")

    row("Recall@N", "recall_at_n")
    row("Precision@N", "precision_at_n")
    row("MRR", "mrr", is_pct=False)

    print("-" * 60)
    lat = summary["avg_latency_sec"]
    print(f"Avg vector search latency : {lat['vector_search']}s")
    print(f"Avg re-ranking latency    : {lat['reranking']}s")
    print("=" * 60 + "\n")


def main():
    questions = load_test_set()
    print(f"Loaded {len(questions)} test questions from {TEST_SET_PATH}")

    results = []
    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Evaluating: {item['question'][:60]}...")
        results.append(evaluate_question(item))

    summary = aggregate(results)
    print_report(summary)

    RESULTS_PATH.write_text(json.dumps(
        {"summary": summary, "per_question": results}, indent=2
    ))
    print(f"Full results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
