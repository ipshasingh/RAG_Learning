import sys
import json
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Make project root available for imports
sys.path.insert(0, str(PROJECT_ROOT))


from retrieval.bm25 import retrieve as bm25_retrieve
from retrieval.dense import retrieve as dense_retrieve


# Load queries
queries_path = PROJECT_ROOT / "data" / "processed" / "queries.json"

with open(queries_path, "r", encoding="utf-8") as f:
    queries = json.load(f)


def recall_at_k(results, relevant_doc_id, k):
    """
    Returns 1 if the relevant document appears
    in the top-k results, otherwise 0.
    """

    retrieved_ids = [
        result["id"]
        for result in results[:k]
    ]

    return int(relevant_doc_id in retrieved_ids)


def reciprocal_rank(results, relevant_doc_id):
    """
    Returns the reciprocal rank of the relevant document.
    Returns 0 if the document was not retrieved.
    """

    for rank, result in enumerate(results, start=1):

        if result["id"] == relevant_doc_id:
            return 1 / rank

    return 0


def evaluate_retriever(retriever, name):

    recall_1 = 0
    recall_5 = 0
    recall_10 = 0

    reciprocal_ranks = []

    for i, query in enumerate(queries):

        results = retriever(
            query["text"],
            k=10
        )

        relevant_doc_id = query["relevant_doc_id"]

        recall_1 += recall_at_k(
            results,
            relevant_doc_id,
            1
        )

        recall_5 += recall_at_k(
            results,
            relevant_doc_id,
            5
        )

        recall_10 += recall_at_k(
            results,
            relevant_doc_id,
            10
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                results,
                relevant_doc_id
            )
        )

        if (i + 1) % 500 == 0:
            print(
                f"{name}: "
                f"processed {i + 1}/{len(queries)} queries"
            )

    total = len(queries)

    return {
        "Recall@1": recall_1 / total,
        "Recall@5": recall_5 / total,
        "Recall@10": recall_10 / total,
        "MRR": sum(reciprocal_ranks) / total
    }


if __name__ == "__main__":

    print("\nEvaluating BM25...\n")

    bm25_results = evaluate_retriever(
        bm25_retrieve,
        "BM25"
    )

    print("\nEvaluating Dense Retrieval...\n")

    dense_results = evaluate_retriever(
        dense_retrieve,
        "Dense"
    )

    print("\n================ RESULTS ================\n")

    print("BM25:")

    for metric, value in bm25_results.items():
        print(f"{metric}: {value:.4f}")

    print("\nDense:")

    for metric, value in dense_results.items():
        print(f"{metric}: {value:.4f}")

    # Combine results
    all_results = {
        "BM25": bm25_results,
        "Dense": dense_results
    }

    # Save results
    results_path = RESULTS_DIR / "retrieval_results.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            all_results,
            f,
            indent=4
        )

    print(f"\nResults saved to: {results_path}")