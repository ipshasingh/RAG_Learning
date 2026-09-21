import sys
import json
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from retrieval.bm25 import retrieve as bm25_retrieve
from retrieval.dense import retrieve as dense_retrieve


# Load queries
queries_path = PROJECT_ROOT / "data" / "processed" / "queries.json"

with open(queries_path, "r", encoding="utf-8") as f:
    queries = json.load(f)


def contains_relevant(results, relevant_doc_id, k=5):
    """
    Check whether the relevant document appears
    in the top-k results.
    """

    retrieved_ids = [
        result["id"]
        for result in results[:k]
    ]

    return relevant_doc_id in retrieved_ids


def get_category(bm25_success, dense_success):

    if bm25_success and dense_success:
        return "both_success"

    if bm25_success and not dense_success:
        return "bm25_only"

    if not bm25_success and dense_success:
        return "dense_only"

    return "both_fail"


def main():

    categories = {
        "both_success": [],
        "bm25_only": [],
        "dense_only": [],
        "both_fail": []
    }

    for i, query in enumerate(queries):

        bm25_results = bm25_retrieve(
            query["text"],
            k=5
        )

        dense_results = dense_retrieve(
            query["text"],
            k=5
        )

        relevant_doc_id = query["relevant_doc_id"]

        bm25_success = contains_relevant(
            bm25_results,
            relevant_doc_id,
            k=5
        )

        dense_success = contains_relevant(
            dense_results,
            relevant_doc_id,
            k=5
        )

        category = get_category(
            bm25_success,
            dense_success
        )

        categories[category].append({
            "query_id": query["id"],
            "query": query["text"],
            "relevant_doc_id": relevant_doc_id
        })

        if (i + 1) % 500 == 0:
            print(
                f"Processed {i + 1}/{len(queries)} queries"
            )

    print("\n================ ERROR ANALYSIS ================\n")

    total = len(queries)

    for category, examples in categories.items():

        percentage = (
            len(examples) / total
        ) * 100

        print(
            f"{category}: "
            f"{len(examples)} "
            f"({percentage:.2f}%)"
        )

    print("\n================ EXAMPLES ================\n")

    for category in [
        "bm25_only",
        "dense_only",
        "both_fail"
    ]:

        print(f"\n--- {category.upper()} ---")

        for example in categories[category][:5]:

            print(
                f"\nQuery: {example['query']}"
            )

            print(
                f"Ground truth: "
                f"{example['relevant_doc_id']}"
            )


if __name__ == "__main__":
    main()