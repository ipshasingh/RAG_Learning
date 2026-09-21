import json
from pathlib import Path

from rank_bm25 import BM25Okapi


# Find project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load processed documents
documents_path = PROJECT_ROOT / "data" / "processed" / "documents.json"

with open(documents_path, "r", encoding="utf-8") as f:
    documents = json.load(f)


# Tokenize documents
tokenized_documents = [
    document["text"].lower().split()
    for document in documents
]


# Build BM25 index
bm25 = BM25Okapi(tokenized_documents)


def retrieve(query, k=5):
    """
    Retrieve the top-k documents for a query using BM25.
    """

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = scores.argsort()[::-1][:k]

    results = []

    for index in ranked_indices:
        results.append({
            "id": documents[index]["id"],
            "score": float(scores[index]),
            "text": documents[index]["text"]
        })

    return results


if __name__ == "__main__":

    query = "Which NFL team represented the AFC at Super Bowl 50?"

    results = retrieve(query, k=5)

    print(f"\nQuery: {query}\n")

    for rank, result in enumerate(results, start=1):
        print(f"Rank {rank}:")
        print(f"Document ID: {result['id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:200]}...")
        print()