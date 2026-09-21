import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load documents
documents_path = PROJECT_ROOT / "data" / "processed" / "documents.json"

with open(documents_path, "r", encoding="utf-8") as f:
    documents = json.load(f)


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Encode documents
document_texts = [document["text"] for document in documents]

document_embeddings = model.encode(
    document_texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)


def retrieve(query, k=5):
    """
    Retrieve the top-k documents using dense semantic similarity.
    """

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Cosine similarity because embeddings are normalized
    scores = document_embeddings @ query_embedding

    ranked_indices = np.argsort(scores)[::-1][:k]

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