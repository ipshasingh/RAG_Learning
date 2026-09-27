import pymupdf
import re
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, CrossEncoder


# =========================================================
# 1. LOAD DOCUMENT
# =========================================================

pdf_path = "rag_search/data/inception.pdf"

doc = pymupdf.open(pdf_path)

text = ""

for page in doc:
    text += page.get_text() + "\n"


# =========================================================
# 2. CLEAN TEXT
# =========================================================

text = re.sub(r"\n+", "\n", text)
text = re.sub(r"[ \t]+", " ", text)
text = text.strip()


# =========================================================
# 3. CREATE CHUNKS
# =========================================================

chunk_size = 500
overlap = 50

chunks = []

start = 0

while start < len(text):

    end = start + chunk_size

    chunks.append(text[start:end])

    start = end - overlap


print("Number of chunks:", len(chunks))


# =========================================================
# 4. CREATE EMBEDDINGS
# =========================================================

embedding_model = SentenceTransformer(
    "Snowflake/snowflake-arctic-embed-xs"
)

embeddings = embedding_model.encode(
    chunks,
    normalize_embeddings=True
)

print("Embedding shape:", embeddings.shape)


# =========================================================
# 5. LOAD RERANKER
# =========================================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# =========================================================
# 6. TEST QUERIES
# =========================================================

queries = [
    "How does extraction work?",
    "Why does Cobb struggle with Mal?",
    "Why is the spinning top important?"
]


# =========================================================
# 7. EXPERIMENT SETTINGS
# =========================================================

initial_k = 10
final_k = 5

k_values = [1, 3, 5, 10]


# =========================================================
# 8. RUN RETRIEVAL + RERANKING
# =========================================================

all_results = {}


for query in queries:

    print("\n" + "=" * 70)
    print("QUERY:", query)
    print("=" * 70)


    # -----------------------------------------------------
    # STEP 1 — DENSE RETRIEVAL
    # -----------------------------------------------------

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    similarities = np.dot(
        embeddings,
        query_embedding
    )

    ranked_indices = np.argsort(similarities)[::-1]

    before_ranking = ranked_indices[:initial_k]


    print("\n--- BEFORE RERANKING ---")

    for rank, index in enumerate(
        before_ranking,
        start=1
    ):

        print(
            f"Rank {rank} | "
            f"Chunk {index} | "
            f"Score: {similarities[index]:.4f}"
        )


    # -----------------------------------------------------
    # STEP 2 — CROSS-ENCODER RERANKING
    # -----------------------------------------------------

    pairs = [
        (query, chunks[index])
        for index in before_ranking
    ]

    reranker_scores = reranker.predict(pairs)

    reranked_results = sorted(
        zip(before_ranking, reranker_scores),
        key=lambda x: x[1],
        reverse=True
    )

    after_ranking = [
        index
        for index, score in reranked_results
    ]


    print("\n--- AFTER RERANKING ---")

    for rank, (index, score) in enumerate(
        reranked_results[:final_k],
        start=1
    ):

        print(
            f"Rank {rank} | "
            f"Chunk {index} | "
            f"Score: {score:.4f}"
        )


    # -----------------------------------------------------
    # STEP 3 — RANK CHANGES
    # -----------------------------------------------------

    original_ranks = {
        index: rank
        for rank, index in enumerate(
            before_ranking,
            start=1
        )
    }

    rank_changes = {}

    for new_rank, index in enumerate(
        after_ranking,
        start=1
    ):

        old_rank = original_ranks[index]

        rank_changes[index] = {
            "before": old_rank,
            "after": new_rank
        }


    print("\n--- RANK CHANGES ---")

    for index, ranks in rank_changes.items():

        change = (
            ranks["before"]
            - ranks["after"]
        )

        print(
            f"Chunk {index}: "
            f"{ranks['before']} → "
            f"{ranks['after']} "
            f"({change:+d})"
        )


    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    all_results[query] = {
        "before": before_ranking,
        "after": np.array(after_ranking),
        "rank_changes": rank_changes
    }


# =========================================================
# 9. MANUAL RELEVANCE EVALUATION
# =========================================================

relevance_results = {}


for query in queries:

    print("\n" + "=" * 70)
    print("RELEVANCE EVALUATION")
    print("QUERY:", query)
    print("=" * 70)

    before_ranking = all_results[query]["before"]
    after_ranking = all_results[query]["after"]

    # Evaluate every unique chunk that appears
    # in either ranking.

    all_indices = list(
        dict.fromkeys(
            list(before_ranking)
            + list(after_ranking)
        )
    )

    relevance_labels = {}


    for index in all_indices:

        print("\n" + "-" * 60)
        print(f"CHUNK {index}")
        print("-" * 60)

        print(chunks[index])

        while True:

            label = input(
                "\nRelevant to this query? "
                "(1 = Yes, 0 = No): "
            ).strip()

            if label in ["0", "1"]:

                relevance_labels[index] = int(label)

                break

            print("Please enter 1 or 0.")


    relevance_results[query] = relevance_labels


# =========================================================
# 10. PRECISION@K
# =========================================================

evaluation_results = {}


for query in queries:

    before_ranking = all_results[query]["before"]
    after_ranking = all_results[query]["after"]

    relevance_labels = relevance_results[query]

    before_scores = {}
    after_scores = {}


    for k in k_values:

        before_top_k = before_ranking[:k]
        after_top_k = after_ranking[:k]


        before_relevant = sum(
            relevance_labels[index]
            for index in before_top_k
        )

        after_relevant = sum(
            relevance_labels[index]
            for index in after_top_k
        )


        before_scores[k] = (
            before_relevant / k
        )

        after_scores[k] = (
            after_relevant / k
        )


    evaluation_results[query] = {
        "before": before_scores,
        "after": after_scores
    }


# =========================================================
# 11. FINAL RESULTS
# =========================================================

print("\n" + "=" * 70)
print("FINAL RERANKING EVALUATION")
print("=" * 70)


for query, results in evaluation_results.items():

    print("\nQUERY:", query)

    for k in k_values:

        before = results["before"][k]
        after = results["after"][k]

        change = after - before

        print(
            f"Precision@{k}: "
            f"{before:.2f} → "
            f"{after:.2f} "
            f"({change:+.2f})"
        )


# =========================================================
# 12. VISUALIZE RANK CHANGES
# =========================================================

for query, data in all_results.items():

    before = data["before"]
    after = data["after"]

    chunk_ids = list(before)

    before_ranks = {
        index: rank
        for rank, index in enumerate(
            before,
            start=1
        )
    }

    after_ranks = {
        index: rank
        for rank, index in enumerate(
            after,
            start=1
        )
    }


    x = np.arange(len(chunk_ids))


    plt.figure(figsize=(12, 6))


    plt.plot(
        x,
        [
            before_ranks[index]
            for index in chunk_ids
        ],
        marker="o",
        label="Before Reranking"
    )


    plt.plot(
        x,
        [
            after_ranks[index]
            for index in chunk_ids
        ],
        marker="o",
        label="After Reranking"
    )


    plt.xticks(
        x,
        chunk_ids,
        rotation=45
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Chunk ID")
    plt.ylabel("Retrieval Rank")

    plt.title(
        f"Before vs After Reranking\n{query}"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()