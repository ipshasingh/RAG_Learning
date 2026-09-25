import pymupdf  # Imports PyMuPDF for reading and extracting text from the PDF.
import re  # Provides regular-expression tools for cleaning the extracted text.
import numpy as np  # Provides numerical operations and similarity calculations.
import matplotlib.pyplot as plt  # Used to visualize retrieval scores.
from sentence_transformers import SentenceTransformer  # Loads the embedding model.


# =========================================================
# 1. LOAD THE DOCUMENT
# =========================================================

pdf_path = "rag_search/data/inception.pdf"  # Stores the path to the Inception screenplay.

doc = pymupdf.open(pdf_path)  # Opens the PDF so we can access its pages.

text = ""  # Creates an empty string to store the complete screenplay.

for page in doc:
    # Iterates through every page in the PDF.

    text += page.get_text() + "\n"
    # Extracts the text from the current page and adds a newline.


# =========================================================
# 2. CLEAN THE EXTRACTED TEXT
# =========================================================

text = re.sub(r"\n+", "\n", text)
# Replaces multiple consecutive newlines with a single newline.

text = re.sub(r"[ \t]+", " ", text)
# Replaces repeated spaces and tabs with a single space.

text = text.strip()
# Removes unnecessary whitespace from the beginning and end.


# =========================================================
# 3. SPLIT THE DOCUMENT INTO CHUNKS
# =========================================================

chunk_size = 500
# Defines the maximum number of characters in each chunk.

overlap = 50
# Keeps 50 characters from the previous chunk
# so that some context is preserved between chunks.

chunks = []
# Creates a list to store all document chunks.

start = 0
# Marks the starting character position of the first chunk.

while start < len(text):
    # Continues creating chunks until the entire document is processed.

    end = start + chunk_size
    # Calculates where the current chunk should end.

    chunk = text[start:end]
    # Extracts the text between the start and end positions.

    chunks.append(chunk)
    # Adds the current chunk to the list.

    start = end - overlap
    # Moves the starting position forward while keeping
    # the specified overlap with the previous chunk.


# =========================================================
# 4. CREATE DOCUMENT EMBEDDINGS
# =========================================================

model = SentenceTransformer("Snowflake/snowflake-arctic-embed-xs")
# Loads the embedding model that converts text into numerical vectors.

embeddings = model.encode(
    chunks,
    normalize_embeddings=True
)
# Converts every document chunk into an embedding vector.
#
# normalize_embeddings=True normalizes the vectors so that
# their dot product corresponds to cosine similarity.


# =========================================================
# 5. DEFINE TEST QUERIES
# =========================================================

queries = [
    "How does extraction work?",
    "Why does Cobb struggle with Mal?",
    "How are dream levels created?",
    "What is a kick?",
    "Why is the spinning top important?"
]
# Defines the questions we will use to test semantic retrieval.


top_k_values = [1, 3, 5, 10]
# Defines the different numbers of chunks we want to retrieve.
# This lets us compare retrieval at different depths.


# =========================================================
# 6. RUN THE RETRIEVAL EXPERIMENT
# =========================================================

results = {}
# Stores the Top-10 similarity scores for every query.

retrieved_chunks = set()
# Stores every unique chunk that appears in the Top-10 results.
# A set automatically removes duplicate chunk indices.


for query in queries:
    # Runs the retrieval experiment separately for every query.

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )
    # Converts the query into the same embedding space
    # as the document chunks.

    similarities = np.dot(embeddings, query_embedding)
    # Calculates cosine similarity between the query
    # and every document chunk.
    #
    # embeddings shape:
    # (number_of_chunks, 384)
    #
    # query_embedding shape:
    # (384,)
    #
    # Result:
    # one similarity score for every chunk.

    ranked_indices = np.argsort(similarities)[::-1]
    # Sorts all chunk indices from highest similarity
    # to lowest similarity.


    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)


    # -----------------------------------------------------
    # Compare different retrieval depths
    # -----------------------------------------------------

    for top_k in top_k_values:
        # Tests one retrieval depth at a time.

        top_indices = ranked_indices[:top_k]
        # Selects the top-k most similar chunks.

        print(f"\n--- TOP {top_k} ---")

        for rank, index in enumerate(top_indices, start=1):
            # Goes through the retrieved chunks
            # and assigns their retrieval rank.

            print(
                f"Rank {rank} | "
                f"Chunk {index} | "
                f"Score: {similarities[index]:.4f}"
            )
            # Prints the rank, chunk number, and similarity score.


    # -----------------------------------------------------
    # Store Top-10 results for visualization
    # -----------------------------------------------------

    top_indices = ranked_indices[:10]
    # Selects the ten highest-scoring chunks.

    scores = similarities[top_indices]
    # Extracts the similarity scores of those ten chunks.

    results[query] = scores
    # Stores the scores using the query as the dictionary key.


    for index in top_indices:
        # Goes through every Top-10 retrieved chunk.

        retrieved_chunks.add(index)
        # Adds the chunk index to the set of unique chunks.


# =========================================================
# 7. VISUALIZE RETRIEVAL SCORES
# =========================================================

plt.figure(figsize=(10, 6))
# Creates the figure for the retrieval-score visualization.


for query, scores in results.items():
    # Goes through every query and its Top-10 scores.

    ranks = range(1, len(scores) + 1)
    # Creates rank numbers from 1 to 10.

    plt.plot(
        ranks,
        scores,
        marker="o",
        label=query
    )
    # Plots similarity score against retrieval rank.


plt.xlabel("Retrieval Rank")
# Labels the x-axis.

plt.ylabel("Cosine Similarity")
# Labels the y-axis.

plt.title("Similarity Score Across Retrieved Chunks")
# Gives the graph a descriptive title.

plt.xticks(range(1, 11))
# Displays every retrieval rank from 1 to 10.

plt.legend()
# Shows which line corresponds to each query.

plt.grid(True, alpha=0.3)
# Adds a light grid to make the graph easier to read.

plt.tight_layout()
# Adjusts spacing so labels do not get cut off.

plt.show()
# Displays the graph.


# =========================================================
# 8. INSPECT UNIQUE RETRIEVED CHUNKS
# =========================================================

print("\n" + "=" * 70)
print("UNIQUE RETRIEVED CHUNKS")
print("=" * 70)


for index in sorted(retrieved_chunks):
    # Goes through every unique chunk that appeared
    # in a Top-10 result.

    print(f"\n--- CHUNK {index} ---")
    # Prints the chunk number.

    print(chunks[index])
    # Prints the actual text of the chunk so we can
    # manually judge whether the retrieved result is relevant.