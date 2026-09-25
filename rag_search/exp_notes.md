# RAG Experiment: How Many Chunks Should We Retrieve?

## 1. Experiment Overview

### The question

When a user asks a question to a RAG system, we usually don't retrieve the entire knowledge base.

Instead, we retrieve the **Top-K most relevant chunks** and pass those chunks to the language model.

But what should `K` be?

Should we retrieve:

- 1 chunk?
- 3 chunks?
- 5 chunks?
- 10 chunks?

Retrieving too few chunks might cause the system to miss useful information.

Retrieving too many might introduce irrelevant information and noise.

### Our experiment

We use the *Inception* screenplay as our knowledge base and test semantic retrieval with:

```text
K = 1, 3, 5, 10
```

We ask five different questions and observe:

1. Which chunks are retrieved?
2. What similarity scores do they receive?
3. How does similarity change as retrieval rank increases?
4. Are the additional retrieved chunks actually useful?

---

# 2. Where This Fits in Our RAG Series

This is part of:

### 15 RAG Investigations

**01–04 — Understand**

1. What is RAG actually doing?
2. RAG-Sequence vs RAG-Token
3. How does retrieval work?
4. Lexical vs Semantic Retrieval

**05–08 — Build**

5. BM25 vs Dense Retrieval
6. What is RAG actually retrieving? → Chunking
7. Embeddings + semantic retrieval
8. Build semantic search from scratch

**09–11 — Improve**

9. **How many documents should RAG retrieve? ← THIS EXPERIMENT**
10. Reranking
11. Hybrid Retrieval + experiment

**12–13 — Evaluate & Fail**

12. How do we know retrieval is good?
13. When retrieval fails

**14–15 — Research**

14. Reproduce a research result
15. My own RAG experiment / extension

So this experiment is about **retrieval depth**.

---

# 3. The RAG Pipeline We Have Built So Far

Our current system looks like:

```text
Inception screenplay
        ↓
Extract text from PDF
        ↓
Clean text
        ↓
Split into chunks
        ↓
Create embeddings
        ↓
Store embeddings
        ↓
User asks a question
        ↓
Embed the question
        ↓
Compare query embedding
with document embeddings
        ↓
Rank chunks by similarity
        ↓
Retrieve Top-K chunks
```

We are currently experimenting with the **last part**:

```text
Ranked chunks
      ↓
How many should we keep?
      ↓
Top-1 / Top-3 / Top-5 / Top-10
```

---

# 4. Concept: Knowledge Base

A RAG system needs some external source of information.

For this experiment, our knowledge base is:

> The *Inception* screenplay.

The PDF contains approximately:

```text
28,995 words
```

After extracting and cleaning the text, we divide it into smaller pieces.

These pieces are called **chunks**.

---

# 5. Concept: Chunking

A language model or retrieval system does not necessarily work best when we treat an entire document as one giant piece of text.

Instead, we divide the document into smaller sections.

We use:

```python
chunk_size = 500
overlap = 50
```

This means each chunk contains approximately 500 characters.

The next chunk begins 50 characters before the previous chunk ends.

For example:

```text
Chunk 1:
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                    ↑
                 overlap
                         BBBBBBBBBBBBBBBBBBBBBBBBBBB
                         Chunk 2
```

More conceptually:

```text
Chunk 1
[----------------------500 chars----------------------]

                         Chunk 2
                         [----------------------500 chars----------------------]
                         ↑
                       50 chars
                       overlap
```

### Why overlap?

Imagine an important sentence is split exactly at the boundary between two chunks.

Without overlap:

```text
Chunk 1:
"The dream can become unstable when the subject..."

Chunk 2:
"...realizes that they are dreaming."
```

The complete meaning is split between the chunks.

With overlap, both chunks retain some surrounding context.

This can improve retrieval because relevant information is less likely to be separated completely.

---

# 6. Loading the PDF

We use PyMuPDF:

```python
import pymupdf
```

Then:

```python
doc = pymupdf.open(pdf_path)
```

This opens the screenplay.

We iterate through every page:

```python
for page in doc:
    text += page.get_text() + "\n"
```

The logic is:

```text
PDF
 ↓
Page 1 → extract text
Page 2 → extract text
Page 3 → extract text
...
Page N → extract text
 ↓
One large text string
```

---

# 7. Cleaning the Text

PDF extraction often produces unnecessary whitespace.

We use:

```python
text = re.sub(r"\n+", "\n", text)
```

This reduces repeated newline characters.

Then:

```python
text = re.sub(r"[ \t]+", " ", text)
```

This replaces repeated spaces and tabs with a single space.

Finally:

```python
text = text.strip()
```

This removes unnecessary whitespace at the beginning and end.

The goal is simply:

```text
messy extracted PDF text
             ↓
       cleaner text
             ↓
       chunking
```

---

# 8. Creating the Chunks

We start at character position 0:

```python
start = 0
```

Then:

```python
while start < len(text):
```

continues until we reach the end of the screenplay.

We calculate the end:

```python
end = start + chunk_size
```

Then extract:

```python
chunk = text[start:end]
```

and store it:

```python
chunks.append(chunk)
```

The important line is:

```python
start = end - overlap
```

Suppose:

```text
chunk_size = 500
overlap = 50
```

Then:

```text
Chunk 1:
0 → 500

Chunk 2:
450 → 950

Chunk 3:
900 → 1400
```

So each consecutive chunk shares 50 characters with the previous one.

---

# 9. Concept: Embeddings

Now we have chunks of text.

But computers need a numerical representation to compare their meanings.

This is where **embeddings** come in.

We use:

```python
from sentence_transformers import SentenceTransformer
```

and load:

```python
model = SentenceTransformer(
    "Snowflake/snowflake-arctic-embed-xs"
)
```

The embedding model converts text into vectors.

For example, conceptually:

```text
"How does extraction work?"
             ↓
       Embedding model
             ↓
[0.12, -0.04, 0.81, ...]
```

Our model produces vectors with:

```text
384 dimensions
```

So each chunk becomes a vector:

```text
Chunk 1 → [384 numbers]
Chunk 2 → [384 numbers]
Chunk 3 → [384 numbers]
...
```

---

# 10. Document Embeddings

We create embeddings for every chunk:

```python
embeddings = model.encode(
    chunks,
    normalize_embeddings=True
)
```

If we have 360 chunks, conceptually we get:

```text
360 chunks × 384 dimensions
```

Therefore:

```python
embeddings.shape
```

is approximately:

```text
(360, 384)
```

The first dimension represents:

> number of chunks

The second represents:

> dimensions in each embedding vector.

---

# 11. Why Normalize the Embeddings?

We use:

```python
normalize_embeddings=True
```

This normalizes the vectors.

That lets us calculate cosine similarity conveniently using a dot product.

Instead of thinking about the raw text:

```text
Chunk A
Chunk B
Query
```

we can compare their numerical representations:

```text
Vector A
Vector B
Vector Query
```

---

# 12. Concept: Semantic Search

Our retrieval system is **semantic**, rather than simply keyword-based.

Suppose our query is:

> "How does extraction work?"

A keyword search might primarily look for the exact word:

```text
extraction
```

Semantic retrieval instead represents the query as an embedding and searches for chunks whose **meaning** is similar.

So a chunk discussing:

> stealing information from someone's subconscious while they are dreaming

can be relevant even if it doesn't contain exactly the same wording as the query.

This is one of the important ideas behind modern RAG systems.

---

# 13. Defining Our Queries

We use five questions:

```python
queries = [
    "How does extraction work?",
    "Why does Cobb struggle with Mal?",
    "How are dream levels created?",
    "What is a kick?",
    "Why is the spinning top important?"
]
```

These intentionally cover different parts of the screenplay.

We don't want our experiment to depend on one lucky query.

We want to see whether retrieval behaves consistently across different questions.

---

# 14. Concept: Query Embedding

The query also needs to be converted into an embedding.

We do:

```python
query_embedding = model.encode(
    query,
    normalize_embeddings=True
)
```

Now both sides are represented in the same vector space:

```text
Document chunks
      ↓
[384-dimensional vectors]


Query
      ↓
[384-dimensional vector]
```

This makes numerical comparison possible.

---

# 15. Concept: Cosine Similarity

Now we need to determine:

> Which chunks are most similar to the question?

We calculate:

```python
similarities = np.dot(
    embeddings,
    query_embedding
)
```

Because the embeddings are normalized, the dot product corresponds to cosine similarity.

Conceptually:

```text
Query
  ↓
[vector]
  ↓
compare against
  ↓
Chunk 1 vector
Chunk 2 vector
Chunk 3 vector
...
```

The result is one similarity score per chunk.

For example:

```text
Chunk 42 → 0.91
Chunk 17 → 0.87
Chunk 93 → 0.72
Chunk 201 → 0.41
```

Higher score means the embedding model considers the chunk more semantically similar to the query.

---

# 16. Important: Similarity Is Not the Same as Relevance

This is one of the most important lessons from this experiment.

A high similarity score does **not automatically mean**:

> "This chunk contains the answer."

It means:

> "The embedding model considers this chunk semantically similar to the query."

Those are not exactly the same thing.

For example, a query about Cobb and Mal might retrieve several chunks involving:

- Cobb
- Mal
- dreams
- reality
- guilt

Some may directly answer the question.

Others may merely discuss related concepts.

Therefore, we need to inspect the retrieved text.

---

# 17. Ranking the Chunks

We rank all chunks:

```python
ranked_indices = np.argsort(similarities)[::-1]
```

Breaking this down:

### `np.argsort()`

Sorts the indices according to similarity.

By default:

```text
lowest → highest
```

### `[::-1]`

Reverses the result:

```text
highest → lowest
```

So we end up with:

```text
Rank 1 → highest similarity
Rank 2 → second highest
Rank 3 → third highest
...
```

---

# 18. The Main Experiment: Top-K

This is the core of today's investigation.

We define:

```python
top_k_values = [1, 3, 5, 10]
```

We are asking:

> What happens when we increase the number of retrieved chunks?

### Top-1

Retrieve only:

```text
1 chunk
```

### Top-3

Retrieve:

```text
3 chunks
```

### Top-5

Retrieve:

```text
5 chunks
```

### Top-10

Retrieve:

```text
10 chunks
```

---

# 19. Why Does K Matter?

Imagine the ideal answer requires information contained in three different chunks.

If we retrieve only:

```text
Top-1
```

we might miss important information.

Increasing K could help:

```text
Top-3
```

But eventually we may start retrieving chunks that aren't particularly useful.

So there is a potential trade-off:

```text
Too little retrieval
        ↓
Miss useful information

        vs.

Too much retrieval
        ↓
Introduce irrelevant information
```

The experiment is trying to investigate this trade-off.

---

# 20. Collecting the Top-10 Scores

For every query, we retrieve the top 10:

```python
top_indices = ranked_indices[:10]
```

Then:

```python
scores = similarities[top_indices]
```

extracts their similarity scores.

We store them:

```python
results[query] = scores
```

This gives us data we can visualize.

Conceptually:

```text
Query 1 → [score1, score2, ..., score10]

Query 2 → [score1, score2, ..., score10]

Query 3 → [score1, score2, ..., score10]
...
```

---

# 21. Visualizing the Results

We plot:

```python
plt.plot(
    ranks,
    scores,
    marker="o",
    label=query
)
```

The graph has:

### X-axis

```text
Retrieval Rank
```

Meaning:

```text
1 → most similar chunk
2 → second most similar
...
10 → tenth most similar
```

### Y-axis

```text
Cosine Similarity
```

Meaning:

> How similar the retrieved chunk's embedding is to the query embedding.

---

# 22. How to Read the Graph

Suppose the graph looks roughly like:

```text
Similarity
   |
0.85| ●
    |   ●
0.84|      ●
    |         ●
0.83|            ●
    |                ●
0.82|                    ●
    |
    +-------------------------
      1  2  3  4  5 ... 10
             Rank
```

The general pattern would tell us:

> As we move down the ranking, similarity decreases.

This is expected because the system is ranking chunks from most similar to least similar.

But the **shape of the decline** is interesting.

If scores drop sharply:

```text
0.90
0.88
0.72
0.60
0.51
...
```

then the first few chunks are substantially more similar than the later ones.

If they remain close:

```text
0.85
0.84
0.84
0.83
0.83
...
```

then several chunks are similarly close to the query.

---

# 23. What Our Current Results Show

For example, our query:

> "Why does Cobb struggle with Mal?"

produced:

```text
Rank 1 → 0.8500
Rank 2 → 0.8472
Rank 3 → 0.8457
Rank 4 → 0.8423
Rank 5 → 0.8408
```

Notice how close those numbers are.

This is interesting because the retriever isn't saying:

```text
"This chunk is relevant."

"This one definitely isn't."
```

Instead, several chunks receive relatively similar scores.

That means we cannot determine the optimal K from similarity scores alone.

---

# 24. The Problem With Only Looking at the Graph

This is a crucial experimental distinction.

The graph tells us about:

> **retrieval similarity**

It does NOT directly tell us about:

> **retrieval usefulness**

For example:

```text
Rank 1 → 0.850 → Relevant
Rank 2 → 0.847 → Relevant
Rank 3 → 0.846 → Relevant
Rank 4 → 0.842 → Irrelevant
Rank 5 → 0.841 → Irrelevant
```

The scores barely changed.

But the usefulness did.

Therefore:

```text
Similarity score
        ≠
Actual relevance
```

This is why our next stage is **manual relevance evaluation**.

---

# 25. Unique Retrieved Chunks

We also maintain:

```python
retrieved_chunks = set()
```

Why a set?

Because the same chunk can be retrieved for multiple queries.

For example:

```text
Query 1 → chunks 6, 64, 100
Query 2 → chunks 64, 100, 326
```

We don't want to print chunk 64 and 100 repeatedly.

A set automatically removes duplicates.

So:

```python
retrieved_chunks.add(index)
```

collects every unique chunk that appeared in our Top-10 retrievals.

---

# 26. Inspecting the Actual Chunks

Finally:

```python
for index in sorted(retrieved_chunks):
    print(f"\n--- CHUNK {index} ---")
    print(chunks[index])
```

prints the actual retrieved text.

This is important because now we can move from:

```text
numerical retrieval
```

to:

```text
human evaluation
```

We can actually read what the system retrieved.

---

# 27. What We Have Actually Built

At this point, we have essentially built a small **semantic retrieval engine**.

It can:

```text
Take a document
      ↓
Split it into chunks
      ↓
Embed the chunks
      ↓
Take a natural-language query
      ↓
Embed the query
      ↓
Calculate semantic similarity
      ↓
Rank the chunks
      ↓
Return Top-K results
```

This is the retrieval component of RAG.

We haven't yet added the LLM generation stage.

So currently we have:

```text
              RAG
               |
       ┌───────┴───────┐
       ↓               ↓
  Retrieval         Generation
       ↑               ↑
       |               |
    WE ARE           Later
    HERE
```

---

# 28. The Experimental Hypothesis

The experiment is ultimately asking:

> **Does increasing K consistently improve the useful information retrieved?**

We can think about three possible outcomes.

### Outcome A — More is better

```text
Top-1  → insufficient
Top-3  → better
Top-5  → better
Top-10 → best
```

This would suggest retrieving more chunks helps for our dataset and queries.

### Outcome B — There is a useful middle ground

```text
Top-1  → insufficient
Top-3  → good
Top-5  → good
Top-10 → noisy
```

This would suggest there may be a practical retrieval sweet spot.

### Outcome C — More retrieval adds mostly noise

```text
Top-1  → useful
Top-3  → still useful
Top-5  → mixed
Top-10 → lots of irrelevant material
```

This would demonstrate why blindly increasing K isn't necessarily beneficial.

We shouldn't assume which outcome we'll get.

**The experiment should determine that.**

---

# 29. What We Should Measure Next

The similarity graph is only the first visualization.

Now we need to manually evaluate the retrieved chunks.

For each query, we can create something like:

| Query | K | Relevant chunks | Total chunks | Precision |
|---|---:|---:|---:|---:|
| Extraction | 1 | ? | 1 | ? |
| Extraction | 3 | ? | 3 | ? |
| Extraction | 5 | ? | 5 | ? |
| Extraction | 10 | ? | 10 | ? |

Where:

```text
Precision@K =
relevant retrieved chunks / K
```

For example, if Top-5 contains:

```text
3 relevant
2 irrelevant
```

then:

```text
Precision@5 = 3 / 5 = 0.60
```

This gives us something much more meaningful than similarity scores alone.

---

# 30. The Bigger RAG Lesson

The important lesson isn't simply:

> "Use K = 5."

Our dataset is tiny and our evaluation set is tiny, so we shouldn't generalize that way.

The deeper lesson is:

> **Retrieval is not just about finding similar text. It is about finding enough useful evidence without overwhelming the generation step with irrelevant context.**

That is one of the central engineering decisions in a RAG system.

---

# 31. Current Experiment Status

### Completed

- [x] Load screenplay
- [x] Extract text
- [x] Clean text
- [x] Chunk document
- [x] Add chunk overlap
- [x] Generate embeddings
- [x] Embed queries
- [x] Calculate cosine similarity
- [x] Rank chunks
- [x] Test K = 1, 3, 5, 10
- [x] Visualize similarity scores
- [x] Collect unique retrieved chunks
- [x] Inspect actual retrieved text

### Next

- [ ] Label retrieved chunks as relevant / irrelevant
- [ ] Calculate Precision@1
- [ ] Calculate Precision@3
- [ ] Calculate Precision@5
- [ ] Calculate Precision@10
- [ ] Compare retrieval depth
- [ ] Visualize relevance against K
- [ ] Interpret the result
- [ ] Turn the experiment into RAG Post #9

---

# 32. Key Concepts to Remember

### Chunking

Breaking a large document into smaller pieces for retrieval.

### Chunk overlap

Keeping some text between consecutive chunks to preserve context.

### Embedding

A numerical vector representation of text that captures semantic information.

### Semantic retrieval

Retrieving text based on meaning rather than only exact keyword matches.

### Cosine similarity

A measure of how similar two vectors are in direction.

### Ranking

Ordering document chunks from highest to lowest similarity to the query.

### Top-K retrieval

Keeping only the K highest-ranked chunks.

### Retrieval depth

How many chunks we choose to retrieve.

### Similarity ≠ relevance

A high embedding similarity score does not guarantee that a chunk actually answers the question.

### Precision@K

The proportion of the Top-K retrieved chunks that are actually relevant.

---

# 33. Mental Model

The entire experiment can be remembered as:

```text
DOCUMENT
   ↓
CHUNK
   ↓
EMBED
   ↓
QUERY
   ↓
EMBED
   ↓
COMPARE
   ↓
RANK
   ↓
TOP-K
   ↓
ARE THESE CHUNKS ACTUALLY USEFUL?
   ↓
EVALUATE
   ↓
LEARN
```

And that last step is the important transition in this RAG series:

> **We're moving from "I implemented retrieval" to "I can experimentally evaluate retrieval."**