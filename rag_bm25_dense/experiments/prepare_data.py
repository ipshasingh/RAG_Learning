import json
from pathlib import Path
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
output_dir = PROJECT_ROOT / "data" / "processed"

dataset = load_dataset("squad")

validation = dataset["validation"]

# Create unique documents
document_to_id = {}
documents = []

for row in validation:
    context = row["context"]

    if context not in document_to_id:
        doc_id = f"doc_{len(documents)}"
        document_to_id[context] = doc_id
        documents.append({
            "id": doc_id,
            "text": context
        })

print("Number of unique documents:", len(documents))

# Create queries and relevance mapping
queries = []

for row in validation:
    context = row["context"]

    queries.append({
        "id": row["id"],
        "text": row["question"],
        "relevant_doc_id": document_to_id[context]
    })

print("Number of queries:", len(queries))

print("\nExample document:")
print(documents[0])

print("\nExample query:")
print(queries[0])



output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "documents.json", "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

with open(output_dir / "queries.json", "w", encoding="utf-8") as f:
    json.dump(queries, f, ensure_ascii=False, indent=2)

print("Saved documents to:", output_dir / "documents.json")
print("Saved queries to:", output_dir / "queries.json")