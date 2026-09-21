import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load experiment results
# --------------------------------------------------

results_path = RESULTS_DIR / "retrieval_results.json"

with open(results_path, "r", encoding="utf-8") as f:
    results = json.load(f)


# --------------------------------------------------
# Create retrieval metrics table
# --------------------------------------------------

metrics = [
    "Recall@1",
    "Recall@5",
    "Recall@10",
    "MRR"
]

rows = []

for metric in metrics:

    bm25_value = results["BM25"][metric]
    dense_value = results["Dense"][metric]

    rows.append({
        "Metric": metric,
        "BM25": bm25_value,
        "Dense": dense_value,
        "Difference": dense_value - bm25_value
    })


metrics_df = pd.DataFrame(rows)


# Save table
metrics_path = TABLES_DIR / "retrieval_metrics.csv"

metrics_df.to_csv(
    metrics_path,
    index=False
)


print(f"Saved: {metrics_path}")


# --------------------------------------------------
# Plot retrieval comparison
# --------------------------------------------------

x = range(len(metrics))

width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    metrics_df["BM25"],
    width=width,
    label="BM25"
)

plt.bar(
    [i + width / 2 for i in x],
    metrics_df["Dense"],
    width=width,
    label="Dense"
)

plt.xticks(
    list(x),
    metrics
)

plt.ylabel("Score")

plt.title(
    "BM25 vs Dense Retrieval"
)

plt.ylim(0, 1)

plt.legend()

plt.tight_layout()


figure_path = FIGURES_DIR / "retrieval_comparison.png"

plt.savefig(
    figure_path,
    dpi=300
)

plt.close()


print(f"Saved: {figure_path}")


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n================ RETRIEVAL RESULTS ================\n")

print(metrics_df.to_string(index=False))

print("\nStep 5A complete.")