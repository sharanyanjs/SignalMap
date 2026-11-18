
import pandas as pd
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

results = []

for method in ["kmeans", "agglomerative"]:

    path = f"outputs/clustered_reviews_{method}.csv"
    df = pd.read_csv(path)

    texts = df["text_clean"].fillna("").astype(str).tolist()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    counts = df["cluster_id"].value_counts()

    score = silhouette_score(
        embeddings,
        df["cluster_id"],
        metric="cosine",
        sample_size=min(1000, len(df)),
        random_state=42,
    )

    results.append({
        "method": method,
        "silhouette": round(score, 4),
        "clusters": len(counts),
        "largest_cluster_pct": round(
            counts.max() / len(df) * 100, 1
        ),
        "small_clusters_lt10": int((counts < 10).sum()),
    })

comparison = pd.DataFrame(results)

print("\nCLUSTERING MODEL COMPARISON")
print(comparison.to_string(index=False))

comparison.to_csv(
    "outputs/model_comparison.csv",
    index=False,
)

print("\nSaved: outputs/model_comparison.csv")
