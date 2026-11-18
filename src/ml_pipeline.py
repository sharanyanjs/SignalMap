
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS,
)
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DOMAIN_STOP_WORDS = [
    "app",
    "application",
    "apps",
    "applications",
    "using",
    "used",
    "use",
    "really",
    "just",
    "got",
    "get",
    "one",
    "thing",
    "things",
]


def _embed_with_transformer(
    texts: list[str],
    batch_size: int = 64,
) -> np.ndarray:

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings


def _embed_with_tfidf(texts: list[str]):

    vectorizer = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(texts)

    return normalize(matrix)


def create_embeddings(
    texts: list[str],
    use_transformer: bool = True,
):

    if not texts:
        raise ValueError("No review texts were provided.")

    if use_transformer:
        try:
            embeddings = _embed_with_transformer(texts)

            return (
                embeddings,
                f"Sentence Transformers: {MODEL_NAME}",
            )

        except Exception as exc:
            print(
                "Transformer model unavailable. "
                f"Using TF-IDF fallback instead. Details: {exc}"
            )

    embeddings = _embed_with_tfidf(texts)

    return (
        embeddings,
        "TF-IDF fallback (not a transformer model)",
    )


def cluster_embeddings(
    embeddings,
    n_clusters: int = 10,
    random_state: int = 42,
    silhouette_sample_size: int = 1000,
    method: str = "kmeans",
):

    n_samples = embeddings.shape[0]

    if n_samples < 3:
        raise ValueError(
            "At least 3 reviews are required for clustering."
        )

    if n_clusters < 2:
        raise ValueError(
            "n_clusters must be at least 2."
        )

    method = method.lower().strip()

    if method not in {"kmeans", "agglomerative"}:
        raise ValueError(
            "method must be 'kmeans' or 'agglomerative'."
        )

    k = min(int(n_clusters), n_samples - 1)

    if method == "kmeans":

        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
        )

    else:

        model = AgglomerativeClustering(
            n_clusters=k,
            metric="cosine",
            linkage="average",
        )

    labels = model.fit_predict(embeddings)

    score = None

    if len(set(labels)) > 1:

        sample_size = min(
            silhouette_sample_size,
            n_samples,
        )

        try:
            score = float(
                silhouette_score(
                    embeddings,
                    labels,
                    metric="cosine",
                    sample_size=sample_size,
                    random_state=random_state,
                )
            )

        except ValueError as exc:
            print(
                "Silhouette evaluation could not be completed: "
                f"{exc}"
            )

    return labels, score, model


def label_clusters(
    texts: list[str],
    labels: np.ndarray,
    top_terms: int = 6,
) -> dict[int, str]:

    if len(texts) != len(labels):
        raise ValueError(
            "The number of texts must match the number of labels."
        )

    frame = pd.DataFrame({
        "text": texts,
        "cluster": labels,
    })

    custom_stop_words = list(
        ENGLISH_STOP_WORDS.union(DOMAIN_STOP_WORDS)
    )

    vectorizer = TfidfVectorizer(
        stop_words=custom_stop_words,
        ngram_range=(1, 2),
        max_features=15000,
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
    )

    try:
        matrix = vectorizer.fit_transform(frame["text"])

    except ValueError:
        return {
            int(cluster_id): f"Theme {int(cluster_id) + 1}"
            for cluster_id in sorted(frame["cluster"].unique())
        }

    terms = vectorizer.get_feature_names_out()

    overall_mean = np.asarray(
        matrix.mean(axis=0)
    ).ravel()

    names = {}

    for cluster_id, indices in frame.groupby("cluster").groups.items():

        cluster_matrix = matrix[list(indices)]

        cluster_mean = np.asarray(
            cluster_matrix.mean(axis=0)
        ).ravel()

        distinctive_scores = cluster_mean - overall_mean

        best_indices = np.argsort(
            distinctive_scores
        )[::-1]

        selected_terms = []

        for index in best_indices:

            if distinctive_scores[index] <= 0:
                break

            term = terms[index]

            if term not in selected_terms:
                selected_terms.append(term)

            if len(selected_terms) >= top_terms:
                break

        if selected_terms:
            names[int(cluster_id)] = " · ".join(
                selected_terms[:4]
            )

        else:
            names[int(cluster_id)] = (
                f"Theme {int(cluster_id) + 1}"
            )

    return names


def run_theme_discovery(
    df: pd.DataFrame,
    n_clusters: int = 10,
    use_transformer: bool = True,
    clustering_method: str = "kmeans",
):

    if df.empty:
        raise ValueError(
            "Cannot discover themes from an empty dataset."
        )

    if "text_clean" not in df.columns:
        raise ValueError(
            "Input dataset must contain a text_clean column."
        )

    texts = (
        df["text_clean"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    if any(not text.strip() for text in texts):
        raise ValueError(
            "Empty review text found. Clean the dataset first."
        )

    embeddings, model_name = create_embeddings(
        texts,
        use_transformer=use_transformer,
    )

    labels, silhouette, _ = cluster_embeddings(
        embeddings,
        n_clusters=n_clusters,
        method=clustering_method,
    )

    names = label_clusters(
        texts,
        labels,
    )

    result = df.copy()

    result["cluster_id"] = labels

    result["theme"] = result["cluster_id"].map(names)

    metrics = {
        "model": model_name,
        "clustering_method": clustering_method,
        "silhouette": silhouette,
        "clusters": int(len(set(labels))),
        "records_clustered": int(len(result)),
    }

    return result, metrics
