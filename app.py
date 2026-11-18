
from pathlib import Path
import sys
from io import StringIO

import streamlit as st
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_reviews, validate_reviews
from src.preprocessing import clean_reviews
from src.ml_pipeline import run_theme_discovery
from src.prioritization import build_priority_table
from src.analysis import weekly_theme_trends, representative_reviews


st.set_page_config(page_title="SignalMap", layout="wide")
st.title("SignalMap")
st.caption("From customer feedback to evidence-backed product decisions.")
st.info(
    "Research prototype. Priority factors are transparent proxies, "
    "not validated measures of actual business impact."
)

DEFAULT_DATA = ROOT / "data" / "raw" / "health_fitness_app_reviews.csv"
SUBTHEME_REVIEWS = ROOT / "outputs" / "subtheme_reviews.csv"
SUBTHEME_SUMMARY = ROOT / "outputs" / "subtheme_summary.csv"

INVESTIGATION_PROMPTS = {
    "Subscription and billing problems": [
        "Audit trial-to-paid conversion and unexpected charge complaints.",
        "Review cancellation flows, confirmation messages, and refund handling.",
        "Check whether billing expectations are clear before purchase.",
    ],
    "Paywall and free-tier dissatisfaction": [
        "Map which commonly requested features are restricted to premium.",
        "Review whether free-tier limits are communicated clearly.",
        "Compare complaints with free-to-paid conversion and retention data.",
    ],
    "Excessive advertising": [
        "Measure ad frequency and interruption points in key user journeys.",
        "Check whether ads block workouts or other core tasks.",
        "Separate ad complaints from performance and stability complaints.",
    ],
    "Privacy and product experience": [
        "Review data-sharing permissions and privacy communication.",
        "Investigate complaints about social prompts and interface changes.",
        "Manually separate privacy concerns from feature and paywall feedback.",
    ],
}


@st.cache_data(show_spinner=False)
def load_data(path: str):
    return load_reviews(path)


@st.cache_data(show_spinner=False)
def load_subtheme_data(reviews_path: str, summary_path: str):
    reviews = pd.read_csv(reviews_path)
    summary = pd.read_csv(summary_path)
    return reviews, summary


with st.sidebar:
    st.header("Analysis settings")
    uploaded = st.file_uploader("Upload a CSV (optional)", type=["csv"])
    sample_size = st.slider("Reviews to analyse", 500, 20000, 5000, 500)
    clusters = st.slider("Number of themes", 3, 20, 10)
    transformer = st.checkbox(
        "Use Sentence Transformers (recommended)", value=True
    )
    run = st.button(
        "Run SignalMap analysis",
        type="primary",
        use_container_width=True,
    )
    st.caption(
        "The first transformer run downloads the model. If unavailable, "
        "the pipeline falls back to TF-IDF and labels that fallback clearly."
    )


if uploaded is not None:
    try:
        raw = load_reviews(
            StringIO(uploaded.getvalue().decode("utf-8-sig"))
        )
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        st.stop()
elif DEFAULT_DATA.exists():
    raw = load_data(str(DEFAULT_DATA))
else:
    st.warning(
        "Place the dataset at data/raw/health_fitness_app_reviews.csv "
        "or upload it from the sidebar."
    )
    st.stop()


stats = validate_reviews(raw)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews loaded", f"{stats['rows']:,}")
c2.metric("Apps", stats["apps"])
c3.metric("Missing review text", f"{stats['missing_text']:,}")
c4.metric("Duplicate review text", f"{stats['duplicate_review_text']:,}")

st.caption(
    f"Dataset dates: {stats['date_min']} to {stats['date_max']} | "
    f"Full-row duplicates: {stats['duplicate_rows']:,}"
)


if "clustered" not in st.session_state:
    st.session_state.clustered = None
    st.session_state.metrics = None


if run:
    cleaned = clean_reviews(raw)

    if len(cleaned) > sample_size:
        cleaned = cleaned.sample(
            n=sample_size, random_state=42
        ).sort_index()

    with st.spinner(
        f"Creating embeddings and discovering themes "
        f"from {len(cleaned):,} reviews..."
    ):
        clustered, metrics = run_theme_discovery(
            cleaned,
            n_clusters=clusters,
            use_transformer=transformer,
        )
        st.session_state.clustered = clustered
        st.session_state.metrics = metrics


clustered = st.session_state.clustered

if clustered is None:
    st.subheader("Start with the data, then discover themes")
    st.write(
        "1. Confirm the dataset summary above.\n"
        "2. Choose a sample size.\n"
        "3. Click **Run SignalMap analysis**."
    )
    st.stop()


metrics = st.session_state.metrics

st.success(
    f"Analysis complete | {metrics['records_clustered']:,} reviews | "
    f"{metrics['clusters']} themes"
)
st.caption(
    f"Embedding method: {metrics['model']} | "
    f"Silhouette score: "
    f"{metrics['silhouette'] if metrics['silhouette'] is not None else 'Not calculated'}"
)

priority = build_priority_table(clustered)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Overview",
        "Theme explorer",
        "Sub-theme insights",
        "Trends",
        "Prioritisation",
    ]
)


with tab1:
    a, b, c = st.columns(3)
    a.metric("Reviews analysed", f"{len(clustered):,}")
    b.metric("Themes discovered", metrics["clusters"])
    c.metric(
        "Negative-rating share",
        f"{(clustered['score'] <= 2).mean() * 100:.1f}%",
    )

    theme_counts = (
        clustered["theme"]
        .value_counts()
        .rename_axis("theme")
        .reset_index(name="reviews")
    )

    st.plotly_chart(
        px.bar(
            theme_counts,
            x="reviews",
            y="theme",
            orientation="h",
            title="Feedback volume by discovered theme",
        ),
        width="stretch",
    )

    st.caption(
        "Theme names are generated from representative terms. "
        "Inspect original reviews before interpreting them."
    )


with tab2:
    selected = st.selectbox(
        "Choose a theme",
        sorted(clustered["theme"].unique()),
    )

    examples = representative_reviews(clustered, selected, 12)

    st.write(
        f"**{selected}** | "
        f"{int((clustered['theme'] == selected).sum()):,} reviews"
    )

    st.dataframe(
        examples,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "Download analysed reviews",
        clustered.to_csv(index=False).encode("utf-8"),
        "signalmap_clustered_reviews.csv",
        "text/csv",
    )


with tab3:
    st.subheader("Sub-theme evidence and investigation")
    st.caption(
        "Explore the precomputed sub-theme analysis or discover sub-themes "
        "directly from the themes in the current SignalMap run."
    )

    mode = st.radio(
        "Sub-theme analysis",
        ["Precomputed results", "Discover from current analysis"],
        horizontal=True,
        key="subtheme_mode",
    )

    if mode == "Precomputed results":
        if not SUBTHEME_REVIEWS.exists() or not SUBTHEME_SUMMARY.exists():
            st.warning(
                "Precomputed sub-theme files were not found. "
                "Choose 'Discover from current analysis' to create a fresh analysis."
            )
        else:
            try:
                sub_reviews, sub_summary = load_subtheme_data(
                    str(SUBTHEME_REVIEWS),
                    str(SUBTHEME_SUMMARY),
                )

                required_review_cols = {"subtheme", "score", "content", "appName"}
                required_summary_cols = {
                    "subtheme", "review_count", "average_rating",
                    "negative_share_pct",
                }

                if not required_review_cols.issubset(sub_reviews.columns):
                    st.error("The sub-theme review file is missing required columns.")
                elif not required_summary_cols.issubset(sub_summary.columns):
                    st.error("The sub-theme summary file is missing required columns.")
                else:
                    st.info(
                        "These are precomputed results from the earlier "
                        "210-review sub-theme analysis. They are separate from "
                        "the analysis run above and do not automatically update "
                        "when a new dataset is uploaded."
                    )

                    total_sub_reviews = int(sub_summary["review_count"].sum())
                    average_rating = (
                        pd.to_numeric(sub_reviews["score"], errors="coerce").mean()
                        if not sub_reviews.empty else float("nan")
                    )
                    overall_negative = (
                        (
                            pd.to_numeric(sub_reviews["score"], errors="coerce") <= 2
                        ).mean() * 100
                        if not sub_reviews.empty else 0
                    )

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Reviews in sub-themes", f"{total_sub_reviews:,}")
                    m2.metric(
                        "Average rating",
                        f"{average_rating:.2f}" if pd.notna(average_rating) else "N/A",
                    )
                    m3.metric("Negative ratings", f"{overall_negative:.1f}%")

                    st.markdown("#### Sub-theme comparison")
                    chart = px.bar(
                        sub_summary.sort_values("negative_share_pct"),
                        x="negative_share_pct",
                        y="subtheme",
                        orientation="h",
                        text="negative_share_pct",
                        title="Share of 1- and 2-star reviews by sub-theme",
                        labels={
                            "negative_share_pct": "Negative reviews (%)",
                            "subtheme": "Sub-theme",
                        },
                    )
                    chart.update_traces(texttemplate="%{text:.1f}%")
                    chart.update_layout(xaxis_range=[0, 100])
                    st.plotly_chart(chart, width="stretch")

                    st.markdown("#### Sub-theme summary")
                    st.dataframe(
                        sub_summary.sort_values(
                            "negative_share_pct", ascending=False
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                    selected_subtheme = st.selectbox(
                        "Explore a sub-theme",
                        sorted(sub_reviews["subtheme"].dropna().unique()),
                        key="precomputed_subtheme_choice",
                    )

                    selected_rows = sub_reviews[
                        sub_reviews["subtheme"] == selected_subtheme
                    ].copy()
                    selected_rows["score"] = pd.to_numeric(
                        selected_rows["score"], errors="coerce"
                    )
                    selected_rows = selected_rows.dropna(subset=["score"])
                    if "thumbsUpCount" in selected_rows.columns:
                        selected_rows = selected_rows.sort_values(
                            ["score", "thumbsUpCount"],
                            ascending=[True, False],
                        )
                    else:
                        selected_rows = selected_rows.sort_values("score")

                    st.markdown("#### Low-rated review examples")
                    st.caption(
                        "These are the lowest-rated examples, not a statistically "
                        "representative sample."
                    )
                    display_cols = [
                        col for col in [
                            "appName", "score", "content", "thumbsUpCount", "at"
                        ]
                        if col in selected_rows.columns
                    ]
                    st.dataframe(
                        selected_rows[display_cols].head(8),
                        width="stretch",
                        hide_index=True,
                    )

                    prompts = INVESTIGATION_PROMPTS.get(
                        selected_subtheme,
                        ["Review the evidence and validate the underlying user problem."],
                    )
                    st.markdown("#### Product manager investigation prompts")
                    for prompt in prompts:
                        st.markdown(f"- {prompt}")

                    st.caption(
                        "Sub-theme labels were assigned to discovered clusters "
                        "and may contain mixed feedback. Validate examples "
                        "before treating a label as a confirmed problem."
                    )
                    st.download_button(
                        "Download sub-theme summary",
                        sub_summary.to_csv(index=False).encode("utf-8"),
                        "signalmap_subtheme_summary.csv",
                        "text/csv",
                    )
                    st.download_button(
                        "Download sub-theme review evidence",
                        sub_reviews.to_csv(index=False).encode("utf-8"),
                        "signalmap_subtheme_reviews.csv",
                        "text/csv",
                    )
            except Exception as exc:
                st.error(f"Could not load precomputed sub-theme outputs: {exc}")

    else:
        st.markdown("#### Discover sub-themes from the current analysis")
        st.caption(
            "This uses the reviews already clustered in the current dashboard run. "
            "It does not change the parent theme clustering."
        )

        available_themes = sorted(clustered["theme"].dropna().unique())
        chosen_theme = st.selectbox(
            "Choose a parent theme",
            available_themes,
            key="dynamic_parent_theme",
        )
        d1, d2 = st.columns(2)
        with d1:
            dynamic_clusters = st.slider(
                "Number of sub-themes", 2, 8, 4, key="dynamic_subtheme_count"
            )
        with d2:
            representative_count = st.slider(
                "Representative reviews per sub-theme",
                2, 10, 5, key="dynamic_representative_count"
            )

        st.caption(
            "Representative reviews are selected by proximity to each cluster "
            "centre in embedding space. They are examples of typical wording, "
            "not necessarily the most negative reviews."
        )

        if st.button(
            "Discover sub-themes",
            type="primary",
            key="run_dynamic_subthemes",
        ):
            theme_rows = clustered[
                clustered["theme"] == chosen_theme
            ].copy().reset_index(drop=True)

            text_column = (
                "text_clean" if "text_clean" in theme_rows.columns else "content"
            )
            theme_rows[text_column] = theme_rows[text_column].fillna("").astype(str)
            theme_rows = theme_rows[
                theme_rows[text_column].str.strip().str.len() > 0
            ].reset_index(drop=True)

            if len(theme_rows) < dynamic_clusters:
                st.error(
                    f"Only {len(theme_rows)} usable reviews are available. "
                    "Choose fewer sub-themes or a larger parent theme."
                )
            else:
                try:
                    from sklearn.cluster import KMeans
                    from sentence_transformers import SentenceTransformer

                    @st.cache_resource(show_spinner=False)
                    def load_subtheme_embedding_model():
                        return SentenceTransformer("all-MiniLM-L6-v2")

                    with st.spinner(
                        f"Embedding {len(theme_rows):,} reviews and discovering "
                        f"{dynamic_clusters} sub-themes..."
                    ):
                        embedding_model = load_subtheme_embedding_model()
                        embeddings = embedding_model.encode(
                            theme_rows[text_column].tolist(),
                            batch_size=32,
                            show_progress_bar=False,
                            normalize_embeddings=True,
                        )
                        subtheme_model = KMeans(
                            n_clusters=dynamic_clusters,
                            random_state=42,
                            n_init=10,
                        )
                        labels = subtheme_model.fit_predict(embeddings)

                    theme_rows["subtheme_id"] = labels
                    theme_rows["subtheme"] = [
                        f"Sub-theme {int(label) + 1}" for label in labels
                    ]
                    theme_rows["score"] = pd.to_numeric(
                        theme_rows["score"], errors="coerce"
                    )

                    summary_rows = []
                    representative_frames = []
                    for cluster_id in range(dynamic_clusters):
                        mask = labels == cluster_id
                        cluster_rows = theme_rows.loc[mask].copy()
                        ratings = cluster_rows["score"].dropna()
                        summary_rows.append({
                            "subtheme": f"Sub-theme {cluster_id + 1}",
                            "review_count": len(cluster_rows),
                            "average_rating": (
                                round(float(ratings.mean()), 2)
                                if not ratings.empty else float("nan")
                            ),
                            "negative_share_pct": (
                                round(float((ratings <= 2).mean() * 100), 1)
                                if not ratings.empty else float("nan")
                            ),
                        })

                        positions = [i for i, value in enumerate(labels) if value == cluster_id]
                        distances = (
                            (embeddings[positions] - subtheme_model.cluster_centers_[cluster_id])
                            ** 2
                        ).sum(axis=1) ** 0.5
                        nearest_order = distances.argsort()[:representative_count]
                        selected_positions = [positions[i] for i in nearest_order]
                        representatives = theme_rows.iloc[selected_positions].copy()
                        representatives["representative_rank"] = range(
                            1, len(representatives) + 1
                        )
                        representative_frames.append(representatives)

                    summary_df = pd.DataFrame(summary_rows).sort_values(
                        "negative_share_pct", ascending=False
                    )
                    representatives_df = pd.concat(
                        representative_frames, ignore_index=True
                    )
                    st.session_state.dynamic_subtheme_result = {
                        "parent_theme": chosen_theme,
                        "summary": summary_df,
                        "reviews": theme_rows,
                        "representatives": representatives_df,
                    }
                except Exception as exc:
                    st.error(f"Sub-theme discovery failed: {exc}")

        result = st.session_state.get("dynamic_subtheme_result")
        if result is not None:
            st.markdown(f"#### Results for: {result['parent_theme']}")
            st.info(
                "Sub-theme labels are generic cluster IDs. Inspect representative "
                "reviews before assigning human-readable problem labels."
            )

            st.dataframe(
                result["summary"],
                width="stretch",
                hide_index=True,
            )
            chart = px.bar(
                result["summary"].sort_values("negative_share_pct"),
                x="negative_share_pct",
                y="subtheme",
                orientation="h",
                title="Negative-rating share by discovered sub-theme",
                labels={
                    "negative_share_pct": "Negative reviews (%)",
                    "subtheme": "Discovered sub-theme",
                },
            )
            chart.update_layout(xaxis_range=[0, 100])
            st.plotly_chart(chart, width="stretch")

            chosen_subtheme = st.selectbox(
                "Inspect representative reviews",
                sorted(result["representatives"]["subtheme"].unique()),
                key="dynamic_subtheme_review_choice",
            )
            selected_representatives = result["representatives"][
                result["representatives"]["subtheme"] == chosen_subtheme
            ].copy()

            rep_columns = [
                col for col in [
                    "representative_rank", "appName", "score", "content",
                    "thumbsUpCount", "at",
                ]
                if col in selected_representatives.columns
            ]
            st.dataframe(
                selected_representatives[rep_columns],
                width="stretch",
                hide_index=True,
            )

            st.download_button(
                "Download discovered sub-theme summary",
                result["summary"].to_csv(index=False).encode("utf-8"),
                "signalmap_discovered_subtheme_summary.csv",
                "text/csv",
            )
            st.download_button(
                "Download all discovered sub-theme reviews",
                result["reviews"].to_csv(index=False).encode("utf-8"),
                "signalmap_discovered_subtheme_reviews.csv",
                "text/csv",
            )
            st.download_button(
                "Download representative reviews",
                result["representatives"].to_csv(index=False).encode("utf-8"),
                "signalmap_representative_reviews.csv",
                "text/csv",
            )


with tab4:
    trends = weekly_theme_trends(clustered)

    if trends.empty:
        st.warning("No valid dates available for trend analysis.")
    else:
        chosen = st.multiselect(
            "Themes to display",
            sorted(trends["theme"].unique()),
            default=list(trends["theme"].value_counts().head(5).index),
        )

        view = trends[trends["theme"].isin(chosen)]

        st.plotly_chart(
            px.line(
                view,
                x="week",
                y="reviews",
                color="theme",
                title="Weekly review volume by theme",
            ),
            width="stretch",
        )

        st.caption(
            "This chart shows review volume, not the rate of complaints "
            "among all users. Changes in app popularity or data collection "
            "can affect it."
        )


with tab5:
    st.dataframe(
        priority,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Priority scores use review-derived proxy signals. "
        "Impact, severity and confidence are not independently validated. "
        "Effort requires human assessment."
    )

    st.download_button(
        "Download priority table",
        priority.to_csv(index=False).encode("utf-8"),
        "signalmap_priority_table.csv",
        "text/csv",
    )
