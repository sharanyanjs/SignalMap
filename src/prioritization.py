import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {"impact": 0.30, "severity": 0.25, "frequency": 0.20, "confidence": 0.15, "trend": 0.10}

def build_priority_table(clustered: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Priority weights must sum to a positive value.")
    weights = {k: v / total for k, v in weights.items()}

    rows = []
    for theme, group in clustered.groupby("theme"):
        n = len(group)
        avg_rating = float(group["score"].mean())
        negative_share = float((group["score"] <= 2).mean() * 100)
        # These are transparent proxy signals, not ground-truth user impact/severity.
        frequency = min(100.0, 100 * n / max(1, clustered["theme"].value_counts().quantile(.90)))
        severity = negative_share
        impact = min(100.0, 0.7 * severity + 0.3 * frequency)
        confidence = min(100.0, 35 + 8 * np.log1p(n))
        dated = group.dropna(subset=["review_date"]).copy()
        trend = 0.0
        if len(dated) >= 8:
            latest = dated["review_date"].max()
            recent = dated[dated["review_date"] >= latest - pd.Timedelta(days=90)]
            previous = dated[
                (dated["review_date"] < latest - pd.Timedelta(days=90)) &
                (dated["review_date"] >= latest - pd.Timedelta(days=180))
            ]
            if len(previous) > 0:
                trend = float(np.clip((len(recent) - len(previous)) / len(previous) * 100, -100, 100))
        trend_score = float(np.clip(50 + trend / 2, 0, 100))
        score = (
            weights["impact"] * impact +
            weights["severity"] * severity +
            weights["frequency"] * frequency +
            weights["confidence"] * confidence +
            weights["trend"] * trend_score
        )
        rows.append({
            "theme": theme, "review_count": n, "average_rating": round(avg_rating, 2),
            "negative_share_pct": round(negative_share, 1),
            "impact_proxy": round(impact, 1), "severity_proxy": round(severity, 1),
            "frequency_score": round(frequency, 1), "confidence_proxy": round(confidence, 1),
            "trend_score": round(trend_score, 1), "priority_score": round(score, 1),
            "effort_estimate": "Needs PM assessment",
        })
    return pd.DataFrame(rows).sort_values("priority_score", ascending=False).reset_index(drop=True)
