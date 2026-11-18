# SignalMap — Model Evaluation Log

## Dataset
Health & Fitness App Reviews (23 apps), supplied as a CSV by the project owner.

## Initial data profile
- Rows: 180,754
- Apps: 23
- Columns: `tag`, `content`, `score`, `thumbsUpCount`, `at`, `appName`
- Missing `content`: 28
- Exact duplicate rows: 0
- Duplicate review text: 14,163
- Date range: 2013-11-01 21:48:09 to 2026-05-15 00:32:48
- Review text length: median 152 characters; mean about 193.8 characters

These values were calculated from the supplied CSV. Re-run the data validation after changing the file.

## Baseline to run
1. TF-IDF + K-Means
2. Sentence Transformer embeddings + K-Means
3. Later: compare HDBSCAN if environment/package support permits

## Record for each experiment
- Date and code version
- Sample size and sampling method
- Model name/version
- Preprocessing choices
- Number of clusters / parameters
- Silhouette score (where meaningful)
- Human assessment of theme coherence
- Examples of good and poor groupings
- Known failure cases

## Do not claim
- That a cluster is correct based on silhouette score alone.
- That star rating is a trained sentiment model.
- That review volume equals unique-user prevalence.
- That proxy priority scores represent actual business impact.
- Any user research, usability outcomes or time savings that have not been measured.
