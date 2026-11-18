# SignalMap

**From customer feedback to evidence-backed product decisions.**

SignalMap is a product-intelligence research prototype. It analyses app reviews, discovers recurring feedback themes using NLP, displays original evidence, explores review-volume trends and creates a transparent prioritisation view.

## Dataset included

`data/raw/health_fitness_app_reviews.csv` — the health & fitness app reviews CSV supplied for this project.

Expected columns: `tag`, `content`, `score`, `thumbsUpCount`, `at`, `appName`.

The app validates the columns at load time. The original review text is retained for traceability.

## Requirements

- Windows 10/11
- Python 3.11 or 3.12 recommended for the smoothest ML package compatibility
- VS Code

Python 3.13 may work with current packages, but if PyTorch or Sentence Transformers installation fails, use Python 3.11/3.12 in a fresh virtual environment.

## Run in VS Code (PowerShell)

Open the extracted `SignalMap` folder in VS Code. In the terminal:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If you already use Python 3.13 and want to try it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this once in that terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## First run

1. Streamlit opens the local app in your browser.
2. Confirm the dataset summary.
3. In the sidebar, choose a sample size (start with 2,000–5,000).
4. Keep **Use Sentence Transformers** enabled.
5. Click **Run SignalMap analysis**.
6. The first run downloads `all-MiniLM-L6-v2`; allow time for the download.
7. Explore themes, inspect the original reviews, view weekly volume and download outputs.

The app uses a sample by default because embedding all 180K reviews locally can take substantial time and memory. Increase the sample after the first run works.

## CLI alternative

```powershell
python run_analysis.py --sample 5000 --clusters 10
```

TF-IDF baseline (does not download a transformer model):
```powershell
python run_analysis.py --sample 5000 --clusters 10 --tfidf-only
```

Outputs are saved under `outputs/`.

## What the ML does

- **Semantic embeddings:** Sentence Transformers `all-MiniLM-L6-v2`
- **Clustering:** K-Means over the embedding vectors
- **Theme labels:** top TF-IDF terms from each discovered cluster
- **Evaluation:** silhouette score on a sample
- **Baseline:** TF-IDF vectors + K-Means, selectable through the CLI fallback

This is an initial baseline, not a claim that K-Means is the best clustering method. Compare algorithms and manually label a review subset before making quality claims.

## Important product / research limitations

- Star rating is used as a coarse sentiment proxy: 1–2 negative, 3 neutral, 4–5 positive. It is not an independently validated sentiment model.
- Priority inputs are explicitly labelled **proxies** derived from review data. They do not establish actual user impact, business impact or engineering effort.
- Review volume is not the same as user prevalence. One person may post multiple reviews, and app popularity/data collection can vary.
- Weekly volume charts show counts, not normalised complaint rates.
- Theme labels are generated from frequent terms and need human review.
- The dataset spans different apps and years. Compare themes within app/time context before making product recommendations.
- This prototype does not yet include user interviews, usability-test results, a validated severity model or production security controls.

## Project structure

```text
SignalMap/
├── app.py
├── run_analysis.py
├── requirements.txt
├── README.md
├── data/
│   └── raw/
│       └── health_fitness_app_reviews.csv
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── ml_pipeline.py
│   ├── prioritization.py
│   └── analysis.py
├── outputs/
└── tests/
```

## Product North Star (proposed)

**Weekly Evidence-Backed Decisions:** the number of issues reviewed and accepted for further investigation or prioritisation in a week, with traceable supporting feedback.

This is a proposed metric. It must be validated with actual product users before claiming it measures product value.
