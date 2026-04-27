# Airbnb US Listings: Three Lenses on the Short-Term Rental Market

**Author:** Parth Deshmukh
**Course:** CSCE 676 — Data Mining, Spring 2026

> **Start here:** [`main_notebook.ipynb`](main_notebook.ipynb) — the curated, end-to-end story of the project.

> **Project video:** https://youtu.be/Fb_VxVX3mAA

## Overview

Airbnb is simultaneously **hyper-local** (prices change block by block) and **globally networked** (a small group of professional hosts run portfolios that span the country). This project takes 280,673 listings across 34 US cities (22 states) from [Inside Airbnb](http://insideairbnb.com/get-the-data/) and studies that duality through three complementary data-mining lenses — graph mining, anomaly detection, and spatial autocorrelation — then ties them together in a cross-RQ synthesis. The goal is not just three separate analyses, but a single coherent story about who the structurally important hosts are, which listings are genuinely mispriced relative to their context, and where price hot/cold spots actually live on the map.

## Research Questions

| | RQ1 — Host Market Reach | RQ2 — Pricing Anomalies | RQ3 — Spatial Patterns |
|---|---|---|---|
| **Question** | Do hosts with higher PageRank in the host–neighbourhood bipartite graph charge systematically different prices than peripheral hosts? | Which listings are anomalously priced relative to their room type, amenity profile, and capacity — and where are they concentrated? | Is Airbnb pricing spatially autocorrelated, and where are the statistically significant price hot spots and cold spots? |
| **Mining task** | Graph mining — node centrality | Anomaly detection | Spatial autocorrelation (external technique) |
| **Algorithm** | PageRank on bipartite graph | Isolation Forest + Local Outlier Factor | Global + Local Moran's I / LISA |
| **Library** | `networkx` | `scikit-learn` | `esda`, `libpysal` |

RQ1 and RQ2 use techniques from the course; RQ3 uses Moran's I / LISA (Anselin, 1995) to satisfy the external-technique requirement.

## Data

- **Source:** [Inside Airbnb](http://insideairbnb.com/get-the-data/) — independent non-commercial scrape of public Airbnb data, city by city.
- **Scope:** 34 US cities (NYC, Chicago, Boston, Hawaii, Broward County FL, Nashville, Austin, Clark County NV, …), 22 states, **280,673 listings** scraped September–December 2025.
- **Working columns (13 of 85):** `id, name, description, host_name, host_id, price, neighbourhood_cleansed, latitude, longitude, room_type, accommodates, review_scores_rating, amenities`.
- **Preprocessing performed in the notebook:**
  - `price_clean` parsed from strings like `"$1,250.00"`; `log_price = log(1 + price_clean)`.
  - `amenity_count` parsed from the JSON-style `amenities` field (range 0–122).
  - `city` and `state` reverse-geocoded from `(lat, lon)` via `reverse_geocoder` (offline K-D tree). The formatted CSV is cached on Google Drive and re-loaded on every run.
- **RQ-specific subsets** (derived from the cleaned master frame, not by global filtering):
  - `df_rq1` — 279,604 rows (99.6%), any row with `host_id` and a non-empty neighbourhood.
  - `df_rq2` — 157,090 rows (55.9%), priced listings (`0 < price < 99th pct`) with non-null features.
  - `df_rq3` — 124,647 rows (44.4%), `df_rq2` restricted to the contiguous US (lat 24–50, lon −125 to −66) so the spatial-weights graph stays connected.

## Results Summary

| RQ | Technique | Headline result |
|---|---|---|
| **RQ1** | PageRank on host–neighbourhood bipartite graph (33.7K nodes, 52.9K edges) | Spearman ρ(PageRank, median price) = **−0.144**, p ≈ 10⁻⁹⁶ — more negative than the degree baseline (−0.032). Top hosts are corporate property managers: Blueground (4,383 listings / 221 nbhds), Evolve, LuxurybookingsFZE, Suiteness. |
| **RQ2** | Isolation Forest + Local Outlier Factor on (log_price, amenity_count, accommodates, room_type) | At 2% contamination: IF = 3,161, LOF = 3,161, **IF ∩ LOF = 296**. **76% of those anomalies are missed by the |z|>3 baseline** — they are contextual, not univariate extreme. |
| **RQ3** | Global + Local Moran's I on 732 neighbourhood centroids (k = 8, 999 permutations) | Global **I = 0.387**, z = 22.1, p = 0.001. Stable across k ∈ {5, 8, 15}. **247 of 732 neighbourhoods (34%)** sit in significant LISA clusters: 98 HH coastal hot spots (RI/CA), 108 LL Midwest cold spots (Chicago South Side). |
| **Synthesis** | Cross-RQ integration | The HL LISA class carries a 25–30% higher anomaly rate than HH/LL/ns. Top-50 PR hosts are **1.7× over-represented in HH** and **10× under-represented in LL**. **Spatial structure is a stronger predictor of pricing anomalies than host centrality.** |

The full analysis, plots, and interactive maps live in [`main_notebook.ipynb`](main_notebook.ipynb).

## Reproducing the Work

This project was built and run in **Google Colab** (Python 3.11). To reproduce:

1. Open [`main_notebook.ipynb`](main_notebook.ipynb) in Colab.
2. Run cells top-to-bottom. The notebook is self-contained:
   - The first install cell adds the three non-default packages (`!pip install esda libpysal reverse_geocoder -q`).
   - The data-loading cell reads the cached, formatted CSV from Google Drive (the reverse-geocoding step is pre-computed; you don't need to re-run it).
   - RQ1 → RQ2 → RQ3 → cross-RQ synthesis run in sequence; later sections depend on earlier outputs.
3. Pinned package versions are in [`requirements.txt`](requirements.txt) (regenerate with the `!pip freeze` cell at the bottom of the notebook if you want a fresh capture).

The two checkpoint notebooks under [`checkpoints/`](checkpoints/) are included to show how the project evolved over the semester; they are not required to reproduce the final results.

## Key Dependencies

| Package | Purpose |
|---|---|
| `python` 3.11 | Colab default runtime |
| `pandas`, `numpy`, `scipy` | Data wrangling, stats |
| `matplotlib`, `seaborn`, `plotly` | Plots and interactive maps |
| `networkx` | Bipartite graph + PageRank (RQ1) |
| `scikit-learn` | Isolation Forest, LOF, scaling, encoding (RQ2) |
| `esda`, `libpysal` | Global + Local Moran's I / LISA (RQ3) |
| `reverse_geocoder` | Offline (lat, lon) → (city, state) lookup |

The exhaustive list with exact versions lives in [`requirements.txt`](requirements.txt).

## Repo Structure

```
.
├── README.md                  <- you are here
├── main_notebook.ipynb        <- final curated deliverable (run this)
├── requirements.txt           <- pinned environment, exported from Colab
├── .gitignore
└── checkpoints/
    ├── checkpoint_1.ipynb     <- Project Checkpoint 1 (early data exploration)
    └── checkpoint_2.ipynb     <- Project Checkpoint 2 (mid-semester progress)
```

There is no separate `src/`, `scripts/`, or `data/` folder — the notebook reads its cached input CSV directly from Google Drive, and there are no helper modules outside the notebook.

## Honor Statement and Citations

On my honor, I declare the following resources were used in the preparation of this project:

- **Collaborators:** None.
- **Web sources:** Inside Airbnb data assumptions — <https://insideairbnb.com/data-assumptions/>. Used to understand the scraping methodology, data limitations, and the `neighbourhood_cleansed` field semantics.
- **AI tools:** Perplexity and ChatGPT / Claude — used to draft project structure, refine explanations, draft markdown narration, and generate example pandas / networkx / scikit-learn / esda code snippets. All analytical decisions, results, and interpretations are my own.
- **Dataset:** Inside Airbnb. (2025). *United States Airbnb Listings* [Dataset]. Retrieved from <http://insideairbnb.com/get-the-data/>. Multiple US cities, scraped September–December 2025. Licensed under CC BY 4.0.
- **External-technique reference:** Anselin, L. (1995). *Local Indicators of Spatial Association — LISA*. Geographical Analysis, 27(2), 93–115.
