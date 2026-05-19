# NBA Free Throw Predictor

A machine learning system that classifies NBA player-seasons as
**good free-throw shooters** (FT% >= 75%) or **below-average free-throw shooters** (FT% < 75%).
The project uses per-game statistics scraped from Basketball Reference (1975-2025),
applies careful preprocessing to remove data leakage, engineers ratio features,
adds historical free-throw features from prior seasons, and trains six classifiers.
An interactive Streamlit dashboard allows real-time prediction for any custom player profile.

---

## Problem Statement

Free-throw shooting is a critical, high-frequency event in NBA games.
Identifying players likely to be good or below-average at the free-throw line
helps teams make informed rotation and contract decisions.
The challenge is that free-throw percentage is a specific physical and mental skill
that is only partially explained by other box-score statistics.
Naive approaches (including FT or FTA as features, or using the raw season FT%
to derive a target) produce misleading results.

---

## How the Target is Chosen

The target is a binary label on per-season free-throw percentage using a fixed threshold:

- 1 (Good) if FT% >= 0.75
- 0 (Below-Average) if FT% < 0.75

Only player-seasons with at least 1.0 free-throw attempt per game are included,
ensuring each row has a meaningful, stable FT%.

Why 0.75 specifically:

- 0.75 is the universally recognised NBA free-throw benchmark, instantly interpretable to any reader.
- It is defined independently of the dataset. A league-average threshold makes each label depend on aggregate statistics of other rows; a fixed constant does not. This is methodologically cleaner.
- It is era-independent. NBA league-average FT% moved only between 0.728 and 0.784 across all 50 seasons. Label agreement between a fixed 0.75 target and a per-season league-average target is 94.8%, so the simpler fixed definition wins.
- The median FT% of the filtered data is 0.758, producing a well-balanced split (54.3% Good, 45.7% Below-Average).

Every player-season that passes the FTA filter receives a label. No rows are excluded.

Alternative: A stricter elite-vs-weak design (>= 0.85 vs < 0.65) reaches higher headline accuracy (~92%) but only by discarding the ambiguous middle ~47% of players. That framing trains and tests only on clear extremes, which is why the accuracy looks impressive but the model never has to make a hard call. The 0.75 threshold is chosen as the primary target because it classifies every player-season and produces an honest estimate of what machine learning can do on the full population.

---

## Data Leakage Prevention

The following columns are NEVER used as model features:

- FT  (free throws made)
- FTA (free throw attempts)
- FT% (free throw percentage)

These directly determine the target label. Including them would allow models
to re-derive the target from its own formula, producing artificially high accuracy.

The historical features below are safe because they use ONLY data from prior seasons:

- career_ft_pct: cumulative FT% from all seasons before the current one
- prev_ft_pct: FT% in the immediately previous season
- career_seasons: number of prior seasons played (0 for a rookie season)

---

## Dataset

- Source: Basketball Reference (https://www.basketball-reference.com)
- Scraped: per-game statistics for all NBA players, seasons 1975-2025
- Raw size: 26,292 rows x 32 columns
- After traded-player deduplication: 21,275 player-seasons
- After FTA >= 1.0 filter: 15,020 modeling rows
- External data: player height and weight from Basketball Reference player index pages

---

## Project Structure

```
NBA Free Throw Predictor/
|
|-- app.py                          # Streamlit dashboard
|-- requirements.txt                # Python dependencies
|-- README.md
|
|-- data/
|   |-- combined.csv                # Raw scraped stats (26,292 rows)
|   |-- cleaned_data.csv            # Final modeling dataset (15,020 rows, 40 cols)
|   |-- player_bios.csv             # Scraped player height and weight
|
|-- notebooks/
|   |-- 01_data_collection.ipynb    # Web scraping (per-game + player bios)
|   |-- 02_preprocessing_and_eda.ipynb  # Cleaning, feature engineering, EDA
|   |-- 03_knn.ipynb                # K-Nearest Neighbors
|   |-- 04_gradient_boosting.ipynb  # Gradient Boosting
|   |-- 05_random_forest.ipynb      # Random Forest
|   |-- 06_svc.ipynb                # Support Vector Classifier
|   |-- 07_logistic_regression.ipynb
|   |-- 08_ann.ipynb                # Artificial Neural Network
|   |-- 09_model_comparison.ipynb   # Unified evaluation across all 6 models
|
|-- models/
|   |-- feature_columns.pkl         # Exact ordered feature list used in training
|   |-- knn.pkl, knn_scaler.pkl
|   |-- svc.pkl, svc_scaler.pkl
|   |-- logistic.pkl, logistic_scaler.pkl
|   |-- rf.pkl
|   |-- gb.pkl
|   |-- ann.pkl, ann_scaler.pkl
|
|-- images/
|   |-- *_confusion_matrix.png      # One per model
|   |-- model_comparison.png
|   |-- eda_*.png                   # EDA plots
|   |-- streamlit_*.png             # Dashboard screenshots
|
|-- docs/
|   |-- IEEE_Machine_Learning_paper.pdf
|   |-- phase1doc.docx
```

---

## Features Used

Features actually used (39 total):

- Playing time: Age, G, GS, MP
- Shooting: FG, FGA, FG%, 3P, 3PA, 3P%, 2P, 2PA, 2P%, eFG%
- Counting stats: ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS
- Historical FT: career_ft_pct, prev_ft_pct, career_seasons
- Physical: height_in, weight_lb
- Engineered ratios: ast_to_tov, pts_per_fga, threep_rate, reb_per_min, stocks_per_min, scoring_load
- Position (one-hot): Pos_C, Pos_PF, Pos_PG, Pos_SF, Pos_SG

---

## Machine Learning Models

| Model                      | Requires Scaling |
|----------------------------|-----------------|
| KNN                        | Yes             |
| SVC                        | Yes             |
| Logistic Regression        | Yes             |
| Random Forest              | No              |
| Gradient Boosting          | No              |
| Artificial Neural Network  | Yes             |

All models use plain default-style constructors with fixed settings.
No sklearn Pipeline objects are used - all steps are procedural.

---

## Results

All numbers below are on the held-out test set (20% of data, random_state=42).
FT, FTA, and FT% are fully removed from all feature sets.

| Model                     | Accuracy | Precision | Recall  | F1-Score |
|---------------------------|----------|-----------|---------|----------|
| KNN                       | 88.42%   | 63.35%    | 30.33%  | 41.02%   |
| ANN                       | 87.05%   | 51.21%    | 52.88%  | 52.03%   |
| Random Forest             | 57.32%   | 22.35%    | 89.47%  | 35.77%   |
| Gradient Boosting         | 56.89%   | 22.38%    | 90.98%  | 35.92%   |
| SVC                       | 55.86%   | 21.99%    | 91.23%  | 35.44%   |
| Logistic Regression       | 54.59%   | 21.23%    | 89.22%  | 34.30%   |

KNN leads at 88.42% accuracy, with ANN close behind at 87.05%. The strongest
predictor is a player's past free-throw record (career_ft_pct, prev_ft_pct):
free-throw shooting is a stable individual skill, and prior seasons are the best
available signal for the current season.

---

## How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run notebooks in order (01 to 09). Skip the scraping cell in notebook 01
   if `combined.csv` already exists. The bio-scraping cell in notebook 01
   takes a few minutes and requires an internet connection:

```
cd notebooks
jupyter notebook
```

3. Launch the dashboard:

```
streamlit run app.py
```

---

## Team Members

- Habibatallah Mahdi
- Sama Abdelsadek
- Mohamed Allam
- Shaza Hossam
- Ahmed Shaban

---

## Course Info

Machine Learning course project.
