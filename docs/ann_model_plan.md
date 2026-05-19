# ANN Model — Implementation Plan (for Sonnet)

Goal: add a 6th classifier — an Artificial Neural Network — to the NBA Free
Throw Predictor, matching the existing notebook syntax exactly. Do NOT change
project structure, target logic, features, or any other model.

## Decision: use `sklearn.neural_network.MLPClassifier`

Every existing model notebook is pure scikit-learn + `joblib`. To "match the
existing syntax", the ANN must be `MLPClassifier`, NOT Keras/TensorFlow.
This adds NO new dependency (scikit-learn already ships it) and keeps the
`.fit()` / `.predict()` / `joblib.dump` pattern identical to the other 5
notebooks. The ANN needs a `StandardScaler`, exactly like KNN/SVC/LR.

If the user actually wants a Keras model instead, stop and ask — it would
change requirements.txt, the save format, and app.py loading.

## Conventions to copy (from `notebooks/03_knn.ipynb`)

- 12 cells, one statement-group per cell, same blank-line spacing.
- Aligned `=` in the scaler block (`X_test_scaled  = ...`).
- Inline `#` comment on the model constructor line.
- Confusion matrix cmaps already used: KNN=Blues, GB=Greens, RF=Oranges,
  SVC=Purples, LR=Reds. The ANN uses **`Greys`** (next unused single-word cmap).
- `target_names=['Below-Average', 'Good']`, tick labels `['Below-Avg', 'Good']`.

## Step 1 — Renumber so the comparison notebook stays last

1. `git mv notebooks/08_model_comparison.ipynb notebooks/09_model_comparison.ipynb`
   (its internal content does not change).
2. The new ANN notebook becomes `notebooks/08_ann.ipynb`.

## Step 2 — Create `notebooks/08_ann.ipynb`

Build it cell-by-cell as a mirror of `03_knn.ipynb`. Exact cell contents:

**Cell 0 — markdown**
```
# Artificial Neural Network (ANN)
```

**Cell 1 — code**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

**Cell 2 — code**
```python
df = pd.read_csv('../data/cleaned_data.csv')
```

**Cell 3 — code**
```python
X = df.drop(columns=['target'])
y = df['target']
print('Columns:', X.columns.tolist())
```

**Cell 4 — code**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Train:', X_train.shape, '  Test:', X_test.shape)
```

**Cell 5 — code**
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

**Cell 6 — code**
```python
ann = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)  # 2 hidden layers (64, 32)
```

**Cell 7 — code**
```python
ann.fit(X_train_scaled, y_train)
```

**Cell 8 — code**
```python
y_pred = ann.predict(X_test_scaled)
```

**Cell 9 — code**
```python
acc = accuracy_score(y_test, y_pred)
print(f'ANN Test Accuracy: {acc * 100:.2f}%')
print(classification_report(y_test, y_pred, target_names=['Below-Average', 'Good']))
```

**Cell 10 — code**
```python
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greys',
            xticklabels=['Below-Avg', 'Good'],
            yticklabels=['Below-Avg', 'Good'])
plt.title(f'ANN Confusion Matrix (Accuracy: {acc*100:.2f}%)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('../images/ann_confusion_matrix.png', bbox_inches='tight')
plt.show()
```

**Cell 11 — code**
```python
joblib.dump(ann, '../models/ann.pkl')
joblib.dump(scaler, '../models/ann_scaler.pkl')
```

## Step 3 — Execute the notebook

Run all cells (`jupyter nbconvert --to notebook --execute --inplace`, from the
`notebooks/` directory so the `../` paths resolve). This must produce:
- `models/ann.pkl`, `models/ann_scaler.pkl`
- `images/ann_confusion_matrix.png`
- embedded cell outputs in `08_ann.ipynb`

**Record the ANN test accuracy / precision / recall / F1** from the
classification report — every number below depends on it.

## Step 4 — Update `notebooks/09_model_comparison.ipynb`

- Load block: add
  `ann_model  = joblib.load('../models/ann.pkl')` and
  `ann_scaler = joblib.load('../models/ann_scaler.pkl')` (keep the aligned style).
- Eval block: add `X_test_ann = ann_scaler.transform(X_test)` next to the
  other scaled inputs, and append `evaluate('ANN', ann_model, X_test_ann)`
  to the `results` list (keep it last — insertion order, not sorted).
- Bar chart: add a 6th color to the `colors` list, e.g. `'#6B7280'`.
- Re-execute the notebook so `images/model_comparison.png` and the printed
  table refresh.

## Step 5 — Update `app.py`

- Load models: add `ann_model = joblib.load("models/ann.pkl")`.
- Load scalers: add `ann_scaler = joblib.load("models/ann_scaler.pkl")` and
  change the comment to `# (only for KNN, SVC, LR, ANN)`.
- Home page: `st.metric("ML Models", "5")` → `"6"`; add
  `- Artificial Neural Network (ANN)` to the models list; update the
  best-model `st.success(...)` line only if the ANN beats GB's accuracy.
- Prediction page: add `"ANN"` to the `model_choice` selectbox list.
- Add `input_ann = ann_scaler.transform(input_df)` to the scaler block.
- Add a branch: `elif model_choice == "ANN": prediction = ann_model.predict(input_ann)`.
- Model Comparison page: insert ANN into the `data` dict (all 5 parallel
  lists) at its accuracy-sorted rank; add
  `st.image("images/ann_confusion_matrix.png", caption="ANN")` to a column.

## Step 6 — Update `README.md`

- Intro + "Machine Learning Models": "five classifiers" → "six", and the
  count wherever it appears.
- Project structure tree: add `08_ann.ipynb`, rename
  `08_model_comparison.ipynb` → `09_model_comparison.ipynb`; under `models/`
  add `ann.pkl, ann_scaler.pkl`; under `images/` it is already covered by
  the `*_confusion_matrix.png` wildcard.
- Models table: add `| Artificial Neural Network | Yes |`.
- Results table: insert the ANN row at its accuracy-sorted rank using the
  real numbers from Step 3; fix the trailing sentence if the best model
  changes.
- "How to Run": "Run notebooks in order (01 to 08)" → "(01 to 09)".

## Step 7 — Update the paper

- `docs/paper.tex`: add the ANN to the model list, the results table, and any
  "five models" wording (→ six). Use the Step 3 numbers.
- Regenerating `docs/paper.pdf` needs a LaTeX toolchain (`pdflatex`). If it is
  not installed, update `paper.tex` only and tell the user the PDF needs a
  manual rebuild.

## Out of scope — do NOT touch

- `data/`, `feature_columns.pkl`, the target definition, the 39 features.
- Notebooks 01–07 and any other model's `.pkl`.
- No GridSearchCV, no sklearn Pipeline, no `assert` leakage checks.

## Final check

- 6 model `.pkl` pairs load without error.
- `streamlit run app.py` → ANN selectable and predicts on the Prediction page.
- README / app / paper all show the same ANN numbers and a consistent ranking.
