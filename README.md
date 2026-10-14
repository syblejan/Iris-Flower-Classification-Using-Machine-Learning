# Iris Flower Classification — Machine Learning

**Python · Pandas · Scikit-learn · Logistic Regression · Cross-Validation**

An introductory multiclass classification project that predicts the Iris flower species from **four measurements**: sepal length/width and petal length/width.

## Dataset
The original repository CSV is preserved in `data/iris.data.csv`. It contains **150 samples**, **3 species** (50 examples each), **4 numerical input features**, and a class label. It has no CSV header; the evaluation script assigns headers when reading it. This is the classic UCI Iris dataset format, not a real-life production dataset.

## Original Notebook (Preserved)
The original notebook is stored unmodified at `notebooks/original_IrisFlowerClassification.ipynb`. It performed exploratory analysis, a Seaborn pairplot, Logistic Regression using `liblinear`, an unstratified 80/20 split and 5-fold stratified cross-validation.

**Historical notebook outputs:** training accuracy **95.83%**, holdout test accuracy **100% on 30 samples**, and 5-fold out-of-fold accuracy **96.00%**. The small perfect holdout score must **not** be interpreted as guaranteed generalization. The original notebook also uses a Windows-specific absolute CSV path, and the cross-validation report's target name order should not be inferred from `unique()` ordering.

## Improvements
1. **Portable paths:** relative `data/iris.data.csv` rather than the original machine-specific path.
2. **Clear class balance:** stratified 80/20 split (`random_state=42`).
3. **Leakage-aware scaling:** StandardScaler is inside the Pipeline, fit separately on each training split/fold.
4. **Reproducible validation:** shuffled 5-fold stratified out-of-fold predictions.
5. **Honest evaluation:** accuracy, per-class precision/recall/F1 and confusion matrices for both holdout and CV.
6. **Reproducible artifacts:** machine-readable `results/evaluation.json` and `results/holdout_predictions.csv`, created locally upon execution.

## Verified Reference Test
The updated Python workflow was executed using scikit-learn's bundled **standard Iris dataset** in the same 150-row, 4-feature, 3-species CSV format. **These are reference-dataset execution results; the exact GitHub CSV bytes were not independently executed in the Python environment.**

| Evaluation | Accuracy |
|---|---:|
| Stratified 80/20 holdout (30 samples) | **93.33% (28/30)** |
| 5-fold out-of-fold validation (150 samples) | **95.33% (143/150)** |

**Reference holdout confusion matrix** (rows actual, columns predicted; order setosa, versicolor, virginica):

```text
[[10, 0, 0],
 [ 0, 9, 1],
 [ 0, 1, 9]]
```

The difference from the original notebook's 100% arises from changes in splitting, stratification, preprocessing and solver. These results are from **different experiment protocols**, not a performance drop measured on one identical test set.

## Repository Layout
```text
Iris-flower-classification-/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── iris.data.csv
├── src/
│   └── evaluate.py
└── notebooks/
    ├── original_IrisFlowerClassification.ipynb
    └── reproducible_iris_classification.ipynb
```

## Run
```bash
git clone https://github.com/syblejan/Iris-flower-classification-.git
cd Iris-flower-classification-
python -m venv .venv
# Activate your environment (Windows: .venv\\Scripts\\activate)
python -m pip install -r requirements.txt
python src/evaluate.py
```

Or open `notebooks/reproducible_iris_classification.ipynb` from the project root.

Generated results are excluded from Git. No downloaded models or external API are needed.

## Limitations
Iris is a small, easy educational benchmark. A perfect 30-row holdout is not strong evidence of production-level performance. The dataset is not representative of deployed computer-vision image recognition; it uses **measured numerical flower features**, not flower photographs.

## Author
**Syble Jan** — BS Computer Science · Aspiring AI/ML Engineer

[GitHub Profile](https://github.com/syblejan)
