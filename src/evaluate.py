"""Reproducible classification of the repository's headerless Iris CSV."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']
COLUMNS = [*FEATURES, 'species']

def load_data(path):
    data = pd.read_csv(path, header=None, names=COLUMNS)
    if len(data) != 150:
        raise ValueError(f'Expected 150 Iris records, got {len(data)}')
    if data.isnull().any().any() or data['species'].nunique() != 3:
        raise ValueError('Unexpected missing values or number of species')
    if not np.isfinite(data[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError('Features contain non-finite values')
    return data

def evaluate(dataset, output_dir):
    X = dataset[FEATURES]
    y = dataset['species']
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_predictions = cross_val_predict(pipeline, X, y, cv=cv)
    labels = sorted(y.unique().tolist())
    report = {
        'dataset_rows': int(len(dataset)),
        'class_counts': {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        'train_rows': len(X_train), 'test_rows': len(X_test),
        'holdout_accuracy': float(accuracy_score(y_test, predictions)),
        'holdout_classification_report': classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0),
        'holdout_confusion_matrix': confusion_matrix(y_test, predictions, labels=labels).tolist(),
        'cv_accuracy': float(accuracy_score(y, cv_predictions)),
        'cv_classification_report': classification_report(y, cv_predictions, labels=labels, output_dict=True, zero_division=0),
        'cv_confusion_matrix': confusion_matrix(y, cv_predictions, labels=labels).tolist(),
        'labels': labels,
        'protocol': '80/20 stratified holdout and shuffled 5-fold out-of-fold evaluation; scaler fitted per training fold',
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame({'actual': y_test.to_numpy(), 'predicted': predictions}).to_csv(out / 'holdout_predictions.csv', index=False)
    print(json.dumps({k: report[k] for k in ('dataset_rows','train_rows','test_rows','class_counts','holdout_accuracy','cv_accuracy','holdout_confusion_matrix','cv_confusion_matrix')}, indent=2))
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate logistic regression on the local Iris CSV')
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--data', type=Path, default=root / 'data' / 'iris.data.csv')
    parser.add_argument('--out', type=Path, default=root / 'results')
    args = parser.parse_args()
    evaluate(load_data(args.data), args.out)
