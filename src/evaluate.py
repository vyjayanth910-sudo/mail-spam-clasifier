"""Reproducible evaluation of the supplied v3b model on the six hand-written examples.
These examples are a tiny exploratory sanity check, not an accuracy benchmark.
"""
from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from textclean import normalize  # noqa: F401

def main():
    df = pd.read_csv(ROOT / 'data' / 'indian_test.csv')
    labels = df['label'].str.strip().str.lower().map({'ham': 0, 'scam': 1})
    if labels.isna().any():
        raise ValueError('Unexpected labels in indian_test.csv')
    model = joblib.load(ROOT / 'models' / 'email_v3b.joblib')
    pred = model.predict(df['text'].fillna('').astype(str))
    scores = model.predict_proba(df['text'].fillna('').astype(str))[:, 1]
    result = df.copy()
    result['prediction'] = ['scam' if x else 'ham' for x in pred]
    result['model_score'] = scores.round(3)
    print(result.to_string(index=False))
    print('\nConfusion matrix (rows actual, columns predicted; order ham, scam):')
    print(confusion_matrix(labels, pred, labels=[0, 1]))
    print(classification_report(labels, pred, labels=[0, 1], target_names=['ham', 'scam'], zero_division=0))
    print('CAUTION: Six examples are too few to estimate real-world accuracy.')

if __name__ == '__main__':
    main()
