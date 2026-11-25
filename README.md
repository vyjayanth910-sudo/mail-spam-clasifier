# Scam / Phishing Message Detector

A Python + Streamlit machine-learning mini-project that screens pasted text for patterns associated with spam and phishing. **It is an educational classifier, not a verification service.**

## Features
- Streamlit interface for pasting an email or SMS
- Spam/phishing versus non-flagged classification
- Model score and plain-language safety guidance
- Reproducible evaluation script for a small hand-written Indian-message test set

## How it works
The main app uses `models/email_v3b.joblib`, a scikit-learn pipeline with a custom text normalizer, TF-IDF vectorizer, and balanced logistic regression classifier. Text normalization lowercases text, replaces URLs and digits with placeholder tokens, and strips nonalphabetic characters. The classifier uses a 0.50 threshold.

Training code: `src/train_v3.py` (uses historical Enron/CEAS/Nazario/Nigerian Fraud datasets). Earlier SMS experiments are in `src/train_model.py`. The app uses the **email_v3b** model, not the earlier SMS-only model.

## Installation and running
Use Python 3.10–3.12 with compatible packages. From the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run src/app.py
```

If Python 3.12 is unavailable, use another supported version and create the environment using its executable. If PowerShell blocks activation, you can call `.venv\Scripts\python.exe` directly.

## Evaluation

```powershell
python src/evaluate.py
```

The six examples in `data/indian_test.csv` are **exploratory only**. To reproduce the historical email training/test split, run `python src/train_v3.py` from the project root. Training overwrites the existing v3a/v3b model files; keep backups if needed. The script's random held-out results should not be treated as performance on new Indian messages or novel scams. A rigorous next evaluation would use a larger, independently labeled, temporally newer, source-separated test set, with false-positive rate, precision, recall, F1, and calibration checks.

## Folder structure
- `src/app.py`: web UI
- `src/textclean.py`: shared preprocessing
- `src/train_v3.py`: historical email model training
- `src/evaluate.py`: exploratory small-sample evaluation
- `models/email_v3b.joblib`: model used by app
- `data/`: historical training data and small evaluation examples

## Limitations and safety
- False positives and false negatives occur; legitimate OTP and banking alerts can be flagged.
- Model score is **not** a calibrated probability of fraud.
- This model does not open URLs, check domain reputation, verify sender identity, or query banks.
- Training data includes older email sources and may not represent modern SMS/WhatsApp or Indian financial scams.
- Remove private information before pasting messages. This app does not intentionally save pasted text; deployment platforms may have their own logging practices.
- Only load trusted `.joblib` files; untrusted serialized Python objects may execute code when loaded.

## College demonstration
Explain the pipeline as: dataset → cleaning → TF-IDF → logistic regression → saved pipeline → Streamlit prediction. Show both a suspicious example and a legitimate message, and explain why false positives matter. Avoid claiming the model guarantees fraud detection.

## Observed exploratory result (supplied model)
Running `src/evaluate.py` on the six supplied `indian_test.csv` examples flagged **all six as scam**: 3/3 suspicious examples flagged, but also 3/3 legitimate examples incorrectly flagged. This is a critical false-positive issue; the sample is too small for a reliable generalization estimate. **Do not deploy this model for security decisions without further work.**

### Model compatibility
The bundled joblib file reported that it was serialized using scikit-learn **1.9.1**; loading it under scikit-learn **1.8.0** produced a version mismatch warning during this review. For best reproducibility, use the same scikit-learn version that created the model if available, or retrain and save the pipeline using your installed version. Never load untrusted joblib files.

## Final model update (2026-11-22)

The app now loads `models/final_v1.joblib`, containing a retrained TF-IDF + logistic regression model and its validation-selected threshold. The earlier models remain available for comparison. To reproduce training from the project root, run `python src/train_final.py`.

Training draws from Enron, CEAS, Nazario, Nigerian Fraud, and the SMS Spam Collection. It samples up to 2,500 records per source/label group and deduplicates normalized text before splitting into train/validation/test. The decision threshold is selected on validation data, not on the test set. Metrics and six manually labeled Indian examples are in `reports/evaluation.json`.

**Important limitations:** The test set comes from the same underlying source collections as training, not a fully independent modern Indian scam dataset. Six Indian examples are too few to establish real-world performance. The model still flags a legitimate bank debit alert as suspicious. Scores are not calibrated fraud probabilities; users must independently verify messages. Do not paste private account details or OTPs.

The app requires `streamlit`, which can be installed with `python -m pip install -r requirements.txt`. Run with `python -m streamlit run src/app.py`. Models serialized with joblib should only be loaded from trusted sources, and dependency versions should be kept compatible.
