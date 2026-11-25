
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Load the test messages
df = pd.read_csv("data/indian_test.csv")

# Convert text labels into numbers: scam = 1, ham = 0
label_map = {"scam": 1, "ham": 0}
df["label"] = df["label"].str.strip().str.lower().map(label_map)

if df["label"].isna().any():
    raise ValueError("Labels must be 'scam' or 'ham'.")

# Load your existing model
model = joblib.load("models/email_v3b.joblib")

# Predict without training
predictions = model.predict(df["text"].fillna("").astype(str))
probabilities = model.predict_proba(
    df["text"].fillna("").astype(str)
)

df["ham_score"] = probabilities[:, 0].round(3)
df["scam_score"] = probabilities[:, 1].round(3)


print("\nPREDICTIONS")
df["predicted"] = predictions
df["predicted_label"] = df["predicted"].map({0: "ham", 1: "scam"})
print(df[[
    "text", "label", "predicted_label",
    "ham_score", "scam_score"
]].to_string(index=False))

print("\nCONFUSION MATRIX")
print("Rows = actual [ham, scam]; columns = predicted [ham, scam]")
print(confusion_matrix(df["label"], predictions, labels=[0, 1]))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    df["label"],
    predictions,
    labels=[0, 1],
    target_names=["ham", "scam"],
    zero_division=0
))

