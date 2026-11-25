
from pathlib import Path
import pandas as pd

DATA = Path("data")

def load_messages(filename):
    path = DATA / filename

    if filename == "SMSSpamCollection":
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["label", "text"],
            on_bad_lines="skip"
        )
        return df["text"]

    df = pd.read_csv(path, low_memory=False)

    if "text" in df.columns:
        return df["text"]

    if "message" in df.columns:
        return df["message"]

    if "subject" in df.columns and "body" in df.columns:
        return (
            df["subject"].fillna("").astype(str)
            + " "
            + df["body"].fillna("").astype(str)
        )

    return pd.Series(dtype=str)


files = [
    "CEAS_08.csv",
    "emails.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "scam_examples.csv",
    "SMSSpamCollection",
]

messages = {}

for filename in files:
    s = load_messages(filename)
    s = s.fillna("").astype(str).str.lower().str.strip()
    s = s[s != ""]
    messages[filename] = set(s)


print("CROSS-DATASET OVERLAP")
print("=" * 45)

for i, first in enumerate(files):
    for second in files[i + 1:]:
        overlap = messages[first] & messages[second]
        print(f"{first} <-> {second}: {len(overlap)}")

