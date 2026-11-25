import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from textclean import normalize
from train_v2 import load_enron, load_ceas, MAXLEN

def load_phish(path, src):
    d = pd.read_csv(path)
    t = "subject: " + d["subject"].fillna("") + " " + d["body"].fillna("")
    out = pd.DataFrame({"text": t.str[:MAXLEN], "label": 1, "src": src})
    return out.drop_duplicates("text")

def make_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalize, stop_words="english",
                                  min_df=3, max_features=100_000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

def build(variant):
    parts = [load_enron(), load_ceas(), load_phish("data/Nazario.csv", "nazario")]
    if variant == "b":
        parts.append(load_phish("data/Nigerian_Fraud.csv", "nigerian"))
    return pd.concat(parts, ignore_index=True).drop_duplicates("text")

if __name__ == "__main__":
    for variant, name in (("a", "email_v3a"), ("b", "email_v3b")):
        df = build(variant)
        strat = df["src"] + "_" + df["label"].astype(str)
        tr, te = train_test_split(df, test_size=0.2, stratify=strat, random_state=42)
        model = make_model().fit(tr["text"], tr["label"])
        pred = model.predict(te["text"])
        print(f"=== {name}: sources = {sorted(df.src.unique())}, n = {len(df)}")
        print("overall confusion:\n", confusion_matrix(te["label"], pred))
        for s in sorted(df.src.unique()):
            m = (te["src"] == s).values
            y = te["label"].values[m]; p = pred[m]
            if y.sum():
                print(f"  {s}: scam recall {((p==1)&(y==1)).sum()/y.sum():.1%}")
            if (y == 0).sum():
                print(f"  {s}: ham correct {((p==0)&(y==0)).sum()/(y==0).sum():.1%}")
        joblib.dump(model, f"models/{name}.joblib")
        print()
