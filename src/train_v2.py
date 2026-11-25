import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from textclean import normalize

MAXLEN = 2000  # cut long emails so length itself isn't a shortcut

def load_enron():
    d = pd.read_csv("data/emails.csv").drop_duplicates("text")
    return pd.DataFrame({"text": d["text"].str[:MAXLEN], "label": d["spam"], "src": "enron"})

def load_ceas():
    d = pd.read_csv("data/CEAS_08.csv")
    t = "subject: " + d["subject"].fillna("") + " " + d["body"].fillna("")
    out = pd.DataFrame({"text": t.str[:MAXLEN], "label": d["label"], "src": "ceas"})
    return out.drop_duplicates("text")

def make_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalize, stop_words="english",
                                  min_df=3, max_features=100_000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

if __name__ == "__main__":
    df = pd.concat([load_enron(), load_ceas()], ignore_index=True).drop_duplicates("text")
    strat = df["src"] + "_" + df["label"].astype(str)
    tr, te = train_test_split(df, test_size=0.2, stratify=strat, random_state=42)

    model = make_model().fit(tr["text"], tr["label"])
    pred = model.predict(te["text"])
    print("Merged in-distribution test:")
    print(confusion_matrix(te["label"], pred))
    print(classification_report(te["label"], pred, digits=3))
    for s in ("enron", "ceas"):
        m = te["src"] == s
        r = ((pred[m] == 1) & (te["label"][m] == 1)).sum() / (te["label"][m] == 1).sum()
        print(f"  scam recall on {s} part: {r:.1%}")

    joblib.dump(model, "models/email_v2.joblib")
    print("saved models/email_v2.joblib")

    # cross-dataset check: train on one source only, test on the other
    print("\nCross-dataset generalization (scam recall / ham specificity):")
    for a, b in (("ceas", "enron"), ("enron", "ceas")):
        A, B = df[df.src == a], df[df.src == b]
        m = make_model().fit(A["text"], A["label"])
        p = m.predict(B["text"])
        rec = ((p == 1) & (B["label"] == 1)).sum() / (B["label"] == 1).sum()
        spec = ((p == 0) & (B["label"] == 0)).sum() / (B["label"] == 0).sum()
        print(f"  train {a} -> test {b}: scam recall {rec:.1%}, ham correct {spec:.1%}")
