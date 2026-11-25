
import pandas as pd

ORIGINAL_DATA_PATH = "data/SMSSpamCollection"
SCAM_DATA_PATH = "data/scam_examples.csv"


def load_data():
    original_df = pd.read_csv(
        ORIGINAL_DATA_PATH,
        sep="\t",
        names=["label", "message"]
    )

    scam_df = pd.read_csv(SCAM_DATA_PATH)

    # Keep the original ham/spam labels
    # and add our new scam examples
    df = pd.concat(
        [original_df, scam_df],
        ignore_index=True
    )

    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].str.strip().str.lower()
    df["message"] = df["message"].astype(str)

    return df


if __name__ == "__main__":
    df = load_data()

    print("Dataset shape:", df.shape)
    print("\nLabel distribution:")
    print(df["label"].value_counts())
