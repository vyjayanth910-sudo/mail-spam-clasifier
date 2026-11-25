
import pandas as pd

# Load our dataset
df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    names=["label", "message"]
)

# Display the first 5 messages
print(df.head())

# Check how many messages we have
print("\nDataset shape:", df.shape)

# Count spam and ham messages
print("\nLabel distribution:")
print(df["label"].value_counts())
print("\nSample spam messages:")
print(df[df["label"] == "spam"]["message"].head(10).to_string(index=False))

from sklearn.model_selection import train_test_split

# Inputs (messages) and correct answers (labels)
X = df["message"]
y = df["label"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())
# Examine spam examples that the model learned from
spam_messages = df[df["label"] == "spam"]["message"]

print("\nFirst 20 spam messages in the dataset:")
for i, message in enumerate(spam_messages.head(20), start=1):
    print(f"{i}. {message}")
# Search existing messages for banking-related terms
keywords = [
    "bank", "account", "password",
    "verify", "blocked", "security",
    "otp", "credit card"
]

print("\nMessages containing banking-related keywords:")

matches = df[
    df["message"].str.contains(
        "|".join(keywords),
        case=False,
        na=False,
        regex=True
    )
]

if matches.empty:
    print("No matching messages found.")
else:
    print(matches[["label", "message"]].head(30).to_string(index=False))
    print("\nTotal matching messages:", len(matches))