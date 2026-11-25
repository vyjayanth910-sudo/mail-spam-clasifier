from prepare_data import load_data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = load_data()

X = df["message"]
y = df["label"]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create the ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 5. Train the model
model.fit(X_train, y_train)

# 6. Evaluate on unseen test messages
y_pred = model.predict(X_test)

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=["ham", "spam"]))

print("\nClassification report:")
print(classification_report(y_test, y_pred))
joblib.dump(model, "models/scam-detector.joblib")