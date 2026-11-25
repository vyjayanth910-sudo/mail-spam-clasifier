import joblib

# Load our saved model
model = joblib.load("models/scam-detector.joblib")

# Try some messages
messages = [
    "Congratulations! You won a free prize. Call now!",
    "Hey bro, are we meeting after college?",
    "URGENT! Verify your bank account immediately.",
    "Your bank account is blocked. Click this link to verify now.",
    "WINNER! Claim your free mobile prize today.",
    "Your assignment is due tomorrow. Please submit it.",
    "Sir, I need AXIS BANK account no and bank address.",
"You have registered Sinco as Payee. Log in at icicibank.com and enter URN <#> to confirm. Beware of frauds. Do NOT share or disclose URN to anyone.",
"Your account has been credited with 500 FREE Text Messages. To activate, just txt the word: CREDIT to No: 80488 T&Cs www.80488.biz"
]

# Predict each message
predictions = model.predict(messages)
probabilities = model.predict_proba(messages)

for message, prediction, probs in zip(messages, predictions, probabilities):
    print(f"\nMessage: {message}")
    print(f"Prediction: {prediction}")
    print(f"Ham probability:  {probs[0]:.2%}")
    print(f"Spam probability: {probs[1]:.2%}")
# Inspect words the model associates with spam
vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

feature_names = vectorizer.get_feature_names_out()
spam_weights = classifier.coef_[0]

top_spam_indices = spam_weights.argsort()[-15:][::-1]

print("\nTop words associated with spam:")
for index in top_spam_indices:
    print(f"{feature_names[index]}: {spam_weights[index]:.3f}")