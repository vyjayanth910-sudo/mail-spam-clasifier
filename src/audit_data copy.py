import joblib
import numpy as np
from textclean import normalize  # needed so joblib can load the saved pipeline

model = joblib.load("models/email_v3a.joblib")

vec = model.named_steps["tfidf"]
clf = model.named_steps["classifier"]
names = np.array(vec.get_feature_names_out())
idx = np.argsort(clf.coef_[0])

print("most ham:", list(names[idx[:30]]))
print("most scam:", list(names[idx[-30:]]))

probe = [
    "12 months of Spotify Premium Standard at ₹799",
    "Your SBI account will be blocked today. Update KYC at bit.ly/xyz",
    "Rs 450 debited from A/c XX1234 at Swiggy. Avl bal Rs 5,210",
    "Your OTP is 482913. Do not share it with anyone.",
]
for t, p in zip(probe, model.predict_proba(probe)[:, 1]):
    print(f"{p:.1%}  {t}")