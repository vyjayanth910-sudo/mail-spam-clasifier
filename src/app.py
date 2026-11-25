"""Streamlit interface for the email and SMS phishing classifier."""
from pathlib import Path
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import joblib
from textclean import normalize  # noqa: F401: needed for older serialized pipelines

st.set_page_config(page_title="Scam / Phishing Detector", page_icon="🛡️", layout="centered")
MODEL_PATH = ROOT / "models" / "final_v1.joblib"

@st.cache_resource(show_spinner="Loading classifier...")
def load_model():
    # Only load models supplied by a trusted source: joblib files can execute code.
    return joblib.load(MODEL_PATH)

st.title("🛡️ Scam / Phishing Message Detector")
st.write("An educational machine-learning screening tool for suspicious emails and messages.")
st.info("**Important:** A model score is not a verified probability of fraud. This model was trained on historical emails and SMS and can misclassify legitimate alerts, especially modern SMS and Indian banking messages.")

with st.expander("How it works"):
    st.write("Text is normalized, converted into TF-IDF word features, and classified by logistic regression. The model looks for patterns in training examples; it does not verify links, sender identities, or bank accounts.")

message = st.text_area("Paste a message to analyze", height=200, placeholder="Paste an email or SMS here. Remove names, account numbers, OTPs, and other private details first.")
if st.button("Analyze message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Enter a message first.")
    elif len(message) > 30000:
        st.warning("Please shorten the message to 30,000 characters or fewer.")
    elif not MODEL_PATH.is_file():
        st.error("The trained model is missing from models/final_v1.joblib.")
    else:
        try:
            bundle = load_model()
            model = bundle["model"]
            threshold = float(bundle["threshold"])
            score = float(model.predict_proba([message])[0][1])
            prediction = int(score >= threshold)
            st.subheader("Analysis result")
            if prediction:
                st.error("⚠️ Potential spam/phishing patterns detected")
            else:
                st.success("No strong spam/phishing patterns detected by this model")
            st.metric("Model spam/phishing score", f"{score:.1%}")
            st.progress(min(1., max(0., score)))
            st.caption("The model uses its validation-selected decision threshold of {threshold:.2f}. This score is not calibrated as a real-world probability and does not establish whether a message is safe or fraudulent.")
            st.write("**What to do:** Verify unexpected requests independently through the organization's official website or app. Don't share passwords or OTPs. A low score does not guarantee safety.")
        except Exception as exc:
            st.error("Unable to analyze this message. Check the installed dependencies and model file.")
            with st.expander("Technical details"):
                st.code(f"{type(exc).__name__}: {exc}")

st.divider()
st.caption("Educational ML mini-project · TF-IDF + Logistic Regression · Not a security guarantee")
