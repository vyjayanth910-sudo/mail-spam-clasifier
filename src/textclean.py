import re

URL = re.compile(r"(https?://\S+|www\s?\.\s?\S+|\b\S+\s?\.\s?(com|net|org|in|co)\b\S*)", re.I)

def normalize(text: str) -> str:
    """Same cleaning for training data and for anything the user pastes."""
    t = str(text).lower()
    t = re.sub(r"^\s*subject\s*:", " ", t)
    t = URL.sub(" urltoken ", t)
    t = re.sub(r"\d+", " numtoken ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()
