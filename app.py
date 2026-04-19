from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import math
import re
import os

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Stopwords
stopwords = {
    "the","is","and","a","to","of","in","on","for",
    "with","that","this","it","as","at","by","an",
    "be","are","was","were","from","or","but",
    "have","has","had","do","does","did","so","if",
    "about","into","over","after","before"
}

model = {}
class_doc = {}
class_total_words = {}

# 🔥 Load model safely (important for deployment)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.txt")

with open(MODEL_PATH) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue

        key, value = parts
        value = int(value)

        if key.endswith("|DOC"):
            label = key.split("|")[0]
            class_doc[label] = value

        elif key.endswith("|TOTAL"):
            label = key.split("|")[0]
            class_total_words[label] = value

        else:
            model[key] = value

total_docs = sum(class_doc.values())

# Vocabulary size
vocab = set(k.split("|",1)[1] for k in model if "|" in k)
V = len(vocab)


# 🔹 Preprocess
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()

    clean = []
    for w in words:
        if len(w) < 2 or len(w) > 15:
            continue
        if w in stopwords:
            continue
        if re.search(r'(.)\1{3,}', w):
            continue
        clean.append(w)

    if not clean:
        clean = ["empty"]

    return clean


# 🔹 Softmax
def softmax(scores):
    max_score = max(scores.values())
    exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
    total = sum(exp_scores.values())
    return {k: exp_scores[k] / total for k in exp_scores}


# 🔹 Prediction
def predict_with_confidence(text):
    words = preprocess(text)
    scores = {}

    for label in class_doc:
        scores[label] = math.log((class_doc[label] + 1) / (total_docs + len(class_doc)))

        total_words = class_total_words.get(label, 1)

        for word in words:
            count = model.get(f"{label}|{word}", 0)
            prob = (count + 1) / (total_words + V)
            scores[label] += math.log(prob)

    probs = softmax(scores)
    predicted = max(probs, key=probs.get)

    return predicted, probs


# 🔥 Serve frontend (THIS FIXES YOUR ISSUE)
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    INDEX_PATH = os.path.join(BASE_DIR, "index.html")
    with open(INDEX_PATH, "r") as f:
        return f.read()


# 🔥 Prediction API
@app.post("/predict")
def classify(data: dict):
    text = data.get("text", "")

    label, probs = predict_with_confidence(text)
    confidence = probs[label]

    # ⚠️ Low confidence handling
    if confidence < 0.5:
        return {
            "label": "Uncertain",
            "message": "Text not enough to classify",
            "confidence": round(confidence, 4),
            "all_scores": {k: round(v, 4) for k, v in probs.items()}
        }

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "all_scores": {k: round(v, 4) for k, v in probs.items()}
    }
