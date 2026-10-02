"""
Pipeline : text cleaning -> TF-IDF -> Naive Bayes / Logistic Regression (best one kept)
Outputs  : accuracy, classification report, confusion matrix (printed + saved as PNG)
Interface: Flask web page + JSON API

Install : pip install -r requirements.txt
Train   : python app.py --train
Web app : python app.py            then open http://127.0.0.1:5000
Terminal: python app.py --predict "Congratulations! You have won N500,000. Claim now"
"""
import argparse
import os
import re
import string

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "spam.csv")
MODEL_FILE = os.path.join(BASE_DIR, "spam_model.joblib")
MATRIX_FILE = os.path.join(BASE_DIR, "static", "confusion_matrix.png")


# ----------------------------------------------------------------------------
# 1. TEXT PREPROCESSING
# ----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, normalise links/money/numbers, strip punctuation and stop words."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)           # links
    text = re.sub(r"[₦$£€]\s?\d[\d,\.]*", " moneytoken ", text)       # currency amounts
    text = re.sub(r"\b\d{5,}\b", " longnumtoken ", text)              # phone numbers / short codes
    text = re.sub(r"\d+", " num ", text)                              # other numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # punctuation
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)


# ----------------------------------------------------------------------------
# 2. DATA LOADING
# ----------------------------------------------------------------------------
def builtin_sample() -> pd.DataFrame:
    """Tiny fallback dataset, used only if spam.csv is missing."""
    spam = [
        "Congratulations! You have won N500,000. Claim now",
        "WINNER!! You have been selected for a free prize. Call 08012345678 now",
        "Urgent! Your account has been suspended. Click http://bit.ly/verify to restore",
        "Free entry in 2 a weekly competition to win FA Cup tickets. Text WIN to 87121",
        "Dear customer your ATM card is blocked. Send your PIN to unblock",
        "Get a loan of N200,000 within 5 minutes without collateral. Apply now",
        "Your BVN needs update. Click this link immediately or lose access",
        "Final notice: claim your lottery winnings of 2,500,000 before it expires",
    ]
    ham = [
        "Hey, are we still meeting at 5pm today?",
        "Please send me the lecture notes when you get a chance",
        "I will be home late, don't wait for me for dinner",
        "Happy birthday! Hope you have a wonderful day",
        "Your SIWES logbook has to be signed by Friday",
        "The meeting has been moved to 10am tomorrow at the main office",
        "Don't forget to bring your laptop and charger tomorrow",
        "See you at the lab at 2, I'll bring the sensors",
    ]
    return pd.DataFrame({"label": ["spam"] * len(spam) + ["ham"] * len(ham),
                         "text": spam + ham})


def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE, encoding="utf-8")
        except UnicodeDecodeError:  # the original Kaggle file is latin-1
            df = pd.read_csv(DATA_FILE, encoding="latin-1")
        if {"v1", "v2"}.issubset(df.columns):
            df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
        elif {"label", "text"}.issubset(df.columns):
            df = df[["label", "text"]]
        else:
            df = df.iloc[:, :2]
            df.columns = ["label", "text"]
        print(f"Loaded {len(df)} messages from spam.csv")
    else:
        print("spam.csv not found - using the tiny built-in sample dataset.")
        df = builtin_sample()
    df = df.dropna().drop_duplicates()
    df["label"] = df["label"].str.strip().str.lower()
    return df


# ----------------------------------------------------------------------------
# 3. TRAINING + EVALUATION
# ----------------------------------------------------------------------------
def build_model(kind: str) -> Pipeline:
    if kind == "nb":
        clf = MultinomialNB(alpha=0.1)
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),  # text is cleaned beforehand with clean_text()
        ("clf", clf),
    ])


def train(show_plot: bool = False):
    df = load_data()
    print("\nClass distribution:\n", df["label"].value_counts(), "\n")

    df["clean"] = df["text"].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    best_model, best_acc, best_name = None, -1, ""
    for name, kind in [("Naive Bayes", "nb"), ("Logistic Regression", "lr")]:
        model = build_model(kind)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"{name:<22} accuracy: {acc:.4f}")
        if acc > best_acc:
            best_model, best_acc, best_name = model, acc, name

    print(f"\nBest model: {best_name}")
    preds = best_model.predict(X_test)
    print("\nClassification report:\n", classification_report(y_test, preds, zero_division=0))

    labels = ["ham", "spam"]
    cm = confusion_matrix(y_test, preds, labels=labels)
    print("Confusion matrix (rows = actual, cols = predicted) [ham, spam]:\n", cm)

    try:
        import matplotlib
        if not show_plot:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(MATRIX_FILE), exist_ok=True)
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {best_name}")
        plt.savefig(MATRIX_FILE, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix image -> {MATRIX_FILE}")
        if show_plot:
            plt.show()
        plt.close()
    except ImportError:
        print("(matplotlib not installed - skipping the confusion matrix image)")

    joblib.dump(best_model, MODEL_FILE)
    print(f"Model saved -> {MODEL_FILE}")
    return best_model


def get_model():
    """Load the saved model, or train one if none exists yet."""
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return train()


# ----------------------------------------------------------------------------
# 4. PREDICTION
# ----------------------------------------------------------------------------
def predict_message(model, message: str):
    """Return ('Spam' | 'Not Spam', confidence between 0 and 1)."""
    cleaned = [clean_text(message)]
    label = model.predict(cleaned)[0]
    probs = model.predict_proba(cleaned)[0]
    confidence = float(probs[list(model.classes_).index(label)])
    return ("Spam" if label == "spam" else "Not Spam"), confidence


# ----------------------------------------------------------------------------
# 5. FLASK WEB APP
# ----------------------------------------------------------------------------
app = Flask(__name__)
model = None   # loaded when the app starts (see bottom of file)
history = []   # last few checks, kept in memory


@app.route("/", methods=["GET", "POST"])
def index():
    message, label, conf = "", None, None
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if message:
            label, conf = predict_message(model, message)
            history.insert(0, {"label": label, "text": message})
            del history[5:]
    return render_template("index.html", message=message, label=label, conf=conf,
                           history=history, has_matrix=os.path.exists(MATRIX_FILE))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "message is required"}), 400
    label, conf = predict_message(model, message)
    return jsonify({"prediction": label, "confidence": conf})


# ----------------------------------------------------------------------------
# 6. ENTRY POINT
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMS/Email spam detector")
    parser.add_argument("--train", action="store_true", help="retrain and show confusion matrix")
    parser.add_argument("--predict", metavar="MESSAGE", help="classify one message in the terminal")
    args = parser.parse_args()

    if args.train:
        train(show_plot=True)
    elif args.predict:
        label, conf = predict_message(get_model(), args.predict)
        print(f"{label} ({conf * 100:.1f}% confidence)")
    else:
        model = get_model()
        print("Open http://127.0.0.1:5000 in your browser")
        app.run(debug=False)