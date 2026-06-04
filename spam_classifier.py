# ============================================================
#   PROJECT 2: Spam Email Classifier
#   Dataset  : UCI SMS Spam Collection (loaded inline)
#   Algorithm: Naive Bayes + TF-IDF
#   Author   : ML Assignment
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)

# ── 1. Load Dataset ──────────────────────────────────────────
# Using the classic SMS Spam Collection dataset from UCI
URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

print("=" * 60)
print("         SPAM EMAIL / SMS CLASSIFIER PROJECT")
print("=" * 60)

try:
    df = pd.read_csv(URL, sep="\t", header=None, names=["label", "message"])
    print("\n✅ Dataset loaded from URL")
except Exception:
    # Fallback: generate representative sample dataset
    print("\n⚠️  URL failed – using built-in sample dataset")
    spam_msgs = [
        "WINNER!! You've been selected for a $1000 prize. Call now!",
        "Free entry in 2 weekly competitions! Text WIN to 87121",
        "Congratulations! You won a FREE vacation. Claim now!",
        "Urgent! Your account will be suspended. Verify now at link",
        "You have won £1,000 cash or a prize. To claim call 09061743806",
        "SIX chances to win CASH! From 100 to 20,000 pounds txt CSH11",
        "URGENT: You have won a 1 week FREE membership. Call 09050002311",
        "Awarded a £2000 prize GUARANTEED. Call 09061790121 now",
        "Win a £1000 cash prize or a prize worth £5000",
        "Free ringtone! Reply with your name and DOB to claim",
        "Your loan is APPROVED. Get £5000 now. Call 08001137397",
        "Claim your free prize. Send CLAIM to 85023 now!",
        "You are a winner! Click here to get your prize today",
        "Amazing offer! Get 50% off. Limited time. Buy now!",
        "Congratulations! ur mobile number has been awarded £900 prize",
    ] * 20

    ham_msgs = [
        "Hey, are you coming to the party tonight?",
        "Can you pick up some milk on your way home?",
        "The meeting has been rescheduled to 3pm tomorrow",
        "Thanks for your help yesterday, really appreciated it",
        "I'll be there in 10 minutes, just stuck in traffic",
        "Did you see the game last night? It was amazing!",
        "Happy birthday! Hope you have a wonderful day",
        "Let me know when you're free for lunch this week",
        "The project report is due on Friday, let me know if you need help",
        "I forgot my umbrella, can you bring it to school?",
        "What time does the movie start? I'll book tickets",
        "The doctor appointment is at 2pm on Thursday",
        "Call me when you get home so I know you're safe",
        "I've attached the file you asked for. Let me know if it's correct",
        "Sorry, I can't make it today. Can we reschedule?",
    ] * 23

    df = pd.DataFrame({
        "label"  : ["spam"] * len(spam_msgs) + ["ham"] * len(ham_msgs),
        "message": spam_msgs + ham_msgs
    }).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n📊 Dataset Shape  : {df.shape}")
print(f"📋 Label counts   :\n{df['label'].value_counts().to_string()}")
print(f"\n🔍 Sample messages:")
print(df.sample(5, random_state=1)[["label", "message"]].to_string(index=False))

# ── 2. EDA ───────────────────────────────────────────────────
df["msg_length"]  = df["message"].apply(len)
df["word_count"]  = df["message"].apply(lambda x: len(x.split()))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Spam Classifier – Exploratory Data Analysis",
             fontsize=14, fontweight="bold")

# Class distribution
counts = df["label"].value_counts()
axes[0].pie(counts, labels=counts.index, autopct="%1.1f%%",
            colors=["#e74c3c", "#2ecc71"], startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[0].set_title("Class Distribution", fontweight="bold")

# Message length distribution
for label, color in [("spam", "#e74c3c"), ("ham", "#2ecc71")]:
    subset = df[df["label"] == label]["msg_length"]
    axes[1].hist(subset, alpha=0.7, label=label, color=color, bins=30, edgecolor="white")
axes[1].set_title("Message Length Distribution", fontweight="bold")
axes[1].set_xlabel("Character Count")
axes[1].set_ylabel("Frequency")
axes[1].legend()
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

# Word count box plot
df.boxplot(column="word_count", by="label", ax=axes[2],
           notch=False, patch_artist=True,
           boxprops=dict(facecolor="#3498db", alpha=0.6))
axes[2].set_title("Word Count by Class", fontweight="bold")
axes[2].set_xlabel("Label")
axes[2].set_ylabel("Word Count")
plt.suptitle("")

plt.tight_layout()
plt.savefig("spam_eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ EDA chart saved → spam_eda.png")

# ── 3. Text Preprocessing ────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)               # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()      # remove extra spaces
    return text

df["clean_msg"] = df["message"].apply(preprocess)

print("\n📝 Text preprocessing done:")
print(f"  Original : {df['message'].iloc[0][:60]}...")
print(f"  Cleaned  : {df['clean_msg'].iloc[0][:60]}...")

# ── 4. Feature Extraction (TF-IDF) ───────────────────────────
X = df["clean_msg"]
y = (df["label"] == "spam").astype(int)   # spam=1, ham=0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"\n🔀 Train size      : {X_train_tfidf.shape[0]} samples")
print(f"🔀 Test size       : {X_test_tfidf.shape[0]} samples")
print(f"📐 TF-IDF features : {X_train_tfidf.shape[1]}")

# ── 5. Model Training (2 models compared) ────────────────────
models = {
    "Naive Bayes"        : MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

results = {}
for name, clf in models.items():
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
    results[name] = {
        "model" : clf,
        "y_pred": y_pred,
        "acc"   : accuracy_score(y_test, y_pred),
    }

print("\n" + "=" * 60)
print("                  MODEL COMPARISON")
print("=" * 60)
for name, r in results.items():
    print(f"\n🤖 {name}")
    print(f"   Accuracy : {r['acc'] * 100:.2f}%")
    report = classification_report(y_test, r["y_pred"], target_names=["Ham", "Spam"])
    for line in report.split("\n"):
        print("   " + line)

# Best model
best_name  = max(results, key=lambda k: results[k]["acc"])
best_model = results[best_name]["model"]
best_pred  = results[best_name]["y_pred"]
print(f"🏆 Best Model : {best_name} ({results[best_name]['acc']*100:.2f}%)")

# ── 6. Confusion Matrix ──────────────────────────────────────
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"],
            linewidths=1, linecolor="white")
plt.title(f"Confusion Matrix – {best_name}", fontsize=13, fontweight="bold", pad=10)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("spam_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Confusion matrix → spam_confusion_matrix.png")

# ── 7. ROC Curve ─────────────────────────────────────────────
plt.figure(figsize=(7, 5))
for name, clf in models.items():
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_test_tfidf)[:, 1]
    else:
        proba = clf.decision_function(X_test_tfidf)
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
plt.xlabel("False Positive Rate", fontsize=11)
plt.ylabel("True Positive Rate", fontsize=11)
plt.title("ROC Curve – Spam Classifier", fontsize=13, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("spam_roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ ROC curve        → spam_roc_curve.png")

# ── 8. Top Spam Keywords (Naive Bayes) ───────────────────────
nb_model = models["Naive Bayes"]
feature_names = np.array(tfidf.get_feature_names_out())
top_spam = feature_names[np.argsort(nb_model.feature_log_prob_[1])[-20:]][::-1]
top_ham  = feature_names[np.argsort(nb_model.feature_log_prob_[0])[-20:]][::-1]

print(f"\n🔴 Top Spam words : {', '.join(top_spam[:10])}")
print(f"🟢 Top Ham words  : {', '.join(top_ham[:10])}")

# ── 9. Live Prediction ───────────────────────────────────────
def predict_message(text):
    cleaned = preprocess(text)
    vec     = tfidf.transform([cleaned])
    pred    = best_model.predict(vec)[0]
    prob    = best_model.predict_proba(vec)[0]
    label   = "🔴 SPAM" if pred == 1 else "🟢 HAM (Not Spam)"
    conf    = max(prob) * 100
    return label, conf

print("\n" + "=" * 60)
print("           LIVE PREDICTION DEMO")
print("=" * 60)
test_msgs = [
    "Congratulations! You've won a £1000 prize. Call now to claim!",
    "Hey man, are you coming to dinner tonight?",
    "FREE entry! Win cash prizes worth £5000. Text WIN to 87121",
    "The meeting is postponed to Monday. Please update your calendar.",
]
for msg in test_msgs:
    label, conf = predict_message(msg)
    print(f"\n  Input  : {msg[:55]}...")
    print(f"  Result : {label}  (Confidence: {conf:.1f}%)")

print("\n✅ Project 2 complete!\n")
