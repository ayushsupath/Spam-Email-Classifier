# 📧 Spam Email / SMS Classifier

A Natural Language Processing (NLP) project to classify messages as **Spam** or **Ham (Not Spam)** using TF-IDF vectorization and machine learning classifiers.

---

## 📌 Objective

To build a text classification system that can automatically detect spam messages using NLP techniques — cleaning raw text, extracting features with TF-IDF, and training ML models to classify messages accurately.

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Name** | UCI SMS Spam Collection Dataset |
| **Source** | Auto-loaded from GitHub (raw CSV) |
| **Total Samples** | 5,572 messages |
| **Classes** | Ham (4,825) and Spam (747) |
| **Format** | Tab-separated: label, message |
| **Class Balance** | Imbalanced (~87% Ham, ~13% Spam) |

> **Note:** If internet is unavailable, the script falls back to a built-in representative sample dataset automatically.

---

## 🧠 Algorithms Used

Two classifiers are trained and compared:

### 1. Naive Bayes (MultinomialNB) — Best Model ✅
A probabilistic classifier based on Bayes' theorem. Works exceptionally well for text classification tasks.

| Hyperparameter | Value |
|---|---|
| `alpha` (Laplace smoothing) | 0.1 |

### 2. Logistic Regression
A linear classifier that models the probability of a class using the logistic (sigmoid) function.

| Hyperparameter | Value |
|---|---|
| `max_iter` | 1000 |
| `random_state` | 42 |

---

## ⚙️ Tech Stack

| Library | Purpose |
|---|---|
| `scikit-learn` | TF-IDF, model training, evaluation |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting charts |
| `seaborn` | Confusion matrix heatmap |
| `re`, `string` | Text cleaning and preprocessing |

---

## 🔄 Project Workflow

```
Load Dataset → EDA → Text Preprocessing → TF-IDF Vectorization → Model Training → Evaluation → Live Prediction
```

1. **Load Dataset** — Load UCI SMS Spam dataset from URL (with fallback)
2. **EDA** — Class distribution, message length histogram, word count boxplot
3. **Text Preprocessing** — Lowercase, remove numbers, remove punctuation, strip whitespace
4. **Feature Extraction** — TF-IDF Vectorizer (5000 features, unigrams + bigrams)
5. **Model Training** — Naive Bayes and Logistic Regression trained and compared
6. **Evaluation** — Accuracy, Classification Report, Confusion Matrix, ROC Curve
7. **Live Prediction** — Classify new messages in real-time

---

## 🧹 Text Preprocessing Steps

| Step | Example |
|---|---|
| Original | `"WIN £1000 CASH!! Call 09061 now!"` |
| Lowercase | `"win £1000 cash!! call 09061 now!"` |
| Remove Numbers | `"win £ cash!! call  now!"` |
| Remove Punctuation | `"win  cash call  now"` |
| Strip Whitespace | `"win cash call now"` |

---

## 📊 Results

### Model Comparison

| Model | Accuracy | Ham Precision | Spam Recall |
|---|---|---|---|
| **Naive Bayes** | **98.21%** | 0.98 | 0.89 |
| Logistic Regression | 96.59% | 0.96 | 0.74 |

### Best Model — Naive Bayes

| Metric | Ham | Spam |
|---|---|---|
| **Precision** | 0.98 | 0.98 |
| **Recall** | 1.00 | 0.89 |
| **F1-Score** | 0.99 | 0.93 |

---

## 🔴 Top Spam Keywords Identified

```
free, txt, stop, claim, text, ur, mobile, reply, prize, won
```

## 🟢 Top Ham Keywords Identified

```
im, ok, ill, come, got, just, dont, good, know, call
```

---

## 📁 Output Files Generated

| File | Description |
|---|---|
| `spam_eda.png` | Class distribution pie chart, message length histogram, word count boxplot |
| `spam_confusion_matrix.png` | Confusion matrix heatmap for best model |
| `spam_roc_curve.png` | ROC curves comparing both models with AUC scores |

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

### 2. Run the Script
```bash
python spam_classifier.py
```

Dataset is automatically fetched from the internet. If unavailable, a built-in fallback dataset is used.

---

## 🔍 Key Findings

- **Naive Bayes** outperforms Logistic Regression for this task — a common result in text classification
- Spam messages are significantly **longer** and use more **urgent/promotional** language
- **TF-IDF with bigrams** captures two-word patterns like "free entry", "win cash", "call now"
- The model achieves **98.21% accuracy** on 1,115 test samples
- Class imbalance (~13% spam) is handled via stratified train-test split

---

## 📚 References

- Almeida, T.A., Gómez Hidalgo, J.M. (2011). *SMS Spam Collection Dataset*. UCI Machine Learning Repository
- [Scikit-learn TF-IDF Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Multinomial Naive Bayes — sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
