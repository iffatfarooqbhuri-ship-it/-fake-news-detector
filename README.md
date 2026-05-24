# 🧠 Fake News Detection using LSTM & Deep Learning

A deep learning project that automatically classifies news articles as **Real** or **Fake** using LSTM and Dense neural networks.

Built as part of **SMIT Batch 10 — AI & Data Science** program.

---

## 🎯 What This Project Does

Given a news headline or article, this model predicts whether it is **Real ✅** or **Fake ❌** — with a confidence score.

**Example predictions:**
```
✅ Real | NASA confirms discovery of water on Mars
❌ Fake | Aliens have taken over the White House
✅ Real | Federal Reserve raises interest rates
❌ Fake | Scientists discover drinking bleach cures cancer
```

---

## 🛠️ Technologies Used

- **Python** — core language
- **TensorFlow / Keras** — model building & training
- **NLTK** — text preprocessing (stopwords, lemmatization)
- **Scikit-learn** — train/test split, evaluation metrics
- **Pandas & NumPy** — data handling
- **Matplotlib & Seaborn** — training graphs

---

## 📁 Project Pipeline

| Step | Description |
|------|-------------|
| 1 | Load & explore dataset (`fake_real_news.csv`) |
| 2 | Text cleaning — lowercase, remove punctuation & numbers |
| 3 | Stopword removal & Lemmatization |
| 4 | Tokenization & Padding (vocab size: 10,000 / max length: 200) |
| 5 | Train/Test Split (80/20) |
| 6 | LSTM Model — training for 9 epochs |
| 7 | Dense Model — trained for comparison |
| 8 | Evaluation — Accuracy, Precision, Recall, F1 Score |
| 9 | Visualization — Accuracy & Loss graphs |
| 10 | Prediction function — test on custom headlines |

---

## 🤖 Models Built

### LSTM Model
```
Embedding → LSTM(64) → Dropout(0.5) → Dense(32, ReLU) → Dense(1, Sigmoid)
```

### Dense Model (for comparison)
```
Embedding → GlobalAveragePooling → Dropout(0.3) → Dense(64) → Dense(32) → Dense(1, Sigmoid)
```

Both models trained with **Adam optimizer** and **Binary Crossentropy loss**.

---

## 📊 Results

- Training graphs show **Accuracy vs Epochs** and **Loss vs Epochs** for both models
- LSTM outperforms Dense model on sequential text data
- Evaluation includes full **Classification Report** (Precision, Recall, F1)

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**
2. Upload `fake_real_news.csv` dataset
3. Run all cells in order
4. Use the `predict_news()` function to test your own headlines!

```python
predict_news("The president signed a new bill today in Congress")
# Output: Prediction: Real (confidence: 0.87)
```

---

## 📂 Dataset

- **Source:** Kaggle — Fake and Real News Dataset
- **Columns used:** `text`, `target` (label)
- **Classes:** Fake (0), Real (1)

---

## 👩‍💻 Author
==" Iffat Farooq Bhuri "==

**Iffat Farooq**  
AI & Data Science Student — SMIT Batch 10  
[GitHub](https://github.com/iffatfarooqbhuri-ship-it)
