# IMDB Sentiment Analysis
### Binary Sentiment Classification using BiLSTM and BiGRU

A complete end-to-end deep learning project for sentiment 
analysis on 50,000 IMDB movie reviews using sequence models.

---

## Project Overview
This project builds and compares two Bidirectional sequence 
models — LSTM and GRU — for binary sentiment classification.
Starting from raw text we implement a complete NLP pipeline
from preprocessing to an interactive Gradio demo.

---

## Results

| Metric | LSTM | GRU | Winner |
|--------|------|-----|--------|
| Test Accuracy | 86.51% | 86.95% | GRU ✓ |
| Precision | 85.49% | 85.29% | LSTM ✓ |
| Recall | 87.94% | 89.30% | GRU ✓ |
| F1 Score | 86.70% | 87.25% | GRU ✓ |
| Training Time | 1.9m | 2.2m | LSTM ✓ |

**Selected Model: GRU — wins 3 out of 5 metrics**

---

## Pipeline

Raw Text → Clean → Tokenize → Pad → BiGRU → Predict

1. Text cleaning (HTML removal, lowercase, punctuation)
2. Tokenization (top 10,000 words vocabulary)
3. Padding (MAX_LEN=200)
4. Bidirectional GRU model
5. Binary classification (sigmoid output)

---

## Model Architecture

Embedding(10000, 128)
↓
SpatialDropout1D(0.3)
↓
Bidirectional GRU(64, return_sequences=True)
↓
Bidirectional GRU(32, return_sequences=False)
↓
Dropout(0.3)
↓
Dense(64, relu)
↓
Dense(1, sigmoid)

---

## Dataset
- **Source:** IMDB Dataset of 50K Movie Reviews (Kaggle)
- **Size:** 50,000 reviews (25,000 positive, 25,000 negative)
- **Split:** 80% train, 20% test
- **Balance:** Perfectly balanced dataset

---

## Technologies
- Python 3.12
- TensorFlow / Keras
- Pandas, NumPy
- Plotly (interactive visualizations)
- Gradio (interactive demo)
- Scikit-learn (evaluation metrics)

---

## Key Findings
1. Both models achieve ~87% accuracy on unseen data
2. GRU converges faster and generalizes slightly better
3. Main failure cases: sarcasm, irony, mixed sentiment
4. Very short reviews under 20 words are unreliable
5. Model is sometimes more confident when wrong (0.563)
   than when correct (0.514)

---

## Benchmark Context

| Approach | Accuracy |
|----------|----------|
| Simple RNN (baseline) | ~80-82% |
| Our BiLSTM / BiGRU | ~87% ✓ |
| LSTM + Pretrained Embeddings | ~90-92% |
| Fine-tuned BERT | ~93-95% |

---

## Kaggle Notebook
[View full notebook on Kaggle](https://www.kaggle.com/code/manarhamdy/imdb-sentiment-analysis-lstm-vs-gru)

---

## References
- Maas et al. (2011) — Learning Word Vectors for Sentiment Analysis
- Hochreiter & Schmidhuber (1997) — Long Short-Term Memory
- Cho et al. (2014) — Learning Phrase Representations using RNN Encoder-Decoder
- Vaswani et al. (2017) — Attention Is All You Need
- Chollet (2021) — Deep Learning with Python
