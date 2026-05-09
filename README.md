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
| Test Accuracy | 86.12% | 87.05% | GRU ✓ |
| Precision | 86.71% | 86.43% | LSTM ✓ |
| Recall | 85.32% | 87.90% | GRU ✓ |
| F1 Score | 86.01% | 87.16% | GRU ✓ |
| Training Time | 2.0m | 1.9m | GRU ✓ |

**Selected Model: GRU — wins 4 out of 5 metrics**

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
- Matplotlib / Seaborn (visualizations)
- Gradio (interactive demo)
- Scikit-learn (evaluation metrics)

---

## Key Findings
1. GRU outperforms LSTM on 4 out of 5 metrics
2. Both models achieve ~87% accuracy on unseen data
3. Main failure cases: sarcasm, irony, mixed sentiment
4. Very short reviews under 20 words are unreliable
5. Model is sometimes more confident when wrong (0.534)
   than when correct (0.506)

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
