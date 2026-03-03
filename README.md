# Analisis Sentimen - Review BPJS Mobile
 
![Python](https://img.shields.io/badge/Python-3.12+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Project Overview

**Sentiment Analysis - BPJS Mobile Reviews** adalah proyek yang mengklasifikasikan sentimen ulasan pengguna aplikasi BPJS Mobile dari Google Play Store. Proyek ini menggunakan pendekatan deep learning dengan arsitektur Bidirectional LSTM (BiLSTM) dan Natural Language Processing untuk menganalisis sentimen dalam bahasa Indonesia.

### 🎯 Tujuan Utama
- Mengklasifikasikan ulasan menjadi 3 kategori sentimen: Positif, Netral, dan Negatif
- Membantu memahami persepsi pengguna terhadap aplikasi BPJS Mobile
- Mengidentifikasi area yang perlu ditingkatkan berdasarkan sentimen negatif

---

## 🛠️ Installation and Setup

### Codes and Resources Used

| Deskripsi | Detail |
|-----------|--------|
| **Editor** | Visual Studio Code |
| **Python Version** | 3.12+ |
| **Environment Manager** | uv (atau pip) |
| **Virtual Environment** | .venv (opsional) |

### Python Packages Used

#### General Purpose
```
emoji>=2.15.0
regex>=2026.2.19
requests>=2.32.5
unidecode>=1.4.0
tqdm>=4.67.3
```

#### Data Manipulation & Processing
```
pandas>=3.0.1
numpy>=1.24.0
```

#### Data Visualization
```
matplotlib>=3.10.8
seaborn>=0.13.2
wordcloud>=1.9.6
```

#### Natural Language Processing (NLP)
```
nltk>=3.9.3
sastrawi>=1.0.1
spacy>=3.8.11
stanza>=1.11.0
transformers>=5.2.0
nlpaug>=1.1.11
gensim>=4.4.0
fasttext-wheel>=0.9.2
```

#### Machine Learning & Deep Learning
```
scikit-learn>=1.8.0
tensorflow>=2.20.0
keras>=3.13.2
keras-tuner>=1.4.8
google-play-scraper>=1.2.7
```

### Quick Setup

```bash
# 1. Navigate to project directory
cd "d:\document\coding\project\Sentimen Analisis"

# 2. Create virtual environment (optional)
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -e .

# or using uv
uv sync
```

---

## 📊 Data

### Source Data

| Data | Sumber | Deskripsi | File |
|------|--------|-----------|------|
| **BPJS Mobile Reviews** | Google Play Store | Ulasan pengguna aplikasi BPJS Mobile (app.bpjs.mobile) | `bpjs_reviews_1.csv` - `bpjs_reviews_5.csv` |
| **FastText Embeddings** | Facebook Research | Pre-trained Indonesian word embeddings (300-dimensional) | `embedding/cc.id.300.vec/cc.id.300.vec` |

**Dataset Characteristics:**
- Total Reviews: ~4,000+
- Bahasa: Indonesian
- Classes: 3 (Positive, Neutral, Negative)
- Format: CSV dengan kolom (reviewer, rating, text, sentiment)

### Data Acquisition

Menggunakan Google Play Scraper untuk mengekstrak ulasan real-time:

```python
from script.scraping import scrape_playstore_reviews

# Ambil 1000 review terbaru
df = scrape_playstore_reviews(
    app_id='app.bpjs.mobile',
    target_count=1000,
    batch_size=500,
    filter_score=None  # Optional: filter by rating
)

df.to_csv('data/bpjs_reviews_new.csv', index=False)
```

**Rate Limiting:** 1 detik delay antar batch untuk menghindari IP ban

### Data Preprocessing

#### 1. Text Cleaning
- Konversi ke lowercase
- Hapus URLs dan emails
- Hapus special characters dan symbols
- Handle emoji (konversi atau hapus)

#### 2. Tokenization & Normalization
- Tokenisasi ke kata individual (NLTK)
- Hapus stopwords (Indonesia + custom)
- Normalisasi whitespace

#### 3. Stemming
- Apply Sastrawi stemmer (Indonesian)
- Reduce kata ke root form

#### 4. Automatic Labeling (Hugging Face RoBERTa)
- **Model**: `w11wo/indonesian-roberta-base-sentiment-classifier`
- Pre-trained model dari Hugging Face untuk sentiment classification bahasa Indonesia
- Otomatis mengklasifikasi teks ke 3 kelas: Negative, Neutral, Positive
- Digunakan untuk memberikan label awal sebelum dilakukan fine-tuning dengan BiLSTM

#### 5. Sequence Processing
- Konversi teks ke sequence numbers
- Padding ke `max_sequence_length = 60`

#### 6. Word Embeddings
- Load pre-trained FastText embeddings
- Create embedding matrix dari vocabulary
- Embedding dimension: 300

#### 7. Data Augmentation & Balancing
- Gunakan NLPAug untuk augmentasi teks dengan FastText embeddings
- Handle class imbalance dengan oversampling
- Maintain balanced distribution antar kelas

---

## 📁 Code Structure

```
Sentimen Analisis/
├── pyproject.toml                       # Project configuration & dependencies
├── README.md                            # Dokumentasi proyek
│
├── data/                                # Dataset folder
│   ├── bpjs_reviews_1.csv               # Raw data batch 1
│   ├── bpjs_reviews_2.csv               # Raw data batch 2
│   ├── bpjs_reviews_3.csv               # Raw data batch 3
│   ├── bpjs_reviews_4.csv               # Raw data batch 4
│   └── bpjs_reviews_5.csv               # Raw data batch 5
│
├── notebooks/                           # Jupyter notebooks
│   ├── model.ipynb                      # Main notebook: EDA, preprocessing, training
│   └── best_model.keras                 # Trained model weights (best performance)
│
└── script/                              # Python modules
    └── scraping.py                      # Google Play Store scraper

**Key Files:**
- `notebooks/model.ipynb` - Main pipeline (EDA, preprocessing, training, evaluation)
- `notebooks/best_model.keras` - Deploy-ready model
- `script/scraping.py` - Data collection module

---

## 📊 Results and Evaluation

### Workflow Pipeline

Raw Text Data (20,000 reviews)  
↓  
**Text Cleaning & Preprocessing**
- Lowercase + Unidecode
- Remove emoji, URLs, mentions
- Remove special characters
- Tokenization
- Fix slang words
- Handle negation
- Remove stopwords
- Stemming (Sastrawi)
↓  
**Automatic Labeling – Hugging Face RoBERTa**
- Model: indonesian-roberta-base-sentiment-classifier
- 3 Classes: Negative, Neutral, Positive
- Output: Labeled dataset
↓  
**Feature Engineering**
- Sequence to tokens (Tokenizer)
- Padding (max_length=60)
- Embedding matrix (FastText 100‑dim)
- Label encoding (0,1,2)
↓  
**Data Augmentation**
- Positive class: 1× augmentation
- Neutral class: 2× augmentation
- Result: Balanced training set
↓  
**Model Training**
- Architecture: BiLSTM with Embedding
- Optimizer: Adam
- Early Stopping + ReduceLROnPlateau
- Class weights: balanced
- Output: Trained model
↓  
**Evaluation & Inference**
- Metrics: Accuracy, Precision, Recall, F1-Score


#### Accuracy & Loss
```
Training Accuracy: 88.9%
Validation Accuracy: 80.7%

Training Loss: 0.30
Validation Loss: 0.47
```

#### Per-Class Metrics (Validation Set)

| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|----------|
| Negative | 0.82 | 0.89 | 0.85 | 2395 |
| Neutral | 0.59 | 0.45 | 0.51 | 1513 |
| Positive | 0.85 | 0.82 | 0.83 | 2092 |
| **Macro Avg** | **0.75** | **0.72** | **0.73** | **6000** |

#### Classification Insights (Baseline Model)

**Strengths:**
- **Negative sentiment** detection sangat baik (Recall: 89%) - berhasil menangkap 89% dari ulasan komplain sebenarnya
- **Positive sentiment** memiliki precision tinggi (85%) - ketika model prediksi positif, itu reliable
- Model lebih aggressive dalam mengidentifikasi kelas mayoritas

**Weaknesses:**
- **Neutral sentiment** sangat challenging (Recall hanya 45%) - model kesulitan membedakan netral dari positive
- Terjadi **overfitting** (Training Accuracy 88.9% → Validation 80.7%) - model terlalu fit ke data training
- Macro F1-Score 73% menunjukkan performa yang belum seimbang antar kelas

**Trade-off Utama:**
Model baseline fokus menangkap ulasan negatif (recall tinggi) dengan mengorbankan kemampuan membedakan kategori netral dan positif.

### Visualizations

**Training History:**
- Convergence plot menunjukkan training optimal pada epoch ~40
- Validation curve smooth tanpa extreme overfitting

**Confusion Matrix:**
- Strongest pada diagonal (correct predictions)
- Neutral sering disalahartikan sebagai Positive

**Model Architecture:**
```
Input Layer (Variable Length)
    ↓
Embedding (300 dimensions)
    ↓
BiLSTM (128 units, bidirectional)
    ↓
Dropout (0.3)
    ↓
Dense (64 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Output Layer (3 units, Softmax)
```

### Hyperparameter Configuration

#### Baseline Model
```python
# Baseline parameters (BiLSTM)
embedding_dim = 100
max_sequence_length = 60
lstm_units = 96
spatial_dropout = 0.3
lstm_dropout = 0.3
dense_units = 64
dense_dropout = 0.5
learning_rate = 5e-4
batch_size = 64
epochs = 20
optimizer = Adam
loss = sparse_categorical_crossentropy
```

#### Tuning Range
Model hyperparameter tuning menggunakan **Keras Tuner dengan Bayesian Optimization** untuk mencari optimal parameters:
- LSTM Units: [64, 96, 128]
- Spatial Dropout: 0.2 - 0.5
- LSTM Dropout: 0.2 - 0.4
- Dense Dropout: 0.3 - 0.6
- Learning Rate: [5e-4, 3e-4, 1e-4]
- L2 Regularization: [1e-4, 1e-5]

---

## 🚀 Future Work

1. **Model Improvements**
   - [ ] Implement Transformer-based models (BERT, RoBERTa)
   - [ ] Ensemble multiple architectures
   - [ ] Fine-tune pre-trained Indonesian BERT models

2. **Feature Enhancements**
   - [ ] Aspect-based sentiment analysis
   - [ ] Emotion detection (5+ emotions)
   - [ ] Sarcasm detection

3. **Deployment**
   - [ ] Create REST API endpoint
   - [ ] Deploy ke cloud (GCP, AWS, atau Azure)
   - [ ] Build web dashboard untuk visualization
   - [ ] Implement real-time scraping & prediction

4. **Data & Analysis**
   - [ ] Expand ke aplikasi lain (selain BPJS)
   - [ ] Temporal analysis (sentiment trends over time)
   - [ ] Topic modeling on negative reviews
   - [ ] User demographic analysis

5. **Production**
   - [ ] Model versioning dan monitoring
   - [ ] A/B testing untuk model variants
   - [ ] Automated retraining pipeline
   - [ ] Model interpretability (LIME, SHAP)

---

## 📚 Acknowledgments & References

### Data Sources & Tools
- **Google Play Scraper**: [github.com/JoMingyu/google-play-scraper](https://github.com/JoMingyu/google-play-scraper)
- **FastText**: [fasttext.cc](https://fasttext.cc) - Pre-trained Indonesian embeddings
- **Sastrawi**: [github.com/har07/Sastrawi](https://github.com/har07/Sastrawi) - Indonesian stemmer
- **Keras Tuner**: [keras.io/keras_tuner](https://keras.io/keras_tuner/) - Hyperparameter optimization

### Publications & References
- Hochreiter & Schmidhuber (1997) - LSTM architecture
- Graves & Schmidhuber (2005) - Bidirectional RNNs
- Bojanowski et al. (2016) - FastText word embeddings
- Devlin et al. (2018) - BERT (Pre-training for NLP)

### Special Thanks
- BPJS community untuk data yang bermanfaat
- TensorFlow & Keras team untuk excellent tools
- Open-source community untuk NLP libraries

---

## 📝 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

**Dataset License:** 
- BPJS Reviews: Sourced from public Google Play Store data
- FastText Embeddings: Licensed under Creative Commons Attribution 3.0

---

## 👤 Author & Contact

**Project Developer**: Ratuayu Nurfajar
- **GitHub**: @ratuanfajar (https://github.com/ratuanfajar)
- **Email**: ratua4820.email@gmail.com
- **LinkedIn**: Ratuayu Nurfajar (https://www.linkedin.com/in/ratuayunurfajar/)