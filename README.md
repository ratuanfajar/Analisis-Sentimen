# Sentiment Analysis - BPJS Mobile Reviews

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange)
![Keras](https://img.shields.io/badge/Keras-3.13+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Project Overview

**Sentiment Analysis - BPJS Mobile Reviews** is a deep learning project that classifies user reviews from the BPJS Mobile application on Google Play Store into sentiment categories. The project implements Bidirectional LSTM (BiLSTM) with Natural Language Processing for sentiment classification in Indonesian language.

### Objectives
- Classify user reviews into **3 sentiment classes**: Positive, Neutral, and Negative
- Analyze user perception and satisfaction with BPJS Mobile
- Identify improvement areas based on negative sentiment feedback
- Provide actionable insights for product development teams

---

## Installation and Setup

### Requirements & Tools
| Component | Version | Description |
|-----------|---------|-------------|
| **Python** | 3.12+ | Programming language |
| **Editor** | VS Code | Code editor |
| **Package Manager** | uv / pip | Dependency management |
| **Virtual Environment** | .venv | Isolated Python environment |

### Core Dependencies

#### Data & General Processing
```
pandas >= 3.0.1
numpy >= 1.24.0
emoji >= 2.15.0
regex >= 2026.2.19
requests >= 2.32.5
unidecode >= 1.4.0
tqdm >= 4.67.3
```

#### Data Visualization
```
matplotlib >= 3.10.8
seaborn >= 0.13.2
wordcloud >= 1.9.6
```

#### Natural Language Processing (NLP)
```
nltk >= 3.9.3
sastrawi >= 1.0.1
spacy >= 3.8.11
transformers >= 5.2.0
nlpaug >= 1.1.11
gensim >= 4.4.0
fasttext-wheel >= 0.9.2
google-play-scraper >= 1.2.7
```

#### Machine Learning & Deep Learning
```
scikit-learn >= 1.8.0
tensorflow >= 2.20.0
keras >= 3.13.2
keras-tuner >= 1.4.8
```

### Quick Setup

```bash
# Navigate to project directory
cd "d:\document\coding\project\Sentimen Analisis"

# Create virtual environment (optional)
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac

# Install dependencies
pip install -e .

# Or using uv
uv sync
```

---

## Data Pipeline

### Data Sources
| Data | Source | Description | File |
|------|--------|-------------|------|
| **BPJS Mobile Reviews** | Google Play Store | Real user reviews from BPJS Mobile app | `bpjs_reviews_1.csv` - `bpjs_reviews_5.csv` |
| **FastText Embeddings** | Facebook Research | Pre-trained Indonesian word vectors (300-dim) | `embedding/cc.id.300.vec` |

**Dataset Characteristics:**
- **Total Reviews**: ~4,000+
- **Language**: Indonesian
- **Classes**: 3 (Positive, Neutral, Negative)
- **Format**: CSV with columns (reviewer, rating, text, sentiment)

### Data Acquisition
Using Google Play Scraper for real-time review extraction:
```python
from script.scraping import scrape_playstore_reviews

df = scrape_playstore_reviews(
    app_id='app.bpjs.mobile',
    target_count=1000,
    batch_size=500,
    filter_score=None
)
df.to_csv('data/bpjs_reviews_new.csv', index=False)
```
**Note**: 1-second delay between batches to avoid IP blocking

---

### **Data Science Methodology Highlights:**

| Stage | Focus Area | Key Techniques | Output |
|-------|-----------|-----------------|--------|
| **1. EDA** | Understand data distribution | Visualization, Statistical Analysis | Data Insights & Quality Report |
| **2. Cleaning** | Text normalization | Tokenization, Stemming, Stopword Removal | Cleaned Text Corpus |
| **3. Labeling** | Automatic annotation | Transformer-based RoBERTa Model | Sentiment Labels (3-class) |
| **4. Features** | Vector representation | FastText Embeddings (300-dim) | Sequence Embeddings |
| **5. Balancing** | Handle class imbalance | NLPAug Augmentation | Balanced Training Set |
| **6. Training** | Model optimization | BiLSTM + Callbacks | Fine-tuned Neural Network |
| **7. Evaluation** | Performance validation | Metrics & Classification Report | Model Performance Analysis |

---


The project follows a **complete end-to-end data science pipeline** with **7 critical stages**, from raw data ingestion to production-ready sentiment classification. This workflow emphasizes reproducibility, scalability, and interpretability.

### **Stage 1: Exploratory Data Analysis (EDA)**
- Analyze review distribution and sentiment class balance
- Generate word frequency statistics
- Create word clouds for each sentiment class
- Visualize rating distributions and trends
- Identify data quality issues and missing values

**Key Insights:**
- Understand class imbalance (if any)
- Discover common keywords per sentiment
- Profile review length and complexity

### **Stage 2: Data Preprocessing & Cleaning**

#### Text Cleaning
- Convert to lowercase for consistency
- Remove URLs, emails, and mentions
- Remove special characters and symbols
- Handle emoji (convert or remove)
- Fix whitespace normalization

#### Tokenization & Normalization
- Tokenize text into individual words (NLTK)
- Remove Indonesian stopwords + custom stopwords
- Normalize whitespace

#### Stemming (Indonesian)
- Apply **Sastrawi stemmer** (Indonesian-specific)
- Reduce words to root form (e.g., "berlari" → "lari")
- Improves vocabulary consistency

**Example Pipeline:**
```
"Aplikasinya sangat membantu banget!"
↓ (lowercase) → "aplikasinya sangat membantu banget!"
↓ (tokenize) → ["aplikasinya", "sangat", "membantu", "banget"]
↓ (remove stopwords) → ["aplikasinya", "membantu"]
↓ (stemming) → ["aplikasi", "bantu"]
```

### **Stage 3: Automatic Labeling (Auto-Annotation)**
- **Model**: `w11wo/indonesian-roberta-base-sentiment-classifier` (Hugging Face)
- **Purpose**: Pre-label reviews using pre-trained transformer model
- **Output**: 3-class labels (Negative, Neutral, Positive)
- **Advantage**: Provides initial labels for training before fine-tuning

### **Stage 4: Feature Engineering & Vectorization**

#### Sequence Processing
- Convert cleaned text to token sequences
- Pad sequences to `max_sequence_length = 60`
- Standardize input dimensions for neural network

#### Word Embeddings
- Load pre-trained **FastText embeddings** (300-dimensional)
- Create embedding matrix from project vocabulary
- Map tokens to dense vector representations
- **Embedding Dimension**: 300 features per word

**Embedding Process:**
```
Token sequence: [app_id_1, app_id_2, app_id_3, ...]
        ↓
Embedding matrix lookup
        ↓
Dense vectors: [[0.2, -0.1, 0.5, ...], [0.1, 0.3, -0.2, ...], ...]
(Each vector is 300-dimensional)
```

#### Label Encoding
- Encode sentiment classes: Negative=0, Neutral=1, Positive=2

### **Stage 5: Data Augmentation & Class Balancing**
- Use **NLPAug** with FastText embeddings for synonym replacement
- Augment minority classes to balance dataset:
  - Positive class: 1× augmentation
  - Neutral class: 2× augmentation
- Result: Balanced training set across all 3 classes
- **Benefit**: Reduces class imbalance bias during training

### **Stage 6: Model Architecture, Training & Optimization**

#### Neural Network Architecture
```
Input Layer (Variable Length)
    ↓
Embedding Layer (300 dimensions)
    ↓
Bidirectional LSTM (128 units)
    ↓
Dropout (0.3)
    ↓
Dense Layer (64 units, ReLU activation)
    ↓
Dropout (0.3)
    ↓
Output Layer (3 units, Softmax)
    ↓
Classification (Negative | Neutral | Positive)
```

#### Training Configuration
```python
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

#### Optimization Techniques
- **Early Stopping**: Stop training if validation loss plateaus
- **ReduceLROnPlateau**: Reduce learning rate if validation metric stops improving
- **Class Weights**: Balanced weights to handle class imbalance
- **Callbacks**: ModelCheckpoint to save best weights

#### Hyperparameter Tuning
Using **Keras Tuner with Bayesian Optimization** for optimal parameters:
- LSTM Units: [64, 96, 128]
- Spatial Dropout: 0.2 - 0.5
- LSTM Dropout: 0.2 - 0.4
- Dense Dropout: 0.3 - 0.6
- Learning Rate: [5e-4, 3e-4, 1e-4]
- L2 Regularization: [1e-4, 1e-5]

### **Stage 7: Model Evaluation, Validation & Performance Analysis**

#### **Performance Metrics**
```
Training Accuracy:   88.9%
Validation Accuracy: 80.7%

Training Loss:   0.30
Validation Loss: 0.47
```

#### **Per-Class Performance Breakdown**

| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| **Negative** | 0.82 | 0.89 | 0.85 | 2,395 |
| **Neutral** | 0.59 | 0.45 | 0.51 | 1,513 |
| **Positive** | 0.85 | 0.82 | 0.83 | 2,092 |
| **Macro Average** | **0.75** | **0.72** | **0.73** | **6,000** |

#### **Model Strengths & Competitive Advantages**
**Negative Sentiment Detection**: Excellent recall (89%) - captures most complaint reviews
**Positive Sentiment**: High precision (85%) - reliable positive predictions
**Balanced Training**: Uses class weights to handle imbalance effectively

#### **Model Limitations & Challenges**
**Neutral Class Challenge**: Low recall (45%) - struggles to distinguish neutral from positive
**Overfitting**: Training accuracy (88.9%) >> Validation accuracy (80.7%)
**Class Imbalance Impact**: Macro F1-Score (73%) shows uneven performance across classes

#### **Strategic Insight & Business Implication**
The baseline model prioritizes capturing negative sentiment (recall=89%) at the expense of distinguishing neutral and positive sentiments. This trade-off reflects the model's focus on identifying customer complaints.

---

## Project Structure

```
Sentimen Analisis/
├── pyproject.toml                    # Project configuration & dependencies
├── README.md                         # Documentation (this file)
│
├── data/                             # Dataset directory
│   ├── bpjs_reviews_1.csv            # Raw data batch 1
│   ├── bpjs_reviews_2.csv            # Raw data batch 2
│   ├── bpjs_reviews_3.csv            # Raw data batch 3
│   ├── bpjs_reviews_4.csv            # Raw data batch 4
│   └── bpjs_reviews_5.csv            # Raw data batch 5
│
├── notebooks/                        # Jupyter notebooks (core pipeline)
│   ├── model.ipynb                   # Main notebook:
│   │                                 # ├─ Stage 1: EDA & visualization
│   │                                 # ├─ Stage 2-3: Preprocessing & auto-labeling
│   │                                 # ├─ Stage 4-5: Embeddings & augmentation
│   │                                 # ├─ Stage 6: Model architecture & training
│   │                                 # └─ Stage 7: Evaluation & analysis
│   │
│   └── best_model.keras              # Trained model weights (production-ready)
│
└── script/                           # Python modules
    └── scraping.py                   # Google Play Store scraper utility

**Key Files:**
- `notebooks/model.ipynb` - Complete data science pipeline
- `notebooks/best_model.keras` - Production model (fine-tuned BiLSTM)
- `script/scraping.py` - Data collection module
```

---

## Model Notebook Breakdown (`model.ipynb`)

The main notebook implements all 7 stages of the data science workflow:

### Cell Structure
1. **Imports** - Load all required libraries and dependencies
2. **Data Loading** - Read CSV files and combine datasets
3. **Stage 1: EDA** - Statistical analysis and visualizations
4. **Stage 2-3: Preprocessing** - Clean text and apply auto-labeling
5. **Stage 4-5: Feature Engineering** - Create embeddings and augment data
6. **Stage 6: Model Building** - Define and configure BiLSTM architecture
7. **Stage 6: Training** - Train model with callbacks and monitoring
8. **Stage 7: Evaluation** - Generate classification reports and plots

### Visualization Outputs
- **Training History**: Convergence plots showing loss and accuracy over epochs
- **Confusion Matrix**: Heatmap showing prediction patterns per class
- **Word Clouds**: Separate visualizations for each sentiment class
- **ROC Curves**: Performance across decision thresholds
- **Classification Report**: Detailed metrics per class

---

## Usage & Inference

### Making Predictions
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load trained model
model = tf.keras.models.load_model('notebooks/best_model.keras')

# Preprocess new text
new_review = "Aplikasinya sangat membantu dalam memantau kesehatan saya"
# [Apply cleaning, tokenization, embedding steps...]
sequences = tokenizer.texts_to_sequences([new_review])
padded = pad_sequences(sequences, maxlen=60)

# Make prediction
prediction = model.predict(padded)
sentiment_label = ['Negative', 'Neutral', 'Positive'][prediction.argmax()]
confidence = prediction.max()

print(f"Sentiment: {sentiment_label} (Confidence: {confidence:.2%})")
```

---

## Future Improvements

### 1. Model Architecture
- [ ] Implement Transformer-based models (BERT, RoBERTa)
- [ ] Ensemble multiple architectures for robustness
- [ ] Fine-tune Indonesian BERT models (IndoBERT)
- [ ] Experiment with Attention mechanisms

### 2. Advanced Features
- [ ] Aspect-Based Sentiment Analysis (identify what users like/dislike)
- [ ] Multi-emotion detection (expand beyond 3 classes)
- [ ] Sarcasm and irony detection
- [ ] Entity recognition for feature-level insights

### 3. Deployment & API
- [ ] Build REST API endpoint for inference
- [ ] Deploy to cloud (GCP, AWS, or Azure)
- [ ] Create web dashboard for real-time monitoring
- [ ] Implement continuous scraping and prediction pipeline

### 4. Analytics & Insights
- [ ] Expand to other applications (beyond BPJS)
- [ ] Temporal analysis (sentiment trends over time)
- [ ] Topic modeling on negative reviews for root cause analysis
- [ ] User demographic analysis and profiling

### 5. Production & Maintenance
- [ ] Model versioning and experiment tracking (MLflow)
- [ ] Performance monitoring and drift detection
- [ ] A/B testing framework for model variants
- [ ] Automated retraining pipeline
- [ ] Model interpretability (LIME, SHAP) for explainability

---

## References & Acknowledgments

### Key Research & Architecture
- **LSTM Networks**: Hochreiter & Schmidhuber (1997) - [Learning Long-Term Dependencies with Gradient Descent is Difficult](http://www.bioinf.jku.at/publications/older/2604.pdf)
- **Bidirectional RNNs**: Graves & Schmidhuber (2005) - [Framewise phoneme classification with bidirectional LSTM networks](https://ieeexplore.ieee.org/document/1367959)
- **FastText Embeddings**: Bojanowski et al. (2016) - [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)
- **BERT**: Devlin et al. (2018) - [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

### Tools & Libraries
- **[google-play-scraper](https://github.com/JoMingyu/google-play-scraper)** - App store review crawler
- **[Sastrawi](https://github.com/har07/Sastrawi)** - Indonesian stemming library
- **[Keras Tuner](https://keras.io/keras_tuner/)** - Hyperparameter optimization framework
- **[FastText](https://fasttext.cc/)** - Efficient word vectors library
- **[Hugging Face Transformers](https://huggingface.co/transformers/)** - Pre-trained NLP models
  
---

## License

This project is licensed under the **MIT License** - see LICENSE file for details.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

**Data Attribution:**
- **BPJS Reviews**: Sourced from public Google Play Store data
- **FastText Embeddings**: Creative Commons Attribution 3.0 License

---

## 👤 Author & Contact

**Project Developer**: Ratuayu Nurfajar

- **GitHub**: [@ratuayunurfajar](https://github.com/ratuayunurfajar)
- **Email**: ratua4820.email@gmail.com
- **LinkedIn**: [Ratuayu Nurfajar](https://www.linkedin.com/in/ratuayunurfajar/)

---

## Quick Stats
- **Total Notebooks**: 1 (Complete pipeline)
- **Data Science Stages**: 7 (EDA → Preprocessing → Labeling → Features → Training → Evaluation)
- **Model Architecture**: Bidirectional LSTM with FastText Embeddings
- **Best Validation Accuracy**: 80.7%
- **Macro F1-Score**: 0.73
