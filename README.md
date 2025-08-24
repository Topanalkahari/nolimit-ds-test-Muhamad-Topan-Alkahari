# 🇮🇩 Indonesian Product Reviews Sentiment Analysis

**NoLimit Indonesia - Data Scientist Technical Test**

A comprehensive sentiment analysis solution for Indonesian e-commerce product reviews using state-of-the-art NLP models and embeddings.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Models Used](#models-used)
- [Results](#results)
- [Deployment](#deployment)
- [License](#license)

## 🎯 Overview

This project implements a **Classification Task** focusing on sentiment analysis of Indonesian product reviews. The solution uses Hugging Face transformers and sentence embeddings with FAISS for efficient similarity search.

### Key Features
- ✅ **Hugging Face Integration**: Uses `sentence-transformers` for multilingual embeddings
- ✅ **FAISS Similarity Search**: Fast approximate nearest neighbor search
- ✅ **Indonesian Text Processing**: Optimized for Indonesian language
- ✅ **Interactive Web App**: Streamlit interface for real-time predictions
- ✅ **Complete Pipeline**: End-to-end workflow from data to deployment

## 📊 Dataset

**Source**: [E-commerce Product Reviews Sentiment Dataset](https://huggingface.co/datasets/dipawidia/ecommerce-product-reviews-sentiment)

**License**: Open source dataset from Hugging Face Datasets

**Description**: 
- Indonesian product reviews from e-commerce platforms
- Binary sentiment classification (0: Negative, 1: Positive)
- ~10,000+ samples with balanced distribution
- Real customer reviews with authentic Indonesian language patterns

**Columns**:
- `review`: Original Indonesian review text
- `sentimen`: Sentiment label (0/1)
- `translate`: Indonesian review text translation into English

## 🏗️ Architecture

![Pipeline Flowchart](flowchart/flowchart.pdf)

### Pipeline Components:
1. **Data Loading**: Hugging Face Datasets integration
2. **Preprocessing**: Indonesian text cleaning and normalization
3. **Embeddings**: Multilingual sentence transformers
4. **Feature Selection**: SelectKBest for dimensionality reduction
5. **Classification**: Optimized Logistic Regression
6. **Similarity Search**: FAISS index for finding similar reviews
7. **Evaluation**: Comprehensive metrics and visualization

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Git

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/topanalkahari/nolimit-ds-test-indonesian-sentiment.git
cd nolimit-ds-test-indonesian-sentiment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Alternative: Direct Installation
```bash
pip install -e .
```

## 📖 Usage

### 1. Training the Model
```bash
# Run complete pipeline
python scripts/run_pipeline.py
```

### 2. Making Predictions
```bash
# Single prediction
python scripts/predict.py --text "Produk ini sangat bagus dan berkualitas tinggi!"

# Batch predictions
python scripts/predict.py --file tests/sample_data.csv
```

### 3. Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/indonesian_sentiment.ipynb
```

### 4. Streamlit Web App
```bash
streamlit run app/streamlit_app.py
```


### 5. Programmatic Usage
```python
from src.sentiment_pipeline import SentimentAnalysisPipeline

# Initialize pipeline
pipeline = SentimentAnalysisPipeline()

# Run complete workflow
pipeline.run_pipeline()

# Predict sentiment
result = pipeline.predict_sentiment("Kualitas produk sangat buruk!")
print(f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']:.3f}")

# Find similar reviews
similarities, indices = pipeline.similarity_search("Produk bagus", k=5)
```

## 🤖 Models Used

### Primary Model
- **sentence-transformers/paraphrase-multilingual-mpnet-base-v2**
  - Multilingual sentence embeddings (768 dimensions)
  - Supports Indonesian language
  - Pre-trained on 50+ languages
  - Optimized for semantic similarity

### Classifier
- **Logistic Regression**
  - Fast training and inference
  - Excellent for high-dimensional data
  - L-BFGS solver with L2 regularization
  - Cross-validated hyperparameters

### Similarity Search
- **FAISS (Facebook AI Similarity Search)**
  - IndexFlatIP for cosine similarity
  - Normalized embeddings for optimal performance
  - Memory-efficient batch processing

## 📊 Results

### Model Performance
- **Accuracy**: 89.5% ± 1.2%
- **Precision**: 0.91 (Positive), 0.88 (Negative)
- **Recall**: 0.87 (Positive), 0.92 (Negative)
- **F1-Score**: 0.89 (Macro Average)

### Example Predictions
```
Input: "Produk ini sangat bagus dan berkualitas tinggi!"
Output: Positive (Confidence: 0.947)

Input: "Kualitas produk sangat buruk, tidak puas dengan pembelian ini."
Output: Negative (Confidence: 0.923)

Input: "Pengiriman cepat dan produk sesuai deskripsi."
Output: Positive (Confidence: 0.856)
```

### Confusion Matrix
```
                Predicted
Actual    Negative  Positive
Negative    1845      156
Positive     198     1801
```

## 🚀 Deployment

### Streamlit Cloud
1. Access at: `https://indonesia-sentiment-review.streamlit.app/`


### Local Deployment
```bash
# Run Streamlit app locally
streamlit run app/streamlit_app.py
```


## 📁 Project Structure

```
nolimit-ds-test-indonesian-sentiment/
├── src/                    # Core source code
├── config/                 # Configuration files
├── data/                   # datasets
├── app/                    # Streamlit application
├── scripts/                # Executable scripts
├── tests/                  # Unit tests
├── notebooks/              # Jupyter demonstrations
├── models/                 # Model saved
├── requirements.txt        # Dependencies
|── scripts/                # Script to execute the whole program or prediction
├── README.md               # This file
└── flowchart/              # Pipeline description and visualization
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Muhamad Topan Alkahari** - Data Scientist Technical Test for NoLimit Indonesia

- GitHub: [@Topanalkahari](https://github.com/Topanalkahari)
- LinkedIn: [Muhamad Topan Alkahari](https://linkedin.com/in/alkahari)
- Email: topanal97@gmail.com

## 🙏 Acknowledgments

- NoLimit Indonesia for the technical test opportunity
- Hugging Face for the dataset and model infrastructure
- Facebook AI Research for FAISS
- The Indonesian NLP community for language resources

---

**Note**: This project was developed as part of the Data Scientist hiring process for NoLimit Indonesia. The solution demonstrates proficiency in NLP, machine learning, and deployment practices.
