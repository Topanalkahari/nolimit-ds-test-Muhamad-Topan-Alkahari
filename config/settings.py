import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


DATASET_CONFIG = {
    "name": "dipawidia/ecommerce-product-reviews-sentiment",
    "split": "train",
    "cache_dir": str(DATA_DIR / "cache")
}


MODEL_CONFIG = {
    "sentence_transformer": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "embedding_dim": 768,
    "batch_size": 128,
    "normalize_embeddings": True
}


CLASSIFIER_CONFIG = {
    "model_type": "logistic_regression",
    "params": {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42,
        "n_jobs": -1
    },
    "feature_selection": {
        "k": 500,
        "score_func": "f_classif"
    }
}

# FAISS configuration
FAISS_CONFIG = {
    "index_type": "IndexFlatIP",
    "batch_size": 1000,
    "normalize_l2": True
}


TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "stratify": True,
    "cv_folds": 3
}


PREPROCESSING_CONFIG = {
    "min_text_length": 5,
    "remove_duplicates": True,
    "clean_patterns": {
        "special_chars": r'[^a-zA-Z\s]',
        "extra_whitespace": r'\s+',
        "lowercase": True
    }
}


SIMILARITY_CONFIG = {
    "default_k": 5,
    "max_k": 50
}


STREAMLIT_CONFIG = {
    "title": "🇮🇩 Indonesian Sentiment Analysis",
    "subtitle": "NoLimit Indonesia - Data Scientist Technical Test",
    "max_text_length": 1000,
    "examples": [
        "Produk ini sangat bagus dan berkualitas tinggi!",
        "Kualitas produk sangat buruk, tidak puas dengan pembelian ini.",
        "Pengiriman cepat dan produk sesuai deskripsi.",
        "Harga terjangkau dengan kualitas yang memuaskan.",
        "Barang tidak sesuai foto, sangat kecewa!"
    ]
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(LOGS_DIR / "sentiment_analysis.log")
}


EVALUATION_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1"],
    "target_names": ["Negative", "Positive"],
    "plot_confusion_matrix": True,
    "save_results": True
}


HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)
WANDB_API_KEY = os.getenv("WANDB_API_KEY", None)

# Random seed for reproducibility
RANDOM_SEED = 42