import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import joblib
import os
from pathlib import Path

from .data_loader import DataLoader
from .preprocessor import TextPreprocessor
from .model_handler import ModelHandler
from .classifier import SentimentClassifier
from .similarity_search import SimilaritySearch
from config.settings import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalysisPipeline:
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        self.data_loader = DataLoader()
        self.preprocessor = TextPreprocessor()
        self.model_handler = ModelHandler()
        self.classifier = SentimentClassifier()
        self.similarity_search = SimilaritySearch()
        
        self.is_trained = False
        self.dataset = None
        self.embeddings = None
        
        np.random.seed(RANDOM_SEED)
        
        logger.info("Sentiment Analysis Pipeline initialized")
    
    def load_data(self) -> bool:
        logger.info("Loading dataset...")
        
        self.dataset = self.data_loader.load_dataset()
        
        if self.dataset is None:
            logger.error("Failed to load dataset")
            return False
        
        if not self.data_loader.validate_dataset():
            logger.error("Dataset validation failed")
            return False
        
        logger.info("Dataset loaded and validated successfully")
        return True
    
    def preprocess_data(self) -> bool:
        if self.dataset is None:
            logger.error("No dataset loaded. Call load_data() first.")
            return False
        
        logger.info("Preprocessing data...")
        
        self.dataset = self.preprocessor.preprocess_dataframe(self.dataset)
        
        if self.dataset is None or len(self.dataset) == 0:
            logger.error("Preprocessing failed or resulted in empty dataset")
            return False
        
        logger.info(f"Data preprocessed successfully: {len(self.dataset)} samples")
        return True
    
    def generate_embeddings(self) -> bool:
        if self.dataset is None:
            logger.error("No dataset available. Call load_data() and preprocess_data() first.")
            return False
        
        logger.info("Generating embeddings...")
        
        if not self.model_handler.load_model():
            logger.error("Failed to load sentence transformer model")
            return False
        
        texts = self.dataset['review_clean'].tolist()
        self.embeddings = self.model_handler.encode_texts(texts)
        
        if self.embeddings is None:
            logger.error("Failed to generate embeddings")
            return False
        
        logger.info(f"Embeddings generated: {self.embeddings.shape}")
        return True
    
    def train_classifier(self) -> bool:
        if self.embeddings is None or self.dataset is None:
            logger.error("Embeddings or dataset not available")
            return False
        
        logger.info("Training classifier...")
        
        # Prepare data
        X = self.embeddings
        y = self.dataset['sentimen'].values
        
        # Train classifier
        results = self.classifier.train(X, y)
        
        if results is None:
            logger.error("Classifier training failed")
            return False
        
        self.is_trained = True
        logger.info(f"Classifier trained successfully")
        logger.info(f"Training accuracy: {results['train_accuracy']:.3f}")
        logger.info(f"Validation accuracy: {results['val_accuracy']:.3f}")
        
        return True
    
    def build_similarity_index(self) -> bool:
        if self.embeddings is None or self.dataset is None:
            logger.error("Embeddings or dataset not available")
            return False
        
        logger.info("Building similarity search index...")
        
        success = self.similarity_search.build_index(
            self.embeddings, 
            self.dataset
        )
        
        if not success:
            logger.error("Failed to build similarity index")
            return False
        
        logger.info("Similarity search index built successfully")
        return True
    
    def run_pipeline(self) -> bool:
        logger.info("Starting complete sentiment analysis pipeline")
        logger.info("=" * 60)
        
        if not self.load_data():
            return False
        
        if not self.preprocess_data():
            return False
        
        if not self.generate_embeddings():
            return False
        
        if not self.train_classifier():
            return False
        
        if not self.build_similarity_index():
            return False
        
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        
        self._display_pipeline_summary()
        
        return True
    
    def predict_sentiment(self, text: str) -> Optional[Dict]:
        if not self.is_trained:
            logger.error("Model not trained. Run the pipeline first.")
            return None
        
        try:
            clean_text = self.preprocessor.clean_text(text)
            
            if not clean_text or len(clean_text.strip()) < 3:
                logger.warning("Text too short or empty after preprocessing")
                return None
            
            embedding = self.model_handler.encode_texts([clean_text])
            
            if embedding is None:
                logger.error("Failed to generate embedding")
                return None
            
            prediction_result = self.classifier.predict(embedding)
            
            if prediction_result is None:
                logger.error("Failed to predict sentiment")
                return None
            
            result = {
                'text': text,
                'clean_text': clean_text,
                'sentiment': "Positive" if prediction_result['prediction'][0] == 1 else "Negative",
                'confidence': float(prediction_result['confidence'][0]),
                'probabilities': {
                    'negative': float(prediction_result['probabilities'][0][0]),
                    'positive': float(prediction_result['probabilities'][0][1])
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in sentiment prediction: {e}")
            return None
    
    def find_similar_reviews(self, text: str, k: int = 5) -> Optional[List[Dict]]:
        if not self.similarity_search.is_built:
            logger.error("❌ Similarity index not built")
            return None
        
        try:
            clean_text = self.preprocessor.clean_text(text)
            
            embedding = self.model_handler.encode_texts([clean_text])
            
            results = self.similarity_search.search(embedding, k=k)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return None
    
    def save_model(self, save_path: str) -> bool:
        if not self.is_trained:
            logger.error("No trained model to save")
            return False
        
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            self.classifier.save_model(save_path / "classifier.joblib")
            
            self.similarity_search.save_index(save_path / "similarity_index")
            
            joblib.dump(
                self.preprocessor.get_settings(), 
                save_path / "preprocessor_settings.joblib"
            )
            
            metadata = {
                'model_name': self.model_handler.model_name,
                'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else None,
                'dataset_size': len(self.dataset) if self.dataset is not None else None,
                'is_trained': self.is_trained,
                'config': self.config
            }
            joblib.dump(metadata, save_path / "pipeline_metadata.joblib")
            
            logger.info(f"Model saved successfully to: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, load_path: str) -> bool:
        try:
            load_path = Path(load_path)
            
            if not load_path.exists():
                logger.error(f"Model path does not exist: {load_path}")
                return False
            
            if not self.classifier.load_model(load_path / "classifier.joblib"):
                return False
            
            if not self.similarity_search.load_index(load_path / "similarity_index"):
                return False
            
            preprocessor_settings = joblib.load(load_path / "preprocessor_settings.joblib")
            self.preprocessor.load_settings(preprocessor_settings)
            
            metadata = joblib.load(load_path / "pipeline_metadata.joblib")
            self.config = metadata.get('config', {})
            
            model_name = metadata.get('model_name', MODEL_CONFIG['sentence_transformer'])
            self.model_handler = ModelHandler(model_name)
            
            if not self.model_handler.load_model():
                return False
            
            self.is_trained = True
            logger.info(f"Model loaded successfully from: {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def evaluate_model(self) -> Optional[Dict]:
        if not self.is_trained:
            logger.error("Model not trained")
            return None
        
        return self.classifier.evaluate()
    
    def get_pipeline_info(self) -> Dict:
        info = {
            'is_trained': self.is_trained,
            'dataset_loaded': self.dataset is not None,
            'dataset_size': len(self.dataset) if self.dataset is not None else 0,
            'embeddings_generated': self.embeddings is not None,
            'embedding_shape': self.embeddings.shape if self.embeddings is not None else None,
            'similarity_index_built': self.similarity_search.is_built,
            'model_name': self.model_handler.model_name if self.model_handler.model else None
        }
        
        return info
    
    def _display_pipeline_summary(self):
        info = self.get_pipeline_info()
        
        logger.info("Pipeline Summary:")
        logger.info(f"   Dataset loaded: {info['dataset_size']:,} samples")
        logger.info(f"   Embeddings shape: {info['embedding_shape']}")
        logger.info(f"   Model trained: {info['is_trained']}")
        logger.info(f"   Similarity index: {info['similarity_index_built']}")
        logger.info(f"   Model: {info['model_name']}")
        
        eval_results = self.evaluate_model()
        if eval_results:
            logger.info(f"   Test Accuracy: {eval_results['accuracy']:.3f}")
            logger.info(f"   F1-Score: {eval_results.get('f1_score', 'N/A')}")

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = SentimentAnalysisPipeline()
    
    # Run complete pipeline
    success = pipeline.run_pipeline()
    
    if success:
        # Test predictions
        test_texts = [
            "Produk ini sangat bagus dan berkualitas tinggi!",
            "Kualitas produk sangat buruk, tidak puas dengan pembelian ini.",
            "Pengiriman cepat dan produk sesuai deskripsi."
        ]
        
        logger.info("Testing predictions:")
        for text in test_texts:
            result = pipeline.predict_sentiment(text)
            if result:
                logger.info(f"Text: {text}")
                logger.info(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.3f})")
                logger.info("-" * 50)