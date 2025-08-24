import numpy as np
import logging
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
from config.settings import MODEL_CONFIG

logger = logging.getLogger(__name__)

class ModelHandler:
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or MODEL_CONFIG['sentence_transformer']
        self.model = None
        self.device = self._get_device()
        
        logger.info(f"ModelHandler initialized with: {self.model_name}")
        logger.info(f"Using device: {self.device}")
    
    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def load_model(self) -> bool:
        logger.info(f"Loading sentence transformer model: {self.model_name}")
        
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            embedding_dim = self.model.get_sentence_embedding_dimension()
            max_seq_length = self.model.max_seq_length
            
            logger.info(f"Model loaded successfully!")
            logger.info(f"   Embedding dimension: {embedding_dim}")
            logger.info(f"   Max sequence length: {max_seq_length}")
            logger.info(f"   Device: {self.device}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def encode_texts(self, texts: Union[str, List[str]], **kwargs) -> Optional[np.ndarray]:
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return None
        
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            logger.warning("No texts provided for encoding")
            return None
        
        logger.info(f"Encoding {len(texts)} texts...")
        
        try:
            encode_params = {
                'batch_size': MODEL_CONFIG.get('batch_size', 128),
                'show_progress_bar': len(texts) > 100,
                'convert_to_numpy': True,
                'normalize_embeddings': MODEL_CONFIG.get('normalize_embeddings', True),
                'device': self.device
            }
            
            encode_params.update(kwargs)
            
            embeddings = self.model.encode(texts, **encode_params)
            
            logger.info(f"Encoding completed: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            return None
    
    def encode_single_text(self, text: str, **kwargs) -> Optional[np.ndarray]:
        embeddings = self.encode_texts([text], **kwargs)
        
        if embeddings is not None:
            return embeddings[0]
        
        return None
    
    def batch_encode_texts(self, texts: List[str], batch_size: int = None) -> Optional[np.ndarray]:
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        if not texts:
            return None
        
        batch_size = batch_size or MODEL_CONFIG.get('batch_size', 128)
        
        logger.info(f"Batch encoding {len(texts)} texts (batch_size={batch_size})")
        
        try:
            all_embeddings = []
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
                batch_texts = texts[i:i + batch_size]
                
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=MODEL_CONFIG.get('normalize_embeddings', True),
                    show_progress_bar=False,
                    device=self.device
                )
                
                all_embeddings.append(batch_embeddings)
            
            embeddings = np.vstack(all_embeddings)
            
            logger.info(f"Batch encoding completed: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in batch encoding: {e}")
            return None
    
    def get_similarity(self, text1: str, text2: str) -> Optional[float]:
        embeddings = self.encode_texts([text1, text2])
        
        if embeddings is None or len(embeddings) != 2:
            return None
        
        embedding1 = embeddings[0]
        embedding2 = embeddings[1]
        
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        return float(similarity)
    
    def get_model_info(self) -> dict:
        if self.model is None:
            return {
                'model_name': self.model_name,
                'loaded': False
            }
        
        try:
            info = {
                'model_name': self.model_name,
                'loaded': True,
                'embedding_dimension': self.model.get_sentence_embedding_dimension(),
                'max_seq_length': self.model.max_seq_length,
                'device': self.device,
                'model_type': type(self.model).__name__
            }
            
            if hasattr(self.model, '_modules'):
                modules = list(self.model._modules.keys())
                info['modules'] = modules
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                'model_name': self.model_name,
                'loaded': True,
                'error': str(e)
            }
    
    def warm_up(self, sample_texts: List[str] = None) -> bool:
        if self.model is None:
            logger.error("Model not loaded")
            return False
        
        if sample_texts is None:
            sample_texts = [
                "Produk ini sangat bagus dan berkualitas.",
                "Kualitas produk buruk, tidak memuaskan.",
                "Pengiriman cepat dan sesuai deskripsi."
            ]
        
        logger.info("Warming up model...")
        
        try:
            _ = self.encode_texts(sample_texts, show_progress_bar=False)
            
            logger.info("Model warm-up completed")
            return True
            
        except Exception as e:
            logger.error(f"Error during model warm-up: {e}")
            return False
    
    def compute_embeddings_statistics(self, embeddings: np.ndarray) -> dict:
        if embeddings is None or len(embeddings) == 0:
            return {}
        
        try:
            stats = {
                'shape': embeddings.shape,
                'dtype': str(embeddings.dtype),
                'mean': float(np.mean(embeddings)),
                'std': float(np.std(embeddings)),
                'min': float(np.min(embeddings)),
                'max': float(np.max(embeddings)),
                'norm_mean': float(np.mean(np.linalg.norm(embeddings, axis=1))),
                'norm_std': float(np.std(np.linalg.norm(embeddings, axis=1)))
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error computing embeddings statistics: {e}")
            return {}
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str) -> bool:
        try:
            np.save(filepath, embeddings)
            logger.info(f"Embeddings saved to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return False
    
    def load_embeddings(self, filepath: str) -> Optional[np.ndarray]:
        try:
            embeddings = np.load(filepath)
            logger.info(f"Embeddings loaded from: {filepath}")
            logger.info(f"Shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return None
    
    def __del__(self):
        if hasattr(self, 'model') and self.model is not None:
            if self.device == 'cuda':
                try:
                    torch.cuda.empty_cache()
                except:
                    pass

# Example usage
if __name__ == "__main__":
    # Initialize model handler
    handler = ModelHandler()
    
    # Load model
    if handler.load_model():
        # Test texts
        test_texts = [
            "Produk ini sangat bagus dan berkualitas tinggi!",
            "Kualitas produk sangat buruk, tidak puas dengan pembelian ini.",
            "Pengiriman cepat dan produk sesuai deskripsi.",
            "Harga terjangkau dengan kualitas yang memuaskan."
        ]
        
        print("Testing model handler:")
        print("-" * 50)
        
        # Test encoding
        embeddings = handler.encode_texts(test_texts)
        
        if embeddings is not None:
            print(f"Embeddings generated: {embeddings.shape}")
            
            # Test single text encoding
            single_embedding = handler.encode_single_text(test_texts[0])
            if single_embedding is not None:
                print(f"Single text encoded: {single_embedding.shape}")
            
            # Test similarity
            similarity = handler.get_similarity(test_texts[0], test_texts[2])
            if similarity is not None:
                print(f"Similarity calculated: {similarity:.3f}")
            
            # Get model info
            info = handler.get_model_info()
            print(f"Model info: {info}")
            
            # Compute statistics
            stats = handler.compute_embeddings_statistics(embeddings)
            print(f"Embeddings stats:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        else:
            print("Failed to generate embeddings")
    else:
        print("Failed to load model")