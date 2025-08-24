import numpy as np
import pandas as pd
import faiss
import logging
from typing import List, Dict, Optional, Tuple
import joblib
from pathlib import Path

from config.settings import FAISS_CONFIG, SIMILARITY_CONFIG

logger = logging.getLogger(__name__)

class SimilaritySearch:
    
    def __init__(self, config: dict = FAISS_CONFIG):
        self.config = config
        self.index = None
        self.embeddings = None
        self.metadata = None
        self.is_built = False
        self.dimension = None
        
        logger.info("SimilaritySearch initialized")
    
    def build_index(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> bool:
        logger.info(f"Building FAISS index from {len(embeddings)} embeddings...")
        
        try:
            if embeddings is None or len(embeddings) == 0:
                logger.error("Empty embeddings provided")
                return False
            
            if metadata is None or len(metadata) != len(embeddings):
                logger.error("Metadata size doesn't match embeddings")
                return False
            
            self.embeddings = embeddings.astype('float32')
            self.metadata = metadata.copy()
            self.dimension = embeddings.shape[1]
            
            index_type = self.config.get('index_type', 'IndexFlatIP')
            
            if index_type == 'IndexFlatIP':
                self.index = faiss.IndexFlatIP(self.dimension)
            elif index_type == 'IndexFlatL2':
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                logger.warning(f"Unknown index type: {index_type}, using IndexFlatIP")
                self.index = faiss.IndexFlatIP(self.dimension)
            
            if self.config.get('normalize_l2', True):
                faiss.normalize_L2(self.embeddings)
            
            batch_size = self.config.get('batch_size', 1000)
            
            logger.info(f"Adding vectors to index in batches of {batch_size}...")
            for i in range(0, len(self.embeddings), batch_size):
                batch_embeddings = self.embeddings[i:i + batch_size]
                self.index.add(batch_embeddings)
            
            self.is_built = True
            
            logger.info(f"FAISS index built successfully!")
            logger.info(f"   Total vectors: {self.index.ntotal}")
            logger.info(f"   Dimension: {self.dimension}")
            logger.info(f"   Index type: {type(self.index).__name__}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            return False
    
    def search(self, query_embeddings: np.ndarray, k: int = 5) -> Optional[List[Dict]]:
        if not self.is_built:
            logger.error("Index not built. Call build_index() first.")
            return None
        
        try:
            # Ensure query is 2D array
            if query_embeddings.ndim == 1:
                query_embeddings = query_embeddings.reshape(1, -1)
            
            query_embeddings = query_embeddings.astype('float32')
            
            # Normalize query if index uses normalized embeddings
            if self.config.get('normalize_l2', True):
                faiss.normalize_L2(query_embeddings)
            
            k = min(k, self.index.ntotal)
            k = max(1, k)  # At least 1
            
            similarities, indices = self.index.search(query_embeddings, k)
            
            results = []
            for i in range(len(query_embeddings)):
                query_results = []
                for j in range(k):
                    idx = indices[i][j]
                    similarity = float(similarities[i][j])
                    
                    if idx >= 0 and idx < len(self.metadata):
                        # Get metadata for this item
                        item_data = self.metadata.iloc[idx]
                        
                        result = {
                            'index': int(idx),
                            'similarity': similarity,
                            'text': item_data.get('review', ''),
                            'clean_text': item_data.get('review_clean', ''),
                            'sentiment': 'Positive' if item_data.get('sentimen', 0) == 1 else 'Negative',
                            'sentiment_label': int(item_data.get('sentimen', 0))
                        }
                        
                        query_results.append(result)
                
                if len(query_embeddings) == 1:
                    return query_results
                else:
                    results.append(query_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return None
    
    def search_by_text(self, model_handler, query_text: str, k: int = 5) -> Optional[List[Dict]]:
        try:
            query_embedding = model_handler.encode_texts([query_text])
            
            if query_embedding is None:
                logger.error("Failed to encode query text")
                return None
            
            return self.search(query_embedding, k=k)
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return None
    
    def get_random_samples(self, n: int = 10) -> Optional[List[Dict]]:
        if not self.is_built or self.metadata is None:
            logger.error("Index not built or no metadata available")
            return None
        
        try:
            # Sample random indices
            n = min(n, len(self.metadata))
            random_indices = np.random.choice(len(self.metadata), size=n, replace=False)
            
            samples = []
            for idx in random_indices:
                item_data = self.metadata.iloc[idx]
                sample = {
                    'index': int(idx),
                    'text': item_data.get('review', ''),
                    'clean_text': item_data.get('review_clean', ''),
                    'sentiment': 'Positive' if item_data.get('sentimen', 0) == 1 else 'Negative',
                    'sentiment_label': int(item_data.get('sentimen', 0))
                }
                samples.append(sample)
            
            return samples
            
        except Exception as e:
            logger.error(f"Failed to get random samples: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        if not self.is_built:
            return {'is_built': False}
        
        stats = {
            'is_built': True,
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'index_type': type(self.index).__name__ if self.index else None,
            'metadata_size': len(self.metadata) if self.metadata is not None else 0
        }
        
        if self.metadata is not None and 'sentimen' in self.metadata.columns:
            sentiment_dist = self.metadata['sentimen'].value_counts()
            stats['sentiment_distribution'] = sentiment_dist.to_dict()
        
        return stats
    
    def save_index(self, save_dir: str) -> bool:
        if not self.is_built:
            logger.error("No index to save")
            return False
        
        try:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            faiss.write_index(self.index, str(save_path / "faiss_index"))
            
            if self.metadata is not None:
                self.metadata.to_parquet(save_path / "metadata.parquet")
            
            if self.embeddings is not None:
                np.save(save_path / "embeddings.npy", self.embeddings)
            
            index_info = {
                'dimension': self.dimension,
                'config': self.config,
                'is_built': self.is_built,
                'index_type': type(self.index).__name__
            }
            joblib.dump(index_info, save_path / "index_info.joblib")
            
            logger.info(f"Index saved to: {save_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self, load_dir: str) -> bool:
        try:
            load_path = Path(load_dir)
            
            if not load_path.exists():
                logger.error(f"Load directory doesn't exist: {load_dir}")
                return False
            
            index_info = joblib.load(load_path / "index_info.joblib")
            self.dimension = index_info['dimension']
            self.config = index_info.get('config', self.config)
            
            self.index = faiss.read_index(str(load_path / "faiss_index"))
            
            if (load_path / "metadata.parquet").exists():
                self.metadata = pd.read_parquet(load_path / "metadata.parquet")
            
            if (load_path / "embeddings.npy").exists():
                self.embeddings = np.load(load_path / "embeddings.npy")
            
            self.is_built = True
            
            logger.info(f"Index loaded from: {load_dir}")
            logger.info(f"   Total vectors: {self.index.ntotal}")
            logger.info(f"   Dimension: {self.dimension}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def find_duplicates(self, threshold: float = 0.95) -> Optional[List[Tuple[int, int, float]]]:
        if not self.is_built:
            logger.error("Index not built")
            return None
        
        logger.info(f"Finding duplicates with similarity > {threshold}")
        
        try:
            duplicates = []
            batch_size = 1000 
            
            for i in range(0, len(self.embeddings), batch_size):
                batch_embeddings = self.embeddings[i:i + batch_size]
                
                similarities, indices = self.index.search(batch_embeddings, k=10) 
                
                for j, (sim_scores, idx_list) in enumerate(zip(similarities, indices)):
                    current_idx = i + j
                    
                    for k, (other_idx, sim_score) in enumerate(zip(idx_list[1:], sim_scores[1:])):
                        if sim_score > threshold and current_idx < other_idx:  # Avoid duplicates
                            duplicates.append((current_idx, int(other_idx), float(sim_score)))
            
            logger.info(f"Found {len(duplicates)} potential duplicates")
            return duplicates
            
        except Exception as e:
            logger.error(f"Failed to find duplicates: {e}")
            return None
    
    def get_cluster_centers(self, n_clusters: int = 10) -> Optional[List[Dict]]:
        if not self.is_built:
            logger.error("❌ Index not built")
            return None
        
        try:
            kmeans = faiss.Kmeans(self.dimension, n_clusters, niter=20, verbose=False)
            kmeans.train(self.embeddings)
            
            centroids = kmeans.centroids
            
            _, closest_indices = self.index.search(centroids, k=1)
            
            centers = []
            for i, (centroid, closest_idx) in enumerate(zip(centroids, closest_indices.flatten())):
                if closest_idx >= 0 and closest_idx < len(self.metadata):
                    item_data = self.metadata.iloc[closest_idx]
                    center_info = {
                        'cluster_id': i,
                        'centroid': centroid,
                        'closest_item_index': int(closest_idx),
                        'text': item_data.get('review', ''),
                        'sentiment': 'Positive' if item_data.get('sentimen', 0) == 1 else 'Negative'
                    }
                    centers.append(center_info)
            
            logger.info(f"Generated {len(centers)} cluster centers")
            return centers
            
        except Exception as e:
            logger.error(f"Failed to get cluster centers: {e}")
            return None
    
    def benchmark_search(self, n_queries: int = 100, k: int = 5) -> Dict:
        if not self.is_built:
            logger.error("Index not built")
            return {}
        
        logger.info(f"Benchmarking search with {n_queries} queries...")
        
        try:
            import time
            
            query_indices = np.random.choice(len(self.embeddings), size=n_queries, replace=False)
            query_embeddings = self.embeddings[query_indices]
            
            start_time = time.time()
            similarities, indices = self.index.search(query_embeddings, k)
            search_time = time.time() - start_time
            
            results = {
                'n_queries': n_queries,
                'k': k,
                'total_time': search_time,
                'avg_time_per_query': search_time / n_queries,
                'queries_per_second': n_queries / search_time
            }
            
            logger.info(f"Benchmark completed:")
            logger.info(f"   {results['queries_per_second']:.1f} queries/second")
            logger.info(f"   {results['avg_time_per_query']*1000:.2f} ms/query")
            
            return results
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    # Create dummy data for testing
    np.random.seed(42)
    
    # Generate synthetic embeddings and metadata
    n_items = 1000
    embedding_dim = 768
    embeddings = np.random.randn(n_items, embedding_dim).astype('float32')
    
    # Create dummy metadata
    texts = [f"Sample review text {i}" for i in range(n_items)]
    sentiments = np.random.randint(0, 2, n_items)
    metadata = pd.DataFrame({
        'review': texts,
        'review_clean': texts,
        'sentimen': sentiments
    })
    
    print("Testing similarity search:")
    print("-" * 50)
    
    # Initialize similarity search
    similarity_search = SimilaritySearch()
    
    # Build index
    if similarity_search.build_index(embeddings, metadata):
        print("Index built successfully")
        
        # Test search
        query_embedding = embeddings[:1]  # Use first embedding as query
        results = similarity_search.search(query_embedding, k=5)
        
        if results:
            print("Search completed")
            print(f"Found {len(results)} similar items")
            for i, result in enumerate(results):
                print(f"   {i+1}. Similarity: {result['similarity']:.3f}")
        
        # Get statistics
        stats = similarity_search.get_statistics()
        print("Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Benchmark search
        benchmark = similarity_search.benchmark_search(n_queries=100)
        if benchmark:
            print("Benchmark completed")
    else:
        print("Failed to build index")