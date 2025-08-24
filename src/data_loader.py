import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import Optional, Tuple
import logging
from config.settings import DATASET_CONFIG, RANDOM_SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    
    def __init__(self, dataset_config: dict = DATASET_CONFIG):
        self.dataset_config = dataset_config
        self.dataset = None
        self.df = None
        
    def load_dataset(self) -> Optional[pd.DataFrame]:
        logger.info("Loading dataset from Hugging Face...")
        
        try:
            self.dataset = load_dataset(
                self.dataset_config["name"],
                split=self.dataset_config["split"],
                cache_dir=self.dataset_config.get("cache_dir")
            )
            
            self.df = self.dataset.to_pandas()
            
            logger.info(f"Dataset loaded successfully!")
            logger.info(f"Dataset shape: {self.df.shape}")
            logger.info(f"Columns: {list(self.df.columns)}")
            
            self._display_dataset_info()
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
    
    def _display_dataset_info(self) -> None:
        if self.df is not None:
            logger.info("Dataset Information:")
            logger.info(f"   • Total samples: {len(self.df):,}")
            logger.info(f"   • Missing values: {self.df.isnull().sum().sum()}")
            
            if 'sentimen' in self.df.columns:
                sentiment_counts = self.df['sentimen'].value_counts().sort_index()
                logger.info("Sentiment Distribution:")
                for label, count in sentiment_counts.items():
                    sentiment_name = "Positive" if label == 1 else "Negative"
                    percentage = (count / len(self.df)) * 100
                    logger.info(f"   • {sentiment_name}: {count:,} ({percentage:.1f}%)")
            
            if 'review' in self.df.columns:
                text_lengths = self.df['review'].str.len()
                logger.info("Text Length Statistics:")
                logger.info(f"   • Mean: {text_lengths.mean():.1f}")
                logger.info(f"   • Median: {text_lengths.median():.1f}")
                logger.info(f"   • Min: {text_lengths.min()}")
                logger.info(f"   • Max: {text_lengths.max()}")
    
    def get_sample_data(self, n_samples: int = 1000, stratify: bool = True) -> pd.DataFrame:
        if self.df is None:
            logger.error("Dataset not loaded. Call load_dataset() first.")
            return None
        
        if n_samples >= len(self.df):
            logger.warning(f"Requested samples ({n_samples}) >= dataset size ({len(self.df)})")
            return self.df
        
        if stratify and 'sentimen' in self.df.columns:
            sample_df = self.df.groupby('sentimen', group_keys=False).apply(
                lambda x: x.sample(
                    min(len(x), n_samples // self.df['sentimen'].nunique()),
                    random_state=RANDOM_SEED
                )
            )
        else:
            sample_df = self.df.sample(n=n_samples, random_state=RANDOM_SEED)
        
        logger.info(f"Sample created: {len(sample_df)} samples")
        return sample_df.reset_index(drop=True)
    
    def save_sample_data(self, output_path: str, n_samples: int = 100) -> bool:
        try:
            sample_df = self.get_sample_data(n_samples)
            if sample_df is not None:
                sample_df.to_csv(output_path, index=False)
                logger.info(f"Sample data saved to: {output_path}")
                return True
        except Exception as e:
            logger.error(f"Error saving sample data: {e}")
        
        return False
    
    def validate_dataset(self) -> bool:
        if self.df is None:
            logger.error("No dataset loaded")
            return False
        
        required_columns = ['review', 'sentimen']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        empty_reviews = self.df['review'].isnull().sum() + (self.df['review'].str.strip() == '').sum()
        if empty_reviews > 0:
            logger.warning(f"Found {empty_reviews} empty reviews")
        
        # Check sentiment labels
        unique_labels = self.df['sentimen'].unique()
        expected_labels = {0, 1}
        
        if not set(unique_labels).issubset(expected_labels):
            logger.error(f"Invalid sentiment labels: {unique_labels}")
            return False
        
        logger.info("Dataset validation passed")
        return True
    
    def get_dataset_statistics(self) -> dict:
        if self.df is None:
            return {}
        
        stats = {
            'total_samples': len(self.df),
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'sentiment_distribution': self.df['sentimen'].value_counts().to_dict(),
            'text_stats': {
                'mean_length': self.df['review'].str.len().mean(),
                'median_length': self.df['review'].str.len().median(),
                'min_length': self.df['review'].str.len().min(),
                'max_length': self.df['review'].str.len().max()
            }
        }
        
        return stats

# Example usage
if __name__ == "__main__":
    # Initialize data loader
    loader = DataLoader()
    
    # Load dataset
    df = loader.load_dataset()
    
    if df is not None:
        # Validate dataset
        is_valid = loader.validate_dataset()
        
        # Get statistics
        stats = loader.get_dataset_statistics()
        print("Dataset Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Save sample data
        loader.save_sample_data("data/sample_data.csv", n_samples=100)