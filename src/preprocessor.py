import pandas as pd
import numpy as np
import re
import logging
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from config.settings import PREPROCESSING_CONFIG

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)

class TextPreprocessor:
    
    def __init__(self, config: dict = PREPROCESSING_CONFIG):
        self.config = config
        
        try:
            self.stopwords = set(stopwords.words('indonesian'))
        except:
            logger.warning("Indonesian stopwords not available, using empty set")
            self.stopwords = set()
        
        indonesian_stopwords = {
            'yang', 'ini', 'itu', 'adalah', 'dengan', 'untuk', 'dari', 'ke', 'dalam',
            'pada', 'sebagai', 'oleh', 'akan', 'telah', 'sudah', 'atau', 'dan', 'juga',
            'saya', 'kamu', 'dia', 'kami', 'kalian', 'mereka', 'nya', 'ku', 'mu',
            'si', 'sang', 'para', 'pak', 'bu', 'bapak', 'ibu', 'mas', 'mbak'
        }
        self.stopwords.update(indonesian_stopwords)
        
        self.slang_dict = {
            'gak': 'tidak',
            'ga': 'tidak', 
            'nggak': 'tidak',
            'ngga': 'tidak',
            'emg': 'emang',
            'emang': 'memang',
            'bgt': 'banget',
            'bgt': 'banget',
            'tp': 'tapi',
            'trus': 'terus',
            'udah': 'sudah',
            'udh': 'sudah',
            'krn': 'karena',
            'dgn': 'dengan',
            'utk': 'untuk',
            'sm': 'sama',
            'gt': 'gitu',
            'gini': 'begini',
            'gitu': 'begitu',
            'klo': 'kalau',
            'kalo': 'kalau',
            'gmn': 'gimana',
            'gimana': 'bagaimana',
            'knp': 'kenapa',
            'kenapa': 'mengapa'
        }
        
        logger.info(f"Text preprocessor initialized with {len(self.stopwords)} stopwords")
    
    def clean_text(self, text: str) -> str:
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        text = str(text).strip()
        
        if not text:
            return ""
        
        if self.config['clean_patterns']['lowercase']:
            text = text.lower()
        
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        text = re.sub(r'\S+@\S+', '', text)
        
        pattern = self.config['clean_patterns']['special_chars']
        text = re.sub(pattern, ' ', text)
        
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        whitespace_pattern = self.config['clean_patterns']['extra_whitespace']
        text = re.sub(whitespace_pattern, ' ', text)
        
        text = self._normalize_slang(text)
        
        return text.strip()
    
    def _normalize_slang(self, text: str) -> str:
        words = text.split()
        normalized_words = []
        
        for word in words:
            if word.lower() in self.slang_dict:
                normalized_words.append(self.slang_dict[word.lower()])
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def remove_stopwords(self, text: str, remove_stopwords: bool = False) -> str:
        if not remove_stopwords or not text:
            return text
        
        try:
            words = word_tokenize(text)
            
            filtered_words = [word for word in words if word.lower() not in self.stopwords]
            
            return ' '.join(filtered_words)
        except:
            words = text.split()
            filtered_words = [word for word in words if word.lower() not in self.stopwords]
            return ' '.join(filtered_words)
    
    def preprocess_text(self, text: str, remove_stopwords: bool = False) -> str:
        clean_text = self.clean_text(text)
        
        if remove_stopwords:
            clean_text = self.remove_stopwords(clean_text, remove_stopwords=True)
        
        return clean_text
    
    def preprocess_texts(self, texts: List[str], remove_stopwords: bool = False) -> List[str]:
        processed_texts = []
        
        for text in texts:
            processed_text = self.preprocess_text(text, remove_stopwords=remove_stopwords)
            processed_texts.append(processed_text)
        
        return processed_texts
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing DataFrame...")
        
        df_clean = df.copy()
        
        if self.config.get('remove_duplicates', True):
            initial_size = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            logger.info(f"   Removed {initial_size - len(df_clean)} duplicates")
        
        df_clean = df_clean.dropna(subset=['review'])
        
        logger.info("   Cleaning text...")
        df_clean['review_clean'] = df_clean['review'].apply(self.clean_text)
        
        min_length = self.config.get('min_text_length', 5)
        initial_size = len(df_clean)
        df_clean = df_clean[df_clean['review_clean'].str.len() >= min_length]
        logger.info(f"   Removed {initial_size - len(df_clean)} texts shorter than {min_length} characters")
        
        if 'sentimen' in df_clean.columns:
            df_clean['sentimen'] = pd.to_numeric(df_clean['sentimen'], errors='coerce')
            # Remove rows with invalid sentiment labels
            df_clean = df_clean.dropna(subset=['sentimen'])
            df_clean['sentimen'] = df_clean['sentimen'].astype(int)
        
        logger.info(f"Preprocessing completed: {len(df_clean)} samples remaining")
        
        return df_clean.reset_index(drop=True)
    
    def get_text_statistics(self, texts: List[str]) -> dict:
        if not texts:
            return {}
        
        lengths = [len(text) for text in texts if text]
        word_counts = [len(text.split()) for text in texts if text]
        
        stats = {
            'total_texts': len(texts),
            'empty_texts': sum(1 for text in texts if not text or text.strip() == ''),
            'avg_length': np.mean(lengths) if lengths else 0,
            'median_length': np.median(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'avg_words': np.mean(word_counts) if word_counts else 0,
            'median_words': np.median(word_counts) if word_counts else 0
        }
        
        return stats
    
    def get_settings(self) -> dict:
        return {
            'config': self.config,
            'stopwords': list(self.stopwords),
            'slang_dict': self.slang_dict
        }
    
    def load_settings(self, settings: dict) -> None:
        self.config = settings.get('config', self.config)
        self.stopwords = set(settings.get('stopwords', self.stopwords))
        self.slang_dict = settings.get('slang_dict', self.slang_dict)
        
        logger.info("Preprocessor settings loaded")

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Test texts
    test_texts = [
        "Produk ini sangat bagussss dan berkualitas tinggi!!!",
        "kualitas produk sangat buruk, ga puas dgn pembelian ini :(",
        "pengiriman cepet bgt, produk juga sesuai deskripsi 👍",
        "harga mahal tp kualitas biasa aja, kurang recommended deh",
        "BARANG TIDAK SESUAI FOTO!!! SANGAT KECEWA!!!"
    ]
    
    print("Testing text preprocessing:")
    print("-" * 50)
    
    for i, text in enumerate(test_texts, 1):
        clean_text = preprocessor.clean_text(text)
        print(f"{i}. Original: {text}")
        print(f"   Cleaned:  {clean_text}")
        print()
    
    # Test statistics
    stats = preprocessor.get_text_statistics(test_texts)
    print("Text Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")