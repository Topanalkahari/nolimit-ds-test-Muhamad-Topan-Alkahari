import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, f_classif
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

print("🇮🇩 Indonesian Product Reviews Sentiment Analysis")
print("Technical Test for NoLimit Indonesia - Data Scientist Position")
print("Full dataset version with optimizations")

class SentimentAnalysisPipeline:
    """
    Optimized sentiment analysis pipeline for Indonesian product reviews using full dataset
    """
    
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None
        self.classifier = None
        self.faiss_index = None
        self.embeddings = None
        self.labels = None
        self.feature_selector = None

    def load_data(self):
        """Load full dataset from Hugging Face"""
        print("Loading dataset...")
        try:
            dataset = load_dataset("dipawidia/ecommerce-product-reviews-sentiment", "default", split="train")
            df = dataset.to_pandas()
            
            print(f"Dataset loaded successfully!")
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Sentiment distribution:")
            print(df['sentimen'].value_counts())
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def preprocess_data(self, df):
        """Clean and preprocess the data with enhanced text cleaning"""
        print("Preprocessing data...")
        
        # Remove duplicates and missing values
        df_clean = df.drop_duplicates().dropna()
        df_clean['sentimen'] = df_clean['sentimen'].astype(int)
        
        # Enhanced text cleaning
        def clean_text(text):
            if pd.isna(text):
                return ""
            text = str(text).strip()
            
            # Remove special characters but keep Indonesian words
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            return text.lower().strip()
        
        df_clean['review_clean'] = df_clean['review'].apply(clean_text)
        
        # Remove empty texts after cleaning
        df_clean = df_clean[df_clean['review_clean'].str.len() > 5]
        
        print(f"Data cleaned: {len(df_clean)} samples remaining")
        return df_clean

    def load_model(self):
        """Load the sentence transformer model"""
        print("Loading sentence transformer model...")
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded: {self.model_name}")
            print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def generate_embeddings(self, texts, batch_size=128):
        """Generate embeddings for the texts with larger batch size"""
        print("Generating embeddings...")
        try:
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size, 
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            print(f"Embeddings generated: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return None
    
    def build_faiss_index(self, embeddings, labels):
        """Build FAISS index for similarity search"""
        print("Building FAISS index...")
        try:
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add vectors to index in batches to avoid memory issues
            batch_size = 1000
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i+batch_size].astype('float32')
                self.faiss_index.add(batch)
            
            # Store embeddings and labels
            self.embeddings = embeddings
            self.labels = labels
            
            print(f"FAISS index built: {self.faiss_index.ntotal} vectors")
            return True
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            return False
    
    def select_features(self, X, y, k=500):
        """Select most important features to reduce dimensionality"""
        print("Selecting most important features...")
        try:
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
            X_selected = self.feature_selector.fit_transform(X, y)
            print(f"Feature selection completed: {X_selected.shape}")
            return X_selected
        except Exception as e:
            print(f"Error in feature selection: {e}")
            return X
    
    def train_classifier(self, X_train, y_train):
        """Train a classifier on the embeddings with optimized parameters"""
        print("Training classifier...")
        try:
            # Use faster classifiers with optimized parameters
            # Start with Logistic Regression as it's generally faster for high-dimensional data
            self.classifier = LogisticRegression(
                C=1.0, 
                solver='lbfgs', 
                max_iter=1000, 
                random_state=42,
                n_jobs=-1
            )
            
            # Train the classifier
            self.classifier.fit(X_train, y_train)
            
            # Cross-validation score with fewer folds for speed
            cv_scores = cross_val_score(self.classifier, X_train, y_train, cv=3, n_jobs=-1)
            print(f"Classifier trained successfully")
            print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            return True
        except Exception as e:
            print(f"Error training classifier: {e}")
            return False

    def evaluate_model(self, X_test, y_test):
        """Evaluate the model performance"""
        print("Evaluating model performance...")
        try:
            # Predictions
            y_pred = self.classifier.predict(X_test)
            y_pred_proba = self.classifier.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Accuracy: {accuracy:.3f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            self.plot_confusion_matrix(cm)
            
            return {
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return None
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def similarity_search(self, query_text, k=5):
        """Find similar reviews using FAISS"""
        print(f"Finding similar reviews for: '{query_text}'")
        try:
            # Generate embedding for query
            query_embedding = self.model.encode([query_text])
            faiss.normalize_L2(query_embedding)
            
            # Search similar vectors
            similarities, indices = self.faiss_index.search(
                query_embedding.astype('float32'), k
            )
            
            print(f"Found {k} similar reviews:")
            for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
                print(f"{i+1}. Similarity: {sim:.3f}")
                print(f"   Review: {self.labels.iloc[idx]['review'][:100]}...")
                print(f"   Sentiment: {'Positive' if self.labels.iloc[idx]['sentimen'] == 1 else 'Negative'}")
                print()
            
            return similarities[0], indices[0]
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return None, None
    
    def predict_sentiment(self, text):
        """Predict sentiment for new text"""
        print(f"Predicting sentiment for: '{text}'")
        try:
            # Generate embedding
            embedding = self.model.encode([text])
            
            # Apply feature selection if available
            if self.feature_selector:
                embedding = self.feature_selector.transform(embedding)
            
            # Predict using classifier
            prediction = self.classifier.predict(embedding)[0]
            probability = self.classifier.predict_proba(embedding)[0]
            
            sentiment = "Positive" if prediction == 1 else "Negative"
            confidence = probability[prediction]
            
            print(f"Prediction: {sentiment}")
            print(f"Confidence: {confidence:.3f}")
            
            return {
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
            }
        except Exception as e:
            print(f"Error predicting sentiment: {e}")
            return None

    def run_pipeline(self):
        """Run the complete pipeline with full dataset"""
        print("Starting Sentiment Analysis Pipeline")
        print("=" * 50)
        
        # 1. Load data
        df = self.load_data()
        if df is None:
            return False
        
        # 2. Preprocess data
        df_clean = self.preprocess_data(df)
        
        # 3. Load model
        self.load_model()
        if self.model is None:
            return False
        
        # 4. Generate embeddings for Indonesian text
        print("\n🇮🇩 Generating embeddings for Indonesian reviews...")
        indonesian_embeddings = self.generate_embeddings(df_clean['review_clean'].tolist())
        
        # 5. Apply feature selection to reduce dimensionality
        X_selected = self.select_features(
            indonesian_embeddings, 
            df_clean['sentimen'], 
            k=500  # Keep top 500 features
        )
        
        # 6. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, 
            df_clean['sentimen'], 
            test_size=0.2, 
            random_state=42,
            stratify=df_clean['sentimen']
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # 7. Train classifier
        if not self.train_classifier(X_train, y_train):
            return False
        
        # 8. Evaluate model
        results = self.evaluate_model(X_test, y_test)
        if results is None:
            return False
        
        # 9. Build FAISS index for similarity search (using full embeddings)
        if not self.build_faiss_index(indonesian_embeddings, df_clean):
            return False
        
        print("Pipeline completed successfully!")
        return True

    def interactive_testing(self):
        """Interactive testing interface"""
        print("Interactive Testing Mode")
        print("=" * 30)
        print("Enter Indonesian text to analyze sentiment (type 'quit' to exit)")
        
        while True:
            text = input("Enter text: ").strip()
            
            if text.lower() == 'quit':
                break
            
            if not text:
                continue
            
            # Predict sentiment
            result = self.predict_sentiment(text)
            
            if result:
                # Find similar reviews
                print("Finding similar reviews...")
                self.similarity_search(text, k=3)
            
            print("-" * 50)

# Initialize pipeline with a model that produces 768-dim embeddings
pipeline = SentimentAnalysisPipeline(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

print("🇮🇩 Indonesian Product Reviews Sentiment Analysis")
print("=" * 60)
print("Technical Test for NoLimit Indonesia - Data Scientist Position")
print("=" * 60)

# Run the complete pipeline with full dataset
success = pipeline.run_pipeline()

if success:
    print("Pipeline Results Summary:")
    print("Full dataset loaded and preprocessed")
    print("Sentence transformer model loaded")
    print("Embeddings generated")
    print("Feature selection applied")
    print("Classifier trained with optimized parameters")
    print("FAISS index built")
    print("Model evaluated")
    
    # Interactive testing
    pipeline.interactive_testing()
else:
    print("Pipeline failed. Please check the error messages above.")

# Example: Test with sample Indonesian text
if success:
    print("Testing with sample Indonesian text:")
    
    sample_texts = [
        "Produk ini sangat bagus dan berkualitas tinggi!",
        "Kualitas produk sangat buruk, tidak puas dengan pembelian ini.",
        "Pengiriman cepat dan produk sesuai deskripsi."
    ]
    
    for text in sample_texts:
        print(f"Testing: {text}")
        result = pipeline.predict_sentiment(text)
        if result:
            print(f"Result: {result['sentiment']} (Confidence: {result['confidence']:.3f})")
        
        # Find similar reviews
        print("Similar reviews:")
        pipeline.similarity_search(text, k=2)
        print("-" * 50)