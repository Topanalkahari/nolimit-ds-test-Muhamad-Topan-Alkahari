import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
import faiss
import re


st.set_page_config(
    page_title="🇮🇩 Indonesian Sentiment Analysis",
    page_icon="🇮🇩",
    layout="wide"
)


st.title("🇮🇩 Indonesian Product Reviews Sentiment Analysis")
st.markdown("### Technical Test for NoLimit Indonesia - Data Scientist Position")


if 'model' not in st.session_state:
    st.session_state.model = None
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'review_data' not in st.session_state:
    st.session_state.review_data = None
if 'embeddings_matrix' not in st.session_state:
    st.session_state.embeddings_matrix = None

def load_and_preprocess_data():
    with st.spinner("📊 Loading dataset..."):
        try:
            dataset = load_dataset("dipawidia/ecommerce-product-reviews-sentiment", "default", split="train")
            df = dataset.to_pandas()
            
            df_clean = df.drop_duplicates().dropna()
            df_clean['sentimen'] = df_clean['sentimen'].astype(int)
            
            def clean_text(text):
                if pd.isna(text):
                    return ""
                text = str(text).strip()
                text = re.sub(r'[^a-zA-Z\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text)
                return text.lower().strip()
            
            df_clean['review_clean'] = df_clean['review'].apply(clean_text)
            df_clean = df_clean[df_clean['review_clean'].str.len() > 5]
            
            st.success(f"✅ Dataset loaded: {len(df_clean)} samples")
            return df_clean
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return None

def build_faiss_index(embeddings, df_clean):
    with st.spinner("🔍 Building FAISS index for similarity search..."):
        try:
            faiss.normalize_L2(embeddings.astype(np.float32))
            
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings.astype(np.float32))
            
            st.session_state.faiss_index = index
            st.session_state.review_data = df_clean.copy()
            st.session_state.embeddings_matrix = embeddings
            
            st.success(f"✅ FAISS index built with {index.ntotal} vectors")
            return index
        except Exception as e:
            st.error(f"❌ FAISS index error: {e}")
            return None

def train_model(df_clean):
    with st.spinner("🤖 Training model..."):
        try:
            model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
            st.session_state.model = model
            
            texts = df_clean['review_clean'].tolist()
            embeddings = model.encode(texts, show_progress_bar=False)
            
            build_faiss_index(embeddings.copy(), df_clean)
            
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, df_clean['sentimen'], test_size=0.2, random_state=42
            )
            
            classifier = LogisticRegression(random_state=42, max_iter=1000)
            classifier.fit(X_train, y_train)
            st.session_state.classifier = classifier
            
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.success(f"✅ Model trained! Accuracy: {accuracy:.3f}")
            st.session_state.is_trained = True
            
            return accuracy
            
        except Exception as e:
            st.error(f"❌ Training error: {e}")
            return None

def predict_sentiment(text):
    if st.session_state.model is None or st.session_state.classifier is None:
        return None
    
    try:
        embedding = st.session_state.model.encode([text])
        
        prediction = st.session_state.classifier.predict(embedding)[0]
        probability = st.session_state.classifier.predict_proba(embedding)[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = probability[prediction]
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': probability
        }
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None

def search_similar_reviews(query_text, k=5):
    if (st.session_state.model is None or 
        st.session_state.faiss_index is None or 
        st.session_state.review_data is None):
        return None
    
    try:
        query_embedding = st.session_state.model.encode([query_text])
        
        faiss.normalize_L2(query_embedding.astype(np.float32))
        
        similarities, indices = st.session_state.faiss_index.search(
            query_embedding.astype(np.float32), k
        )
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(st.session_state.review_data):
                review_data = st.session_state.review_data.iloc[idx]
                results.append({
                    'rank': i + 1,
                    'similarity': similarity,
                    'review': review_data['review'],
                    'review_clean': review_data['review_clean'],
                    'sentiment': 'Positive' if review_data['sentimen'] == 1 else 'Negative',
                    'sentiment_score': review_data['sentimen']
                })
        
        return results
        
    except Exception as e:
        st.error(f"❌ Search error: {e}")
        return None

def calculate_embedding_stats():
    if st.session_state.embeddings_matrix is None:
        return None
    
    try:
        embeddings = st.session_state.embeddings_matrix
        
        sample_size = min(1000, len(embeddings))
        sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[sample_indices]
        
        normalized_embeddings = sample_embeddings / np.linalg.norm(sample_embeddings, axis=1, keepdims=True)
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        similarity_matrix = similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)]
        
        stats = {
            'embedding_dimension': embeddings.shape[1],
            'total_vectors': embeddings.shape[0],
            'avg_similarity': np.mean(similarity_matrix),
            'std_similarity': np.std(similarity_matrix),
            'min_similarity': np.min(similarity_matrix),
            'max_similarity': np.max(similarity_matrix)
        }
        
        return stats
        
    except Exception as e:
        st.error(f"❌ Stats calculation error: {e}")
        return None

tab1, tab2, tab3, tab4 = st.tabs(["🚀 Setup", "🎯 Predict", "🔍 Search", "📊 Analysis"])

with tab1:
    st.header("🚀 Setup Pipeline")
    
    if st.button("🚀 Load Data & Train Model", type="primary"):
        df_clean = load_and_preprocess_data()
        
        if df_clean is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Reviews", len(df_clean))
            with col2:
                st.metric("Positive", len(df_clean[df_clean['sentimen'] == 1]))
            with col3:
                st.metric("Negative", len(df_clean[df_clean['sentimen'] == 0]))
 
            accuracy = train_model(df_clean)
            
            if accuracy:
                st.success("🎉 Pipeline ready! You can now use the Predict and Search tabs.")
                
                stats = calculate_embedding_stats()
                if stats:
                    st.subheader("📊 Embedding Space Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Embedding Dimension", stats['embedding_dimension'])
                        st.metric("Total Vectors", stats['total_vectors'])
                    with col2:
                        st.metric("Avg Similarity", f"{stats['avg_similarity']:.3f}")
                        st.metric("Std Similarity", f"{stats['std_similarity']:.3f}")
                    with col3:
                        st.metric("Min Similarity", f"{stats['min_similarity']:.3f}")
                        st.metric("Max Similarity", f"{stats['max_similarity']:.3f}")

with tab2:
    st.header("🎯 Predict Sentiment")
    
    if not st.session_state.is_trained:
        st.warning("⚠️ Please run the pipeline first in the Setup tab.")
    else:
        text_input = st.text_area(
            "Enter Indonesian text to analyze:",
            placeholder="Masukkan teks dalam bahasa Indonesia untuk dianalisis...",
            height=100
        )
        
        if st.button("🎯 Predict", type="primary"):
            if text_input.strip():
                result = predict_sentiment(text_input.strip())
                
                if result:
                    col1, col2 = st.columns(2)
                    with col1:
                        color = "green" if result['sentiment'] == "Positive" else "red"
                        st.markdown(f"<h2 style='color: {color};'>{result['sentiment']}</h2>", 
                                  unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.3f}")
                    
                    st.subheader("📊 Sentiment Probabilities")
                    prob_df = pd.DataFrame({
                        'Sentiment': ['Negative', 'Positive'],
                        'Probability': result['probabilities']
                    })
                    st.bar_chart(prob_df.set_index('Sentiment'))
            else:
                st.warning("Please enter some text to analyze.")

with tab3:
    st.header("🔍 Semantic Search")
    
    if not st.session_state.is_trained:
        st.warning("⚠️ Please run the pipeline first in the Setup tab.")
    else:
        st.markdown("**Find reviews similar to your input using semantic embeddings!**")
        
        search_query = st.text_area(
            "Enter text to find similar reviews:",
            placeholder="Masukkan teks untuk mencari review serupa...",
            height=80
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            num_results = st.slider("Number of results", min_value=3, max_value=15, value=5)
        
        if st.button("🔍 Search Similar Reviews", type="primary"):
            if search_query.strip():
                results = search_similar_reviews(search_query.strip(), k=num_results)
                
                if results:
                    st.subheader(f"📋 Top {len(results)} Similar Reviews")
                    
                    for result in results:
                        with st.expander(f"#{result['rank']} - Similarity: {result['similarity']:.3f} - {result['sentiment']}"):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write("**Original Review:**")
                                st.write(result['review'])
                            with col2:
                                sentiment_color = "green" if result['sentiment'] == "Positive" else "red"
                                st.markdown(f"**Sentiment:** <span style='color: {sentiment_color};'>{result['sentiment']}</span>", 
                                          unsafe_allow_html=True)
                                st.write(f"**Similarity:** {result['similarity']:.3f}")
                    
                    st.subheader("📊 Similarity Scores")
                    similarity_df = pd.DataFrame({
                        'Review': [f"#{r['rank']}" for r in results],
                        'Similarity': [r['similarity'] for r in results]
                    })
                    st.bar_chart(similarity_df.set_index('Review'))
                    
                else:
                    st.error("No similar reviews found.")
            else:
                st.warning("Please enter some text to search.")
        
        with st.expander("💡 Search Tips"):
            st.markdown("""
            - **Semantic search** finds reviews with similar *meaning*, not just similar words
            - Try searching with product names, emotions, or specific features
            - The search uses multilingual embeddings, so it works well with Indonesian text
            - Similarity scores range from 0 (completely different) to 1 (identical)
            - Higher similarity scores indicate more semantically similar content
            """)

with tab4:
    st.header("📊 Dataset Analysis")
    
    if not st.session_state.is_trained:
        st.warning("⚠️ Please run the pipeline first in the Setup tab.")
    else:
        try:
            dataset = load_dataset("dipawidia/ecommerce-product-reviews-sentiment", "default", split="train")
            df = dataset.to_pandas()
            
            st.subheader("🎯 Sentiment Distribution")
            sentiment_counts = df['sentimen'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(sentiment_counts)
            
            with col2:
                st.bar_chart(sentiment_counts)
            
            st.subheader("📝 Sample Reviews")
            sample_df = df.sample(n=5)[['review', 'sentimen']]
            sample_df['sentiment'] = sample_df['sentimen'].map({1: 'Positive', 0: 'Negative'})
            st.dataframe(sample_df, use_container_width=True)
            
            if st.session_state.faiss_index is not None:
                st.subheader("🔍 FAISS Index Information")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Index Type", "IndexFlatIP")
                with col2:
                    st.metric("Total Vectors", st.session_state.faiss_index.ntotal)
                with col3:
                    st.metric("Vector Dimension", st.session_state.faiss_index.d)
                
                st.info("💡 Using FAISS IndexFlatIP for exact cosine similarity search")
            
        except Exception as e:
            st.error(f"Error in analysis: {e}")

st.markdown("---")
st.markdown("🇮🇩 **Indonesian Sentiment Analysis with Semantic Search** - Built with Streamlit & FAISS")
st.markdown("Technical Test for NoLimit Indonesia Data Scientist Position")