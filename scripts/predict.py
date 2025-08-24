import sys
import argparse
import pandas as pd
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.sentiment_pipeline import SentimentAnalysisPipeline
from config.settings import MODELS_DIR


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Indonesian Sentiment Analysis Prediction"
    )
    
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        '--text', '-t',
        type=str,
        help='Single text to analyze'
    )
    text_group.add_argument(
        '--file', '-f',
        type=str,
        help='CSV file with texts to analyze (must have "review" column)'
    )
    
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        default=str(MODELS_DIR / "sentiment_model"),
        help='Path to trained model (default: models/sentiment_model)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path for predictions (CSV format)'
    )
    
    parser.add_argument(
        '--similarity', '-s',
        action='store_true',
        help='Include similarity search results'
    )
    
    parser.add_argument(
        '--similarity-k',
        type=int,
        default=3,
        help='Number of similar reviews to return (default: 3)'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.5,
        help='Minimum confidence threshold for predictions (default: 0.5)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🇮🇩 Indonesian Sentiment Analysis - Prediction")
    print("=" * 50)
    
    try:
        logger.info("Initializing prediction pipeline...")
        pipeline = SentimentAnalysisPipeline()
        
        logger.info(f"Loading trained model from: {args.model_path}")
        if not pipeline.load_model(args.model_path):
            logger.error("Failed to load model")
            return False
        
        logger.info("Model loaded successfully")
        
        if args.text:
            result = predict_single_text(pipeline, args.text, args)
            if result:
                display_single_result(result, args)
        
        elif args.file:
            results = predict_batch_file(pipeline, args.file, args)
            if results:
                display_batch_results(results, args)
                
                if args.output:
                    save_results(results, args.output)
        
        return True
        
    except KeyboardInterrupt:
        logger.info("Prediction interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return False

def predict_single_text(pipeline, text: str, args) -> dict:
    logger.info(f"Analyzing text: {text[:50]}...")
    
    try:
        result = pipeline.predict_sentiment(text)
        
        if not result:
            logger.error("Failed to predict sentiment")
            return None
        
        if args.similarity:
            logger.info("Finding similar reviews...")
            similar_reviews = pipeline.find_similar_reviews(text, k=args.similarity_k)
            result['similar_reviews'] = similar_reviews
        
        return result
        
    except Exception as e:
        logger.error(f"Error in single text prediction: {e}")
        return None

def predict_batch_file(pipeline, file_path: str, args) -> list:
    logger.info(f"Processing file: {file_path}")
    
    try:
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            logger.error("Unsupported file format. Use CSV or Excel files.")
            return None
        
        if 'review' not in df.columns:
            logger.error("File must contain 'review' column")
            return None
        
        logger.info(f"Processing {len(df)} texts...")
        
        results = []
        for i, row in df.iterrows():
            text = row['review']
            
            if pd.isna(text) or not str(text).strip():
                logger.warning(f"Skipping empty text at row {i}")
                continue
            
            result = pipeline.predict_sentiment(str(text))
            
            if result:
                result['row_index'] = i
                result['original_text'] = text
                
                if args.similarity and i < 10:
                    similar_reviews = pipeline.find_similar_reviews(text, k=args.similarity_k)
                    result['similar_reviews'] = similar_reviews
                
                results.append(result)
            else:
                logger.warning(f"Failed to predict sentiment for row {i}")
        
        logger.info(f"Processed {len(results)} texts successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return None

def display_single_result(result: dict, args):
    print("Prediction Results:")
    print("-" * 30)
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Probabilities:")
    print(f"  • Negative: {result['probabilities']['negative']:.3f}")
    print(f"  • Positive: {result['probabilities']['positive']:.3f}")
    
    if result['confidence'] < args.confidence_threshold:
        print(f"Warning: Low confidence ({result['confidence']:.3f} < {args.confidence_threshold})")
    
    if 'similar_reviews' in result and result['similar_reviews']:
        print(f"Similar Reviews:")
        print("-" * 20)
        for i, similar in enumerate(result['similar_reviews'], 1):
            print(f"{i}. Similarity: {similar['similarity']:.3f}")
            print(f"   Text: {similar['text'][:100]}{'...' if len(similar['text']) > 100 else ''}")
            print(f"   Sentiment: {similar['sentiment']}")
            print()

def display_batch_results(results: list, args):
    print(f"Batch Prediction Results ({len(results)} items):")
    print("-" * 50)
    
    sentiments = [r['sentiment'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    positive_count = sum(1 for s in sentiments if s == 'Positive')
    negative_count = sum(1 for s in sentiments if s == 'Negative')
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    low_confidence_count = sum(1 for c in confidences if c < args.confidence_threshold)
    
    print(f"Summary:")
    print(f"  • Positive: {positive_count} ({positive_count/len(results)*100:.1f}%)")
    print(f"  • Negative: {negative_count} ({negative_count/len(results)*100:.1f}%)")
    print(f"  • Average Confidence: {avg_confidence:.3f}")
    print(f"  • Low Confidence (<{args.confidence_threshold}): {low_confidence_count}")
    
    print(f"Sample Results (first 5):")
    for i, result in enumerate(results[:5]):
        print(f"\n{i+1}. Row {result.get('row_index', i)}:")
        print(f"   Text: {result['text'][:80]}{'...' if len(result['text']) > 80 else ''}")
        print(f"   Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.3f})")

def save_results(results: list, output_path: str):
    try:
        data = []
        for result in results:
            row = {
                'original_text': result.get('original_text', result['text']),
                'clean_text': result.get('clean_text', ''),
                'predicted_sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'negative_probability': result['probabilities']['negative'],
                'positive_probability': result['probabilities']['positive'],
                'row_index': result.get('row_index', '')
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif output_path.endswith(('.xlsx', '.xls')):
            df.to_excel(output_path, index=False)
        else:
            output_path += '.csv'
            df.to_csv(output_path, index=False)
        
        logger.info(f"Results saved to: {output_path}")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")

if __name__ == "__main__":
    success = main()
    
    if success:
        print("Prediction completed successfully!")
        sys.exit(0)
    else:
        print("Prediction failed!")
        sys.exit(1)