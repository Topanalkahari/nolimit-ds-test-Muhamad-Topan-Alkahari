import sys
import argparse
import logging
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent))

from src.sentiment_pipeline import SentimentAnalysisPipeline
from config.settings import MODELS_DIR, RANDOM_SEED

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Indonesian Sentiment Analysis Pipeline - NoLimit Indonesia Technical Test"
    )
    
    parser.add_argument(
        '--save-model', 
        type=str, 
        default=str(MODELS_DIR / "sentiment_model"),
        help="Path to save the trained model (default: models/sentiment_model)"
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help="Use a sample of the dataset for faster testing (default: use full dataset)"
    )
    
    parser.add_argument(
        '--skip-similarity',
        action='store_true',
        help="Skip building similarity search index to save time"
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help="Run interactive testing mode after training"
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=" * 60)
    print("🇮🇩 INDONESIAN SENTIMENT ANALYSIS PIPELINE")
    print("NoLimit Indonesia - Data Scientist Technical Test")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        logger.info("Initializing sentiment analysis pipeline...")
        pipeline = SentimentAnalysisPipeline()
        
        logger.info("Step 1: Loading dataset...")
        if not pipeline.load_data():
            logger.error("Failed to load dataset")
            return False
        
        if args.sample_size:
            logger.info(f"Using sample size: {args.sample_size}")
            pipeline.dataset = pipeline.data_loader.get_sample_data(args.sample_size)
        
        logger.info("Step 2: Preprocessing data...")
        if not pipeline.preprocess_data():
            logger.error("Failed to preprocess data")
            return False
        
        logger.info("Step 3: Generating embeddings...")
        if not pipeline.generate_embeddings():
            logger.error("Failed to generate embeddings")
            return False
        
        logger.info("Step 4: Training classifier...")
        if not pipeline.train_classifier():
            logger.error("Failed to train classifier")
            return False
        
        if not args.skip_similarity:
            logger.info("Step 5: Building similarity search index...")
            if not pipeline.build_similarity_index():
                logger.error("Failed to build similarity index")
                return False
        else:
            logger.info("Step 5: Skipping similarity search index")
        
        if args.save_model:
            logger.info(f"Saving model to: {args.save_model}")
            if pipeline.save_model(args.save_model):
                logger.info("Model saved successfully")
            else:
                logger.error("Failed to save model")
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        eval_results = pipeline.evaluate_model()
        if eval_results:
            print(f"Model Performance:")
            print(f"   • Accuracy: {eval_results['accuracy']:.3f}")
            print(f"   • F1-Score: {eval_results.get('f1_score', 'N/A'):.3f}")
            print(f"   • Precision: {eval_results.get('precision', 'N/A'):.3f}")
            print(f"   • Recall: {eval_results.get('recall', 'N/A'):.3f}")
        
        print(f"Total Duration: {duration:.2f} seconds")
        print(f"Model saved to: {args.save_model}")
        
        if args.interactive:
            interactive_mode(pipeline, args.skip_similarity)
        
        return True
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        return False

def interactive_mode(pipeline, skip_similarity):
    print("Interactive Testing Mode")
    print("=" * 40)
    print("Enter Indonesian text to analyze sentiment (type 'quit' to exit)")
    print("Commands: 'examples' to see sample texts, 'help' for help")
    
    while True:
        try:
            text = input("Enter text: ").strip()
            
            if not text:
                continue
            
            if text.lower() == 'quit':
                print("Goodbye!")
                break
            
            if text.lower() == 'help':
                print_help()
                continue
            
            if text.lower() == 'examples':
                print_examples()
                continue
            
            print("Analyzing sentiment...")
            result = pipeline.predict_sentiment(text)
            
            if result:
                print(f"Result:")
                print(f"   • Sentiment: {result['sentiment']}")
                print(f"   • Confidence: {result['confidence']:.3f}")
                print(f"   • Probabilities:")
                print(f"     - Negative: {result['probabilities']['negative']:.3f}")
                print(f"     - Positive: {result['probabilities']['positive']:.3f}")
                
                if not skip_similarity:
                    print("Finding similar reviews...")
                    similar = pipeline.find_similar_reviews(text, k=3)
                    if similar:
                        for i, sim in enumerate(similar):
                            print(f"{i+1}. Similarity: {sim['similarity']:.3f}")
                            print(f"   Text: {sim['text'][:100]}...")
                            print(f"   Sentiment: {sim['sentiment']}")
            else:
                print("Failed to analyze sentiment")
            
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def print_help():
    print("Help:")
    print("• Enter any Indonesian text to analyze its sentiment")
    print("• Type 'examples' to see sample texts")
    print("• Type 'quit' to exit")
    print("• Examples of Indonesian text:")
    print("  - 'Produk ini sangat bagus!'")
    print("  - 'Kualitas buruk, tidak recommended'")

def print_examples():
    examples = [
        "Produk ini sangat bagus dan berkualitas tinggi!",
        "Kualitas produk sangat buruk, tidak puas.",
        "Pengiriman cepat dan sesuai deskripsi.",
        "Harga mahal tapi kualitas biasa saja.",
        "Sangat puas dengan pembelian ini, recommended!"
    ]
    
    print("Example Indonesian Texts:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")

if __name__ == "__main__":
    success = main()
    
    if success:
        print("Pipeline execution completed successfully!")
        sys.exit(0)
    else:
        print("Pipeline execution failed!")
        sys.exit(1)