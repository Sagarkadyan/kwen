import argparse
import sys
from scripts import crawl_channels, fetch_transcripts, clean_transcripts, train_tokenizer
from utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description="Haryanvi YouTube Transcript Pipeline")
    parser.add_argument("--step", choices=["crawl", "fetch", "clean", "tokenize", "all"], default="all",
                        help="Choose which step of the pipeline to run.")
    
    args = parser.parse_args()
    
    try:
        if args.step in ["crawl", "all"]:
            logger.info(">>> STEP 1: Crawling channels for video IDs...")
            crawl_channels.main()
            
        if args.step in ["fetch", "all"]:
            logger.info(">>> STEP 2: Fetching transcripts (API + ASR fallback)...")
            fetch_transcripts.main()
            
        if args.step in ["clean", "all"]:
            logger.info(">>> STEP 3: Cleaning transcripts and building dataset...")
            clean_transcripts.main()
            
        if args.step in ["tokenize", "all"]:
            logger.info(">>> STEP 4: Training tokenizer...")
            train_tokenizer.train_tokenizer()
            
        logger.info("Pipeline execution finished successfully.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
