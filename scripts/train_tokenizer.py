import os
import sentencepiece as spm
from utils.config import DATASET_DIR
from utils.logger import logger

def train_tokenizer(vocab_size=32000):
    """
    Trains a SentencePiece tokenizer on the cleaned dataset.
    """
    input_file = os.path.join(DATASET_DIR, "train.txt")
    model_prefix = os.path.join(DATASET_DIR, "haryanvi_tokenizer")
    
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} not found. Run clean_transcripts.py first.")
        return
        
    logger.info(f"Training tokenizer with vocab_size={vocab_size} on {input_file}...")
    
    spm_args = [
        f"--input={input_file}",
        f"--model_prefix={model_prefix}",
        f"--vocab_size={vocab_size}",
        "--model_type=unigram", # Better for morphologically rich languages
        "--character_coverage=0.9995", # Default for non-CJK languages
        "--pad_id=0",
        "--unk_id=1",
        "--bos_id=2",
        "--eos_id=3",
        "--pad_piece=[PAD]",
        "--unk_piece=[UNK]",
        "--bos_piece=[BOS]",
        "--eos_piece=[EOS]",
    ]
    
    try:
        spm.SentencePieceTrainer.train(" ".join(spm_args))
        logger.info(f"Tokenizer training complete. Model saved at {model_prefix}.model")
    except Exception as e:
        logger.error(f"Tokenizer training failed: {str(e)}")

if __name__ == "__main__":
    train_tokenizer()
