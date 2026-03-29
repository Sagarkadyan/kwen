import os
import json
import re
import random
from utils.config import RAW_DIR, PROCESSED_DIR, DATASET_DIR
from utils.logger import logger

def clean_text(text):
    """
    Cleaning pipeline for transcript text.
    """
    # Remove noise tokens
    text = re.sub(r'\[.*?\]', '', text) # Remove [Music], [Applause], etc.
    text = re.sub(r'\(.*?\)', '', text) # Remove (laughs), etc.
    
    # Remove metadata/timestamps if any leaked in
    text = re.sub(r'\d{1,2}:\d{2}', '', text)
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    # Basic normalization for Hindi/Haryanvi (optional specific rules can be added here)
    # text = text.replace("v", "b") # Common in Haryanvi dialects sometimes
    
    return text.strip()

def process_transcripts():
    """
    Cleans all JSON transcripts and merges them into one text per video.
    """
    logger.info("Starting text cleaning and dataset construction...")
    
    stats = {
        "videos_processed": 0,
        "total_words": 0,
        "source_counts": {"api": 0, "api-generated": 0, "asr": 0}
    }
    
    all_sentences = []
    
    for filename in os.listdir(RAW_DIR):
        if not filename.endswith(".json") or filename == "video_ids.json":
            continue
            
        with open(os.path.join(RAW_DIR, filename), "r") as f:
            data = json.load(f)
            
        source = data.get("source", "unknown")
        transcript_data = data.get("transcript", [])
        
        video_text = []
        for segment in transcript_data:
            cleaned = clean_text(segment["text"])
            if cleaned:
                video_text.append(cleaned)
                
        if video_text:
            # Join segments into natural sentences (heuristically)
            full_text = " ".join(video_text)
            # Simple sentence splitting on punctuation
            sentences = re.split(r'[।!?]', full_text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
            
            all_sentences.extend(sentences)
            
            stats["videos_processed"] += 1
            stats["total_words"] += len(full_text.split())
            stats["source_counts"][source] = stats["source_counts"].get(source, 0) + 1
            
    # Deduplicate repeated lines
    unique_sentences = list(set(all_sentences))
    logger.info(f"Collected {len(all_sentences)} lines, {len(unique_sentences)} unique lines.")
    
    return unique_sentences, stats

def main():
    sentences, stats = process_transcripts()
    
    if not sentences:
        logger.error("No sentences extracted! Check your raw data.")
        return
        
    # Split into train/validation (95/5)
    random.shuffle(sentences)
    split_idx = int(len(sentences) * 0.95)
    train_data = sentences[:split_idx]
    val_data = sentences[split_idx:]
    
    # Save splits
    with open(os.path.join(DATASET_DIR, "train.txt"), "w") as f:
        f.write("\n".join(train_data))
        
    with open(os.path.join(DATASET_DIR, "validation.txt"), "w") as f:
        f.write("\n".join(val_data))
        
    # Save statistics
    stats["total_sentences"] = len(sentences)
    stats["avg_transcript_length"] = stats["total_words"] / stats["videos_processed"] if stats["videos_processed"] > 0 else 0
    
    with open(os.path.join(DATASET_DIR, "metadata.json"), "w") as f:
        json.dump(stats, f, indent=4)
        
    logger.info("Dataset construction complete.")
    logger.info(f"Stats: {stats}")

if __name__ == "__main__":
    main()
