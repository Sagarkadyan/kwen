import os
import json
import subprocess
from utils.config import CHANNELS, RAW_DIR
from utils.logger import logger

def extract_video_ids(channel_id):
    """
    Extracts all video IDs from a YouTube channel using yt-dlp.
    """
    logger.info(f"Extracting video IDs for channel: {channel_id}")
    
    # Flat playlist means it won't try to download everything, just list items.
    command = [
        "yt-dlp",
        "--get-id",
        "--flat-playlist",
        f"https://www.youtube.com/channel/{channel_id}"
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        video_ids = result.stdout.strip().split("\n")
        logger.info(f"Found {len(video_ids)} videos for channel {channel_id}")
        return video_ids
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting video IDs for channel {channel_id}: {e.stderr}")
        return []

def main():
    video_data = {}
    
    for channel in CHANNELS:
        video_ids = extract_video_ids(channel)
        if video_ids:
            video_data[channel] = video_ids
            
    output_path = os.path.join(RAW_DIR, "video_ids.json")
    with open(output_path, "w") as f:
        json.dump(video_data, f, indent=4)
        
    logger.info(f"Video IDs saved to {output_path}")

if __name__ == "__main__":
    main()
