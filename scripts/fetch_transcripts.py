import os
import json
import time
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from youtube_transcript_api import YouTubeTranscriptApi
from faster_whisper import WhisperModel
from utils.config import RAW_DIR, MAX_WORKERS, DELAY_RANGE, ASR_MODEL_NAME, COMPUTE_TYPE
from utils.logger import logger

# Initialize ASR Model (Shared across threads)
logger.info(f"Loading Whisper model: {ASR_MODEL_NAME}")
model = WhisperModel(ASR_MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE) # Change to "cuda" if GPU is available

def download_audio(video_id, output_path):
    """
    Downloads low-bitrate audio from YouTube using yt-dlp.
    """
    command = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "128K",
        "-o", output_path,
        f"https://www.youtube.com/watch?v={video_id}"
    ]
    try:
        subprocess.run(command, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading audio for {video_id}: {e.stderr}")
        return False

def get_asr_transcript(video_id):
    """
    Downloads audio and performs ASR using faster-whisper.
    """
    audio_path = os.path.join(RAW_DIR, f"{video_id}.mp3")
    if not download_audio(video_id, audio_path):
        return None
    
    try:
        segments, info = model.transcribe(audio_path, beam_size=5)
        transcript = []
        for segment in segments:
            transcript.append({
                "text": segment.text,
                "start": segment.start,
                "duration": segment.end - segment.start
            })
        
        # Cleanup audio file after transcription
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        return transcript
    except Exception as e:
        logger.error(f"ASR failed for {video_id}: {str(e)}")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return None

def fetch_single_transcript(video_id):
    """
    Attempts to fetch transcript via API, falls back to ASR.
    """
    # Randomized delay to avoid rate limits
    time.sleep(random.uniform(*DELAY_RANGE))
    
    output_file = os.path.join(RAW_DIR, f"{video_id}.json")
    if os.path.exists(output_file):
        logger.info(f"Transcript already exists for {video_id}, skipping.")
        return
        
    try:
        # Prefer Hindi/Indian languages
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(['hi', 'hi-IN', 'en']).fetch()
            source = "api"
        except:
            # Try manual/generated translations
            transcript = transcript_list.find_generated_transcript(['hi', 'en']).fetch()
            source = "api-generated"
            
        with open(output_file, "w") as f:
            json.dump({"video_id": video_id, "source": source, "transcript": transcript}, f, indent=4)
        logger.info(f"Fetched API transcript for {video_id}")
        
    except Exception as e:
        logger.warning(f"No API transcript for {video_id}, falling back to ASR. Error: {str(e)}")
        transcript = get_asr_transcript(video_id)
        if transcript:
            with open(output_file, "w") as f:
                json.dump({"video_id": video_id, "source": "asr", "transcript": transcript}, f, indent=4)
            logger.info(f"Completed ASR transcript for {video_id}")
        else:
            logger.error(f"Failed to get any transcript for {video_id}")

def main():
    video_ids_path = os.path.join(RAW_DIR, "video_ids.json")
    if not os.path.exists(video_ids_path):
        logger.error("No video_ids.json found. Run crawl_channels.py first.")
        return
        
    with open(video_ids_path, "r") as f:
        video_data = json.load(f)
        
    all_video_ids = [vid for vids in video_data.values() for vid in vids]
    logger.info(f"Starting transcript fetching for {len(all_video_ids)} videos...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(fetch_single_transcript, all_video_ids)

if __name__ == "__main__":
    main()
