import os
import json
import re
import time
import random
import subprocess
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# --- STEP 0: COLAB SETUP & IMPORTS ---
try:
    from google.colab import drive
    COLAB = True
except ImportError:
    COLAB = False

def setup_colab():
    print(">>> Installing/Updating dependencies...")
    # Force update to the absolute latest yt-dlp to bypass YouTube's latest blocks
    subprocess.run(["pip", "install", "-U", "yt-dlp", "youtube-transcript-api", "faster-whisper", "sentencepiece", "tqdm"], check=True)
    subprocess.run(["apt-get", "install", "-y", "ffmpeg"], capture_output=True)
    
    if COLAB:
        drive.mount('/content/drive', force_remount=True)
        base_path = "/content/drive/MyDrive/Haryanvi_Data"
    else:
        base_path = "./data"
    
    os.makedirs(base_path, exist_ok=True)
    return base_path

# --- STEP 1: CONFIGURATION ---
class Config:
    CHANNELS = [
        "UC_aEa8K-EO63dx_S27H868g", # Amit Bhadana
        "UCGv_pB8n0AqyOQ2Lp_o-L4A", # Swadu Staff
        "UC2Y_VqfFhU_l0e5oD0sMv7Q", # Nav Haryanvi
        "UC-pE0z6xWzE6y_p06_m2gGg", # NDJ Film Official
        "UCRWp0iB9iG-H1R40O9v53jQ", # Sonotek Music
    ]
    ASR_MODEL_NAME = "base" 
    MAX_WORKERS = 4 # Slow and steady to avoid IP bans
    DELAY_RANGE = (3, 7) 
    VOCAB_SIZE = 32000

# --- STEP 2: LOGGING ---
def get_logger(base_path):
    log_dir = os.path.join(base_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("HaryanviPipeline")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        f_handler = logging.FileHandler(os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(f_handler)
    return logger

# --- STEP 3: PIPELINE CLASS ---
class HaryanviPipeline:
    def __init__(self, base_path):
        self.base_path = base_path
        self.raw_dir = os.path.join(base_path, "raw")
        self.dataset_dir = os.path.join(base_path, "dataset")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.logger = get_logger(base_path)
        
        # GPU detection
        device = "cuda" if subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0 else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.logger.info(f"Using {device.upper()} with {compute_type} precision.")
        
        from faster_whisper import WhisperModel
        self.model = WhisperModel(Config.ASR_MODEL_NAME, device=device, compute_type=compute_type)

    def crawl_channels(self):
        self.logger.info("Step 1: Crawling channels (using flat extraction)...")
        video_data = {}
        
        # Mimic a real browser to avoid blocks
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        
        for channel_id in Config.CHANNELS:
            url = f"https://www.youtube.com/channel/{channel_id}"
            self.logger.info(f"Crawling: {url}")
            
            # More robust yt-dlp flags for Colab
            cmd = [
                "yt-dlp",
                "--get-id",
                "--flat-playlist",
                "--extract-flat",
                "--playlist-end", "50", # Start with 50 per channel for testing
                "--ignore-errors",
                "--no-check-certificates",
                "--user-agent", user_agent,
                url
            ]
            
            try:
                res = subprocess.run(cmd, capture_output=True, text=True)
                ids = [i.strip() for i in res.stdout.split("\n") if i.strip()]
                
                if ids:
                    video_data[channel_id] = ids
                    self.logger.info(f"Found {len(ids)} videos for {channel_id}")
                else:
                    # Fallback: try the /videos path
                    self.logger.info(f"Retrying with /videos path for {channel_id}...")
                    cmd[-1] = f"{url}/videos"
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    ids = [i.strip() for i in res.stdout.split("\n") if i.strip()]
                    if ids:
                        video_data[channel_id] = ids
                        self.logger.info(f"Found {len(ids)} videos for {channel_id} (via /videos)")
                    else:
                        self.logger.warning(f"No videos found for {channel_id}. YouTube might be throttling the IP.")
            except Exception as e:
                self.logger.error(f"Error crawling {channel_id}: {e}")
            
            time.sleep(5) # Delay between channels
        
        with open(os.path.join(self.raw_dir, "video_ids.json"), "w") as f:
            json.dump(video_data, f, indent=4)

    def fetch_single(self, video_id):
        if not video_id: return
        out_path = os.path.join(self.raw_dir, f"{video_id}.json")
        if os.path.exists(out_path): return

        time.sleep(random.uniform(*Config.DELAY_RANGE))
        
        from youtube_transcript_api import YouTubeTranscriptApi
        try:
            # API Fetch
            t_list = YouTubeTranscriptApi.list_transcripts(video_id)
            try:
                t_obj = t_list.find_transcript(['hi', 'hi-IN', 'en'])
            except:
                t_obj = t_list.find_generated_transcript(['hi', 'en'])
            
            transcript = t_obj.fetch()
            with open(out_path, "w") as f:
                json.dump({"video_id": video_id, "source": "api", "transcript": transcript}, f)
            self.logger.info(f"SUCCESS: Fetched API for {video_id}")
            
        except Exception:
            # ASR Fallback
            self.logger.info(f"FALLBACK: Starting ASR for {video_id}...")
            audio_path = os.path.join(self.raw_dir, f"{video_id}.mp3")
            dl_cmd = ["yt-dlp", "-x", "--audio-format", "mp3", "-o", audio_path, f"https://www.youtube.com/watch?v={video_id}"]
            subprocess.run(dl_cmd, capture_output=True)
            
            if os.path.exists(audio_path):
                try:
                    segments, _ = self.model.transcribe(audio_path, beam_size=5)
                    transcript = [{"text": s.text} for s in segments]
                    with open(out_path, "w") as f:
                        json.dump({"video_id": video_id, "source": "asr", "transcript": transcript}, f)
                    self.logger.info(f"SUCCESS: Completed ASR for {video_id}")
                except Exception as e:
                    self.logger.error(f"FAILED: ASR for {video_id}: {e}")
                finally:
                    if os.path.exists(audio_path): os.remove(audio_path)

    def run_fetch(self):
        ids_path = os.path.join(self.raw_dir, "video_ids.json")
        if not os.path.exists(ids_path): return

        with open(ids_path, "r") as f:
            data = json.load(f)
        
        all_ids = list(set([i for sub in data.values() for i in sub if i]))
        self.logger.info(f"Step 2: Processing {len(all_ids)} videos...")
        
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            executor.map(self.fetch_single, all_ids)

    def clean_and_tokenize(self):
        self.logger.info("Step 3: Cleaning and Tokenizing...")
        all_sentences = []
        
        for fn in os.listdir(self.raw_dir):
            if not fn.endswith(".json") or fn == "video_ids.json": continue
            with open(os.path.join(self.raw_dir, fn), "r") as f:
                data = json.load(f)
                video_text = " ".join([re.sub(r'\[.*?\]|\(.*?\)', '', s['text']) for s in data['transcript']])
                sentences = re.split(r'[।!?.]', video_text)
                all_sentences.extend([s.strip() for s in sentences if len(s.strip()) > 10])
        
        if not all_sentences:
            self.logger.error("No data found to process.")
            return

        train_path = os.path.join(self.dataset_dir, "train.txt")
        with open(train_path, "w") as f:
            f.write("\n".join(list(set(all_sentences))))
        
        import sentencepiece as spm
        spm.SentencePieceTrainer.train(
            input=train_path,
            model_prefix=os.path.join(self.dataset_dir, "haryanvi_spm"),
            vocab_size=Config.VOCAB_SIZE,
            model_type="unigram"
        )
        self.logger.info("DONE!")

if __name__ == "__main__":
    base = setup_colab()
    pipeline = HaryanviPipeline(base)
    pipeline.crawl_channels()
    pipeline.run_fetch()
    pipeline.clean_and_tokenize()
    print(f"\n✅ PIPELINE COMPLETE! Folder: {base}")
