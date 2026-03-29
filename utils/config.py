import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, DATASET_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# List of Haryanvi YouTube Channels (examples - user should provide real IDs)
CHANNELS = [
    # Example channel IDs (replace with actual Haryanvi channels)
    "UC_aEa8K-EO63dx_S27H868g", # Amit Bhadana
    "UCGv_pB8n0AqyOQ2Lp_o-L4A", # Swadu Staff
    "UC2Y_VqfFhU_l0e5oD0sMv7Q", # Nav Haryanvi
    "UC-pE0z6xWzE6y_p06_m2gGg", # NDJ Film Official
    "UCRWp0iB9iG-H1R40O9v53jQ", # Sonotek Music
]

# ASR Model Configuration
ASR_MODEL_NAME = "base" # "base", "small", "medium", "large-v3"
COMPUTE_TYPE = "float32" # "float16", "int8_float16", "int8" (int8 for CPU, float16 for GPU)

# Pipeline Settings
MAX_WORKERS = 10
DELAY_RANGE = (1, 5) # seconds
RETRY_ATTEMPTS = 3
EXPONENTIAL_BACKOFF = 2 # multiplier
