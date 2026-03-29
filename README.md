# Haryanvi YouTube Transcript Pipeline

A fast, reliable, and scalable pipeline to collect, clean, and prepare Haryanvi language transcripts for SLM training.

## Features
- **Channel Crawler**: Efficiently extracts video IDs using `yt-dlp`.
- **Hybrid Downloader**: Attempts to use `youtube-transcript-api` first, then falls back to `faster-whisper` (ASR) if captions are missing.
- **ASR Fallback**: Downloads low-bitrate audio and transcribes using `faster-whisper`.
- **Cleaning & Normalization**: Removes noise tokens, deduplicates lines, and splits text into sentences.
- **Tokenizer Training**: Trains a SentencePiece (Unigram) tokenizer on the processed data.
- **Resume Capability**: Skips already processed videos to avoid duplicate work.

## Setup
1. Create a virtual environment and install dependencies:
```bash
python -m venv kwen
source kwen/bin/activate
pip install -r requirements.txt
```

2. Update `utils/config.py` with the list of Haryanvi YouTube channel IDs you wish to crawl.

## Usage
Run the entire pipeline:
```bash
python main.py --step all
```

Or run individual steps:
1. **Crawl**: `python main.py --step crawl`
2. **Fetch**: `python main.py --step fetch`
3. **Clean**: `python main.py --step clean`
4. **Tokenize**: `python main.py --step tokenize`

## Optimization Strategies

### Avoiding YouTube Rate Limits
- **Randomized Delays**: The pipeline uses `random.uniform(1, 5)` delays between API calls.
- **User-Agent Rotation**: You can configure `yt-dlp` to use different user agents or proxies if you hit severe limits.
- **Proxy Support**: If scaling to 100k+ videos, it is recommended to pass a list of proxies to `yt-dlp` and `YouTubeTranscriptApi`.

### Scaling to 100k+ Videos
- **Sharding**: Divide the channel list across multiple machines.
- **Database Backend**: Use Redis or a database (SQLAlchemy) instead of JSON files to track processed video IDs for better concurrency.
- **Worker Scaling**: Increase `MAX_WORKERS` in `config.py` (ensure your CPU/GPU can handle it).

### Reducing Whisper Compute Cost
- **VAD (Voice Activity Detection)**: `faster-whisper` has built-in VAD to skip silence.
- **Quantization**: Use `int8` or `int8_float16` compute types in `config.py` to speed up CPU inference.
- **Model Size**: Use "base" or "small" for high-speed processing. Only use "large-v3" if maximum accuracy is required and you have ample GPU resources.

## Small Language Model (SLM) Training Suggestions
- **Base Architecture**: Start with a decoder-only model like **TinyLlama (1.1B)** or **Phi-3-mini**.
- **Data Weighting**: If you have a mix of Hindi and Haryanvi, up-sample the Haryanvi-specific transcripts.
- **Context Length**: Since transcripts are conversational, a context length of 2048 or 4096 is usually sufficient.
- **Fine-Tuning**: Use LoRA/QLoRA for memory efficiency if training on consumer hardware.
