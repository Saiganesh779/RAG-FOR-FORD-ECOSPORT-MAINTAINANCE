from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================
# MANUAL
# ============================================================

MANUAL_ROOT = PROJECT_ROOT / "2022 Ford EcoSport Titanium"

PAGES_DIR = MANUAL_ROOT / "pages"
IMAGES_DIR = MANUAL_ROOT / "images"

# ============================================================
# PROCESSED / DATABASE
# ============================================================

PROCESSED_DIR = PROJECT_ROOT / "processed"

JSONL_PATH = PROCESSED_DIR / "manual_chunks.jsonl"
ERROR_LOG = PROCESSED_DIR / "errors.log"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"

DATABASE_DIR = PROJECT_ROOT / "database" / "chroma"

COLLECTION_NAME = "ford_ecosport_2022"

# ============================================================
# EMBEDDING
# ============================================================

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================
# RETRIEVAL
# ============================================================

VECTOR_RESULTS = 6
EXACT_RESULTS = 6

# ============================================================
# OLLAMA
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

# LOW-MEMORY MODEL
OLLAMA_MODEL = "gemma3:4b"

# Keep the context modest for a low-memory machine
OLLAMA_NUM_CTX = 4096

# ============================================================
# VOICE
# ============================================================

# Path to whisper.cpp executable
WHISPER_EXE = PROJECT_ROOT / "whisper" / "whisper-cli.exe"

# Path to Whisper tiny English model
WHISPER_MODEL = (
    PROJECT_ROOT
    / "whisper"
    / "models"
    / "ggml-tiny.en.bin"
)

# Recording settings
SAMPLE_RATE = 16000
CHANNELS = 1

# Maximum recording duration in seconds
MAX_RECORD_SECONDS = 15

# Stop recording after this amount of silence
SILENCE_SECONDS = 1.2