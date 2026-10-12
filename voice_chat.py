import sys
import re
import json
import time
import webbrowser
import tempfile
import asyncio
import os
import subprocess
import math
import pickle
from pathlib import Path
from datetime import datetime
from urllib.parse import unquote

import numpy as np

# CPU tuning for Whisper on the Ryzen 7. These environment variables must
# be set before PyTorch is imported so the underlying math libraries can
# honor them. Start conservatively at 8 threads; this avoids excessive
# contention with the rest of the RAG/voice application.
CPU_THREADS = max(1, int(os.environ.get("ECOSPORT_CPU_THREADS", "8")))
CPU_INTEROP_THREADS = max(1, int(os.environ.get("ECOSPORT_CPU_INTEROP_THREADS", "2")))
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_THREADS))

import torch
torch.set_num_threads(CPU_THREADS)
try:
    torch.set_num_interop_threads(CPU_INTEROP_THREADS)
except RuntimeError:
    # PyTorch only allows this to be changed before parallel work starts.
    pass

import sounddevice as sd
import soundfile as sf
import whisper
import edge_tts
import pygame


# ============================================================
# PROJECT PATH
# ============================================================

if getattr(sys, "frozen", False):
    # PyInstaller: keep all packaged data beside the executable.
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    # Development layout: voice_chat.py is in project\scripts.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# EXISTING RAG SYSTEM
# ============================================================

import config as _config

DATABASE_DIR = _config.DATABASE_DIR
COLLECTION_NAME = _config.COLLECTION_NAME
EMBED_MODEL = _config.EMBED_MODEL
OLLAMA_URL = _config.OLLAMA_URL
OLLAMA_MODEL = _config.OLLAMA_MODEL
JSONL_PATH = _config.JSONL_PATH

# In a frozen build, config.py may have been authored around the
# development project layout. Prefer packaged data beside the EXE when
# the configured path does not exist there.
if getattr(sys, "frozen", False):
    packaged_db = PROJECT_ROOT / "chroma_db"
    packaged_jsonl = PROJECT_ROOT / "manual_chunks.jsonl"

    try:
        if not Path(DATABASE_DIR).exists() and packaged_db.exists():
            DATABASE_DIR = packaged_db
            _config.DATABASE_DIR = DATABASE_DIR
    except Exception:
        pass

    try:
        if not Path(JSONL_PATH).is_file() and packaged_jsonl.is_file():
            JSONL_PATH = packaged_jsonl
            _config.JSONL_PATH = JSONL_PATH
    except Exception:
        pass

# The build process can store the embedding model locally here. When it
# exists, use it directly so the packaged application does not need to
# download a Hugging Face model at runtime.
PACKAGED_EMBED_MODEL = PROJECT_ROOT / "models" / "embeddings"
if (PACKAGED_EMBED_MODEL / "modules.json").is_file():
    EMBED_MODEL = str(PACKAGED_EMBED_MODEL)
    _config.EMBED_MODEL = EMBED_MODEL

# Import the existing RAG/diagnostic engine. Keep this import here in
# voice_chat.py; DO NOT put it inside rag_chat.py (that would create a
# circular import).
from rag_chat import (
    retrieve,
    render_step,
    build_context,
    ask_ollama,
    get_source_page,
    procedure_page_for,
)

# ============================================================
# CONFIGURATION
# ============================================================

# Whisper model. medium.en is intentionally used for better recognition of
# automotive speech, DTCs, and technical terminology.
WHISPER_MODEL_NAME = "medium.en"

# Force Whisper to CPU so the RTX 3050's 4 GB VRAM is left available for
# Ollama/Gemma. This is intentional for the current laptop configuration.
WHISPER_DEVICE = "cpu"
WHISPER_FP16 = False

SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 8

SILENCE_THRESHOLD = 0.012
PAD_SECONDS = 0.35

WHISPER_LANGUAGE = "en"

# --- silence / hallucination protection -------------------------------
# Audio is sent to Whisper ONLY if at least this much speech-level audio
# (frames whose RMS exceeds SILENCE_THRESHOLD) is present.
SPEECH_MIN_SECONDS = 0.25

# Whisper segments that are probably silence/noise are discarded.
NO_SPEECH_PROB_LIMIT = 0.6
SEGMENT_LOGPROB_LIMIT = -0.8
SEGMENT_COMPRESSION_LIMIT = 2.4

# --- diagram opening ----------------------------------------------------
# Only these image types are ever opened as diagrams (never SVG / HTML).
OPENABLE_DIAGRAM_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Upper limit of image windows opened by one request.
MAX_DIAGRAM_IMAGES = 8

WHISPER_PROMPT = """
Automotive diagnostic conversation for a 2022 Ford EcoSport Titanium.
Ford service manual, diagnostic trouble codes, DTCs, PCM, BCM, CAN,
Pinpoint Test KF, Pinpoint Test KN, fan control relay, cooling fan,
variable speed electric cooling fan motor, voltage, resistance,
continuity, short to ground, short to voltage, open circuit,
connector, wiring harness, circuit, fuse, relay, ground.

Common commands:
yes, no, continue, next, repeat, go back, restart,
show diagram, show wiring diagram, open diagram,
show source, show sources, explain, stop, exit.

DTC examples:
P0480, P028D, P028E.
"""

TTS_VOICE = "en-IN-NeerjaNeural"
TTS_RATE = "+0%"
TTS_VOLUME = "+0%"

VOICE_TEMP_DIR = Path(tempfile.gettempdir()) / "ecosport_voice"
VOICE_TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MANUAL LOCATION
# ============================================================

MANUAL_ROOT = (
    PROJECT_ROOT
    / "2022 Ford EcoSport Titanium"
    / "pages"
)

# The preprocessed JSONL already contains the image references
# extracted from every manual HTML page. Use it as the authoritative
# page -> image index instead of guessing/ranking image files.
# Same file build_index.py reads (config.JSONL_PATH, in processed\).
MANUAL_JSONL = Path(JSONL_PATH)

# Diagram/image files may be opened ONLY from this folder.
IMAGE_ROOT = (
    PROJECT_ROOT
    / "2022 Ford EcoSport Titanium"
    / "images"
)

# Lazy-loaded once on the first "show diagram" request.
_MANUAL_IMAGE_INDEX = None

# Cache for deterministic HTML fallback extraction. A number of manual
# records intentionally have no JSONL image list; when we have already
# selected that exact manual page, this cache stores the raster image
# references extracted from that page so repeated requests do not reread
# the HTML file.
_HTML_DIAGRAM_IMAGE_CACHE = {}


# ============================================================
# DIAGRAM SEARCH CONFIGURATION
# ============================================================

# Words that strongly suggest a diagram page/image.
DIAGRAM_TERMS = {
    "wiring": 20,
    "wiring diagram": 35,
    "schematic": 30,
    "circuit diagram": 35,
    "electrical": 15,
    "connector": 18,
    "connector view": 25,
    "connector pin": 25,
    "pinout": 30,
    "circuit": 12,
    "diagram": 25,
    "illustration": 15,
    "harness": 18,
    "relay": 12,
    "cooling fan": 25,
    "fan control": 25,
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".svg",
}

HTML_EXTENSIONS = {
    ".html",
    ".htm",
}

# Avoid obvious UI images.
BAD_IMAGE_WORDS = {
    "icon",
    "logo",
    "button",
    "arrow",
    "bullet",
    "menu",
    "home",
    "print",
    "close",
    "next",
    "previous",
}


# ============================================================
# WHISPER
# ============================================================

print()
print("=" * 70)
print("Loading Whisper...")
print(f"Model:  {WHISPER_MODEL_NAME}")
print("Device: cpu")
print(f"CPU threads: {CPU_THREADS}")
print(f"PyTorch inter-op threads: {CPU_INTEROP_THREADS}")
print("Whisper is intentionally running on CPU so the RTX 3050 is kept available for Ollama/Gemma.")
print("=" * 70)

WHISPER_CACHE_DIR = PROJECT_ROOT / "models" / "whisper"
WHISPER_MODEL_FILE = WHISPER_CACHE_DIR / f"{WHISPER_MODEL_NAME}.pt"


def _find_existing_whisper_model():
    """
    Reuse an already-downloaded medium.en model before asking Whisper to
    download anything. Older versions of this program used Whisper's
    default cache (usually %USERPROFILE%/.cache/whisper), while the newer
    packaged layout uses PROJECT_ROOT/models/whisper. Check both so the
    model is never downloaded again just because the cache location changed.
    """

    candidates = []

    # New packaged/project location.
    candidates.append(WHISPER_MODEL_FILE)

    # OpenAI Whisper's normal default cache location.
    candidates.append(Path.home() / ".cache" / "whisper" / f"{WHISPER_MODEL_NAME}.pt")

    # Also check common Windows cache locations.
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.append(
            Path(local_appdata) / "whisper" / f"{WHISPER_MODEL_NAME}.pt"
        )
        candidates.append(
            Path(local_appdata) / ".cache" / "whisper" / f"{WHISPER_MODEL_NAME}.pt"
        )

    seen = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except Exception:
            candidate = Path(candidate)

        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)

        try:
            if candidate.is_file() and candidate.stat().st_size > 100_000_000:
                return candidate
        except OSError:
            continue

    return None


_existing_whisper_model = _find_existing_whisper_model()

if _existing_whisper_model is not None:
    WHISPER_MODEL_FILE = _existing_whisper_model
    print(f"Using existing Whisper model: {WHISPER_MODEL_FILE}")
    print("No Whisper download is required.")
    WHISPER = whisper.load_model(
        str(WHISPER_MODEL_FILE),
        device="cpu",
    )
else:
    if getattr(sys, "frozen", False):
        print()
        print("ERROR: bundled Whisper model was not found:")
        print(f"  {WHISPER_CACHE_DIR / (WHISPER_MODEL_NAME + '.pt')}")
        print("Build the application with prepare_models.py before packaging.")
        raise FileNotFoundError(
            str(WHISPER_CACHE_DIR / (WHISPER_MODEL_NAME + ".pt"))
        )

    # Development fallback: download only when the model truly does not
    # exist in any of the known local locations. The downloaded file is
    # stored in the project cache so later runs reuse it.
    WHISPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Existing medium.en model not found in known local caches.")
    print(f"Downloading Whisper once to: {WHISPER_CACHE_DIR}")
    WHISPER = whisper.load_model(
        WHISPER_MODEL_NAME,
        device="cpu",
        download_root=str(WHISPER_CACHE_DIR),
    )


print()
print("Whisper loaded on: cpu")
print()


# ============================================================
# PYGAME
# ============================================================

pygame.mixer.init()


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text):
    if not text:
        return ""

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def normalize_dtc(text):
    """
    Recognize DTCs such as:

        P0480
        p 0 4 8 0
        P zero four eight zero
    """

    if not text:
        return None

    original = text.lower().strip()

    cleaned = re.sub(
        r"[^a-z0-9]",
        "",
        original,
    )

    match = re.search(
        r"([pbcu][0-9a-f]{4})",
        cleaned,
    )

    if match:
        return match.group(1).upper()

    replacements = {
        "zero": "0",
        "oh": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }

    spoken = original

    for word, digit in replacements.items():

        spoken = re.sub(
            rf"\b{word}\b",
            digit,
            spoken,
            flags=re.IGNORECASE,
        )

    spoken = re.sub(
        r"[^a-z0-9]",
        "",
        spoken,
    )

    match = re.search(
        r"([pbcu][0-9a-f]{4})",
        spoken,
    )

    if match:
        return match.group(1).upper()

    return None


# ============================================================
# COMMAND RECOGNITION
# ============================================================

def dtc_base(dtc):
    """P0480:00 -> P0480"""

    if not dtc:
        return None

    m = re.match(r"\s*([PBCU][0-9A-F]{4})", str(dtc).upper())

    return m.group(1) if m else str(dtc).upper().strip()[:5]


SPOKEN_DIGITS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

STRICT_DTC_RE = re.compile(
    r"\b([pbcu])[\s\-:]*([0-3])[\s\-]*([0-9a-f])[\s\-]*"
    r"([0-9a-f])[\s\-]*([0-9a-f])(?![0-9a-z])"
)


def _spoken_to_digits(text):

    t = str(text or "").lower()

    for word, digit in SPOKEN_DIGITS.items():
        t = re.sub(rf"\b{word}\b", digit, t)

    return t


def extract_spoken_dtc(text):
    """
    Strict DTC finder for diagram requests: 'P0563', 'p 0 5 6 3',
    'P zero five six three'. Returns e.g. 'P0563' or None.
    """

    m = STRICT_DTC_RE.search(_spoken_to_digits(text))

    if not m:
        return None

    return "".join(m.groups()).upper()


DIAGRAM_NOUN_RE = re.compile(
    r"\b(diagrams?|schematics?|wiring|pinouts?|pin outs?|"
    r"connector views?|pictures?|images?)\b"
)

DIAGRAM_VERB_RE = re.compile(
    r"\b(show|open|display|pull up|bring up|view)\b"
)


def is_diagram_request(text):
    """
    True for 'show diagram', 'show cooling fan diagrams',
    'P0563, show diagrams', 'open wiring', ...
    """

    t = normalize_text(text)

    if not t:
        return False

    return bool(
        DIAGRAM_NOUN_RE.search(t)
        and DIAGRAM_VERB_RE.search(t)
    )


DIAGRAM_COMMAND_WORDS = {
    "show", "open", "display", "pull", "up", "bring", "view", "views",
    "me", "the", "a", "an", "of", "for", "please", "can", "you",
    "could", "would", "diagram", "diagrams", "schematic", "schematics",
    "wiring", "pinout", "pinouts", "pin", "outs", "picture", "pictures",
    "image", "images", "and", "to", "i", "want", "need", "see", "with",
    "on", "about", "this", "that", "current", "here", "it", "ford",
    "ecosport", "code", "dtc", "trouble", "all", "any", "some", "my",
    "is", "are", "what", "where", "give", "get", "find", "now",
}


def diagram_topic_words(text):
    """
    Words left in a diagram request after removing command words and
    the DTC, e.g. 'show cooling fan diagrams' -> ['cooling', 'fan'].
    """

    t = _spoken_to_digits(text)

    t = STRICT_DTC_RE.sub(" ", t)

    words = []

    for word in re.findall(r"[a-z0-9]+", t):

        if word in DIAGRAM_COMMAND_WORDS:
            continue

        if word not in words:
            words.append(word)

    return words


def command_from_text(text):

    text = normalize_text(text)

    if not text:
        return None

    # --------------------------------------------------------
    # DIAGRAM REQUEST (checked first so that words such as
    # "stop" or "close" inside a topic do not end the session)
    # --------------------------------------------------------

    if is_diagram_request(text):
        return "diagram"

    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    for phrase in [
        "exit",
        "quit",
        "stop",
        "close",
        "goodbye",
        "end session",
        "end diagnostic",
    ]:

        if phrase in text:
            return "exit"

    # --------------------------------------------------------
    # DIAGRAM
    # --------------------------------------------------------

    for phrase in [
        "show diagram",
        "open diagram",
        "display diagram",
        "show the diagram",
        "open the diagram",
        "show wiring diagram",
        "open wiring diagram",
        "show wiring",
        "open wiring",
        "show schematic",
        "open schematic",
        "show picture",
        "show image",
        "open picture",
        "open image",
    ]:

        if phrase in text:
            return "diagram"

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    for phrase in [
        "show source",
        "show sources",
        "show manual page",
        "open source",
        "open manual page",
    ]:

        if phrase in text:
            return "sources"

    # --------------------------------------------------------
    # EXPLAIN
    # --------------------------------------------------------

    for phrase in [
        "explain",
        "explain that",
        "why",
        "why is that",
        "what does that mean",
    ]:

        if phrase in text:
            return "explain"

    # --------------------------------------------------------
    # RESTART
    # --------------------------------------------------------

    for phrase in [
        "restart",
        "start over",
        "restart test",
        "restart diagnostic",
        "begin again",
    ]:

        if phrase in text:
            return "restart"

    # --------------------------------------------------------
    # REPEAT
    # --------------------------------------------------------

    for phrase in [
        "repeat",
        "repeat that",
        "say that again",
        "say again",
        "what was that",
    ]:

        if phrase in text:
            return "repeat"

    # --------------------------------------------------------
    # BACK
    # --------------------------------------------------------

    for phrase in [
        "go back",
        "previous",
        "previous step",
        "go to previous step",
    ]:

        if phrase in text:
            return "back"

    # --------------------------------------------------------
    # CONTINUE
    # --------------------------------------------------------

    for phrase in [
        "continue",
        "next",
        "next step",
        "go ahead",
        "proceed",
        "keep going",
        "move on",
        "okay continue",
        "ok continue",
    ]:

        if phrase in text:
            return "continue"

    # --------------------------------------------------------
    # YES
    # --------------------------------------------------------

    for phrase in [
        "yes",
        "yeah",
        "yep",
        "yup",
        "correct",
        "that's right",
        "that is right",
        "it does",
        "it is",
        "works",
        "working",
        "present",
        "found",
    ]:

        if text == phrase or text.startswith(
            phrase + " "
        ):

            return "yes"

    # --------------------------------------------------------
    # NO
    # --------------------------------------------------------

    for phrase in [
        "no",
        "nope",
        "nah",
        "not working",
        "doesn't work",
        "does not work",
        "not present",
        "not found",
        "doesn't operate",
        "does not operate",
    ]:

        if text == phrase or text.startswith(
            phrase + " "
        ):

            return "no"

    return None


# ============================================================
# RECORDING
# ============================================================

def record_audio(
    seconds=RECORD_SECONDS,
    sample_rate=SAMPLE_RATE,
):

    print()
    print("🎤 Listening...")

    try:

        audio = sd.rec(
            int(seconds * sample_rate),
            samplerate=sample_rate,
            channels=CHANNELS,
            dtype="float32",
        )

        sd.wait()

    except Exception as exc:

        print(
            f"Microphone error: {exc}"
        )

        return None

    if audio is None:
        return None

    return audio.reshape(-1)


def trim_silence(
    audio,
    sample_rate=SAMPLE_RATE,
    threshold=SILENCE_THRESHOLD,
    pad_seconds=PAD_SECONDS,
):

    if audio is None or len(audio) == 0:
        return audio

    absolute = np.abs(audio)

    window = max(
        1,
        int(sample_rate * 0.02),
    )

    if len(absolute) <= window:
        return audio

    rms = np.sqrt(
        np.convolve(
            absolute ** 2,
            np.ones(window) / window,
            mode="valid",
        )
    )

    indices = np.where(
        rms > threshold
    )[0]

    if len(indices) == 0:
        return audio

    start = int(indices[0])
    end = int(indices[-1] + window)

    padding = int(
        sample_rate * pad_seconds
    )

    start = max(
        0,
        start - padding,
    )

    end = min(
        len(audio),
        end + padding,
    )

    return audio[start:end]


def save_recording(audio):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    path = (
        VOICE_TEMP_DIR
        / f"input_{timestamp}.wav"
    )

    sf.write(
        str(path),
        audio,
        SAMPLE_RATE,
    )

    return path


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def has_speech(
    audio,
    sample_rate=SAMPLE_RATE,
    threshold=SILENCE_THRESHOLD,
    min_seconds=SPEECH_MIN_SECONDS,
):
    """
    Energy gate. True only if enough 20 ms frames exceed the speech
    threshold. Silence/room noise never reaches Whisper.
    """

    if audio is None or len(audio) == 0:
        return False

    frame = max(1, int(sample_rate * 0.02))

    count = len(audio) // frame

    if count == 0:
        return False

    frames = audio[: count * frame].reshape(count, frame)

    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    voiced = int(np.sum(rms > threshold))

    return voiced * 0.02 >= min_seconds


_PROMPT_WORDS = " ".join(
    re.findall(r"[a-z0-9']+", WHISPER_PROMPT.lower())
)


def is_whisper_hallucination(text):
    """
    Reject typical Whisper silence hallucinations: repeated words or
    phrases ('ParchmentS, ParchmentS, ...'), punctuation-only output
    and echoes of the initial prompt.
    """

    if not text:
        return False

    words = re.findall(r"[a-z0-9']+", text.lower())

    if not words:
        return True

    # Echo of the initial prompt.
    if len(words) >= 5 and " ".join(words) in _PROMPT_WORDS:
        return True

    # Very low vocabulary diversity in a longer utterance.
    if len(words) >= 6 and len(set(words)) / len(words) <= 0.4:
        return True

    # Same word four or more times in a row.
    run = 1

    for a, b in zip(words, words[1:]):

        run = run + 1 if a == b else 1

        if run >= 4:
            return True

    # A 2-4 word phrase repeated three or more times.
    for n in (2, 3, 4):

        grams = [
            " ".join(words[i:i + n])
            for i in range(len(words) - n + 1)
        ]

        if grams and max(grams.count(g) for g in set(grams)) >= 3:
            return True

    return False


def transcribe_audio(audio):

    if audio is None:
        return ""

    # Silence is detected BEFORE Whisper is ever called.
    if not has_speech(audio):
        return ""

    audio = trim_silence(audio)

    if (
        audio is None
        or len(audio)
        < int(SAMPLE_RATE * 0.15)
    ):
        return ""

    wav_path = save_recording(
        audio
    )

    try:

        result = WHISPER.transcribe(
            str(wav_path),
            language=WHISPER_LANGUAGE,
            task="transcribe",
            # CUDA uses FP16 to reduce VRAM use and improve throughput.
            # CPU mode automatically uses FP32.
            fp16=WHISPER_FP16,
            temperature=0,
            condition_on_previous_text=False,
            initial_prompt=WHISPER_PROMPT,
            no_speech_threshold=0.45,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

        segments = result.get("segments") or []

        if segments:

            kept = []

            for segment in segments:

                if (
                    segment.get("no_speech_prob", 0.0)
                    > NO_SPEECH_PROB_LIMIT
                    and segment.get("avg_logprob", 0.0)
                    < SEGMENT_LOGPROB_LIMIT
                ):
                    continue

                if (
                    segment.get("compression_ratio", 0.0)
                    > SEGMENT_COMPRESSION_LIMIT
                ):
                    continue

                kept.append(
                    segment.get("text", "")
                )

            text = " ".join(kept).strip()

        else:

            text = result.get(
                "text",
                "",
            ).strip()

        if is_whisper_hallucination(text):

            print(
                "(ignored: Whisper produced repeated/noise text)"
            )

            return ""

        return text

    except Exception as exc:

        print(
            f"Whisper error: {exc}"
        )

        return ""

    finally:

        try:
            wav_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass


def listen():

    audio = record_audio()

    if audio is None:
        return ""

    text = transcribe_audio(
        audio
    )

    if text:

        print(
            f"🗣️ You: {text}"
        )

    return text


# ============================================================
# TEXT TO SPEECH
# ============================================================

def clean_for_speech(text):
    """
    Remove file names, links and source labels so they are neither
    read aloud nor printed as the assistant's reply. Keeps link labels:
    '[PINPOINT TEST KN](../pages/16138.html#...)' -> 'PINPOINT TEST KN'.
    """

    t = str(text or "")

    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(
        r"(?im)^\s*(?:sources?|files?)\s*:.*$", "", t
    )
    t = re.sub(
        r"\(\s*(?:sources?|files?)\b[^)]*\)",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[^\s()\[\]]*\.html?[^\s()\[\]]*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"[^\s()\[\]]*\.\./[^\s()\[\]]*", "", t)
    t = re.sub(r"%[0-9A-Fa-f]{2}", " ", t)
    t = re.sub(r"\[?\bSOURCES?\s*\d+\]?\s*:?", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:sources?|files?)\s*:", "", t, flags=re.IGNORECASE)
    t = re.sub(r"[*#`_]+", " ", t)
    t = re.sub(r"\(\s*\)|\[\s*\]", "", t)
    t = re.sub(r"\s+([.,;:])", r"\1", t)
    t = re.sub(r"\s+", " ", t)

    return t.strip()


def speak(text):

    if not text:
        return

    text = clean_for_speech(text)

    if not text:
        return

    print(
        f"🤖 EcoSport: {text}"
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    mp3_path = (
        VOICE_TEMP_DIR
        / f"response_{timestamp}.mp3"
    )

    async def generate():

        communicator = edge_tts.Communicate(
            text,
            TTS_VOICE,
            rate=TTS_RATE,
            volume=TTS_VOLUME,
        )

        await communicator.save(
            str(mp3_path)
        )

    try:

        asyncio.run(
            generate()
        )

        pygame.mixer.music.load(
            str(mp3_path)
        )

        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():

            time.sleep(0.05)

        try:
            pygame.mixer.music.unload()
        except Exception:
            pass

    except Exception as exc:

        print(
            f"TTS error: {exc}"
        )

    finally:

        time.sleep(0.05)

        try:
            mp3_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass


# ============================================================
# MANUAL FILE HELPERS
# ============================================================

def page_path(page):

    if not page:
        return None

    page = str(page).strip()

    if not page.lower().endswith(
        ".html"
    ):
        page += ".html"

    path = (
        MANUAL_ROOT
        / page
    )

    if path.exists():
        return path

    return None


def _launch_and_check(command):
    """
    Start a launcher process; report failure if it exits with an error
    within a second (a viewer that stays running counts as success).
    """

    try:

        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )

    except Exception as exc:

        print(f"    launcher failed ({command[0]}): {exc}")

        return False

    time.sleep(1.0)

    code = process.poll()

    if code not in (None, 0):

        print(f"    launcher exited with code {code} ({command[0]})")

        return False

    return True


def open_local_file(path):
    """
    Open one exact local image (already restricted to IMAGE_ROOT by
    the caller) with the system's default image viewer.

    Windows order: os.startfile (default app) -> cmd start ->
    Paint -> Explorer. The method that worked is printed.
    """

    if not path:
        return False

    path = Path(path).resolve()

    if not path.is_file():
        print(
            f"Cannot open file because it does not exist: {path}"
        )
        return False

    if os.name == "nt":

        try:

            os.startfile(str(path))

            print("    opened via Windows default app")

            return True

        except Exception as exc:

            print(f"    os.startfile failed: {exc}")

        if _launch_and_check(
            ["cmd", "/c", "start", "", str(path)]
        ):
            print("    opened via cmd start")
            return True

        if _launch_and_check(["mspaint.exe", str(path)]):
            print("    opened via Paint")
            return True

        if _launch_and_check(["explorer.exe", str(path)]):
            print("    opened via Explorer")
            return True

        return False

    try:

        return bool(
            webbrowser.open(
                path.as_uri(),
                new=0,
                autoraise=True,
            )
        )

    except Exception as exc:

        print(f"Could not open image: {exc}")

        return False


def open_diagram_images(images):
    """
    Open the exact raster diagram images selected from the manual.

    Safety rules:
      * Only files inside IMAGE_ROOT may be opened.
      * Only JPG/JPEG/PNG files are allowed.
      * The supplied order is preserved.
      * Duplicate image paths are opened only once.
      * At most MAX_DIAGRAM_IMAGES images are opened per request.
    """

    if not images:
        return 0

    opened = 0
    seen = set()
    image_root = IMAGE_ROOT.resolve()

    print()
    print("Opening ONLY image files referenced by the matched manual page:")

    for raw_path in images:
        if opened >= MAX_DIAGRAM_IMAGES:
            print(
                f"Reached diagram image limit ({MAX_DIAGRAM_IMAGES}); "
                "remaining images were not opened."
            )
            break

        try:
            image_path = Path(raw_path).resolve()
            image_path.relative_to(image_root)
        except (TypeError, ValueError, OSError):
            print(f"  Refusing unsafe diagram path: {raw_path}")
            continue

        if image_path.suffix.lower() not in OPENABLE_DIAGRAM_EXTENSIONS:
            print(f"  Skipping non-raster diagram asset: {image_path}")
            continue

        if not image_path.is_file():
            print(f"  Missing diagram image: {image_path}")
            continue

        key = str(image_path).lower()
        if key in seen:
            continue
        seen.add(key)

        print(f"  {image_path}")
        if open_local_file(image_path):
            opened += 1

    print()
    print(f"Opened {opened} diagram image(s).")
    print()

    return opened


def read_html(path):

    try:

        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    except Exception:

        return ""


# ============================================================
# HTML LINK EXTRACTION
# ============================================================

def extract_local_links(
    html_path,
    html_text=None,
):

    if html_text is None:
        html_text = read_html(
            html_path
        )

    if not html_text:
        return []

    links = []

    # href="..."
    patterns = [
        r'href\s*=\s*["\']([^"\']+)["\']',
        r'src\s*=\s*["\']([^"\']+)["\']',
        r'data-src\s*=\s*["\']([^"\']+)["\']',
        r'data-href\s*=\s*["\']([^"\']+)["\']',
    ]

    for pattern in patterns:

        for match in re.finditer(
            pattern,
            html_text,
            flags=re.IGNORECASE,
        ):

            raw = match.group(1).strip()

            if not raw:
                continue

            if raw.startswith(
                (
                    "http://",
                    "https://",
                    "javascript:",
                    "data:",
                    "#",
                )
            ):
                continue

            raw = unquote(
                raw.split("#")[0]
            )

            # HTML entities commonly found in paths.
            raw = raw.replace(
                "&amp;",
                "&",
            )

            candidate = (
                html_path.parent
                / raw
            ).resolve()

            if not candidate.exists():
                continue

            if (
                candidate.suffix.lower()
                in HTML_EXTENSIONS
                or candidate.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                links.append(
                    candidate
                )

    # Deduplicate.
    unique = []

    seen = set()

    for path in links:

        key = str(path).lower()

        if key not in seen:

            seen.add(key)
            unique.append(path)

    return unique


# ============================================================
# IMAGE EXTRACTION
# ============================================================

def _resolve_html_image_reference(html_path, raw_reference):
    """Resolve one HTML image reference strictly inside IMAGE_ROOT."""

    if not raw_reference:
        return None

    raw = unquote(str(raw_reference).strip())
    raw = raw.replace("&amp;", "&")
    raw = raw.split("#", 1)[0].split("?", 1)[0].strip()

    if not raw or raw.lower().startswith(("http://", "https://", "data:", "javascript:")):
        return None

    # Some pages use CSS-style url(...), which should also be handled.
    if raw.lower().startswith("url(") and raw.endswith(")"):
        raw = raw[4:-1].strip().strip("'\"")

    try:
        candidate = (html_path.parent / raw).resolve()
    except OSError:
        return None

    try:
        candidate.relative_to(IMAGE_ROOT.resolve())
    except ValueError:
        return None

    if not candidate.is_file():
        # The HTML may use a different relative spelling. Resolve by exact
        # basename only; never fuzzy-match or score filenames.
        name = Path(raw).name.lower()
        if not name:
            return None
        candidate = _image_file_map().get(name)
        if candidate is None:
            return None
        candidate = Path(candidate).resolve()
        try:
            candidate.relative_to(IMAGE_ROOT.resolve())
        except ValueError:
            return None

    if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
        return None

    return candidate


def extract_images(
    html_path,
    html_text=None,
):
    """Extract local raster/SVG image references from one HTML file.

    This function only *reads* HTML to discover image references. The caller
    is responsible for deciding which image types may actually be opened.
    """

    if html_text is None:
        html_text = read_html(html_path)

    if not html_text:
        return []

    images = []

    # Direct image elements.
    patterns = [
        r'<img[^>]+(?:src|data-src|data-original|data-lazy-src|data-image)=\s*["\']([^"\']+)["\']',
        r'<image[^>]+(?:href|xlink:href)=\s*["\']([^"\']+)["\']',
        r'<object[^>]+data=\s*["\']([^"\']+)["\']',
        r'<source[^>]+src=\s*["\']([^"\']+)["\']',
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, html_text, flags=re.IGNORECASE):
            path = _resolve_html_image_reference(html_path, match.group(1))
            if path is not None:
                images.append(path)

    # Responsive images often put the actual image in srcset.
    for match in re.finditer(
        r'\bsrcset\s*=\s*["\']([^"\']+)["\']',
        html_text,
        flags=re.IGNORECASE,
    ):
        for item in match.group(1).split(","):
            reference = item.strip().split()[0] if item.strip() else ""
            path = _resolve_html_image_reference(html_path, reference)
            if path is not None:
                images.append(path)

    # Some manual pages use an anchor around a full-size image.
    for match in re.finditer(
        r'<a[^>]+href=\s*["\']([^"\']+)["\']',
        html_text,
        flags=re.IGNORECASE,
    ):
        raw = match.group(1)
        if Path(unquote(raw).split("#", 1)[0].split("?", 1)[0]).suffix.lower() in IMAGE_EXTENSIONS:
            path = _resolve_html_image_reference(html_path, raw)
            if path is not None:
                images.append(path)

    # CSS/background image references in the page itself.
    for match in re.finditer(
        r'url\(\s*["\']?([^"\')]+)["\']?\s*\)',
        html_text,
        flags=re.IGNORECASE,
    ):
        path = _resolve_html_image_reference(html_path, match.group(1))
        if path is not None:
            images.append(path)

    unique = []
    seen = set()

    for path in images:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)

    return unique


# ============================================================
# DIAGRAM SCORING
# ============================================================

def score_text(
    text,
    search_terms,
):

    if not text:
        return 0

    lower = text.lower()

    score = 0

    for term, weight in search_terms:

        if term.lower() in lower:
            score += weight

    return score


def image_score(
    image_path,
    context_terms,
):

    name = image_path.name.lower()

    score = 0

    # Context/DTC/step terms.
    for term, weight in context_terms:

        if term.lower() in name:
            score += weight

    # Diagram vocabulary.
    for term, weight in DIAGRAM_TERMS.items():

        if term in name:
            score += weight

    # File type.
    if image_path.suffix.lower() in {
        ".svg",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }:

        score += 5

    # Penalize UI images.
    for bad in BAD_IMAGE_WORDS:

        if bad in name:
            score -= 20

    return score


def html_page_score(
    path,
    html_text,
    context_terms,
):

    score = 0

    name = path.name.lower()

    # Filename relevance.
    for term, weight in context_terms:

        if term.lower() in name:
            score += weight

    # Page content relevance.
    for term, weight in context_terms:

        occurrences = html_text.lower().count(
            term.lower()
        )

        score += min(
            occurrences,
            5,
        ) * weight

    # Diagram terms.
    lower = html_text.lower()

    for term, weight in DIAGRAM_TERMS.items():

        if term in lower:

            score += min(
                lower.count(term),
                3,
            ) * weight

    # Presence of images is a strong signal.
    image_count = len(
        extract_images(
            path,
            html_text,
        )
    )

    score += min(
        image_count,
        5,
    ) * 15

    return score



# ============================================================
# JSONL IMAGE INDEX
# ============================================================

def _normalize_lookup_text(value):
    if not value:
        return ""

    value = str(value).lower()

    value = value.replace(
        "\\",
        " ",
    )

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


_IMAGE_FILE_MAP = None
_UNRESOLVED_REFS = []
_REF_COUNTS = {"total": 0, "resolved": 0}


def _image_file_map():
    """
    {lower-case file name: real path} for every JPEG/PNG below
    IMAGE_ROOT, built once. Used to resolve the EXACT file names that
    manual_chunks.jsonl lists, whatever folder prefix they carry.
    """
    global _IMAGE_FILE_MAP

    if _IMAGE_FILE_MAP is not None:
        return _IMAGE_FILE_MAP

    mapping = {}

    root = IMAGE_ROOT.resolve()

    try:

        for folder, _dirs, files in os.walk(root):

            for name in files:

                if Path(name).suffix.lower() in OPENABLE_DIAGRAM_EXTENSIONS:
                    mapping.setdefault(
                        name.lower(),
                        Path(folder) / name,
                    )

    except Exception as exc:

        print(f"WARNING: could not scan images folder: {exc}")

    print(f"Images folder: {len(mapping)} JPEG/PNG files ({root})")

    _IMAGE_FILE_MAP = mapping

    return mapping


def _safe_image_path_from_json(raw_path):
    """
    Resolve one image reference from manual_chunks.jsonl to a real file
    below IMAGE_ROOT. The reference's file name must exist exactly
    (case-insensitive) in the images folder; nothing is guessed.
    Accepts 'images/x.jpg', '../images/x.jpg', 'x.jpg', URL-encoded,
    backslashes. Only JPEG/PNG are accepted.
    """
    if not raw_path:
        return None

    _REF_COUNTS["total"] += 1

    raw = unquote(str(raw_path).strip()).replace("\\", "/")

    raw = raw.split("#")[0].split("?")[0]

    name = raw.rsplit("/", 1)[-1].strip()

    found = None

    if name and Path(name).suffix.lower() in OPENABLE_DIAGRAM_EXTENSIONS:

        # Exact relative path under IMAGE_ROOT first.
        relative = raw
        while relative.startswith(("./", "../")):
            relative = relative.split("/", 1)[1]
        if relative.lower().startswith("images/"):
            relative = relative.split("/", 1)[1]

        direct = (IMAGE_ROOT / relative).resolve()

        try:
            direct.relative_to(IMAGE_ROOT.resolve())
            if direct.is_file():
                found = direct
        except (ValueError, OSError):
            pass

        if found is None:
            found = _image_file_map().get(name.lower())

    if found is None:

        if len(_UNRESOLVED_REFS) < 8:
            _UNRESOLVED_REFS.append(str(raw_path))

        return None

    try:
        found = Path(found).resolve()
        found.relative_to(IMAGE_ROOT.resolve())
    except (ValueError, OSError):
        return None

    _REF_COUNTS["resolved"] += 1

    return found


# DTC / pinpoint-test relationships learned from manual_chunks.jsonl.
_DTC_INFO = {}
_PINPOINT_TOPICS = {}
_TOKEN_IDF = {}

PINPOINT_TITLE_RE = re.compile(
    r"^\s*pinpoint tests?\s+([a-z0-9]+)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)

DTC_ROW_RE = re.compile(
    r"([PBCU][0-9A-F]{4})(?::[0-9A-F]{2})?\s*\|\s*([^|\n]*?)\s*\|"
    r"\s*[^|\n]*?Pinpoint Tests?\s+([A-Z][A-Z0-9]{0,3})\b",
    re.IGNORECASE,
)

TOKEN_STOPWORDS = {
    "the", "and", "for", "with", "of", "to", "in", "on", "or", "an",
    "pinpoint", "test", "tests",
}


def _topic_tokens(text):
    """Lower-case, lightly stemmed, de-duplicated word tokens."""

    tokens = []

    for token in re.findall(r"[a-z0-9]+", str(text or "").lower()):

        if token in TOKEN_STOPWORDS or len(token) < 2:
            continue

        if token.isdigit():
            continue

        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]

        if token not in tokens:
            tokens.append(token)

    return tokens


def _is_generic_page(entry):
    """Procedure/chart pages are not diagram pages."""

    title = _normalize_lookup_text(entry.get("title", ""))

    return (
        title.startswith("pinpoint test")
        or title.endswith("test procedure")
        or title == "test procedure"
        or title.startswith("diagnostic trouble code")
        or title.startswith("diagnostic methods")
    )

def _clean_manual_title(title):
    """Remove the scraped site suffix from manual page titles."""

    value = _normalize_lookup_text(title)
    value = re.sub(
        r"\s+[—-]\s+2022\s+ford\s+ecosport.*$",
        "",
        value,
    ).strip()
    value = re.sub(r"\s+~\s+lemon\s+manuals.*$", "", value)
    return value


def _entry_combined_text(entry):
    return _normalize_lookup_text(
        " ".join(
            [
                entry.get("title", ""),
                entry.get("section", ""),
                " ".join(entry.get("headings", [])),
                entry.get("search_text", ""),
                entry.get("captions", ""),
            ]
        )
    )


def _entry_is_wiring_page(entry):
    """True only when the manual metadata identifies a wiring-diagram area."""

    if entry.get("wiring"):
        return True

    combined = _entry_combined_text(entry)
    return bool(
        "wiring diagrams" in combined
        or "oem wiring diagrams" in combined
        or "system wiring diagrams" in combined
    )


def _entry_is_connector_page(entry):
    """Connector/pin pages are separate from wiring diagrams."""

    title = _normalize_lookup_text(entry.get("title", ""))
    return bool(
        title.startswith("connector end view")
        or "pinout" in title
        or "connector view" in title
        or ("connector" in title and "inline" in title)
    )


TOPIC_ALIASES = {
    "fog lights": "fog lamps",
    "fog light": "fog lamp",
    "front fog lights": "front fog lamps",
    "front fog light": "front fog lamp",
    "rear fog lights": "rear fog lamps",
    "rear fog light": "rear fog lamp",
    "head lights": "headlamps",
    "head light": "headlamp",
    "head lights": "headlamps",
    "head light": "headlamp",
}


def _canonical_topic_phrase(text):
    """Normalize common spoken component names without using semantic ranking."""

    value = _normalize_lookup_text(text)

    # Longest aliases first so multi-word replacements win.
    for source, target in sorted(
        TOPIC_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        value = re.sub(
            rf"\b{re.escape(source)}\b",
            target,
            value,
        )

    return " ".join(_topic_tokens(value))


def _title_matches_topic(entry, topic):
    """Exact/manual-topic match: title itself, not generic body relevance."""

    topic_norm = _canonical_topic_phrase(topic)
    if not topic_norm:
        return False

    title_norm = _clean_manual_title(entry.get("title", ""))
    return title_norm == topic_norm


def _title_contains_topic(entry, topic):
    topic_norm = _canonical_topic_phrase(topic)
    title_norm = _clean_manual_title(entry.get("title", ""))
    if not topic_norm or not title_norm:
        return False

    topic_tokens = set(topic_norm.split())
    title_tokens = set(_topic_tokens(title_norm))
    return topic_tokens.issubset(title_tokens)


def _diagram_intent(text):
    """Determine what kind of diagram the user asked for."""

    value = _normalize_lookup_text(text)

    connector_words = (
        "connector",
        "pinout",
        "pin out",
        "pin diagram",
        "connector view",
        "end view",
    )

    if any(word in value for word in connector_words):
        return "connector"

    # For a normal "show <component> diagram" request, default to the
    # actual wiring/schematic page rather than a connector/end-view page.
    return "wiring"


def _sorted_manual_pages(index):
    def key(item):
        page = str(item[0])
        m = re.match(r"(\d+)\.html$", page, flags=re.IGNORECASE)
        if m:
            return (0, int(m.group(1)))
        return (1, page.lower())

    return sorted(index.items(), key=key)


def _manual_diagram_page_candidates(
    index,
    topics,
    intent="wiring",
    current_page=None,
):
    """Return deterministic manual-page candidates in priority order."""

    normalized_topics = []
    for topic in topics or []:
        normalized = _canonical_topic_phrase(topic)
        if normalized and normalized not in normalized_topics:
            normalized_topics.append(normalized)

    entries = _sorted_manual_pages(index)
    candidates = []
    seen_pages = set()

    def eligible(entry):
        if _is_generic_page(entry):
            return False
        if intent == "connector":
            return _entry_is_connector_page(entry)
        return _entry_is_wiring_page(entry) and not _entry_is_connector_page(entry)

    def add(page, entry):
        key = str(page).lower()
        if key in seen_pages:
            return
        seen_pages.add(key)
        candidates.append((page, entry))

    # 1. Current page is considered only when it is the requested diagram
    # type and its title/topic actually matches the request.
    if current_page:
        current_key = str(current_page).strip().lower()
        for page, entry in entries:
            if str(page).lower() != current_key:
                continue
            if not eligible(entry):
                break
            if not normalized_topics or any(
                _title_matches_topic(entry, topic)
                or _title_contains_topic(entry, topic)
                for topic in normalized_topics
            ):
                add(page, entry)
            break

    # 2. Exact title match, preserving topic order and manual page order.
    # This is deliberately NOT a relevance score.
    for topic in normalized_topics:
        for page, entry in entries:
            if eligible(entry) and _title_matches_topic(entry, topic):
                add(page, entry)

    # 3. Complete topic in title.
    for topic in normalized_topics:
        for page, entry in entries:
            if eligible(entry) and _title_contains_topic(entry, topic):
                add(page, entry)

    # 4. Complete topic in the page's own metadata/breadcrumb.
    for topic in normalized_topics:
        topic_tokens = set(topic.split())
        if not topic_tokens:
            continue
        for page, entry in entries:
            if not eligible(entry):
                continue
            combined_tokens = set(
                _topic_tokens(
                    " ".join(
                        [
                            entry.get("title", ""),
                            entry.get("section", ""),
                            entry.get("search_text", ""),
                        ]
                    )
                )
            )
            if topic_tokens.issubset(combined_tokens):
                add(page, entry)

    return candidates


def _resolve_manual_diagram_page(
    index,
    topics,
    intent="wiring",
    current_page=None,
):
    """Return the first deterministic diagram-page match."""
    candidates = _manual_diagram_page_candidates(
        index,
        topics,
        intent=intent,
        current_page=current_page,
    )
    return candidates[0] if candidates else (None, None)



def _html_diagram_images_for_page(source_page):
    """Fallback only for a known exact manual page whose JSONL record has no images."""

    key = str(source_page).strip().lower()
    if key in _HTML_DIAGRAM_IMAGE_CACHE:
        return list(_HTML_DIAGRAM_IMAGE_CACHE[key])

    path = page_path(source_page)
    if path is None:
        _HTML_DIAGRAM_IMAGE_CACHE[key] = []
        return []

    html = read_html(path)
    if not html:
        _HTML_DIAGRAM_IMAGE_CACHE[key] = []
        return []

    images = []
    seen = set()

    for image in extract_images(path, html):
        try:
            resolved = Path(image).resolve()
            resolved.relative_to(IMAGE_ROOT.resolve())
        except (ValueError, OSError):
            continue

        if (
            resolved.is_file()
            and resolved.suffix.lower() in OPENABLE_DIAGRAM_EXTENSIONS
        ):
            image_key = str(resolved).lower()
            if image_key not in seen:
                seen.add(image_key)
                images.append(resolved)

    _HTML_DIAGRAM_IMAGE_CACHE[key] = images
    return list(images)


def _topics_from_dtc(dtc):
    """Build a deterministic topic list from manual DTC -> pinpoint metadata."""

    topics = []
    info = _DTC_INFO.get(dtc)
    if not info:
        return topics

    for test in info.get("tests", []):
        topic = _PINPOINT_TOPICS.get(test)
        if topic:
            topics.append(topic)

    desc = info.get("desc", "") or ""
    desc_norm = _normalize_lookup_text(desc)

    # Generic component anchors derived from the manual wording.
    if "fan" in desc_norm:
        topics.append("cooling fan")
    if "fog lamp" in desc_norm or "fog lamps" in desc_norm:
        topics.append("fog lamps")

    unique = []
    for topic in topics:
        key = _canonical_topic_phrase(topic)
        if key and key not in {_canonical_topic_phrase(x) for x in unique}:
            unique.append(topic)

    return unique



def _build_manual_image_index():
    """
    Read manual_chunks.jsonl ONCE and build a complete manual-page index.

    Every page is retained, including pages whose JSONL `images` array is
    empty. This is important because some manual wiring pages are navigation
    leaf pages (for example Fog Lamps) whose raster images are present in the
    corresponding HTML page but were not copied into the JSONL record.

    The JSONL remains authoritative when it contains image references. HTML is
    used only as a deterministic fallback for the exact page selected from the
    manual metadata.
    """
    global _MANUAL_IMAGE_INDEX

    if _MANUAL_IMAGE_INDEX is not None:
        return _MANUAL_IMAGE_INDEX

    index = {}

    print()
    print("Loading complete manual page/image index:")
    print(MANUAL_JSONL)

    if not MANUAL_JSONL.is_file():
        print("WARNING: manual_chunks.jsonl was not found.")
        print("Diagram lookup will be unavailable.")
        _MANUAL_IMAGE_INDEX = {}
        return _MANUAL_IMAGE_INDEX

    def dtc_entry(code):
        return _DTC_INFO.setdefault(
            code,
            {"tests": [], "desc": ""},
        )

    try:
        with MANUAL_JSONL.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as handle:

            for line_number, line in enumerate(handle, start=1):

                if line_number % 20000 == 0:
                    print(f"  ...read {line_number:,} records")

                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except Exception:
                    print(
                        f"WARNING: could not parse JSONL line {line_number}."
                    )
                    continue

                source_page = str(
                    record.get("source_file", "")
                ).strip()

                if not source_page:
                    continue

                title = str(record.get("title", "") or "").strip()
                section = str(record.get("section", "") or "").strip()
                text = str(record.get("text", "") or "")

                # ---- DTC / pinpoint relationships -----------------
                pinpoint = PINPOINT_TITLE_RE.match(title)

                if pinpoint:
                    _PINPOINT_TOPICS.setdefault(
                        pinpoint.group(1).upper(),
                        pinpoint.group(2).strip(),
                    )

                codes = []

                for raw_code in record.get("dtc_codes", []) or []:
                    code = dtc_base(raw_code)
                    if code and re.fullmatch(r"[PBCU][0-9A-F]{4}", code):
                        codes.append(code)

                for code in codes:
                    info = dtc_entry(code)
                    if pinpoint:
                        test = pinpoint.group(1).upper()
                        if test not in info["tests"]:
                            info["tests"].append(test)

                if "Pinpoint Test" in text and "|" in text:
                    for row in DTC_ROW_RE.finditer(text):
                        code = row.group(1).upper()
                        desc = re.sub(
                            r":?\s*No Sub Type Information",
                            "",
                            row.group(2),
                            flags=re.IGNORECASE,
                        ).strip()
                        test = row.group(3).upper()

                        info = dtc_entry(code)
                        if test not in info["tests"]:
                            info["tests"].append(test)
                        if desc and not info["desc"]:
                            info["desc"] = desc

                # ---- page metadata + images -----------------------
                entry = index.setdefault(
                    source_page,
                    {
                        "images": [],
                        "title": "",
                        "section": "",
                        "headings": [],
                        "search_text": "",
                        "captions": "",
                        "dtcs": set(),
                    },
                )

                entry["dtcs"].update(codes)

                for raw_image in record.get("images", []) or []:
                    image_path = _safe_image_path_from_json(raw_image)
                    if image_path and image_path not in entry["images"]:
                        entry["images"].append(image_path)

                for caption in re.findall(
                    r"fig\s+\d+\s*:\s*(.{0,180})",
                    text,
                    flags=re.IGNORECASE,
                ):
                    if len(entry["captions"]) < 4000:
                        entry["captions"] += " " + caption

                headings = [
                    str(x).strip()
                    for x in (record.get("headings", []) or [])
                    if str(x).strip()
                ]

                preview = text[:6000]

                if len(preview) > len(entry["search_text"]):
                    entry["search_text"] = preview

                if title:
                    entry["title"] = title

                if section:
                    entry["section"] = section

                for heading in headings:
                    if heading not in entry["headings"]:
                        entry["headings"].append(heading)

        # Precompute metadata for both image-bearing and metadata-only pages.
        document_frequency = {}
        image_page_count = 0

        for entry in index.values():
            if entry["images"]:
                image_page_count += 1

            entry["title_tokens"] = set(
                _topic_tokens(
                    " ".join(
                        [entry["title"], entry["section"]]
                        + entry["headings"]
                    )
                )
            )

            entry["caption_tokens"] = set(
                _topic_tokens(entry["captions"])
            )

            entry["body_tokens"] = set(
                _topic_tokens(entry["search_text"])
            )

            combined_raw = (
                entry["title"] + " " + entry["section"] + " "
                + " ".join(entry["headings"]) + " "
                + entry["captions"] + " " + entry["search_text"]
            )

            combined_lower = combined_raw.lower()

            entry["wiring"] = (
                "wiring diagram" in combined_lower
                or "wiring diagrams" in combined_lower
                or "oem wiring diagrams" in combined_lower
                or "system wiring diagrams" in combined_lower
            )

            entry["connector"] = _entry_is_connector_page(entry)

            for token in (
                entry["title_tokens"] | entry["caption_tokens"]
            ):
                document_frequency[token] = (
                    document_frequency.get(token, 0) + 1
                )

        total = max(1, len(index))
        _TOKEN_IDF.clear()
        for token, count in document_frequency.items():
            _TOKEN_IDF[token] = math.log((total + 1) / (count + 1)) + 1.0

        _MANUAL_IMAGE_INDEX = index

        print(
            f"Indexed {len(index)} manual pages; "
            f"{image_page_count} contain JSONL image references, "
            f"{len(_DTC_INFO)} DTCs, "
            f"{len(_PINPOINT_TOPICS)} pinpoint tests."
        )

        print(
            f"Image references resolved to real files: "
            f"{_REF_COUNTS['resolved']} of {_REF_COUNTS['total']}."
        )

        if _UNRESOLVED_REFS and _REF_COUNTS["resolved"] < _REF_COUNTS["total"]:
            print("Examples of references NOT found in the images folder:")
            for ref in _UNRESOLVED_REFS:
                print(f"  {ref}")

        return _MANUAL_IMAGE_INDEX

    except Exception as exc:
        print(
            f"WARNING: could not load manual_chunks.jsonl: {exc}"
        )
        _MANUAL_IMAGE_INDEX = {}
        return _MANUAL_IMAGE_INDEX


INDEX_CACHE_VERSION = 2


def _index_cache_path():
    return Path(str(MANUAL_JSONL) + ".diagram_cache.pkl")


def _index_cache_signature():
    """Changes whenever the JSONL or the images folder changes."""

    stat = MANUAL_JSONL.stat()

    try:
        image_stamp = int(IMAGE_ROOT.stat().st_mtime)
    except OSError:
        image_stamp = 0

    return (
        INDEX_CACHE_VERSION,
        stat.st_size,
        int(stat.st_mtime),
        str(IMAGE_ROOT),
        image_stamp,
    )


def _load_index_cache():
    """Return the cached index, or None if missing/stale/corrupt."""
    global _MANUAL_IMAGE_INDEX

    path = _index_cache_path()

    if not path.is_file() or not MANUAL_JSONL.is_file():
        return None

    try:

        with path.open("rb") as handle:
            data = pickle.load(handle)

        if data.get("signature") != _index_cache_signature():
            print("Diagram cache is out of date; rebuilding.")
            return None

        _DTC_INFO.clear()
        _DTC_INFO.update(data["dtc_info"])
        _PINPOINT_TOPICS.clear()
        _PINPOINT_TOPICS.update(data["pinpoint_topics"])
        _TOKEN_IDF.clear()
        _TOKEN_IDF.update(data["token_idf"])
        _REF_COUNTS.update(data["ref_counts"])
        _UNRESOLVED_REFS[:] = data["unresolved"]

        _MANUAL_IMAGE_INDEX = data["index"]

        print(
            f"Diagram index loaded from cache: "
            f"{len(_MANUAL_IMAGE_INDEX)} pages, {len(_DTC_INFO)} DTCs, "
            f"{_REF_COUNTS['resolved']} of {_REF_COUNTS['total']} "
            f"image references resolved."
        )

        return _MANUAL_IMAGE_INDEX

    except Exception as exc:

        print(f"Diagram cache could not be used ({exc}); rebuilding.")

        return None


def _save_index_cache(index):

    try:

        data = {
            "signature": _index_cache_signature(),
            "index": index,
            "dtc_info": _DTC_INFO,
            "pinpoint_topics": _PINPOINT_TOPICS,
            "token_idf": _TOKEN_IDF,
            "ref_counts": dict(_REF_COUNTS),
            "unresolved": list(_UNRESOLVED_REFS),
        }

        temp = _index_cache_path().with_suffix(".tmp")

        with temp.open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        temp.replace(_index_cache_path())

        print(f"Diagram cache saved: {_index_cache_path()}")

    except Exception as exc:

        print(f"WARNING: could not save diagram cache: {exc}")


def load_manual_image_index():
    """
    Cached entry point. First run parses the 75 MB JSONL (slow) and
    saves a small cache next to it; later runs load the cache.
    """

    if _MANUAL_IMAGE_INDEX is not None:
        return _MANUAL_IMAGE_INDEX

    cached = _load_index_cache()

    if cached is not None:
        return cached

    index = _build_manual_image_index()

    if index:
        _save_index_cache(index)

    return index


def _diagram_topic_phrases(
    active_test=None,
    step_text=None,
):
    """
    Build deterministic topic phrases from the current diagnostic
    step. No embeddings, LLM, or image classification is used.
    """
    phrases = []

    def add(
        phrase,
    ):
        phrase = _normalize_lookup_text(
            phrase
        )

        if (
            phrase
            and phrase not in phrases
        ):
            phrases.append(
                phrase
            )

    active = str(
        active_test or ""
    ).upper().strip()

    if active == "KF":
        add(
            "fan control relays"
        )
        add(
            "fan control relay"
        )
        add(
            "cooling fan"
        )

    elif active == "KN":
        add(
            "variable speed electric cooling fan"
        )
        add(
            "cooling fan"
        )

    text = str(
        step_text or ""
    )

    important_phrases = [
        "fan control relay",
        "fan control relays",
        "cooling fan",
        "variable speed electric cooling fan",
        "fan motor",
        "cooling fan motor",
        "relay",
        "relay ground",
        "relay voltage",
        "circuit",
        "open circuit",
        "short to ground",
        "short to voltage",
        "connector view",
        "connector",
        "harness",
        "power",
        "ground",
        "fcv",
        "cacff",
        "lfc",
        "fanpwr",
        "fc1",
        "fc2",
        "fc3",
        "mfc",
        "hfc",
    ]

    lower = text.lower()

    for phrase in important_phrases:

        if phrase in lower:
            add(
                phrase
            )

    return phrases


def _score_json_diagram_page(
    source_page,
    entry,
    active_test=None,
    step_id=None,
    step_text=None,
    current_page=None,
):
    """
    Score only the PAGE metadata in manual_chunks.jsonl.

    The score selects a page whose own extracted text/title matches the
    current diagnostic topic. The image files themselves are never
    selected by filename relevance.
    """
    title = _normalize_lookup_text(
        " ".join(
            [
                entry.get(
                    "title",
                    "",
                ),
                entry.get(
                    "section",
                    "",
                ),
                " ".join(
                    entry.get(
                        "headings",
                        [],
                    )
                ),
            ]
        )
    )

    preview = _normalize_lookup_text(
        entry.get(
            "search_text",
            "",
        )
    )

    step_lower = _normalize_lookup_text(
        step_text or ""
    )

    active = str(
        active_test or ""
    ).upper().strip()

    current = str(
        current_page or ""
    ).strip().lower()

    phrases = _diagram_topic_phrases(
        active_test=active_test,
        step_text=step_text,
    )

    score = 0

    # A directly referenced current page is preferred only when the
    # page itself looks like a real diagram page. This prevents generic
    # Test Procedure pages containing unrelated illustrations from winning.
    if (
        source_page.lower()
        == current
    ):
        if (
            "wiring diagram" in preview
            or "wiring diagram" in title
            or "diagram" in title
        ):
            score += 10000

    # The page explicitly describes a wiring diagram.
    if "wiring diagram" in preview:
        score += 5000

    if "wiring diagram" in title:
        score += 5000

    # A topic-specific title is much stronger than a generic image page.
    for phrase in phrases:

        if phrase in title:
            score += 1800

        elif phrase in preview[:4000]:
            score += 150

    # Exact step/component words in the title.
    if active == "KF":
        if "fan control" in title:
            score += 1800

    if active == "KN":
        if "cooling fan" in title:
            score += 1800

    # Cooling Fan is the exact wiring-diagram topic in this manual.
    if (
        "cooling fan" in title
        and "cooling fan" in step_lower
    ):
        score += 3500

    # Caption-level confirmation.
    caption_text = " ".join(
        re.findall(
            r"fig\s+\d+\s*:\s*(.{0,180})",
            entry.get(
                "search_text",
                "",
            ),
            flags=re.IGNORECASE,
        )
    )

    caption_text = _normalize_lookup_text(
        caption_text
    )

    if "wiring diagram" in caption_text:
        score += 2500

    for phrase in phrases:

        if phrase in caption_text:
            score += 500

    # Generic procedure pages are not diagram pages unless they were
    # directly identified as a diagram page above.
    generic_title = (
        title == "test procedure"
        or title.endswith(
            " test procedure"
        )
    )

    if generic_title:
        score -= 2500

    # Connector-end-view pages are useful when the actual step explicitly
    # asks for a connector view, otherwise prefer wiring diagrams.
    if (
        "connector end view" in title
        and "connector view" not in step_lower
    ):
        score -= 2500

    # More referenced figures is a useful tie-breaker for true multi-panel
    # wiring diagrams, without ever selecting individual files by score.
    score += min(
        len(entry.get("images", [])),
        12,
    ) * 20

    return score


def _safe_manual_page_images(page, entry):
    """Get the exact raster image set for one already-selected manual page."""

    images = list(entry.get("images", []))

    # If preprocessing stored images, those references are authoritative.
    # Never replace them with a fuzzy alternative.
    if not images:
        images = _html_diagram_images_for_page(page)
        if images:
            print(
                "JSONL has no image list for this exact page; extracted "
                "raster references from that exact HTML page."
            )

    safe_images = []
    seen = set()

    for image_path in images:
        try:
            resolved = Path(image_path).resolve()
            resolved.relative_to(IMAGE_ROOT.resolve())
        except (ValueError, OSError):
            continue

        if (
            not resolved.is_file()
            or resolved.suffix.lower() not in OPENABLE_DIAGRAM_EXTENSIONS
        ):
            continue

        key = str(resolved).lower()
        if key not in seen:
            seen.add(key)
            safe_images.append(resolved)

    return safe_images


def _select_first_page_with_images(
    index,
    topics,
    intent="wiring",
    current_page=None,
):
    """Try deterministic page candidates until one has actual raster images."""

    candidates = _manual_diagram_page_candidates(
        index,
        topics,
        intent=intent,
        current_page=current_page,
    )

    for page, entry in candidates:
        images = _safe_manual_page_images(page, entry)
        if images:
            return page, entry, images

    return None, None, []


def find_json_diagram_images(
    dtc=None,
    active_test=None,
    step_id=None,
    step_text=None,
    current_page=None,
):
    """Resolve a diagnostic diagram to a deterministic wiring page and exact images."""

    index = load_manual_image_index()
    if not index:
        return []

    print()
    print("Looking up the exact manual diagram page...")
    print(f"DTC: {dtc}")
    print(f"Pinpoint Test: {active_test}")
    print(f"Step: {step_id}")
    print(f"Current page: {current_page}")

    topics = _diagram_topic_phrases(
        active_test=active_test,
        step_text=step_text,
    )

    # DTC-derived topics are used when the current step did not already give
    # us a concrete component.
    if dtc:
        for topic in _topics_from_dtc(dtc):
            if _canonical_topic_phrase(topic) not in {
                _canonical_topic_phrase(x) for x in topics
            }:
                topics.append(topic)

    topic_keys = {_canonical_topic_phrase(x) for x in topics}
    if any("fan" in key for key in topic_keys):
        if "cooling fan" not in topic_keys:
            topics.append("cooling fan")

    page, entry, images = _select_first_page_with_images(
        index,
        topics,
        intent="wiring",
        current_page=current_page,
    )

    if page is None:
        print("No deterministic wiring-diagram page with raster images matched the diagnostic context.")
        return []

    print(f"Matched manual diagram page: {page}")
    print(f"Page title: {entry.get('title', '')}")
    print("Exact diagram image set:")
    for image_path in images:
        print(f"  {image_path}")

    return images



def find_topic_diagram_images(
    dtc=None,
    topic_words=None,
    request_text=None,
    active_test=None,
    step_text=None,
    current_page=None,
):
    """Resolve a direct component diagram request without semantic/image scoring."""

    index = load_manual_image_index()
    if not index:
        return []

    intent = _diagram_intent(request_text or "")

    topics = []
    raw_topic = " ".join(topic_words or []).strip()

    if raw_topic:
        topics.append(raw_topic)
    elif dtc:
        topics.extend(_topics_from_dtc(dtc))
    elif active_test:
        topics.extend(_diagram_topic_phrases(active_test, step_text))

    topic_keys = {_canonical_topic_phrase(x) for x in topics}
    if any("fan" in key for key in topic_keys):
        if "cooling fan" not in topic_keys:
            topics.append("cooling fan")

    unique_topics = []
    seen_topics = set()
    for topic in topics:
        key = _canonical_topic_phrase(topic)
        if key and key not in seen_topics:
            seen_topics.add(key)
            unique_topics.append(topic)

    print()
    print("Deterministic diagram request")
    print(f"Intent: {intent}")
    print(f"DTC: {dtc}")
    print(f"Topic: {raw_topic or '(from manual DTC/pinpoint context)'}")
    print(f"Candidate manual topics: {unique_topics}")

    if not unique_topics:
        print("No component topic was specified for the direct diagram request.")
        return []

    page, entry, images = _select_first_page_with_images(
        index,
        unique_topics,
        intent=intent,
        current_page=current_page,
    )

    if page is None:
        print("No deterministic manual diagram page with raster images matched the request.")
        return []

    print(f"Matched manual diagram page: {page}")
    print(f"Page title: {entry.get('title', '')}")
    print("Exact diagram image set:")
    for image_path in images:
        print(f"  {image_path}")

    return images



def handle_diagram_request(text, session=None):
    """
    Voice entry point for diagram requests.

      'show cooling fan diagrams'   -> topic lookup
      'P0563, show diagrams'        -> DTC lookup
      'show diagram' (in procedure) -> current step's diagram
    """

    dtc = extract_spoken_dtc(text)

    topic_words = diagram_topic_words(text)

    if dtc:
        print(f"Diagram request for DTC: {dtc}")

    if topic_words:
        print(f"Diagram request topic: {' '.join(topic_words)}")

    if session is not None and not dtc and not topic_words:

        session.show_diagram()

        return

    if not dtc and not topic_words:

        speak(
            "Tell me which diagram you want. For example, say show "
            "cooling fan diagrams, or give a trouble code and say "
            "show diagrams."
        )

        return

    images = find_topic_diagram_images(
        dtc=dtc,
        topic_words=topic_words,
        request_text=text,
        active_test=(session.active_test if session is not None else None),
        step_text=(session.current_step_text() if session is not None else None),
        current_page=(session.current_page() if session is not None else None),
    )

    if not images:

        speak(
            "I could not find a matching diagram in the manual."
        )

        return

    opened = open_diagram_images(images)

    if opened:

        speak(
            f"I opened {opened} diagram "
            f"{'image' if opened == 1 else 'images'} from the manual."
        )

    else:

        speak(
            "I found the diagram references but could not open "
            "the image files."
        )


def find_best_diagram(
    dtc=None,
    active_test=None,
    step_id=None,
    step_text=None,
    current_page=None,
):
    """
    Resolve the current diagnostic diagram request through the same
    deterministic manual-page/image logic used by direct diagram commands.

    No global image scoring is used here. The selected manual page's
    JSONL/HTML image references remain authoritative.
    """

    print()
    print("=" * 70)
    print("EXTRACTING DIAGRAM REFERENCES FROM MANUAL DATA")
    print("=" * 70)

    images = find_json_diagram_images(
        dtc=dtc,
        active_test=active_test,
        step_id=step_id,
        step_text=step_text,
        current_page=current_page,
    )

    if not images:
        print(
            "No diagram images were found for the current diagnostic context."
        )
        return None, None, None

    return "images", images, current_page


def open_diagram(
    dtc=None,
    active_test=None,
    step_id=None,
    step_text=None,
    current_page=None,
):

    result_type, target, source = (
        find_best_diagram(
            dtc=dtc,
            active_test=active_test,
            step_id=step_id,
            step_text=step_text,
            current_page=current_page,
        )
    )

    if not target:
        return False

    if result_type == "images":

        targets = (
            target
            if isinstance(target, list)
            else [target]
        )

        return open_diagram_images(targets) > 0

    # Retain the existing page-opening branch for compatibility with
    # other code, although the new diagram lookup itself returns only images.
    if result_type == "image":

        try:
            Path(
                target
            ).resolve().relative_to(
                IMAGE_ROOT.resolve()
            )
        except ValueError:
            print(
                "Refusing to open an image outside the images folder."
            )
            return False

        if open_local_file(
            target
        ):

            print()
            print(
                "Diagram opened:"
            )
            print(
                target
            )
            print()

            return True

    if result_type == "page":

        if open_local_file(
            target
        ):

            print()
            print(
                "Relevant manual page opened:"
            )
            print(
                target
            )
            print()

            return True

    return False


# ============================================================
# SOURCE DISPLAY
# ============================================================

def show_sources_for_results(
    results
):

    pages = []

    for item in results or []:

        try:

            page = get_source_page(
                item.get(
                    "metadata",
                    {},
                )
            )

        except Exception:

            page = None

        if page and page not in pages:

            pages.append(
                page
            )

    print()
    print("=" * 70)
    print("MANUAL SOURCE PAGES")
    print("=" * 70)

    if not pages:

        print(
            "No source pages found."
        )

    for page in pages:

        print(
            f"  {page}"
        )

    print("=" * 70)
    print()


# ============================================================
# STEP HELPERS
# ============================================================

def extract_step_number(
    text
):

    if not text:
        return None

    match = re.search(
        r"\b([A-Z]{2}\d{1,2})\b",
        text,
        flags=re.IGNORECASE,
    )

    if match:

        return match.group(
            1
        ).upper()

    return None


def extract_dtc_from_results(
    results
):

    for item in results or []:

        dtc = item.get(
            "dtc"
        )

        if dtc:

            base = dtc_base(
                dtc
            )

            if base:
                return base.upper()

        for field in [
            "document",
            "label",
        ]:

            value = item.get(
                field,
                "",
            )

            match = re.search(
                r"\b([PBCU]\d{4})\b",
                str(value),
                flags=re.IGNORECASE,
            )

            if match:

                return match.group(
                    1
                ).upper()

    return None


def get_group_name(
    item
):

    group = item.get(
        "group"
    )

    return str(
        group
    ) if group else ""


def get_test_from_group(
    group
):

    if not group:
        return None

    match = re.search(
        r"Pinpoint Test\s+([A-Z]{2})",
        str(group),
        flags=re.IGNORECASE,
    )

    if match:

        return match.group(
            1
        ).upper()

    return None


# ============================================================
# DIAGNOSTIC SESSION
# ============================================================

class DiagnosticSession:

    def __init__(
        self,
        question,
        results,
    ):

        self.question = question
        self.results = results or []

        self.dtc = (
            extract_dtc_from_results(
                self.results
            )
        )

        self.groups = {}

        self.active_test = None

        self.steps = []

        self.current_index = 0

        self.last_spoken = ""

        self.build_groups()

    # --------------------------------------------------------
    # GROUPS
    # --------------------------------------------------------

    def build_groups(self):

        for item in self.results:

            group = get_group_name(
                item
            )

            if not group:
                continue

            self.groups.setdefault(
                group,
                [],
            ).append(
                item
            )

        if self.dtc:

            routes = []

            for group in self.groups:

                test = get_test_from_group(
                    group
                )

                if (
                    test
                    and test not in routes
                ):

                    routes.append(
                        test
                    )

            if self.dtc == "P0480":

                ordered = []

                for test in [
                    "KF",
                    "KN",
                ]:

                    if test in routes:

                        ordered.append(
                            test
                        )

                for test in routes:

                    if test not in ordered:

                        ordered.append(
                            test
                        )

                routes = ordered

            if routes:

                self.active_test = routes[0]

        if not self.active_test:

            for group in self.groups:

                test = get_test_from_group(
                    group
                )

                if test:

                    self.active_test = test

                    break

    # --------------------------------------------------------
    # ACTIVE ITEMS
    # --------------------------------------------------------

    def active_items(self):

        if not self.active_test:
            return []

        selected = []

        for group, items in self.groups.items():

            test = get_test_from_group(
                group
            )

            if test == self.active_test:

                selected.extend(
                    items
                )

        return selected

    # --------------------------------------------------------
    # BUILD STEPS
    # --------------------------------------------------------

    def build_steps(self):

        items = self.active_items()

        steps = []

        seen = set()

        for item in items:

            if item.get(
                "search_type"
            ) != "procedure_step":

                continue

            document = item.get(
                "document",
                "",
            )

            if not document:
                continue

            key = (
                str(
                    item.get(
                        "metadata",
                        {},
                    )
                ),
                document,
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            steps.append(
                item
            )

        self.steps = steps

    # --------------------------------------------------------
    # CURRENT STEP
    # --------------------------------------------------------

    def current_step(self):

        if not self.steps:
            return None

        self.current_index = max(
            0,
            min(
                self.current_index,
                len(self.steps) - 1,
            ),
        )

        return self.steps[
            self.current_index
        ]

    # --------------------------------------------------------
    # CURRENT PAGE
    # --------------------------------------------------------

    def current_page(self):

        item = self.current_step()

        if not item:
            return None

        try:

            return get_source_page(
                item.get(
                    "metadata",
                    {},
                )
            )

        except Exception:

            return None

    # --------------------------------------------------------
    # CURRENT TEXT
    # --------------------------------------------------------

    def current_step_text(self):

        item = self.current_step()

        if not item:
            return ""

        document = item.get(
            "document",
            "",
        )

        if not document:
            return ""

        try:

            rendered = render_step(
                document,
                self.dtc,
                compact=True,
            )

            if rendered:

                return str(
                    rendered
                ).strip()

        except Exception:
            pass

        return str(
            document
        ).strip()

    # --------------------------------------------------------
    # SPOKEN STEP
    # --------------------------------------------------------

    def spoken_step(self):

        text = self.current_step_text()

        if not text:

            return (
                "I could not extract the current diagnostic step."
            )

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        max_chars = 650

        if len(text) > max_chars:

            shortened = text[
                :max_chars
            ]

            last_period = shortened.rfind(
                "."
            )

            if last_period > 250:

                shortened = shortened[
                    :last_period + 1
                ]

            text = shortened

        return text

    # --------------------------------------------------------
    # DISPLAY PATH
    # --------------------------------------------------------

    def display_path(self):

        print()
        print("=" * 70)
        print("DIAGNOSTIC PATH")
        print("=" * 70)

        if self.dtc:

            print(
                f"DTC: {self.dtc}"
            )

        if self.active_test:

            print(
                f"Active path: Pinpoint Test {self.active_test}"
            )

        print()

        if self.steps:

            for index, item in enumerate(
                self.steps
            ):

                document = item.get(
                    "document",
                    "",
                )

                step_id = extract_step_number(
                    document
                )

                marker = (
                    ">>"
                    if index
                    == self.current_index
                    else "  "
                )

                if step_id:

                    print(
                        f"{marker} {step_id}"
                    )

                else:

                    print(
                        f"{marker} Step {index + 1}"
                    )

        print("=" * 70)
        print()

    # --------------------------------------------------------
    # SPEAK CURRENT STEP
    # --------------------------------------------------------

    def speak_current_step(self):

        spoken = self.spoken_step()

        self.last_spoken = spoken

        speak(
            spoken
        )

        speak(
            "Say continue for the next step, "
            "repeat to hear this again, "
            "or say show diagram to open the relevant manual diagram."
        )

    # --------------------------------------------------------
    # SWITCH TEST
    # --------------------------------------------------------

    def switch_to_test(
        self,
        test,
    ):

        if not test:
            return False

        test = test.upper()

        available = {
            get_test_from_group(
                group
            )
            for group in self.groups
        }

        if test not in available:
            return False

        self.active_test = test

        self.current_index = 0

        self.build_steps()

        return True

    # --------------------------------------------------------
    # NEXT
    # --------------------------------------------------------

    def next_step(self):

        if not self.steps:
            return False

        if (
            self.current_index + 1
            >= len(self.steps)
        ):

            if self.dtc == "P0480":

                if self.active_test == "KF":

                    if self.switch_to_test(
                        "KN"
                    ):

                        speak(
                            "The KF diagnostic path has reached "
                            "its end. I am switching to the KN "
                            "cooling fan motor path."
                        )

                        self.display_path()

                        self.speak_current_step()

                        return True

            speak(
                "The current diagnostic path has reached its end."
            )

            return False

        self.current_index += 1

        self.display_path()

        self.speak_current_step()

        return True

    # --------------------------------------------------------
    # PREVIOUS
    # --------------------------------------------------------

    def previous_step(self):

        if not self.steps:

            return False

        if self.current_index <= 0:

            speak(
                "This is already the first step."
            )

            return False

        self.current_index -= 1

        self.display_path()

        self.speak_current_step()

        return True

    # --------------------------------------------------------
    # REPEAT
    # --------------------------------------------------------

    def repeat(self):

        if self.last_spoken:

            speak(
                self.last_spoken
            )

        else:

            self.speak_current_step()

    # --------------------------------------------------------
    # SHOW DIAGRAM
    # --------------------------------------------------------

    def show_diagram(self):

        step = self.current_step()

        if not step:

            speak(
                "There is no active diagnostic step."
            )

            return

        step_text = self.current_step_text()

        step_id = extract_step_number(
            step_text
        )

        current_page = self.current_page()

        print()
        print(
            f"Current DTC: {self.dtc}"
        )

        print(
            f"Current test: {self.active_test}"
        )

        print(
            f"Current step: {step_id}"
        )

        print(
            f"Current page: {current_page}"
        )

        success = open_diagram(
            dtc=self.dtc,
            active_test=self.active_test,
            step_id=step_id,
            step_text=step_text,
            current_page=current_page,
        )

        if success:

            speak(
                "I found and opened the most relevant "
                "diagram or manual diagram page."
            )

        else:

            speak(
                "I searched the relevant manual pages but "
                "could not find a diagram that I could "
                "confidently associate with this diagnostic step."
            )

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    def show_sources(self):

        show_sources_for_results(
            self.results
        )

        speak(
            "I displayed the manual source pages on the screen."
        )

    # --------------------------------------------------------
    # EXPLAIN
    # --------------------------------------------------------

    def explain(self):

        context = build_context(
            self.question,
            self.results,
        )

        answer = ask_ollama(
            self.question,
            context,
            focus=self.current_step_text(),
        )

        if answer:

            speak(
                answer
            )

        else:

            speak(
                "I could not generate an explanation."
            )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def run(self):

        if not self.results:

            speak(
                "I could not retrieve diagnostic information "
                "for that request."
            )

            return

        self.build_steps()

        if not self.steps:

            speak(
                "I found relevant manual information, "
                "but I could not build an interactive "
                "diagnostic step sequence."
            )

            context = build_context(
                self.question,
                self.results,
            )

            answer = ask_ollama(
                self.question,
                context,
            )

            if answer:
                speak(answer)

            return

        self.display_path()

        if self.dtc:

            speak(
                f"I found diagnostic information for {self.dtc}."
            )

        if self.active_test:

            speak(
                f"I am starting Pinpoint Test {self.active_test}."
            )

        self.speak_current_step()

        while True:

            try:

                text = listen()

            except KeyboardInterrupt:

                print(
                    "Stopping voice session."
                )

                return

            if not text:

                # Silence: keep listening, no assistant response.
                continue

            command = command_from_text(
                text
            )

            # ------------------------------------------------
            # EXIT
            # ------------------------------------------------

            if command == "exit":

                speak(
                    "Diagnostic session ended."
                )

                return

            # ------------------------------------------------
            # DIAGRAM
            # ------------------------------------------------

            if command == "diagram":

                handle_diagram_request(
                    text,
                    session=self,
                )

                continue

            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            if command == "sources":

                self.show_sources()

                continue

            # ------------------------------------------------
            # EXPLAIN
            # ------------------------------------------------

            if command == "explain":

                self.explain()

                continue

            # ------------------------------------------------
            # RESTART
            # ------------------------------------------------

            if command == "restart":

                self.current_index = 0

                self.build_steps()

                self.display_path()

                self.speak_current_step()

                continue

            # ------------------------------------------------
            # REPEAT
            # ------------------------------------------------

            if command == "repeat":

                self.repeat()

                continue

            # ------------------------------------------------
            # BACK
            # ------------------------------------------------

            if command == "back":

                self.previous_step()

                continue

            # ------------------------------------------------
            # CONTINUE
            # ------------------------------------------------

            if command == "continue":

                self.next_step()

                continue

            # ------------------------------------------------
            # YES
            # ------------------------------------------------

            if command == "yes":

                self.next_step()

                continue

            # ------------------------------------------------
            # NO
            # ------------------------------------------------

            if command == "no":

                self.next_step()

                continue

            # ------------------------------------------------
            # MEASUREMENT
            # ------------------------------------------------

            measurement = extract_measurement(
                text
            )

            if measurement:

                print(
                    f"Detected measurement: {measurement}"
                )

                speak(
                    f"I heard {measurement}. "
                    "Continue when you are ready."
                )

                continue

            # ------------------------------------------------
            # UNKNOWN
            # ------------------------------------------------

            print()
            print(
                "Unrecognized command."
            )

            print(
                "Try:"
            )

            print(
                "  continue"
            )

            print(
                "  yes / no"
            )

            print(
                "  repeat"
            )

            print(
                "  back"
            )

            print(
                "  show diagram"
            )

            print(
                "  show sources"
            )

            print(
                "  explain"
            )

            print(
                "  restart"
            )

            print(
                "  exit"
            )

            speak(
                "I didn't understand that command."
            )


# ============================================================
# MEASUREMENT RECOGNITION
# ============================================================

def extract_measurement(
    text
):

    if not text:
        return None

    text = normalize_text(
        text
    )

    patterns = [
        r"\b\d+(?:\.\d+)?\s*(?:volts?|v)\b",
        r"\b\d+(?:\.\d+)?\s*(?:ohms?|Ω)\b",
        r"\b\d+(?:\.\d+)?\s*(?:kilo\s*ohms?|kohms?|kΩ)\b",
        r"\b\d+(?:\.\d+)?\s*(?:amps?|amperes?|a)\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            return match.group(
                0
            )

    return None


# ============================================================
# GENERAL RAG QUESTION
# ============================================================

def answer_general_question(
    question
):

    results = retrieve(
        question
    )

    if not results:

        speak(
            "I could not find relevant information "
            "in the service manual."
        )

        return

    dtc = extract_dtc_from_results(
        results
    )

    if dtc:

        session = DiagnosticSession(
            question,
            results,
        )

        session.run()

        return

    context = build_context(
        question,
        results,
    )

    answer = ask_ollama(
        question,
        context,
    )

    if answer:

        speak(
            answer
        )

    else:

        speak(
            "I could not generate an answer."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("FORD ECOSPORT VOICE DIAGNOSTIC ASSISTANT")
    print("=" * 70)

    print()
    print(
        f"Whisper: {WHISPER_MODEL_NAME} ({WHISPER_DEVICE})"
    )

    print(
        f"Whisper CPU threads: {CPU_THREADS}"
    )

    print(
        f"TTS: {TTS_VOICE}"
    )

    print(
        f"Manual: {MANUAL_ROOT}"
    )

    print()
    print("Voice commands:")
    print("  P0480")
    print("  What does P0480 mean?")
    print("  Continue")
    print("  Yes / No")
    print("  Repeat")
    print("  Go back")
    print("  Show diagram")
    print("  Show wiring diagram")
    print("  Show cooling fan diagrams")
    print("  P0563, show diagrams")
    print("  Show sources")
    print("  Explain")
    print("  Restart")
    print("  Exit")

    print()
    print("=" * 70)
    print()

    # Read manual_chunks.jsonl once now, not in the middle of a request.
    load_manual_image_index()

    speak(
        "Ford EcoSport diagnostic assistant ready."
    )

    while True:

        try:

            print()
            print(
                "Waiting for your question..."
            )

            text = listen()

            if not text:
                continue

            command = command_from_text(
                text
            )

            if command == "exit":

                speak(
                    "Goodbye."
                )

                break

            if command == "diagram":

                handle_diagram_request(
                    text,
                    session=None,
                )

                continue

            answer_general_question(
                text
            )

        except KeyboardInterrupt:

            print()
            print(
                "Exiting..."
            )

            break

        except Exception as exc:

            print()
            print(
                "Unexpected error:"
            )

            print(
                exc
            )

            print()

            speak(
                "An error occurred. "
                "Please try again."
            )


# ============================================================
# CLEANUP
# ============================================================

def cleanup():

    try:
        pygame.mixer.music.stop()
    except Exception:
        pass

    try:
        pygame.mixer.quit()
    except Exception:
        pass


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    finally:

        cleanup()