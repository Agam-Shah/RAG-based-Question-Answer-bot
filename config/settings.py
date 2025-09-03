from pathlib import Path
from transformers import AutoTokenizer

# ===== Models =====
MODEL_NAME: str = "google/flan-t5-base"

# Load once, reuse everywhere (splitter + LLM share the same tokenizer)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

# ===== Paths =====
BASE_DIR: Path = Path(__file__).resolve().parent.parent
UPLOAD_DIR: Path = BASE_DIR / "uploads"
CHROMA_DIR: Path = BASE_DIR / "chroma"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
