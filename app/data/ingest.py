import json
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path(__file__).parent
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR.parent.parent / "chroma_db"
PROVIDERS_FILE = DATA_DIR / "providers.json"

PDF_CATEGORIES = [
    "health_insurance",
    "car_insurance",
    "term_insurance",
    "travel_insurance",
    "other",
]


def load_providers() -> dict:
    with open(PROVIDERS_FILE) as f:
        return json.load(f)


def get_vectorstore() -> Chroma:
    """Return ChromaDB vectorstore; data comes from uploads (ingested via API or ingest_all_uploads)."""
    embeddings = OllamaEmbeddings(model="qwen3-embedding")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="policies",
    )
