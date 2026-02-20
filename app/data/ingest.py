import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path(__file__).parent
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR.parent.parent / "chroma_db"
PROVIDERS_FILE = DATA_DIR / "providers.json"

# Allowed PDF categories for uploads; each gets a subfolder under UPLOADS_DIR
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


def ingest_single_pdf(pdf_path: Path, category: str, source_filename: str | None = None) -> int:
    """Load one PDF, chunk it, and add to ChromaDB with category metadata. Returns chunk count."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    name = source_filename or pdf_path.name
    embeddings = OllamaEmbeddings(model="qwen3-embedding")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    for page in pages:
        page.metadata["category"] = category
        page.metadata["source_file"] = name
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="policies",
    )
    vectorstore.add_documents(chunks)
    return len(chunks)


def ingest_all_uploads(force: bool = False) -> int:
    """Scan UPLOADS_DIR by category and ingest every PDF into ChromaDB. Returns total chunks added."""
    total = 0
    for category in PDF_CATEGORIES:
        category_dir = UPLOADS_DIR / category
        if not category_dir.is_dir():
            continue
        for pdf_path in category_dir.glob("*.pdf"):
            try:
                n = ingest_single_pdf(pdf_path, category, pdf_path.name)
                total += n
                print(f"  Ingested {pdf_path.name} ({n} chunks)")
            except Exception as e:
                print(f"  Skip {pdf_path.name}: {e}")
    return total


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    print("Ingesting all PDFs from uploads...")
    ingest_all_uploads()
