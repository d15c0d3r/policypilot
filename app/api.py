import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

from app.data.ingest import (
    PDF_CATEGORIES,
    UPLOADS_DIR,
    ingest_single_pdf,
)


app = FastAPI(title="PolicyPilot API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
                   "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Upload endpoints ──

def run_ingest(pdf_path: str, category: str, source_filename: str) -> None:
    try:
        count = ingest_single_pdf(Path(pdf_path), category, source_filename)
        print(f"Ingested {source_filename} ({count} chunks) into ChromaDB.")
    except Exception as e:
        print(f"Ingest failed for {pdf_path}: {e}")


@app.get("/api/categories")
def list_categories():
    return {"categories": PDF_CATEGORIES}


@app.post("/api/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form(...),
):
    if category not in PDF_CATEGORIES:
        raise HTTPException(
            400, f"Invalid category. Allowed: {PDF_CATEGORIES}")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "A PDF file is required")

    content = await file.read()
    if not content.startswith(b"%PDF"):
        raise HTTPException(400, "File must be a PDF")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    category_dir = UPLOADS_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}_{file.filename}" if file.filename else f"{uuid.uuid4().hex}.pdf"
    dest = category_dir / safe_name
    dest.write_bytes(content)

    background_tasks.add_task(run_ingest, str(
        dest), category, file.filename or safe_name)
    return {
        "ok": True,
        "message": "File uploaded and ingestion started.",
        "category": category,
        "filename": file.filename,
    }
