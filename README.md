# PolicyPilot

Policy PDF upload and Q&A over policy documents (React + FastAPI + ChromaDB).

## Backend (FastAPI)

```bash
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.api:app --reload --port 8000
```

- **Upload**: `POST /api/upload` with `file` (PDF) and `category` (e.g. `health_insurance`, `car_insurance`, `term_insurance`). Files are stored under `app/data/uploads/{category}/` and ingested into ChromaDB in the background.
- **Categories**: `GET /api/categories` returns allowed PDF types.

## Frontend (React)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Use the form to pick a PDF type and upload a file; ingestion runs asynchronously after upload.

## CLI (existing chat)

```bash
source .venv/bin/activate
python -m app.main
```

Uses only PDFs from `app/data/uploads/{category}/`. Populate by uploading via the React app, or run `python -m app.data.ingest` to ingest all PDFs already in `uploads/`.
