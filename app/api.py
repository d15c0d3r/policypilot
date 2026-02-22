import uuid
import json
import asyncio
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessageChunk

from app.data.ingest import (
    PDF_CATEGORIES,
    UPLOADS_DIR,
    ingest_single_pdf,
)
from app.agent.graph import build_graph


app = FastAPI(title="PolicyPilot API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
                   "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()


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


# ── WebSocket chat with streaming ──

@app.websocket("/ws/chat")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        while True:
            data = await ws.receive_text()
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                payload = {"message": data}

            user_text = payload.get("message", "").strip()
            if not user_text:
                continue

            await ws.send_text(json.dumps({"type": "start"}))

            try:
                final_answer = ""
                async for event in graph.astream_events(
                    {"messages": [HumanMessage(content=user_text)]},
                    config=config,
                    version="v2",
                ):
                    kind = event.get("event")
                    if kind == "on_chat_model_stream":
                        node = event.get("metadata", {}).get("langgraph_node", "")
                        if node == "supervisor":
                            continue
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            raw = chunk.content
                            if isinstance(raw, list):
                                token = "".join(
                                    part.get("text", "") if isinstance(part, dict) else str(part)
                                    for part in raw
                                )
                            else:
                                token = str(raw)
                            if token:
                                final_answer += token
                                await ws.send_text(json.dumps({"type": "token", "content": token}))

                if not final_answer:
                    state = graph.get_state(config)
                    msgs = state.values.get("messages", [])
                    if msgs:
                        raw = msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])
                        if isinstance(raw, list):
                            final_answer = "".join(
                                part.get("text", "") if isinstance(part, dict) else str(part)
                                for part in raw
                            )
                        else:
                            final_answer = str(raw)
                    if not final_answer:
                        final_answer = "I couldn't generate a response. Please try again."
                    await ws.send_text(json.dumps({"type": "token", "content": final_answer}))

                await ws.send_text(json.dumps({"type": "end"}))

            except Exception as e:
                await ws.send_text(json.dumps({"type": "error", "content": str(e)}))

    except WebSocketDisconnect:
        pass
