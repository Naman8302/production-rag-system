from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil, time, os, mlflow

from app.ingestion import ingest_pdf
from app.retriever import hybrid_retrieve
from app.reranker  import rerank
from app.generator import generate_answer
from app.config    import DATA_PATH

app = FastAPI(
    title="Production RAG System",
    description="Hybrid BM25 + Vector Retrieval with Cross-Encoder Reranking",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "RAG System is running!", "status": "healthy"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    os.makedirs(DATA_PATH, exist_ok=True)
    save_path = f"{DATA_PATH}/{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    result = ingest_pdf(save_path)
    return {"status": "success", "filename": file.filename,
            "chunks_stored": result["chunks_stored"]}

@app.post("/query")
async def query_rag(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start      = time.time()
    candidates = hybrid_retrieve(request.question)
    top_docs   = rerank(request.question, candidates)
    result     = generate_answer(request.question, top_docs)
    latency_ms = round((time.time() - start) * 1000, 2)

    mlflow.set_experiment("rag-system")
    with mlflow.start_run():
        mlflow.log_metric("latency_ms",          latency_ms)
        mlflow.log_metric("chunks_retrieved",    len(candidates))
        mlflow.log_metric("chunks_after_rerank", len(top_docs))
        mlflow.log_param("question_preview",     request.question[:80])

    return {
        "question":      request.question,
        "answer":        result["answer"],
        "source_chunks": result["source_chunks"],
        "latency_ms":    latency_ms
    }
