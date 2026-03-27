# Production RAG System

A production-grade Retrieval-Augmented Generation (RAG) system with **Hybrid BM25 + Vector Retrieval**, **Cross-Encoder Reranking**, and **MLflow experiment tracking**.

## Architecture
PDF → Chunking → Embedding → ChromaDBUser Query → BM25 + Vector Retrieval → RRF Fusion → Cross-Encoder Reranking → Groq LLM → Answer


## Tech Stack
- **LangChain** — document loading and orchestration
- **ChromaDB** — persistent vector store
- **HuggingFace Sentence Transformers** — embeddings (MPS accelerated on Apple Silicon)
- **BM25 + RRF** — hybrid sparse-dense retrieval
- **Cross-Encoder** — semantic reranking
- **Groq API (Llama 3)** — LLM inference
- **FastAPI** — REST API backend
- **MLflow** — experiment tracking (latency, retrieval metrics)
- **Docker** — containerised deployment

## Results
- End-to-end query latency: ~1.5–2.5s
- Retrieval pipeline: BM25 (10) + Vector (10) → RRF Fusion → Reranked Top 3

## Setup
```bash
python3.11 -m venv rag-env && source rag-env/bin/activate
export HNSWLIB_NO_NATIVE=1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
