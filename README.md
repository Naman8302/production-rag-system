# 🔍 Production RAG System

A production-grade **Retrieval-Augmented Generation (RAG)** system built with
hybrid BM25 + vector retrieval, cross-encoder reranking, and MLflow experiment
tracking. Upload any PDF and ask natural language questions — powered by
Groq's Llama 3.1 inference API.

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   PDF File  │────▶│   Chunking   │────▶│   ChromaDB      │
│  (Upload)   │     │  512 tokens  │     │  Vector Store   │
└─────────────┘     │  64 overlap  │     └────────┬────────┘
                    └──────────────┘              │
                                                  │ Vector Search (top 10)
┌─────────────┐                                   │
│ User Query  │──────────────────────────────────▶│
└─────────────┘                                   │
       │                                          │
       │ BM25 Sparse Search (top 10)              │
       │                                          ▼
       └──────────────────────────▶ ┌─────────────────────┐
                                    │   RRF Fusion        │
                                    │ (Reciprocal Rank)   │
                                    └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │  Cross-Encoder      │
                                    │  Reranking (top 3)  │
                                    └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │  Groq LLM           │
                                    │  llama-3.1-8b-instant│
                                    └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │  JSON Response      │
                                    │  + Source Chunks    │
                                    │  + Latency (ms)     │
                                    └─────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **API Framework** | FastAPI 0.115 | REST API backend |
| **LLM Inference** | Groq API (llama-3.1-8b-instant) | Fast cloud LLM — 560 tok/sec |
| **Vector Store** | ChromaDB 0.5 | Persistent local vector database |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Document + query embeddings |
| **Sparse Retrieval** | BM25Okapi (rank-bm25) | Keyword-based retrieval |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Merges BM25 + vector results |
| **Reranking** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Semantic reranking of top candidates |
| **Orchestration** | LangChain 0.3 | Document loading + pipeline |
| **Experiment Tracking** | MLflow 2.16 | Latency + retrieval metrics |
| **Containerisation** | Docker (ARM64) | Deployment-ready container |
| **GPU Acceleration** | PyTorch MPS | Apple Silicon M1/M2 native GPU |

---

## 📁 Project Structure

```
production-rag-system/
├── app/
│   ├── __init__.py
│   ├── config.py         # Centralised config + env var loading
│   ├── ingestion.py      # PDF loading, chunking, ChromaDB storage
│   ├── retriever.py      # Hybrid BM25 + vector search + RRF fusion
│   ├── reranker.py       # Cross-encoder reranking
│   ├── generator.py      # Groq LLM answer generation
│   └── main.py           # FastAPI app + endpoints + MLflow logging
├── data/                 # Uploaded PDFs (git-ignored)
├── chroma_db/            # Persisted vector store (git-ignored)
├── requirements.txt
├── Dockerfile
├── .env.example          # Safe template — copy to .env and fill in key
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- macOS with Apple Silicon M1/M2 (or any Linux/Windows machine)
- Python 3.11
- A free [Groq API key](https://console.groq.com)

### 1. Clone the Repository
```bash
git clone https://github.com/Naman8302/production-rag-system.git
cd production-rag-system
```

### 2. Create & Activate Virtual Environment
```bash
python3.11 -m venv rag-env
source rag-env/bin/activate        # Mac/Linux
# rag-env\Scripts\activate         # Windows
```

### 3. Set Up Your API Key (Never Commit This)
```bash
cp .env.example .env
```
Open `.env` and add your real Groq API key:
```
GROQ_API_KEY=your_real_key_here
```

### 4. Install Dependencies
```bash
# Required fix for ChromaDB on Apple Silicon
export HNSWLIB_NO_NATIVE=1

pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the Server
```bash
uvicorn app.main:app --reload --port 8000
```

Server is live at `http://localhost:8000` ✅

---

## 🚀 API Endpoints

### Interactive Docs
Visit `http://localhost:8000/docs` for a full interactive Swagger UI.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/health` | Server status |
| `POST` | `/upload` | Upload a PDF for ingestion |
| `POST` | `/query` | Ask a question about uploaded documents |

### Example: Upload a PDF
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/your/document.pdf"
```

Response:
```json
{
  "status": "success",
  "filename": "document.pdf",
  "chunks_stored": 142
}
```

### Example: Ask a Question
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

Response:
```json
{
  "question": "What is this document about?",
  "answer": "This document covers...",
  "source_chunks": ["...chunk 1...", "...chunk 2...", "...chunk 3..."],
  "latency_ms": 1842.5
}
```

---

## 📊 MLflow Experiment Tracking

Every query automatically logs metrics to MLflow.

```bash
# In a separate terminal tab
source rag-env/bin/activate
mlflow ui --port 5001
```

Visit `http://127.0.0.1:5001` to see:
- `latency_ms` — end-to-end response time per query
- `chunks_retrieved` — number of candidates from hybrid retrieval
- `chunks_after_rerank` — final chunks sent to the LLM
- `question_preview` — first 80 characters of each query

> **Note for Mac users:** If port 5000 is blocked by AirPlay Receiver, use port 5001.

---

## 🐳 Docker Deployment

```bash
# Build (ARM64 for Apple Silicon)
docker build --platform linux/arm64 -t rag-system .

# Run
docker run --platform linux/arm64 \
  -p 8000:8000 \
  --env-file .env \
  rag-system
```

---

## ⚡ Performance (Apple M1 MacBook Pro)

| Stage | Latency |
|---|---|
| Embedding 500 chunks (MPS) | ~5–10 seconds (one-time on ingest) |
| BM25 + Vector retrieval | ~50–150ms |
| Cross-encoder reranking | ~200–400ms |
| Groq LLM inference | ~500ms–1.5s |
| **Full query end-to-end** | **~1–2.5 seconds** |

---

## 🔐 Security

- API keys are stored in `.env` locally — **never committed to GitHub**
- `.env` is listed in `.gitignore` and will never be tracked by Git
- Use `.env.example` as a safe template to share required variable names
- Always run `git status` before pushing to confirm `.env` is not listed

---

## 🐛 Known Issues & Fixes

| Error | Cause | Fix |
|---|---|---|
| `TypeError: proxies argument` | `httpx>=0.28` broke groq | `pip install httpx==0.27.2` |
| `RuntimeError: python-multipart` | Missing file upload dependency | `pip install python-multipart` |
| `model_decommissioned` error | `llama3-8b-8192` was retired | Use `llama-3.1-8b-instant` in config |
| ChromaDB install fails on M1 | hnswlib native build issue | `export HNSWLIB_NO_NATIVE=1` before pip install |
| MLflow page blank | No successful queries logged yet | Make a successful query first, then refresh |

---

## 📈 Resume Bullet

> *"Built a production RAG system with hybrid BM25 + vector retrieval, Reciprocal Rank Fusion, and cross-encoder reranking using LangChain and ChromaDB — served via FastAPI, containerised with Docker, and tracked with MLflow; end-to-end query latency under 2.5 seconds on Apple Silicon"*

---

## 📄 License
MIT
