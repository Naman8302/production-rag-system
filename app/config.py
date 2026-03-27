import os
from dotenv import load_dotenv

load_dotenv()  # reads from .env on your Mac — never from GitHub

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
CHROMA_PATH    = "chroma_db"
DATA_PATH      = "data"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL      = "llama-3.1-8b-instant"
CHUNK_SIZE     = 512
CHUNK_OVERLAP  = 64
TOP_K_RETRIEVE = 10
TOP_N_RERANK   = 3
