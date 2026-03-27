import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from app.config import CHROMA_PATH, EMBED_MODEL, TOP_K_RETRIEVE

device = "mps" if torch.backends.mps.is_available() else "cpu"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device}
    )
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

def vector_search(vectorstore, query: str, k: int = TOP_K_RETRIEVE):
    results = vectorstore.similarity_search(query, k=k)
    print(f"[Retriever] Vector search: {len(results)} results")
    return results

def bm25_search(all_texts: list, query: str, k: int = TOP_K_RETRIEVE):
    tokenized = [t.lower().split() for t in all_texts]
    bm25      = BM25Okapi(tokenized)
    scores    = bm25.get_scores(query.lower().split())
    top_idx   = np.argsort(scores)[::-1][:k]
    docs      = [Document(page_content=all_texts[i]) for i in top_idx]
    print(f"[Retriever] BM25 search: {len(docs)} results")
    return docs

def reciprocal_rank_fusion(vector_docs, bm25_docs, k: int = 60):
    fused_scores = {}
    doc_map      = {}

    for rank, doc in enumerate(vector_docs):
        key              = doc.page_content[:120]
        doc_map[key]     = doc
        fused_scores[key] = fused_scores.get(key, 0) + 1 / (k + rank + 1)

    for rank, doc in enumerate(bm25_docs):
        key              = doc.page_content[:120]
        doc_map[key]     = doc
        fused_scores[key] = fused_scores.get(key, 0) + 1 / (k + rank + 1)

    sorted_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)
    fused       = [doc_map[key] for key in sorted_keys[:TOP_K_RETRIEVE]]
    print(f"[Retriever] RRF fusion: {len(fused)} candidates")
    return fused

def hybrid_retrieve(query: str):
    vectorstore = load_vectorstore()
    raw_texts   = vectorstore.get()["documents"]

    if not raw_texts:
        print("[Retriever] WARNING: ChromaDB is empty — ingest a PDF first.")
        return []

    vector_results = vector_search(vectorstore, query)
    bm25_results   = bm25_search(raw_texts, query)
    return reciprocal_rank_fusion(vector_results, bm25_results)
