import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from sentence_transformers import CrossEncoder
from app.config import RERANK_MODEL, TOP_N_RERANK

reranker = CrossEncoder(RERANK_MODEL)

def rerank(query: str, docs: list, top_n: int = TOP_N_RERANK):
    if not docs:
        return []
    pairs  = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top    = [doc for _, doc in ranked[:top_n]]
    print(f"[Reranker] Top {len(top)} chunks selected")
    return top
