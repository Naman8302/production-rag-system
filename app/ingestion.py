import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import CHROMA_PATH, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Ingestion] Using device: {device}")

def load_and_chunk(pdf_path: str):
    loader    = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter  = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"[Ingestion] Created {len(chunks)} chunks")
    return chunks

def embed_and_store(chunks):
    embeddings  = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device}
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    vectorstore.persist()
    print(f"[Ingestion] Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore

def ingest_pdf(pdf_path: str):
    chunks = load_and_chunk(pdf_path)
    embed_and_store(chunks)
    return {"chunks_stored": len(chunks)}
