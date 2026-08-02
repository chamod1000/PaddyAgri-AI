"""
Multi-Agent Paddy Disease Diagnostic and Fertilizer Recommendation System
Step 2: RAG Pipeline Implementation for Sri Lankan Paddy Farming Data

This script:
1. Recursively loads PDF documents from Data/PDF/ directory subfolders.
2. Applies RecursiveCharacterTextSplitter for document chunking.
3. Uses a HuggingFace embedding model (paraphrase-multilingual-MiniLM-L12-v2)
   to support text embeddings.
4. Stores chunks into a persistent vector store (FAISS / Chroma).
5. Provides an evaluation function running 5 sample agricultural queries.
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Path Definitions
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data" / "PDF"
FAISS_INDEX_DIR = BASE_DIR / "faiss_db"

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def load_text_documents(data_path: Path) -> List[Document]:
    """
    Loads text files from the Data/Learned directory for self-updating RAG.
    """
    documents: List[Document] = []
    if not data_path.exists():
        return documents

    from langchain_community.document_loaders import TextLoader
    txt_files = list(data_path.rglob("*.txt"))
    print(f"[INFO] Found {len(txt_files)} learned text files in {data_path}")

    for txt_path in txt_files:
        try:
            loader = TextLoader(str(txt_path), encoding='utf-8')
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["category"] = "Self-Learned"
                doc.metadata["filename"] = txt_path.name
                doc.metadata["source"] = str(txt_path)
            documents.extend(loaded_docs)
        except Exception as e:
            print(f"  -> [ERROR] Failed to load {txt_path.name}: {e}")
    return documents


def load_pdf_documents(data_path: Path) -> List[Document]:
    """
    Recursively scans the Data/PDF directory and loads all PDF files.
    Attaches category metadata based on the subfolder name.
    """
    documents: List[Document] = []
    if not data_path.exists():
        print(f"[WARNING] Data directory does not exist: {data_path}")
        return documents

    pdf_files = list(data_path.rglob("*.pdf"))
    print(f"[INFO] Found {len(pdf_files)} PDF files in {data_path}")

    for pdf_path in pdf_files:
        # Determine logical category folder (relative to Data/PDF)
        relative_path = pdf_path.relative_to(data_path)
        category = relative_path.parts[0] if len(relative_path.parts) > 1 else "General"

        try:
            loader = PyPDFLoader(str(pdf_path))
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["category"] = category
                doc.metadata["filename"] = pdf_path.name
                doc.metadata["source"] = str(pdf_path)
            documents.extend(loaded_docs)
            print(f"  -> Loaded {len(loaded_docs)} pages from [{category}] {pdf_path.name}")
        except Exception as e:
            print(f"  -> [ERROR] Failed to load {pdf_path.name}: {e}")

    print(f"[INFO] Total pages loaded across all PDFs: {len(documents)}")
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits documents into smaller chunks using RecursiveCharacterTextSplitter.
    
    Chunking Strategy Choice:
    - Chunk Size: 1000 characters (~150-200 words), balancing detailed technical context
      (e.g., specific chemical dosages & disease symptoms) with vector space precision.
    - Chunk Overlap: 200 characters to prevent loss of critical contextual information
      split across chunk boundaries.
    - Separators: ["\n\n", "\n", ". ", " ", ""] to prefer natural paragraph/sentence breaks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[INFO] Created {len(chunks)} text chunks from {len(documents)} document pages.")
    return chunks


_CACHED_EMBEDDINGS = None
_CACHED_VECTOR_STORE = None

import threading
_VECTOR_STORE_WRITE_LOCK = threading.Lock()


def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Initializes multilingual HuggingFace embedding model with singleton caching and offline local-first fallback.
    """
    global _CACHED_EMBEDDINGS
    if _CACHED_EMBEDDINGS is not None:
        return _CACHED_EMBEDDINGS

    print(f"[INFO] Loading multilingual embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu', 'local_files_only': True},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception:
        _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _CACHED_EMBEDDINGS


def create_or_load_vector_store(
    chunks: Optional[List[Document]] = None,
    force_rebuild: bool = False
) -> FAISS:
    """
    Creates a new FAISS vector database from chunks or loads an existing local index with singleton caching.
    """
    global _CACHED_VECTOR_STORE
    if not force_rebuild and _CACHED_VECTOR_STORE is not None:
        return _CACHED_VECTOR_STORE

    embeddings = get_embeddings_model()

    if not force_rebuild and FAISS_INDEX_DIR.exists():
        print(f"[INFO] Loading existing FAISS index from {FAISS_INDEX_DIR}...")
        _CACHED_VECTOR_STORE = FAISS.load_local(
            str(FAISS_INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
        return _CACHED_VECTOR_STORE

    if not chunks:
        print("[INFO] FAISS index missing on disk. Auto-loading PDF documents to build vector store...")
        
        pdf_documents = load_pdf_documents(DATA_DIR)
        
        learned_dir = BASE_DIR / "Data" / "Learned"
        if not learned_dir.exists():
            learned_dir.mkdir(parents=True, exist_ok=True)
        text_documents = load_text_documents(learned_dir)
        
        all_documents = pdf_documents + text_documents
        
        if all_documents:
            chunks = chunk_documents(all_documents, chunk_size=1000, chunk_overlap=200)
        else:
            raise ValueError("No documents found in Data/PDF/ or Data/Learned/ to build vector store.")

    print("[INFO] Generating embeddings and building FAISS vector database...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(FAISS_INDEX_DIR))
    print(f"[SUCCESS] FAISS vector store saved to {FAISS_INDEX_DIR}")
    return vector_store


def auto_learn_text(text: str, source_id: str):
    """
    Dynamically injects new confirmed knowledge into the FAISS vector store
    and saves it to the Data/Learned directory for persistence.
    """
    global _CACHED_VECTOR_STORE
    
    # 1. Save persistently to disk
    learned_dir = BASE_DIR / "Data" / "Learned"
    learned_dir.mkdir(parents=True, exist_ok=True)
    file_path = learned_dir / f"{source_id}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    # 2. Inject into live vector store (thread-safe)
    with _VECTOR_STORE_WRITE_LOCK:
        vector_store = create_or_load_vector_store()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        
        metadatas = [{"category": "Self-Learned", "filename": f"{source_id}.txt", "source": "Auto-Generated"} for _ in chunks]
        
        vector_store.add_texts(texts=chunks, metadatas=metadatas)
        vector_store.save_local(str(FAISS_INDEX_DIR))
    print(f"[SUCCESS] Learned new information and injected into FAISS: {source_id}")


def evaluate_retrieval(vector_store: FAISS, sample_queries: Optional[List[str]] = None, top_k: int = 3):
    """
    Runs sample evaluation queries to test semantic search accuracy and source retrieval.
    """
    if sample_queries is None:
        sample_queries = [
            "What are the common symptoms and control measures for Paddy Blast disease?",
            "What is the recommended NPK fertilizer application rate for Yala season paddy cultivation?",
            "Identify symptoms of brown planthopper damage in paddy.",
            "Seed purity regulations for Certified Seed Paddy in Sri Lanka.",
            "What are the quarantine regulations for importing paddy seed and agricultural items into Sri Lanka?"
        ]

    print("\n" + "=" * 80)
    print("                      RAG RETRIEVAL EVALUATION REPORT                       ")
    print("=" * 80)

    for idx, query in enumerate(sample_queries, start=1):
        print(f"\n[QUERY {idx}]: {query}")
        print("-" * 80)

        # Retrieve relevant chunks with distance scores
        results = vector_store.similarity_search_with_score(query, k=top_k)

        if not results:
            print("  No relevant results found.")
            continue

        for rank, (doc, score) in enumerate(results, start=1):
            category = doc.metadata.get("category", "N/A")
            filename = doc.metadata.get("filename", "N/A")
            page = doc.metadata.get("page", 0) + 1
            snippet = doc.page_content.replace("\n", " ")[:200]

            print(f"  Result #{rank} | Distance Score: {score:.4f}")
            print(f"    Source   : [{category}] {filename} (Page {page})")
            print(f"    Snippet  : {snippet}...\n")

    print("=" * 80 + "\n")


def build_and_evaluate_rag(force_rebuild: bool = True):
    """
    Full pipeline execution function.
    """
    print("[STEP 1/4] Loading PDF documents...")
    documents = load_pdf_documents(DATA_DIR)

    if not documents:
        print("[WARNING] No PDF documents found. Please place PDFs in Data/PDF/ subdirectories.")
        return None

    print("\n[STEP 2/4] Chunking documents...")
    chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)

    print("\n[STEP 3/4] Indexing vector database...")
    vector_store = create_or_load_vector_store(chunks, force_rebuild=force_rebuild)

    print("\n[STEP 4/4] Running retrieval evaluation...")
    evaluate_retrieval(vector_store)

    return vector_store


if __name__ == "__main__":
    build_and_evaluate_rag(force_rebuild=True)
