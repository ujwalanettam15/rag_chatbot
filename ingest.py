"""
RAG ingestion pipeline — Phase 2.

Usage:
  python ingest.py                          # ingest all PDFs in ./docs/
  python ingest.py --url <URL>              # ingest a webpage
  python ingest.py --strategy semantic      # use semantic chunking
  python ingest.py --docs ./my_docs --strategy markdown
"""

import argparse
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_chroma import Chroma

from config import (
    CHROMA_PATH,
    EMBEDDING_DIMS,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
)
from utils.chunking import chunk_documents

load_dotenv()


# ---------------------------------------------------------------------------
# Embedding factory
# ---------------------------------------------------------------------------

def _build_embeddings():
    if EMBEDDING_PROVIDER == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_documents(docs_dir: str = "./docs", url: str = None) -> list:
    docs = []

    if url:
        print(f"  Loading URL: {url}")
        loader = WebBaseLoader(url)
        docs.extend(loader.load())

    pdf_paths = list(Path(docs_dir).glob("*.pdf"))
    for pdf_path in pdf_paths:
        print(f"  Loading PDF: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())

    if not docs:
        print(f"  No documents found in '{docs_dir}' and no URL provided.")

    return docs


# ---------------------------------------------------------------------------
# Idempotent store: delete existing chunks from this source before re-adding
# ---------------------------------------------------------------------------

def _delete_existing_source(vectorstore: Chroma, source_names: set[str]) -> int:
    """Remove all chunks whose metadata 'source' matches any of source_names."""
    deleted = 0
    try:
        collection = vectorstore._collection
        results = collection.get(include=["metadatas"])
        ids_to_delete = [
            doc_id
            for doc_id, meta in zip(results["ids"], results["metadatas"])
            if meta.get("source") in source_names
        ]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            deleted = len(ids_to_delete)
    except Exception:
        pass  # Collection may be empty on first run
    return deleted


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def ingest(
    docs_dir: str = "./docs",
    url: str = None,
    strategy: str = "recursive",
) -> Chroma:
    start = time.time()

    print(f"\nEmbedding provider : {EMBEDDING_PROVIDER}")
    print(f"Embedding model    : {EMBEDDING_MODEL}  ({EMBEDDING_DIMS} dims)")
    print(f"Chunking strategy  : {strategy}")
    print(f"ChromaDB path      : {CHROMA_PATH}\n")

    # 1. Load
    print("Loading documents...")
    docs = load_documents(docs_dir, url)
    print(f"  Loaded {len(docs)} page(s)\n")

    if not docs:
        print("Nothing to ingest. Add PDFs to ./docs/ or pass --url.")
        return None

    # 2. Chunk
    print(f"Chunking ({strategy})...")
    chunks = chunk_documents(docs, strategy=strategy)
    avg_size = round(sum(len(c.page_content) for c in chunks) / len(chunks)) if chunks else 0
    print(f"  Created {len(chunks)} chunks  (avg size: {avg_size} chars)\n")

    # 3. Embed + store (idempotent)
    print("Embedding and storing in ChromaDB...")
    embeddings = _build_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    source_names = {Path(d.metadata.get("source", "")).name for d in docs}
    deleted = _delete_existing_source(vectorstore, source_names)
    if deleted:
        print(f"  Removed {deleted} stale chunk(s) for idempotency")

    vectorstore.add_documents(chunks)

    elapsed = round(time.time() - start, 2)
    print(f"\nDone in {elapsed}s — {len(chunks)} chunks stored at {CHROMA_PATH}")
    return vectorstore


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument("--docs",     default="./docs",      help="Directory of PDFs")
    parser.add_argument("--url",      default=None,          help="Webpage URL to ingest")
    parser.add_argument(
        "--strategy",
        default="recursive",
        choices=["recursive", "semantic", "markdown"],
        help="Chunking strategy (default: recursive)",
    )
    args = parser.parse_args()
    ingest(docs_dir=args.docs, url=args.url, strategy=args.strategy)
