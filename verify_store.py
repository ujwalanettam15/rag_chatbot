"""
Verify ChromaDB store after ingestion.

Prints:
  - Total chunk count
  - 3 random chunk samples with metadata
  - A test similarity search
"""

import random

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma

from config import CHROMA_PATH, EMBEDDING_MODEL, EMBEDDING_PROVIDER

load_dotenv()


def _build_embeddings():
    if EMBEDDING_PROVIDER == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def verify():
    print(f"Connecting to ChromaDB at {CHROMA_PATH} ...\n")
    embeddings = _build_embeddings()
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    collection = vectorstore._collection

    # Total count
    total = collection.count()
    print(f"Total chunks stored: {total}\n")

    if total == 0:
        print("Store is empty — run ingest.py first.")
        return

    # 3 random samples
    all_data = collection.get(include=["documents", "metadatas"])
    sample_size = min(3, total)
    indices = random.sample(range(total), sample_size)

    print(f"--- {sample_size} random chunk samples ---")
    for idx in indices:
        meta = all_data["metadatas"][idx]
        text = all_data["documents"][idx]
        print(f"\n[Chunk #{meta.get('chunk_index', idx)}]")
        print(f"  Source     : {meta.get('source', 'unknown')}")
        print(f"  Page       : {meta.get('page', 'N/A')}")
        print(f"  Size       : {meta.get('chunk_size', len(text))} chars")
        print(f"  Ingested   : {meta.get('ingested_at', 'N/A')}")
        print(f"  Preview    : {text[:200].strip()}...")

    # Similarity search smoke test
    query = "what is this document about?"
    print(f"\n--- Similarity search: '{query}' ---")
    results = vectorstore.similarity_search(query, k=3)
    for i, doc in enumerate(results, 1):
        print(f"\n[Result {i}]")
        print(f"  Source  : {doc.metadata.get('source', 'unknown')}")
        print(f"  Page    : {doc.metadata.get('page', 'N/A')}")
        print(f"  Preview : {doc.page_content[:300].strip()}...")

    print("\nVerification complete — retrieval is working.")


if __name__ == "__main__":
    verify()
