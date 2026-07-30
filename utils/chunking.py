"""
Chunking strategies for the RAG ingestion pipeline.

Three strategies:
  recursive  — RecursiveCharacterTextSplitter (default, good for prose)
  semantic   — SemanticChunker (splits on embedding-similarity shifts)
  markdown   — MarkdownHeaderTextSplitter (preserves section structure)
"""

from datetime import datetime, timezone
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, EMBEDDING_PROVIDER


def _enrich_metadata(chunks: list) -> list:
    """Add source, chunk_index, chunk_size, and ingested_at to every chunk."""
    ingested_at = datetime.now(timezone.utc).isoformat()
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata
        # Normalize source to filename only
        if "source" in meta:
            meta["source"] = Path(meta["source"]).name
        meta["chunk_index"] = i
        meta["chunk_size"] = len(chunk.page_content)
        meta["ingested_at"] = ingested_at
    return chunks


def chunk_recursive(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    return _enrich_metadata(chunks)


def chunk_semantic(docs: list) -> list:
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:
        raise ImportError(
            "langchain-experimental is required for semantic chunking. "
            "Run: pip install langchain-experimental"
        )

    if EMBEDDING_PROVIDER == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    else:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    splitter = SemanticChunker(embeddings)
    chunks = splitter.split_documents(docs)
    return _enrich_metadata(chunks)


def chunk_markdown(docs: list) -> list:
    from langchain.text_splitter import MarkdownHeaderTextSplitter

    headers_to_split_on = [
        ("#",   "h1"),
        ("##",  "h2"),
        ("###", "h3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    chunks = []
    for doc in docs:
        # MarkdownHeaderTextSplitter works on raw text strings
        splits = splitter.split_text(doc.page_content)
        for split in splits:
            # Merge parent doc metadata with header metadata
            split.metadata = {**doc.metadata, **split.metadata}
            chunks.append(split)

    return _enrich_metadata(chunks)


STRATEGIES = {
    "recursive": chunk_recursive,
    "semantic":  chunk_semantic,
    "markdown":  chunk_markdown,
}


def chunk_documents(docs: list, strategy: str = "recursive") -> list:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGIES)}")
    return STRATEGIES[strategy](docs)
