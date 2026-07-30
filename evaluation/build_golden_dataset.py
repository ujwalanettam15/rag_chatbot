"""
Build a golden evaluation dataset from ChromaDB chunks using a local Ollama LLM.

Loads sample chunks from the vector store, generates realistic Q&A pairs,
and writes evaluation/golden_dataset.json for manual review and editing.

Usage:
  # Ollama must be running first: ollama serve
  EMBEDDING_PROVIDER=local LLM_MODEL=llama3.2 python evaluation/build_golden_dataset.py
  EMBEDDING_PROVIDER=local LLM_MODEL=llama3.2 python evaluation/build_golden_dataset.py --chunks 8 --pairs 2
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_PATH, EMBEDDING_MODEL, EMBEDDING_PROVIDER, LLM_MODEL, LLM_PROVIDER

load_dotenv()

OUTPUT_PATH = Path(__file__).parent / "golden_dataset.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_embeddings():
    if EMBEDDING_PROVIDER == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def _build_llm():
    from utils.llm import build_llm
    return build_llm(temperature=0)


def load_chunks(n: int) -> list[Document]:
    """Pull n chunks from ChromaDB."""
    embeddings = _build_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )
    collection = vectorstore._collection
    results = collection.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(results["documents"], results["metadatas"])
    ]
    # Spread sample across the collection
    step = max(1, len(docs) // n)
    return docs[::step][:n]


def generate_pairs(chunk: Document, llm, n_pairs: int) -> list[dict]:
    """Ask the LLM to generate n_pairs Q&A entries from a single chunk."""
    from pathlib import Path as P

    source = P(chunk.metadata.get("source", "unknown")).name
    page = chunk.metadata.get("page", "?")
    text = chunk.page_content.strip()

    prompt = f"""You are building a QA evaluation dataset.

Given the following document excerpt, generate exactly {n_pairs} question-answer pair(s).

Rules:
- Questions must be answerable strictly from the excerpt below — no outside knowledge.
- Answers must be drawn word-for-word or closely paraphrased from the excerpt.
- Make questions realistic: what would a real user ask?
- Mix question types: factual lookups AND inferential questions.
- Output valid JSON only — an array of objects with keys "question" and "ground_truth".
- Do not add any explanation, preamble, or markdown fencing.

Excerpt:
\"\"\"
{text}
\"\"\"

Output (JSON array only):"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Strip markdown fences if the model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    pairs = json.loads(raw)
    return [
        {
            "question": p["question"],
            "ground_truth": p["ground_truth"],
            "source_chunk": text,
            "source_doc": source,
            "source_page": page,
        }
        for p in pairs
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(n_chunks: int = 5, n_pairs: int = 2) -> list[dict]:
    print(f"\nLoading {n_chunks} chunks from ChromaDB...")
    chunks = load_chunks(n_chunks)
    print(f"  Loaded {len(chunks)} chunk(s)\n")

    llm = _build_llm()
    dataset = []

    for i, chunk in enumerate(chunks, 1):
        source = Path(chunk.metadata.get("source", "?")).name
        page = chunk.metadata.get("page", "?")
        print(f"[{i}/{len(chunks)}] Generating {n_pairs} pair(s) from {source} p.{page} ...")
        try:
            pairs = generate_pairs(chunk, llm, n_pairs)
            for p in pairs:
                print(f"  Q: {p['question']}")
                print(f"  A: {p['ground_truth'][:120]}...")
                print()
            dataset.extend(pairs)
        except Exception as exc:
            print(f"  [WARN] Failed to parse LLM output: {exc}\n")

    # Load existing entries (manual ones) and merge
    existing = []
    if OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH) as f:
                existing = json.load(f)
            # Drop the placeholder entry
            existing = [e for e in existing if "REPLACE" not in e.get("question", "")]
        except Exception:
            pass

    merged = existing + dataset
    with open(OUTPUT_PATH, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Saved {len(merged)} total entries to {OUTPUT_PATH}")
    print("\nNext steps:")
    print("  1. Review the file and edit/remove any bad entries")
    print("  2. Add 1-3 out-of-scope questions (answers NOT in docs) to test the fallback")
    print("  3. Run: EMBEDDING_PROVIDER=local LLM_PROVIDER=ollama python evaluation/evaluate.py")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=5, help="Number of chunks to sample")
    parser.add_argument("--pairs",  type=int, default=2, help="Q&A pairs per chunk")
    args = parser.parse_args()
    build(n_chunks=args.chunks, n_pairs=args.pairs)
