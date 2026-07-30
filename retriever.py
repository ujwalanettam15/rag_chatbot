"""
RAG retriever — Phase 3.

Supports three retrieval strategies (set via RETRIEVAL_STRATEGY env var):
  "similarity" — cosine similarity, top-k chunks
  "mmr"        — Maximal Marginal Relevance (default)
  "hybrid"     — BM25 + vector via EnsembleRetriever

Score threshold filtering drops chunks below SCORE_THRESHOLD;
falls back to top-2 if too few pass.

Optional Cohere reranking when USE_RERANKER=true and COHERE_API_KEY is set.

Usage:
  python retriever.py
  RETRIEVAL_STRATEGY=similarity python retriever.py
  RETRIEVAL_STRATEGY=hybrid python retriever.py
"""

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma

from config import (
    CHROMA_PATH,
    COHERE_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    MMR_FETCH_K,
    MMR_LAMBDA,
    RETRIEVAL_STRATEGY,
    SCORE_THRESHOLD,
    TOP_K,
    USE_RERANKER,
)

load_dotenv()


# ---------------------------------------------------------------------------
# Embedding factory (mirrors ingest.py)
# ---------------------------------------------------------------------------

def _build_embeddings():
    if EMBEDDING_PROVIDER == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Score threshold filter
# ---------------------------------------------------------------------------

def _apply_score_threshold(docs_and_scores: list[tuple], threshold: float, min_results: int = 2):
    """
    Filter (doc, score) pairs below threshold.
    Falls back to top min_results if too few pass.
    Returns list of docs only.
    """
    passing = [(doc, score) for doc, score in docs_and_scores if score >= threshold]

    if len(passing) < min_results:
        print(
            f"  [WARNING] Only {len(passing)} chunk(s) above score threshold "
            f"{threshold:.2f}. Falling back to top {min_results}."
        )
        passing = docs_and_scores[:min_results]

    return [doc for doc, _ in passing]


# ---------------------------------------------------------------------------
# Retriever factory
# ---------------------------------------------------------------------------

def build_retriever(strategy: str = RETRIEVAL_STRATEGY, chunks=None):
    """
    Build and return a LangChain retriever.

    Args:
        strategy: "similarity" | "mmr" | "hybrid"
        chunks:   list of Document objects required for hybrid BM25 retriever.
                  If None and strategy is "hybrid", chunks are loaded from the
                  ChromaDB collection.

    Returns a retriever with optional Cohere reranking applied.
    """
    embeddings = _build_embeddings()

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    if strategy == "similarity":
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K},
        )

    elif strategy == "mmr":
        base_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": TOP_K,
                "fetch_k": MMR_FETCH_K,
                "lambda_mult": MMR_LAMBDA,
            },
        )

    elif strategy == "hybrid":
        from langchain_community.retrievers import BM25Retriever

        from utils.fusion import WeightedRRFRetriever

        # Need raw documents for BM25
        if chunks is None:
            collection = vectorstore._collection
            results = collection.get(include=["documents", "metadatas"])
            from langchain_core.documents import Document
            chunks = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(results["documents"], results["metadatas"])
            ]

        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = TOP_K

        vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": TOP_K}
        )

        base_retriever = WeightedRRFRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6],
            top_k=TOP_K,
        )

    else:
        raise ValueError(f"Unknown retrieval strategy: '{strategy}'. Choose similarity | mmr | hybrid.")

    # Optional Cohere reranking
    if USE_RERANKER and COHERE_API_KEY:
        from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
        from langchain_cohere import CohereRerank

        compressor = CohereRerank(
            model="rerank-english-v3.0",
            top_n=TOP_K,
            cohere_api_key=COHERE_API_KEY,
        )
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )

    return base_retriever


# ---------------------------------------------------------------------------
# Retrieve with score threshold applied
# ---------------------------------------------------------------------------

def retrieve(query: str, strategy: str = RETRIEVAL_STRATEGY, chunks=None) -> list:
    """
    Run retrieval for query, apply score threshold filtering.

    For hybrid strategy, threshold is skipped (EnsembleRetriever doesn't
    expose per-chunk scores in a comparable format).

    Returns list of Document objects.
    """
    embeddings = _build_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    if strategy == "hybrid" or USE_RERANKER:
        # EnsembleRetriever and compression retrievers don't expose raw scores
        retriever = build_retriever(strategy=strategy, chunks=chunks)
        return retriever.invoke(query)

    # For similarity / mmr: use similarity_search_with_score for threshold filtering
    if strategy == "mmr":
        # MMR doesn't expose scores via similarity_search_with_score; use retriever directly
        retriever = build_retriever(strategy=strategy)
        return retriever.invoke(query)

    # similarity — can apply threshold
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=TOP_K)
    # Chroma returns L2 distance (lower = better); convert to cosine-like score
    # by treating distance as 1 - distance for filtering (works for normalized embeddings)
    threshold_adapted = [(doc, 1 - score) for doc, score in docs_and_scores]
    return _apply_score_threshold(threshold_adapted, SCORE_THRESHOLD)


# ---------------------------------------------------------------------------
# A/B retriever comparison (Step 10 — diagnostic)
# ---------------------------------------------------------------------------

def _keyword_recall(docs: list, ground_truth: str) -> float:
    """
    Lightweight proxy for context recall: fraction of ground-truth keywords
    found in the combined retrieved text.  No LLM call needed.
    """
    import re
    gt_words = set(re.findall(r"\b[a-z]{4,}\b", ground_truth.lower()))
    if not gt_words:
        return 0.0
    combined = " ".join(d.page_content for d in docs).lower()
    found = sum(1 for w in gt_words if w in combined)
    return round(found / len(gt_words), 3)


def ab_compare(queries_and_truths: list[tuple[str, str]] | None = None) -> None:
    """
    Side-by-side A/B comparison:
      A — current MMR retriever (settings from config)
      B — hybrid BM25+vector with k=7

    Prints keyword-overlap recall estimates for each query under both
    strategies so you can decide whether hybrid improves retrieval before
    running a full RAGAs evaluation.

    Args:
        queries_and_truths: list of (query, ground_truth) tuples.
                            If None, loads the first 5 entries from
                            evaluation/golden_dataset.json, falling back
                            to built-in sample queries when the file is
                            empty or missing.
    """
    import json

    # Load queries from golden dataset if not provided
    if queries_and_truths is None:
        golden_path = Path(__file__).parent / "evaluation" / "golden_dataset.json"
        queries_and_truths = []
        if golden_path.exists():
            try:
                with open(golden_path) as f:
                    entries = json.load(f)
                entries = [e for e in entries if "REPLACE" not in e.get("question", "")]
                queries_and_truths = [
                    (e["question"], e["ground_truth"])
                    for e in entries[:5]
                ]
            except Exception:
                pass

        if not queries_and_truths:
            # Fallback sample queries (no ground truth → recall shows as N/A)
            queries_and_truths = [
                ("What are the main skills mentioned?", ""),
                ("What projects or experience are described?", ""),
                ("What education background does the candidate have?", ""),
                ("What tools or technologies are listed?", ""),
                ("What are the candidate's career goals?", ""),
            ]

    print("\nA/B Retriever Comparison")
    print("  A: MMR          (strategy=mmr, k={TOP_K})".format(TOP_K=TOP_K))
    print("  B: Hybrid       (strategy=hybrid, k=7)")
    print("  Metric: keyword-overlap recall (proxy — no LLM)\n")
    print(f"{'Query':<45} {'A (MMR)':>10} {'B (Hybrid)':>12} {'Winner':>8}")
    print("-" * 80)

    a_scores, b_scores = [], []
    have_ground_truth = any(gt for _, gt in queries_and_truths)

    for query, ground_truth in queries_and_truths:
        try:
            docs_a = build_retriever(strategy="mmr").invoke(query)
        except Exception as exc:
            print(f"  [MMR ERROR] {exc}")
            docs_a = []

        try:
            # Build hybrid retriever with k=7
            embeddings = _build_embeddings()
            from langchain_chroma import Chroma as _Chroma
            vs = _Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            collection = vs._collection
            results = collection.get(include=["documents", "metadatas"])
            from langchain_core.documents import Document as _Doc
            all_chunks = [
                _Doc(page_content=t, metadata=m)
                for t, m in zip(results["documents"], results["metadatas"])
            ]
            from langchain_community.retrievers import BM25Retriever

            from utils.fusion import WeightedRRFRetriever
            bm25 = BM25Retriever.from_documents(all_chunks)
            bm25.k = 7
            vec = vs.as_retriever(search_kwargs={"k": 7})
            hybrid = WeightedRRFRetriever(
                retrievers=[bm25, vec], weights=[0.4, 0.6], top_k=7
            )
            docs_b = hybrid.invoke(query)
        except Exception as exc:
            print(f"  [HYBRID ERROR] {exc}")
            docs_b = []

        if ground_truth:
            score_a = _keyword_recall(docs_a, ground_truth)
            score_b = _keyword_recall(docs_b, ground_truth)
            winner = "B" if score_b > score_a else ("A" if score_a > score_b else "TIE")
        else:
            score_a = len(docs_a)
            score_b = len(docs_b)
            winner = f"B({score_b})" if score_b > score_a else f"A({score_a})"

        a_scores.append(score_a)
        b_scores.append(score_b)
        label = query[:44]
        print(f"  {label:<44} {score_a:>10.3f} {score_b:>12.3f} {winner:>8}")

    if a_scores and have_ground_truth:
        avg_a = sum(a_scores) / len(a_scores)
        avg_b = sum(b_scores) / len(b_scores)
        print("-" * 80)
        print(f"  {'AVERAGE':<44} {avg_a:>10.3f} {avg_b:>12.3f}")
        winner = "Hybrid" if avg_b > avg_a else ("MMR" if avg_a > avg_b else "TIE")
        print(f"\n  Recommendation: use {winner} retriever")
        if winner == "Hybrid":
            print("  → Set RETRIEVAL_STRATEGY=hybrid in .env")
        elif winner == "MMR":
            print("  → Current MMR strategy is already optimal")


# ---------------------------------------------------------------------------
# Test / demo
# ---------------------------------------------------------------------------

def _print_results(label: str, docs: list):
    print(f"\n{'='*60}")
    print(f"Strategy: {label}  ({len(docs)} result(s))")
    print("=" * 60)
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source = Path(meta.get("source", "unknown")).name
        page   = meta.get("page", "?")
        print(f"\n[{i}] source={source}  page={page}")
        print(f"    {doc.page_content[:200].strip()} ...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ab", action="store_true", help="Run A/B retriever comparison")
    args = parser.parse_args()

    if args.ab:
        ab_compare()
    else:
        query = "What are the main skills and projects in this document?"
        print(f"\nQuery: {query}")
        print(f"TOP_K={TOP_K}  SCORE_THRESHOLD={SCORE_THRESHOLD}  USE_RERANKER={USE_RERANKER}\n")
        for strat in ["similarity", "mmr", "hybrid"]:
            try:
                docs = retrieve(query, strategy=strat)
                _print_results(strat, docs)
            except Exception as exc:
                print(f"\n[{strat}] ERROR: {exc}")
