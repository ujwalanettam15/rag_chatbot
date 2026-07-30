"""
RAGAs evaluation — Phase 5.

Scores the RAG chain on four metrics:
  faithfulness       — hallucination rate (answer claims supported by context)
  answer_relevancy   — answer stays on topic
  context_recall     — retriever finds all needed chunks
  context_precision  — retriever avoids noise

Also runs a local NLI-based hallucination check via
cross-encoder/nli-deberta-v3-small (no API calls needed).

Usage (Ollama must be running):
  EMBEDDING_PROVIDER=local LLM_PROVIDER=ollama LLM_MODEL=llama3.2 python evaluation/evaluate.py
  EMBEDDING_PROVIDER=local LLM_PROVIDER=ollama LLM_MODEL=llama3.2 python evaluation/evaluate.py --skip-ragas
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

GOLDEN_PATH  = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


# ---------------------------------------------------------------------------
# LLM / embedding factories for RAGAs (must wrap in RAGAs wrappers)
# ---------------------------------------------------------------------------

def _ragas_llm():
    from ragas.llms import LangchainLLMWrapper
    from utils.llm import build_llm
    return LangchainLLMWrapper(build_llm(temperature=0))


def _ragas_embeddings():
    from ragas.embeddings import LangchainEmbeddingsWrapper
    provider = os.getenv("EMBEDDING_PROVIDER", "local")
    if provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        return LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        )
    from langchain_openai import OpenAIEmbeddings
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings())


# ---------------------------------------------------------------------------
# Collect predictions
# ---------------------------------------------------------------------------

def collect_predictions(golden: list[dict]) -> list[dict]:
    """Run each question through the chain and retriever, return rows."""
    from chain import get_chain
    from retriever import build_retriever

    chain     = get_chain()
    retriever = build_retriever()
    rows = []

    print(f"Running {len(golden)} prediction(s)...\n")
    for i, entry in enumerate(golden, 1):
        q = entry["question"]
        print(f"  [{i}/{len(golden)}] {q[:70]}...")

        answer = chain.invoke(q)
        docs   = retriever.invoke(q)
        rows.append({
            "question":     q,
            "answer":       answer,
            "contexts":     [d.page_content for d in docs],
            "ground_truth": entry["ground_truth"],
        })

    return rows


# ---------------------------------------------------------------------------
# RAGAs scoring
# ---------------------------------------------------------------------------

def run_ragas(rows: list[dict]) -> dict:
    """Score with RAGAs 0.2.x API. Returns summary dict and per-row DataFrame."""
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    ragas_llm  = _ragas_llm()
    ragas_emb  = _ragas_embeddings()

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        )
        for r in rows
    ]
    dataset = EvaluationDataset(samples=samples)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ContextRecall(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
    ]

    print("\nScoring with RAGAs (this calls the LLM multiple times)...\n")
    result = evaluate(dataset=dataset, metrics=metrics)
    df = result.to_pandas()
    return df


# ---------------------------------------------------------------------------
# NLI hallucination check (fully local, no API)
# ---------------------------------------------------------------------------

def check_hallucinations(rows: list[dict]) -> list[dict]:
    """
    Check each answer sentence against its retrieved context using
    cross-encoder/nli-deberta-v3-small NLI model.

    Returns flagged entries: {question, sentence, contradiction_score}.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("[SKIP] sentence-transformers not installed — skipping NLI check.")
        return []

    import re

    print("\nRunning local NLI hallucination check...")
    model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    # Label order for this model: contradiction=0, entailment=1, neutral=2
    CONTRADICTION_IDX = 0
    THRESHOLD = 0.7

    flagged = []
    for row in rows:
        context = " ".join(row["contexts"])
        # Split answer into sentences (simple heuristic)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", row["answer"]) if len(s.strip()) > 10]
        for sent in sentences:
            scores = model.predict([(context, sent)], apply_softmax=True)[0]
            contradiction_score = float(scores[CONTRADICTION_IDX])
            if contradiction_score > THRESHOLD:
                flagged.append({
                    "question":           row["question"],
                    "sentence":           sent,
                    "contradiction_score": round(contradiction_score, 3),
                })

    if flagged:
        print(f"\n  Flagged {len(flagged)} potentially hallucinated sentence(s):")
        for f in flagged:
            print(f"    Q: {f['question'][:60]}...")
            print(f"    Sentence: {f['sentence'][:80]}")
            print(f"    Contradiction score: {f['contradiction_score']}\n")
    else:
        print("  No contradictions flagged above threshold.\n")

    return flagged


# ---------------------------------------------------------------------------
# Diagnostic interpretation
# ---------------------------------------------------------------------------

def interpret(summary: dict):
    print("\nDiagnostic interpretation:")
    thresholds = {
        "faithfulness":      (0.85, "Strengthen system prompt: 'answer ONLY from context'. Lower temperature."),
        "answer_relevancy":  (0.80, "Add 'be concise' to prompt. Check if retriever returns off-topic chunks."),
        "context_recall":    (0.70, "Retriever missing chunks. Increase k, switch to hybrid, revisit chunk size."),
        "context_precision": (0.75, "Retriever fetching noise. Lower k, raise score threshold, or use MMR."),
    }
    all_good = True
    for metric, (threshold, fix) in thresholds.items():
        score = summary.get(metric)
        if score is None:
            continue
        status = "OK" if score >= threshold else "LOW"
        if status == "LOW":
            all_good = False
            print(f"  [{status}] {metric}: {score:.3f} (target >= {threshold})")
            print(f"        Fix: {fix}")
        else:
            print(f"  [{status}]  {metric}: {score:.3f}")
    if all_good:
        print("  All metrics above target thresholds.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation(skip_ragas: bool = False):
    # Load golden dataset
    if not GOLDEN_PATH.exists() or GOLDEN_PATH.stat().st_size < 10:
        print(f"ERROR: {GOLDEN_PATH} is empty or missing.")
        print("Run: EMBEDDING_PROVIDER=local LLM_PROVIDER=ollama python evaluation/build_golden_dataset.py")
        sys.exit(1)

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    placeholder = [e for e in golden if "REPLACE" in e.get("question", "")]
    if placeholder:
        print(f"WARNING: {len(placeholder)} placeholder entry(ies) in golden_dataset.json — remove before evaluating.")
        golden = [e for e in golden if "REPLACE" not in e.get("question", "")]

    if not golden:
        print("No valid entries in golden_dataset.json. Add Q&A pairs first.")
        sys.exit(1)

    print(f"\nEvaluating {len(golden)} golden entry(ies)")
    print("=" * 60)

    rows = collect_predictions(golden)

    summary = {}
    df = None

    if not skip_ragas:
        df = run_ragas(rows)

        cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
        available = [c for c in cols if c in df.columns]

        print("\n" + "=" * 60)
        print("RAGAs Evaluation Results")
        print("=" * 60)
        for col in available:
            summary[col] = float(df[col].mean())
            print(f"  {col:<22} {summary[col]:.3f}")

        # Hallucination candidates from RAGAs
        if "faithfulness" in df.columns:
            low = df[df["faithfulness"] < 0.7]
            if not low.empty:
                print(f"\nHallucination candidates ({len(low)} question(s) with faithfulness < 0.7):")
                q_col = "user_input" if "user_input" in df.columns else "question"
                for _, row in low.iterrows():
                    q_label = row.get(q_col, "?")
                    print(f"  - {str(q_label)[:60]}... (faithfulness: {row['faithfulness']:.2f})")

        interpret(summary)

    # NLI hallucination check (always runs, fully local)
    flagged = check_hallucinations(rows)

    # Save results
    output = {
        "timestamp":  datetime.now().isoformat(),
        "n_questions": len(golden),
        "summary":    summary,
        "nli_flagged": flagged,
        "per_question": df.to_dict(orient="records") if df is not None else [],
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-ragas", action="store_true",
        help="Skip RAGAs scoring (only run NLI hallucination check)"
    )
    args = parser.parse_args()
    run_evaluation(skip_ragas=args.skip_ragas)
