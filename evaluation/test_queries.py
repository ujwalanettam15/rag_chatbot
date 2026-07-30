"""
Generate predictions for ad-hoc test queries without a golden dataset.
Useful for eyeballing answers before formal evaluation.

Queries are written against the current 4-document corpus:
  17361-77195-1-PB.pdf   — Gilpin, "Reasonableness Monitors" (MIT CSAIL)
  CSE_142_HW (1).pdf     — Naive Bayes / Decision Tree on mushroom + voting data
  Fire_Propagation_...   — slope-based fire spread modelling
  resume_masters (17).pdf

Each entry carries the behaviour it should exhibit, so a run can be skimmed for
regressions. The out-of-scope probes are deliberately *topically adjacent* —
terms verified absent from the store (gpu/cuda/gpa: 0 occurrences) — so they test
whether the grounding prompt resists confabulating plausible-sounding detail,
rather than being trivially unrelated to the corpus.

Usage:
  python evaluation/test_queries.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

# (query, expected behaviour)
QUERIES = [
    # --- single-document factual retrieval ---
    (
        "What knowledge base does the reasonableness monitor use, and what is it used for?",
        "cite ConceptNet 5 from the Gilpin paper",
    ),
    (
        "Which two datasets were used to evaluate the Naive Bayes and Decision Tree "
        "classifiers, and how large is each one?",
        "cite mushroom + voting-records sizes from CSE_142_HW",
    ),
    (
        "How does terrain slope affect fire propagation probability?",
        "cite the slope/elevation methodology from Fire_Propagation",
    ),
    # --- the retrieval miss identified earlier: resume is only 8 of 32 chunks ---
    (
        "What degree programs and universities appear in the resume?",
        "cite resume_masters — previously lost to higher-scoring academic chunks",
    ),
    # --- cross-document synthesis ---
    (
        "Which documents discuss checking or monitoring whether a machine "
        "learning system behaves correctly?",
        "span the Gilpin paper AND CSE_142_HW without conflating them",
    ),
    # --- out-of-scope: topically adjacent, terms verified absent from the store ---
    (
        "Which GPU or CUDA hardware was used to train the classifiers?",
        "REFUSE — no gpu/cuda anywhere in the corpus",
    ),
    (
        "What is the candidate's GPA?",
        "REFUSE — plausible for a resume, but no gpa in the corpus",
    ),
]


def run():
    # Deliberately get_chain(), not ConversationalRAGChain: these queries are
    # independent, so history condensing would (a) double the LLM calls — one
    # condense + one answer per query — and (b) rewrite each question against
    # unrelated prior turns, corrupting it. One call per query, no cross-talk.
    from chain import get_chain

    chain = get_chain()

    print("\nTest query run")
    print("=" * 70)
    for i, (q, expected) in enumerate(QUERIES, 1):
        print(f"\n[{i}] Q: {q}")
        print(f"    expect: {expected}")
        try:
            print(f"    A: {chain.invoke(q)}")
        except Exception as exc:
            print(f"    [ERROR] {type(exc).__name__}: {str(exc)[:160]}")
            if "429" in str(exc):
                print("    Daily free-tier quota exhausted — remaining queries skipped.")
                break
        print("-" * 70)


if __name__ == "__main__":
    run()
