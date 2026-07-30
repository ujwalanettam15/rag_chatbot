"""
Weighted Reciprocal Rank Fusion retriever.

Why this exists instead of langchain.retrievers.EnsembleRetriever:
the installed dependency pair (langchain 0.3.25 + langchain-core 1.3.2) is
incompatible — core 1.x dropped langchain_core.memory, which langchain 0.3.x's
chains.base still imports. Importing anything under langchain.retrievers
executes its __init__, which pulls in contextual_compression -> chains, so the
whole package raises ModuleNotFoundError. This module depends only on
langchain_core, which works.

Algorithm (same as EnsembleRetriever's default): for each retriever i with
weight w_i, every returned document gets w_i / (c + rank) added to its score,
where rank is 0-based position in that retriever's result list. Documents are
deduplicated by page_content and returned in descending score order.
"""

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class WeightedRRFRetriever(BaseRetriever):
    """Fuse several retrievers' ranked results via weighted RRF."""

    retrievers: list[BaseRetriever]
    weights: list[float]
    c: int = 60          # RRF damping constant; 60 is the conventional default
    top_k: int = 5

    def _fuse(self, ranked_lists: list[list[Document]]) -> list[Document]:
        scores: dict[str, float] = {}
        by_key: dict[str, Document] = {}

        for docs, weight in zip(ranked_lists, self.weights):
            for rank, doc in enumerate(docs):
                key = doc.page_content
                scores[key] = scores.get(key, 0.0) + weight / (self.c + rank)
                by_key.setdefault(key, doc)

        ordered = sorted(scores, key=scores.get, reverse=True)
        return [by_key[k] for k in ordered[: self.top_k]]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> list[Document]:
        return self._fuse([r.invoke(query) for r in self.retrievers])

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> list[Document]:
        return self._fuse([await r.ainvoke(query) for r in self.retrievers])
