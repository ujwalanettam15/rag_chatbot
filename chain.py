"""
RAG chain — Phase 4.

Implements:
  - get_chain()              LCEL pipe: retriever → prompt → LLM → parser
  - ConversationalRAGChain   multi-turn wrapper with history condensing
  - stream_answer()          synchronous streaming to stdout
  - async_stream()           async generator for FastAPI StreamingResponse

LLM provider is selected via LLM_PROVIDER env var (see utils/llm.py):
  "openai"     → ChatOpenAI  (default, model from config.py)
  "anthropic"  → ChatAnthropic (ANTHROPIC_MODEL)
  "ollama"     → ChatOllama (local server)
  "openrouter" → ChatOpenAI via OpenRouter (OPENROUTER_MODEL)

Usage:
  python chain.py
  LLM_PROVIDER=anthropic python chain.py
  LLM_PROVIDER=openrouter EMBEDDING_PROVIDER=local python chain.py
"""

import asyncio
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from config import LLM_PROVIDER, MAX_HISTORY_TURNS
from prompts import CONDENSE_PROMPT, RAG_PROMPT
from retriever import build_retriever
from utils.llm import build_llm, model_label


# ---------------------------------------------------------------------------
# Document formatter
# ---------------------------------------------------------------------------

def format_docs(docs: list) -> str:
    """Format retrieved chunks with inline source metadata."""
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}, "
        f"p.{doc.metadata.get('page', '?')}]: {doc.page_content}"
        for doc in docs
    )


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def get_llm(streaming: bool = False):
    """Delegate to the shared factory in utils/llm.py."""
    return build_llm(streaming=streaming)


# ---------------------------------------------------------------------------
# Chain factory
# ---------------------------------------------------------------------------

def get_chain():
    """
    Build and return the LCEL RAG chain.

    Flow:
      query
        → RunnableParallel: {context: retriever | format_docs, question: passthrough}
        → RAG_PROMPT
        → LLM
        → StrOutputParser
        → answer string

    LangSmith tracing is enabled automatically when LANGCHAIN_TRACING_V2=true
    and LANGCHAIN_API_KEY are set in .env. Each run is tagged with run_name
    "rag-chatbot" and grouped under LANGCHAIN_PROJECT.
    """
    retriever = build_retriever()
    llm = get_llm()

    rag_chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        })
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    # Tag every invocation for LangSmith observability (no-op when tracing is off)
    return rag_chain.with_config(run_name="rag-chatbot")


# ---------------------------------------------------------------------------
# Conversational wrapper
# ---------------------------------------------------------------------------

class ConversationalRAGChain:
    """
    Multi-turn RAG chain with history condensing.

    Caps history at MAX_HISTORY_TURNS to avoid context overflow.
    Before retrieval, rewrites follow-up questions as standalone queries
    so retrieval always works on a self-contained question.
    """

    def __init__(self):
        self.chain = get_chain()
        self.condense_chain = CONDENSE_PROMPT | get_llm() | StrOutputParser()
        self.history: list[tuple[str, str]] = []

    def invoke(self, question: str) -> str:
        if self.history:
            history_str = "\n".join(
                f"Human: {h}\nAssistant: {a}"
                for h, a in self.history[-MAX_HISTORY_TURNS:]
            )
            standalone = self.condense_chain.invoke({
                "chat_history": history_str,
                "question": question,
            })
        else:
            standalone = question

        answer = self.chain.invoke(standalone)
        self.history.append((question, answer))
        return answer

    def clear_history(self):
        self.history = []


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

def stream_answer(question: str) -> None:
    """Stream answer tokens to stdout in real time."""
    chain = get_chain()
    print("Answer: ", end="", flush=True)
    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)
    print()


async def async_stream(question: str):
    """
    Async generator yielding SSE-formatted chunks.
    Designed for FastAPI's StreamingResponse in Phase 6.
    """
    chain = get_chain()
    async for chunk in chain.astream(question):
        yield f"data: {chunk}\n\n"


# ---------------------------------------------------------------------------
# Test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_queries = [
        "What is the main topic of the documents?",
        "Who are the key people or authors mentioned?",
        "What conclusions or recommendations are made?",
    ]

    print(f"\nRAG Chain demo  ({model_label()})\n")
    print("=" * 60)

    conv = ConversationalRAGChain()

    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        answer = conv.invoke(query)
        print(f"Answer: {answer}")
        print("-" * 60)
