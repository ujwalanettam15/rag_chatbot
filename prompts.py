"""
Prompt templates for the RAG chatbot — Phase 4.

RAG_PROMPT       — main QA prompt; answers only from context, cites sources.
CONDENSE_PROMPT  — rewrites follow-up questions as standalone queries.
CITATIONS_PROMPT — variant of RAG_PROMPT with a formatted "Sources:" section.
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Main RAG prompt
# ---------------------------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise document assistant. Answer the user's \
question using ONLY the information in the context below.

Rules:
- If the answer is in the context, answer directly and cite the source \
filename in parentheses, e.g. (report.pdf, p.3).
- If the context doesn't contain enough information, say exactly: \
"I don't have enough information to answer that based on the provided documents."
- Never use prior knowledge. Never guess. Never extrapolate.
- Keep answers concise and factual.

Context:
{context}"""),
    ("human", "{question}"),
])

# ---------------------------------------------------------------------------
# Condense prompt — multi-turn conversation
# ---------------------------------------------------------------------------

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Given a conversation history and a follow-up question, \
rewrite the follow-up as a standalone question that captures all necessary \
context from the history. Output only the rewritten question, nothing else."""),
    ("human", """Chat history:
{chat_history}

Follow-up question: {question}
Standalone question:"""),
])

# ---------------------------------------------------------------------------
# Citations prompt — structured Sources section
# ---------------------------------------------------------------------------

CITATIONS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise document assistant. Answer the user's \
question using ONLY the information in the context below.

Rules:
- If the answer is in the context, answer directly and cite the source \
filename inline, e.g. (report.pdf, p.3).
- If the context doesn't contain enough information, say exactly: \
"I don't have enough information to answer that based on the provided documents."
- Never use prior knowledge. Never guess. Never extrapolate.
- Keep answers concise and factual.
- After your answer, add a "Sources:" section listing each source document \
and page number you used, one per line, in the format:
  - filename.pdf, p.N

Context:
{context}"""),
    ("human", "{question}"),
])
