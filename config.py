from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM provider: "openai" | "anthropic" | "ollama" | "openrouter"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))

# OpenRouter — OpenAI-compatible /chat/completions, no embeddings endpoint.
# Keep EMBEDDING_PROVIDER=local when using this provider.
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")

# Reasoning trace visibility. NOTE: on models where reasoning is mandatory
# (e.g. openai/gpt-oss-20b:free) it cannot be disabled — sending
# {"enabled": false} returns HTTP 400. false therefore means "exclude the trace
# from the response", not "skip the reasoning tokens". They are still generated.
ENABLE_REASONING = os.getenv("ENABLE_REASONING", "true").lower() == "true"

# Optional OpenRouter leaderboard attribution headers
OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER", "")
OPENROUTER_TITLE   = os.getenv("OPENROUTER_TITLE", "rag-chatbot")
CHROMA_PATH = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

# Embedding provider: "openai" | "openai-large" | "local"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")

_EMBEDDING_CONFIGS = {
    "openai":       {"model": "text-embedding-3-small", "dims": 1536},
    "openai-large": {"model": "text-embedding-3-large", "dims": 3072},
    "local":        {"model": "BAAI/bge-small-en-v1.5",  "dims": 384},
}

EMBEDDING_MODEL = _EMBEDDING_CONFIGS[EMBEDDING_PROVIDER]["model"]
EMBEDDING_DIMS  = _EMBEDDING_CONFIGS[EMBEDDING_PROVIDER]["dims"]

# Retrieval settings
RETRIEVAL_STRATEGY = os.getenv("RETRIEVAL_STRATEGY", "mmr")
SCORE_THRESHOLD    = float(os.getenv("SCORE_THRESHOLD", "0.7"))
MMR_FETCH_K        = int(os.getenv("MMR_FETCH_K", "20"))
MMR_LAMBDA         = float(os.getenv("MMR_LAMBDA", "0.7"))

# Reranker settings
USE_RERANKER   = os.getenv("USE_RERANKER", "false").lower() == "true"
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
