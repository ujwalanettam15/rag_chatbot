"""
Single LLM factory shared by chain.py, evaluation/, and build_golden_dataset.py.

Providers (selected via LLM_PROVIDER):
  "openai"     → ChatOpenAI on api.openai.com, model = LLM_MODEL
  "anthropic"  → ChatAnthropic, model = ANTHROPIC_MODEL
  "ollama"     → ChatOllama (local server), model = LLM_MODEL
  "openrouter" → ChatOpenAI pointed at OpenRouter, model = OPENROUTER_MODEL

OpenRouter exposes an OpenAI-compatible /chat/completions endpoint, so no extra
SDK is needed — only a different base_url and api_key. It does NOT serve an
embeddings endpoint, so EMBEDDING_PROVIDER must stay "local" when using it.

Reasoning models (e.g. openai/gpt-oss-20b:free) accept a "reasoning" body field.
OpenRouter returns the thinking trace in a separate reasoning_details field and
keeps the final answer in message.content, so StrOutputParser still yields a
clean answer. Disable with ENABLE_REASONING=false.
"""

from config import (
    ANTHROPIC_MODEL,
    ENABLE_REASONING,
    LLM_MODEL,
    LLM_PROVIDER,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OPENROUTER_REFERER,
    OPENROUTER_TITLE,
    TEMPERATURE,
)


def _openrouter_headers() -> dict:
    """Optional attribution headers for the OpenRouter leaderboards."""
    headers = {}
    if OPENROUTER_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_REFERER
    if OPENROUTER_TITLE:
        headers["X-Title"] = OPENROUTER_TITLE
    return headers


def build_llm(provider: str = None, model: str = None, temperature: float = None,
              streaming: bool = False):
    """
    Return a LangChain chat model for the configured provider.

    Args are optional overrides; everything falls back to config.py / .env.
    """
    provider    = provider if provider is not None else LLM_PROVIDER
    temperature = temperature if temperature is not None else TEMPERATURE

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        if not OPENROUTER_API_KEY.strip():
            raise ValueError(
                "LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is empty. "
                "Get a key at https://openrouter.ai/keys and add it to .env."
            )

        # Some endpoints (incl. openai/gpt-oss-20b:free) reject {"enabled": False}
        # with "Reasoning is mandatory for this endpoint and cannot be disabled."
        # So ENABLE_REASONING=false does not stop the model reasoning — it only
        # excludes the trace from the response payload. Reasoning always happens
        # and is always billed on such models.
        extra_body = {"reasoning": {"enabled": True}}
        if not ENABLE_REASONING:
            extra_body["reasoning"]["exclude"] = True

        return ChatOpenAI(
            model=model or OPENROUTER_MODEL,
            temperature=temperature,
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            default_headers=_openrouter_headers() or None,
            extra_body=extra_body,
            streaming=streaming,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or ANTHROPIC_MODEL,
            temperature=temperature,
            streaming=streaming,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model or LLM_MODEL, temperature=temperature)

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model or LLM_MODEL,
        temperature=temperature,
        streaming=streaming,
    )


def model_label(provider: str = None) -> str:
    """Human-readable 'provider/model' string for console banners."""
    provider = provider if provider is not None else LLM_PROVIDER
    model = {
        "openrouter": OPENROUTER_MODEL,
        "anthropic":  ANTHROPIC_MODEL,
    }.get(provider, LLM_MODEL)
    return f"{provider}/{model}"
