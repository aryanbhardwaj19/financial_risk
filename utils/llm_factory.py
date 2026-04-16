"""
LLM Factory — returns the correct LangChain chat-model instance
based on ``settings.LLM_PROVIDER``.

Supported providers
-------------------
- ``"groq"``   → ``ChatGroq`` (fast inference, generous free tier)
- ``"gemini"`` → ``ChatGoogleGenerativeAI``
- ``"openai"`` → ``ChatOpenAI``

Rate-limit handling
-------------------
Every returned LLM is wrapped in a retry proxy that catches 429 /
quota-exceeded errors and retries with exponential back-off.

Usage
-----
    from utils.llm_factory import get_llm
    llm = get_llm(temperature=0.0)
"""

from __future__ import annotations

import time
from typing import Any

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Retry tunables ──────────────────────────────────────────────────
_MAX_RETRIES = 5
_INITIAL_BACKOFF_S = 5.0


class _RetryLLMWrapper:
    """Transparent proxy around a LangChain chat-model that retries on
    rate-limit (429) errors with exponential back-off.

    Delegates every attribute access to the wrapped model so it behaves
    identically.
    """

    def __init__(self, llm: Any) -> None:
        # Store on the instance dict directly to avoid __getattr__ loops.
        object.__setattr__(self, "_llm", llm)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_llm"), name)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Call the underlying LLM's ``.invoke()`` with retry logic."""
        llm = object.__getattribute__(self, "_llm")
        backoff = _INITIAL_BACKOFF_S

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return llm.invoke(*args, **kwargs)
            except Exception as exc:
                exc_str = str(exc)
                is_rate_limit = (
                    "429" in exc_str
                    or "quota" in exc_str.lower()
                    or "rate" in exc_str.lower()
                )

                if is_rate_limit and attempt < _MAX_RETRIES:
                    logger.warning(
                        "LLM rate-limited (attempt %d/%d). "
                        "Retrying in %.0fs … [%s]",
                        attempt,
                        _MAX_RETRIES,
                        backoff,
                        exc_str[:150],
                    )
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise

        raise RuntimeError("Exhausted all LLM retry attempts")


def get_llm(temperature: float = 0.0):
    """Return a LangChain chat-model for the configured provider.

    The returned object is wrapped with automatic retry-on-429 logic.

    Parameters
    ----------
    temperature :
        Sampling temperature (0.0 = deterministic).

    Returns
    -------
    BaseChatModel
        A LangChain chat-model instance ready for ``.invoke()``.
    """
    provider = settings.LLM_PROVIDER.lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        logger.info(
            "Using Groq  |  model=%s", settings.LLM_MODEL
        )
        llm = ChatGroq(
            model=settings.LLM_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=temperature,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.info(
            "Using Google Gemini  |  model=%s", settings.LLM_MODEL
        )
        llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature,
            convert_system_message_to_human=True,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        logger.info(
            "Using OpenAI  |  model=%s", settings.LLM_MODEL
        )
        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Set LLM_PROVIDER to 'gemini', 'groq', or 'openai' in your .env file."
        )

    return _RetryLLMWrapper(llm)

