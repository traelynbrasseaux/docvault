"""Anthropic API client wrapper with retries and prompt caching."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docvault.config import Config

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds; doubled on each retry


class LLMClient:
    """Thin wrapper around the Anthropic Messages API.

    Handles authentication, retry logic (exponential back-off for rate-limit
    and network errors), and prompt-cache decoration on the system turn.

    The system message is marked with ``cache_control: ephemeral`` so that
    repeated queries against the same system prompt benefit from Anthropic's
    prompt caching, reducing latency and cost on cache hits.

    Args:
        config: :class:`~docvault.config.Config` instance.  The API key is
            retrieved via :meth:`~docvault.config.Config.require_api_key`
            so an informative error is raised if it is missing.

    Attributes:
        model: Claude model ID being used.
    """

    def __init__(self, config: "Config") -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic SDK is required. Install it with: pip install anthropic"
            ) from exc

        import anthropic as _anthropic

        api_key = config.require_api_key()
        self._client = _anthropic.Anthropic(api_key=api_key)
        self._config = config
        self.model: str = config.llm_model

        logger.info("LLMClient ready — model: '%s'.", self.model)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_api(self, system: str, user: str) -> str:
        """Execute a single Messages API call.

        The system prompt is sent with ``cache_control: ephemeral`` so
        Anthropic can cache it across repeated queries that share the same
        system message.

        Args:
            system: System prompt text.
            user: User turn text (includes retrieved context + question).

        Returns:
            The assistant's reply as a plain string.
        """
        import anthropic

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self._config.llm_max_tokens,
            temperature=self._config.llm_temperature,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, system: str, user: str) -> str:
        """Send a prompt to the Claude API and return the generated text.

        Retries up to :data:`_MAX_RETRIES` times on rate-limit errors,
        connection errors, and transient API errors, using exponential
        back-off.  Authentication errors are surfaced immediately without
        retrying.

        Args:
            system: System message (will be prompt-cached).
            user: User message containing retrieved context and the question.

        Returns:
            The model's reply as a plain string.

        Raises:
            anthropic.AuthenticationError: If the API key is invalid.
            anthropic.RateLimitError: If the rate limit is exhausted after all
                retries.
            anthropic.APIError: For other unrecoverable API errors.
        """
        import anthropic

        delay = _RETRY_BASE_DELAY

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.debug(
                    "LLM call attempt %d/%d (model=%s).",
                    attempt,
                    _MAX_RETRIES,
                    self.model,
                )
                text = self._call_api(system, user)
                logger.info(
                    "LLM response received (%d chars).", len(text)
                )
                return text

            except anthropic.AuthenticationError:
                # Never retry auth failures — the key is wrong.
                logger.error(
                    "Anthropic authentication failed. "
                    "Check that ANTHROPIC_API_KEY is correct."
                )
                raise

            except anthropic.RateLimitError as exc:
                if attempt == _MAX_RETRIES:
                    logger.error(
                        "Rate limit exceeded after %d attempt(s).", _MAX_RETRIES
                    )
                    raise
                logger.warning(
                    "Rate limit hit (attempt %d/%d) — retrying in %.1fs.",
                    attempt,
                    _MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)
                delay *= 2

            except anthropic.APIConnectionError as exc:
                if attempt == _MAX_RETRIES:
                    logger.error(
                        "API connection error after %d attempt(s): %s",
                        _MAX_RETRIES,
                        exc,
                    )
                    raise
                logger.warning(
                    "Connection error (attempt %d/%d) — retrying in %.1fs.",
                    attempt,
                    _MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)
                delay *= 2

            except anthropic.APIStatusError as exc:
                # 5xx errors are transient; 4xx (except 429/401) are not.
                if exc.status_code >= 500:
                    if attempt == _MAX_RETRIES:
                        logger.error(
                            "API server error %d after %d attempt(s).",
                            exc.status_code,
                            _MAX_RETRIES,
                        )
                        raise
                    logger.warning(
                        "API server error %d (attempt %d/%d) — retrying in %.1fs.",
                        exc.status_code,
                        attempt,
                        _MAX_RETRIES,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise

        # Should never reach here, but satisfy the type checker
        raise RuntimeError("generate() exhausted retries without returning.")
