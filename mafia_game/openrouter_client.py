"""Async HTTP client for the OpenRouter API."""

import asyncio
import json
import logging

import aiohttp

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when the model has hit its daily free-tier rate limit (HTTP 429)."""


class OpenRouterClient:
    """Minimal async wrapper around OpenRouter's chat completions endpoint."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        referer: str,
        app_title: str,
         timeout: float,
        max_retries: int,
        reasoning_enabled: bool = True,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": referer,
            "X-OpenRouter-Title": app_title,
            "Content-Type": "application/json",
        }
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.reasoning_enabled = reasoning_enabled
        self.max_tokens = max_tokens

    async def __aenter__(self) -> "OpenRouterClient":
        self._session = aiohttp.ClientSession(
            headers=self.headers, timeout=self.timeout
        )
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> dict:
        """Send a chat completion request.

        Retries on transient failures (timeout, 5xx, 429) with exponential
        backoff of 1s, 2s, 4s.  Returns the full assistant message dict
        (including ``reasoning_details`` if the model produced any).

        Raises:
            RuntimeError: if all retries are exhausted.
        """
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers=self.headers, timeout=self.timeout
            )
            try:
                return await self._do_chat(model, messages, temperature)
            finally:
                await self._session.close()
                self._session = None

        return await self._do_chat(model, messages, temperature)

    async def _do_chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
    ) -> dict:
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_enabled:
            payload["reasoning"] = {"enabled": True}
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                async with self._session.post(
                    f"{self.BASE_URL}/chat/completions",
                    data=json.dumps(payload),
                ) as resp:
                    if resp.status == 429:
                        text = await resp.text()
                        raise RateLimitError(
                            f"Rate limit exceeded for {model}: {text[:300]}"
                        )
                    elif resp.status >= 500:
                        text = await resp.text()
                        last_exc = aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status, message=text
                        )
                        logger.warning(
                            "Attempt %d/%d for %s failed with HTTP %d: %s",
                            attempt + 1, self.max_retries, model, resp.status, text[:200],
                        )
                    else:
                        data = await resp.json()
                        if not data.get("choices"):
                            raise RuntimeError(
                                f"No choices in response: {data}"
                            )
                        message = data["choices"][0]["message"]
                        if not message.get("content"):
                            raise RuntimeError("Empty content in model response")
                        return message
            except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                last_exc = exc
                logger.warning(
                    "Attempt %d/%d for %s raised %s: %s",
                    attempt + 1, self.max_retries, model, type(exc).__name__, exc,
                )
            except RuntimeError:
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Attempt %d/%d for %s unexpected error: %s",
                    attempt + 1, self.max_retries, model, exc,
                )

            if attempt < self.max_retries - 1:
                backoff = 2 ** attempt
                logger.info("Backing off for %.1fs before retry", backoff)
                await asyncio.sleep(backoff)

        raise RuntimeError(
            f"All {self.max_retries} retries exhausted for model {model}: {last_exc}"
        )

    async def chat_many(
        self, requests: list[tuple[str, list[dict]]]
    ) -> list[dict]:
        """Concurrently send multiple requests. Returns results in same order."""
        tasks = [self.chat(model, messages) for model, messages in requests]
        return await asyncio.gather(*tasks)
