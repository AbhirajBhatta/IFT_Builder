"""
LLM API Client
==============
Thin async wrapper around the provider's chat completions endpoint.
Handles retries and exponential back-off so no other file needs to.

Supports any OpenAI-compatible endpoint (OpenAI, Azure OpenAI, local vLLM, etc.)
— just change LLM_BASE_URL and LLM_MODEL in .env.
"""
from __future__ import annotations

import asyncio
import json

import httpx

from app.config import get_settings

settings = get_settings()

MAX_RETRIES = 4
BASE_DELAY  = 2.0   # seconds; doubles on each retry


async def chat_completion(
    messages: list[dict],
    temperature: float = 0.2,   # low temp = more faithful extraction
    max_tokens: int = 2048,
) -> str:
    """
    Call the LLM and return the raw text content of the first choice.

    POSTs to f"{settings.llm_base_url}/chat/completions" with the configured
    model, messages, temperature, and max_tokens. On a 429 or 5xx response,
    retries up to MAX_RETRIES times with exponential back-off
    (BASE_DELAY * 2**attempt seconds), raising RuntimeError if still failing
    after the final attempt. Any other 4xx response raises immediately with
    the response body included, so the caller can log the failure.
    """
    url = f"{settings.llm_base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        for attempt in range(MAX_RETRIES):
            response = await client.post(url, headers=headers, json=payload)

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]

            if response.status_code == 429 or response.status_code >= 500:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(BASE_DELAY * (2 ** attempt))
                    continue
                raise RuntimeError(
                    f"LLM request failed after {MAX_RETRIES} attempts: "
                    f"{response.status_code} {response.text}"
                )

            raise RuntimeError(f"LLM request failed: {response.status_code} {response.text}")

    raise RuntimeError(f"LLM request failed after {MAX_RETRIES} attempts")
