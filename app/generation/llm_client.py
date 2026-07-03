"""
Day 2 — Person A
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

    Implementation guide:
    1.  Build the request payload:
            {
                "model": settings.llm_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

    2.  POST to f"{settings.llm_base_url}/chat/completions"
        with header "Authorization: Bearer {settings.llm_api_key}".
        Use httpx.AsyncClient with a generous timeout (timeout=120.0).

    3.  On success (status 200):
            data = response.json()
            return data["choices"][0]["message"]["content"]

    4.  On 429 (rate limit) or 5xx:
            sleep BASE_DELAY * (2 ** attempt) seconds, then retry.
            Raise RuntimeError after MAX_RETRIES attempts.

    5.  On 4xx other than 429 (bad request, auth error):
            Raise immediately with the response body so the caller can log it.

    Example with httpx:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, headers=headers, json=payload)
    """
    raise NotImplementedError
