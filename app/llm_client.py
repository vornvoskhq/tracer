import asyncio
import os
from typing import Optional

import aiohttp


DEFAULT_MODEL = "mistral/mistral-small-latest"


class OpenRouterClient:
    """
    Minimal async client for OpenRouter.

    Expects OPENROUTER_API_KEY to be set in the environment.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def summarize_function(self, code: str) -> str:
        """
        Generate a concise summary of a Python function.

        This is intentionally simple and low-level; callers are responsible
        for threading / background execution so GUIs remain responsive.
        """
        if not self.api_key:
            return "OpenRouter API key not configured (set OPENROUTER_API_KEY)."

        prompt = (
            "You are an expert Python engineer. Summarize the purpose and behavior "
            "of the following function in concise, technical prose. Focus on:\n"
            "- Overall purpose\n"
            "- Key inputs and outputs\n"
            "- Important side effects (I/O, network, database, etc.)\n"
            "- Non-obvious edge cases or constraints\n\n"
            "Function source:\n"
            "```python\n"
            f"{code}\n"
            "```"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://local-tooling",
            "X-Title": "Execution Trace Viewer",
            "Content-Type": "application/json",
        }

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You summarize Python functions for developers."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=body,
                    headers=headers,
                    timeout=60,
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return f"OpenRouter error {resp.status}: {text[:400]}"
                    data = await resp.json()
            except asyncio.TimeoutError:
                return "OpenRouter request timed out."
            except Exception as exc:
                return f"OpenRouter request failed: {exc}"

        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return "Unexpected response format from OpenRouter."