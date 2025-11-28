import asyncio
import os
from pathlib import Path
from typing import Optional

import aiohttp


DEFAULT_MODEL = "openai/gpt-4o-mini"


def _load_api_key_from_env_file() -> str:
    """
    Best-effort loader for OPENROUTER_API_KEY from a local .env file.

    This avoids adding a python-dotenv dependency while still supporting the
    common pattern of storing the key in a .env file at the project root.
    """
    candidates = []

    # Current working directory .env
    cwd_env = Path.cwd() / ".env"
    candidates.append(cwd_env)

    # Repository-root .env (one level up from this file's directory)
    try:
        repo_root_env = Path(__file__).resolve().parent.parent / ".env"
        if repo_root_env not in candidates:
            candidates.append(repo_root_env)
    except Exception:
        pass

    for env_path in candidates:
        if not env_path.exists():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "OPENROUTER_API_KEY":
                    return value.strip().strip('"').strip("'")
        except OSError:
            continue

    return ""


class OpenRouterClient:
    """
    Minimal async client for OpenRouter.

    Expects OPENROUTER_API_KEY to be set in the environment or in a .env file.
    """

    def __init__(self, model: Optional[str] = None):
        env_model = os.getenv("OPENROUTER_MODEL", "").strip()
        self.model = model or env_model or DEFAULT_MODEL

        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not self.api_key:
            self.api_key = _load_api_key_from_env_file()

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