import asyncio
import os
from pathlib import Path
from typing import Optional

import aiohttp


# Default model used when nothing is configured via environment or settings.
DEFAULT_MODEL = "openai/gpt-4o-mini"


# Very rough price table (USD per 1M tokens) for a few common models.
# This is only used for CLI-side cost estimation and is not guaranteed to be
# accurate for your specific OpenRouter account or pricing tier.
PRICE_TABLE_USD_PER_1M = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-4o": {"input": 5.00, "output": 15.00},
    # Add more entries here if you regularly use other models and know their pricing.
}


DEFAULT_PROMPT_TEMPLATE = (
    "You are an expert Python engineer. Summarize the purpose and behavior "
    "of the following function in concise, technical prose. Focus on:\n"
    "- Overall purpose\n"
    "- Key inputs and outputs\n"
    "- Important side effects (I/O, network, database, etc.)\n"
    "- Non-obvious edge cases or constraints\n\n"
    "Function source:\n"
    "```python\n"
    "{code}\n"
    "```"
)


def _load_api_key_from_env_file() -> str:
    """
    Best-effort loader for OPENROUTER_API_KEY from a local .env file.

    This avoids adding a python-dotenv dependency while still supporting the
    common pattern of storing the key in a .env file at the project root.
    """
    candidates = []

    # Current working directory .env
    candidates.append(Path.cwd() / ".env")

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

    def __init__(
        self,
        model: Optional[str] = None,
        prompt_template: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.1,
    ):
        env_model = os.getenv("OPENROUTER_MODEL", "").strip()
        self.model = model or env_model or DEFAULT_MODEL

        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.max_tokens = max_tokens
        self.temperature = float(temperature)

        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not self.api_key:
            self.api_key = _load_api_key_from_env_file().strip()
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

        prompt = self.prompt_template.format(code=code)

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
            "temperature": self.temperature,
        }
        if self.max_tokens is not None and self.max_tokens > 0:
            body["max_tokens"] = int(self.max_tokens)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=body,
                    headers=headers,
                    timeout=60,
                ) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        return f"OpenRouter error {resp.status}: {text[:400]}"
                    data = await resp.json()
            except asyncio.TimeoutError:
                return "OpenRouter request timed out."
            except Exception as exc:
                return f"OpenRouter request failed: {exc}"

        # Best-effort cost estimation based on token usage (if available)
        try:
            usage = data.get("usage") or {}
            in_tokens = int(usage.get("prompt_tokens", 0) or 0)
            out_tokens = int(usage.get("completion_tokens", 0) or 0)

            price_info = PRICE_TABLE_USD_PER_1M.get(self.model)
            if price_info:
                in_cost = (in_tokens / 1_000_000.0) * price_info["input"]
                out_cost = (out_tokens / 1_000_000.0) * price_info["output"]
                total_cost = in_cost + out_cost
                print(
                    f"[OpenRouter] model={self.model} "
                    f"prompt_tokens={in_tokens}, completion_tokens={out_tokens}, "
                    f"estimated_cost=${total_cost:.6f} "
                    f"(input=${in_cost:.6f}, output=${out_cost:.6f})"
                )
            else:
                if in_tokens or out_tokens:
                    print(
                        f"[OpenRouter] model={self.model} "
                        f"prompt_tokens={in_tokens}, completion_tokens={out_tokens} "
                        f"(no price table entry; cost not estimated)"
                    )
        except Exception:
            # Cost estimation is best-effort only and should never break the UI.
            pass

        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return "Unexpected response format from OpenRouter."