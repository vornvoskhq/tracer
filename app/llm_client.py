import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp


# Default model used when nothing is configured via environment or settings.
DEFAULT_MODEL = "openai/gpt-4o-mini"


# Very rough price table (USD per 1M tokens) for a few common models.
# This is only used for CLI-side cost estimation and is not guaranteed to be
# accurate for your specific OpenRouter account or pricing tier.
PRICE_TABLE_USD_PER_1M: Dict[str, Dict[str, float]] = {
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
    "Function source or trace context:\n"
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


def _log_console_run(
    *,
    model: str,
    preset_id: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
    prompt_tokens: int,
    completion_tokens: int,
    estimated_cost: Optional[float],
    duration_s: Optional[float] = None,
) -> None:
    """
    Emit a compact, single-line summary of an LLM call to the console.
    """
    preset_display = preset_id or "-"
    max_tok_display = str(max_tokens) if isinstance(max_tokens, int) and max_tokens > 0 else "-"
    if estimated_cost is None:
        cost_display = "NA"
    else:
        cost_display = f"${estimated_cost:.6f}"
    if isinstance(duration_s, (int, float)):
        dur_display = f"{duration_s:.2f}s"
    else:
        dur_display = "-"

    print(
        "LLM | "
        f"model={model} | "
        f"preset={preset_display} | "
        f"temp={temperature:.2f} | "
        f"max_tok={max_tok_display} | "
        f"in={prompt_tokens} | "
        f"out={completion_tokens} | "
        f"cost={cost_display} | "
        f"dur={dur_display}"
    )


def _log_file_run(
    *,
    model: str,
    preset_id: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
    prompt_tokens: int,
    completion_tokens: int,
    estimated_cost: Optional[float],
    duration_s: Optional[float] = None,
    prompt: Optional[str] = None,
    response: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a detailed JSON record of an LLM call to logs/llm_runs.jsonl.

    This runs quietly; failures are ignored rather than surfacing in the UI.
    """
    try:
        root = Path(__file__).resolve().parent.parent
        log_dir = root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "llm_runs.jsonl"

        # Shorter timestamp (no microseconds) to keep lines readable.
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # Round cost to a reasonable precision so the value is stable and compact.
        if isinstance(estimated_cost, (int, float)):
            est_cost_val: Optional[float] = round(float(estimated_cost), 6)
        else:
            est_cost_val = None

        record: Dict[str, Any] = {
            "timestamp": ts,
            "model": model,
            "preset": preset_id,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens) if isinstance(max_tokens, int) else None,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "estimated_cost": est_cost_val,
        }

        if isinstance(duration_s, (int, float)):
            record["duration_s"] = round(float(duration_s), 3)

        if prompt is not None:
            record["prompt"] = prompt
        if response is not None:
            record["response"] = response

        if meta:
            # Merge meta fields, without overwriting core keys.
            for key, value in meta.items():
                if key not in record:
                    record[key] = value

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        # Logging must never break the UI.
        pass


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

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def summarize_function(
        self,
        code: str,
        preset_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a concise summary of a Python function or execution path.

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

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You summarize Python functions and execution traces for developers."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None and self.max_tokens > 0:
            body["max_tokens"] = int(self.max_tokens)

        start_ts = time.time()
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
                        # Include a small hint for debugging model-specific issues.
                        return f"OpenRouter error {resp.status} for model {self.model}: {text[:400]}"
                    data = await resp.json()
            except asyncio.TimeoutError:
                return "OpenRouter request timed out."
            except Exception as exc:
                return f"OpenRouter request failed: {exc}"

        duration_s = time.time() - start_ts

        # Extract response text, if available, before logging.
        result_text: Optional[str] = None
        try:
            result_text = data["choices"][0]["message"]["content"]
        except Exception:
            result_text = None

        # Best-effort cost estimation and logging based on token usage (if available)
        try:
            usage = data.get("usage") or {}
            in_tokens = int(usage.get("prompt_tokens", 0) or 0)
            out_tokens = int(usage.get("completion_tokens", 0) or 0)

            price_info = PRICE_TABLE_USD_PER_1M.get(self.model)
            if price_info:
                in_cost = (in_tokens / 1_000_000.0) * price_info["input"]
                out_cost = (out_tokens / 1_000_000.0) * price_info["output"]
                total_cost = in_cost + out_cost
                estimated_cost: Optional[float] = total_cost
            else:
                estimated_cost = None

            _log_console_run(
                model=self.model,
                preset_id=preset_id,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                prompt_tokens=in_tokens,
                completion_tokens=out_tokens,
                estimated_cost=estimated_cost,
                duration_s=duration_s,
            )
            _log_file_run(
                model=self.model,
                preset_id=preset_id,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                prompt_tokens=in_tokens,
                completion_tokens=out_tokens,
                estimated_cost=estimated_cost,
                duration_s=duration_s,
                prompt=prompt,
                response=(result_text or ""),
                meta=meta,
            )
        except Exception:
            # Logging and cost estimation are best-effort only and should never break the UI.
            pass

        if result_text is not None:
            try:
                return result_text.strip()
            except Exception:
                return str(result_text)
        else:
            return "Unexpected response format from OpenRouter."