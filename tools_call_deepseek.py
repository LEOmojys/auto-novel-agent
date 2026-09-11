import os
import json
import time
import datetime
import threading
import requests
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
TOKEN_LOG_PATH = os.path.join(BASE_DIR, "logs", "token_usage.jsonl")
_LOG_LOCK = threading.Lock()

# DeepSeek V4 (2026): official models and OpenAI-compatible endpoint.
# base_url https://api.deepseek.com (``/v1`` is an official alias).
#
# All of these can be overridden via environment variables (loaded from the
# local ``.env`` by ``novel_agent.env.load_env``), so any OpenAI-compatible
# endpoint can be used — e.g. Volcano Ark Coding Plan:
#   LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v3
#   LLM_API_KEY=ark-...
#   LLM_MODEL=<volcano model id>
#   LLM_FALLBACK_MODEL=<volcano fallback model id>
API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-v4-pro"
FALLBACK_MODEL = "deepseek-v4-flash"


def get_base_url() -> str:
    """Resolve the OpenAI-compatible base URL (``LLM_BASE_URL`` overrides).

    The value may be a full endpoint (DeepSeek default, already ends in
    ``/chat/completions``) or a base URL (e.g. Volcano Ark Coding Plan
    ``https://ark.cn-beijing.volces.com/api/coding/v3``) to which
    ``/chat/completions`` is appended at call time.
    """
    return os.environ.get("LLM_BASE_URL", "").strip() or API_URL


def _chat_completions_url(base_url: str) -> str:
    """Normalize a base URL into the full chat-completions endpoint."""
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    return f"{stripped}/chat/completions"


def get_model_name() -> str:
    """Resolve the primary model name (``LLM_MODEL`` overrides)."""
    return os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODEL


def get_fallback_model() -> str:
    """Resolve the fallback model name (``LLM_FALLBACK_MODEL`` overrides)."""
    return os.environ.get("LLM_FALLBACK_MODEL", "").strip() or FALLBACK_MODEL


def get_api_style() -> str:
    """Resolve the API protocol style (``LLM_API_STYLE`` overrides).

    ``chat`` (default) uses the OpenAI chat-completions endpoint; ``responses``
    uses the Responses API (``POST {base}/responses``), which is what Volcano
    Ark Coding Plan exposes.
    """
    style = os.environ.get("LLM_API_STYLE", "").strip().lower()
    return style if style in ("chat", "responses") else "chat"


# Official output limit field is ``max_tokens``; without it long Chinese
# chapters (3000+ chars) can be truncated by the default output cap.
# Long creative outputs (10-chapter outlines with detailed goals/events)
# routinely exceed 16k tokens; keep 32k+ of headroom so the model never gets
# truncated mid-JSON.  DeepSeek V4 accepts larger max_tokens.
MAX_OUTPUT_TOKENS = 32768

# These HTTP codes are not retryable: 401 key invalid, 402 no balance,
# 403 region block, 400/422 malformed request.  Only 429 (rate limit) and
# 5xx / transport errors deserve a backoff retry.
_NON_RETRYABLE_STATUS = frozenset((400, 401, 402, 403, 422))


class ProviderCallError(RuntimeError):
    """A non-retryable DeepSeek API failure with the response status exposed."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code


os.makedirs(os.path.dirname(TOKEN_LOG_PATH), exist_ok=True)


def _get_api_key():
    """Resolve an API key from the environment at call time.

    Read live from ``os.environ`` (never a module-import snapshot) so keys
    set after startup — e.g. by the web console or a test — take effect
    immediately.  ``LLM_API_KEY`` is the generic slot (works with any
    OpenAI-compatible endpoint such as Volcano Ark); ``DEEPSEEK_API_KEY_1``
    and ``DEEPSEEK_API_KEY`` are kept for backward compatibility.
    """
    keys = []
    generic = os.environ.get("LLM_API_KEY", "").strip()
    if generic:
        keys.append(generic)
    primary = os.environ.get("DEEPSEEK_API_KEY_1", "").strip()
    if primary and primary not in keys:
        keys.append(primary)
    fallback = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if fallback and fallback not in keys:
        keys.append(fallback)
    if not keys:
        raise RuntimeError(
            "未找到 LLM API Key。请设置环境变量 LLM_API_KEY（或 DEEPSEEK_API_KEY / DEEPSEEK_API_KEY_1）"
        )
    return keys[hash(str(time.time())) % len(keys)]


def _log_tokens(model: str, prompt_tokens: int, completion_tokens: int, caller: str):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "caller": caller,
    }
    with _LOG_LOCK:
        with open(TOKEN_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _build_request(style: str, base_url: str, model: str, text: str) -> tuple[str, dict]:
    """Build (url, body) for the configured API style."""
    if style == "responses":
        url = f"{base_url.rstrip('/')}/responses"
        body = {
            "model": model,
            "input": [{"role": "user", "content": text}],
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }
        return url, body
    url = _chat_completions_url(base_url)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0.7,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    return url, body


def _parse_response(style: str, response_json: dict, model: str, caller_label: str) -> tuple[str, dict]:
    """Extract (text, usage) from a chat-completions or Responses payload."""
    if style == "responses":
        output = response_json.get("output") or []
        text_parts: list[str] = []
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message":
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text_parts.append(str(part.get("text", "")))
        response_txt = "".join(text_parts)
    else:
        response_txt = response_json["choices"][0]["message"]["content"]
    usage = response_json.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total = prompt_tokens + completion_tokens

    _log_tokens(model, prompt_tokens, completion_tokens, caller_label)

    token_info = {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total,
    }
    return response_txt, token_info


def access_deepseek(text, max_retry=3, model_name=None,
                    return_tokens=False, caller=""):
    """Call the configured OpenAI-compatible API (chat-completions or Responses).

    Endpoint, API key, model names and protocol style are read from the
    environment at call time (``LLM_BASE_URL`` / ``LLM_API_KEY`` /
    ``LLM_MODEL`` / ``LLM_FALLBACK_MODEL`` / ``LLM_API_STYLE``; DeepSeek chat
    defaults when unset).  ``LLM_API_STYLE=responses`` selects the Responses
    API (``POST {base}/responses``) used by Volcano Ark Coding Plan.

    - ``max_tokens`` / ``max_output_tokens`` is always sent so long outputs
      are not truncated.
    - Non-retryable HTTP codes (401/402/403/400/422) fail immediately with the
      status exposed; only 429/5xx/transport errors retry with backoff.
    - A model failure falls back to the configured fallback model.
    """
    base_url = get_base_url()
    style = get_api_style()
    primary_model = model_name or get_model_name()
    fallback_models = [get_fallback_model()]
    if primary_model not in fallback_models:
        fallback_models.insert(0, primary_model)

    last_error = None
    caller_label = caller or "unknown"

    for model in fallback_models:
        for attempt in range(max_retry):
            apikey = _get_api_key()
            try:
                headers = {
                    "Authorization": f"Bearer {apikey}",
                    "Content-Type": "application/json",
                }
                url, data = _build_request(style, base_url, model, text)

                response = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=300,
                )
                if response.status_code in _NON_RETRYABLE_STATUS:
                    raise ProviderCallError(
                        response.status_code,
                        f"LLM API {response.status_code}: {response.text[:300]}",
                    )
                response.raise_for_status()

                response_json = response.json()
                response_txt, token_info = _parse_response(style, response_json, model, caller_label)

                if return_tokens:
                    return response_txt, token_info
                return response_txt

            except ProviderCallError as error:
                # Key/balance/request problems never improve by retrying.
                raise
            except Exception as e:
                last_error = e
                print(f"当前第 {attempt+1} 请求失败, 模型:{model}, 错误: {str(e)}")
                if attempt < max_retry - 1:
                    time.sleep(2 ** attempt)
                continue

    error_msg = (
        f"全部请求失败, 最后一次错误: {str(last_error)}"
    )
    raise Exception(error_msg)


if __name__ == "__main__":
    text = "请用一句话介绍你自己。"
    print(access_deepseek(text))