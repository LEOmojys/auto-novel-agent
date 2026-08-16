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


os.makedirs(os.path.dirname(TOKEN_LOG_PATH), exist_ok=True)


API_POOL = {
    1: os.environ.get("DEEPSEEK_API_KEY_1", ""),
}

_FALLBACK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if _FALLBACK_KEY:
    for key in API_POOL:
        if not API_POOL[key]:
            API_POOL[key] = _FALLBACK_KEY

API_POOL = {key: value for key, value in API_POOL.items() if value}


def _get_api_key():
    if not API_POOL:
        raise RuntimeError(
            "未找到 DeepSeek API Key。请设置环境变量 DEEPSEEK_API_KEY "
            "或 DEEPSEEK_API_KEY_1"
        )
    keys = list(API_POOL.values())
    chosen = keys[hash(str(time.time())) % len(keys)]
    return chosen


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


def access_deepseek(text, max_retry=3, model_name="deepseek-v4-pro",
                    return_tokens=False, caller=""):
    fallback_models = ["deepseek-v4-flash"]

    if model_name not in fallback_models:
        fallback_models.insert(0, model_name)

    last_error = None
    caller_label = caller or "unknown"

    for model in fallback_models:
        for attempt in range(max_retry):
            apikey = _get_api_key()
            try:
                url = "https://api.deepseek.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {apikey}",
                    "Content-Type": "application/json",
                }
                data = {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": text}
                    ],
                    "temperature": 0.7,
                }

                response = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=300,
                )
                response.raise_for_status()

                response_json = response.json()
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

                if return_tokens:
                    return response_txt, token_info
                return response_txt

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