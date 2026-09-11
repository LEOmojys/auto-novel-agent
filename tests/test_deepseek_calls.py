from __future__ import annotations

import json

import pytest

from tools_call_deepseek import (
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    MAX_OUTPUT_TOKENS,
    ProviderCallError,
    access_deepseek,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | str) -> None:
        self.status_code = status_code
        self.text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise _HttpError(self.status_code)

    def json(self) -> dict:
        if isinstance(self._payload, dict):
            return self._payload
        raise ValueError("invalid json")


class _HttpError(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


def _ok_response(content: str = "正常") -> _FakeResponse:
    return _FakeResponse(
        200,
        {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def _error_response(status: int) -> _FakeResponse:
    return _FakeResponse(status, json.dumps({"error": {"message": f"boom {status}"}}))


def _install(monkeypatch: pytest.MonkeyPatch, *responses: _FakeResponse) -> list[dict]:
    captured: list[dict] = []
    queue = list(responses)

    def fake_post(url, headers=None, data=None, timeout=None):
        captured.append(
            {"url": url, "body": json.loads(data), "timeout": timeout, "auth": headers.get("Authorization")}
        )
        return queue.pop(0)

    monkeypatch.setattr("tools_call_deepseek.requests.post", fake_post)
    return captured


def test_request_uses_official_v4_model_and_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    captured = _install(monkeypatch, _ok_response("ok"))

    result = access_deepseek("你好", max_retry=1)

    assert result == "ok"
    body = captured[0]["body"]
    assert body["model"] == DEFAULT_MODEL
    assert body["max_tokens"] == MAX_OUTPUT_TOKENS
    assert captured[0]["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert captured[0]["auth"] == "Bearer sk-test"


def test_non_retryable_401_fails_immediately_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    captured = _install(monkeypatch, _error_response(401))

    with pytest.raises(ProviderCallError) as caught:
        access_deepseek("hi", max_retry=3)

    assert caught.value.status_code == 401
    assert len(captured) == 1  # no retry on auth failures


def test_transient_429_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    captured = _install(monkeypatch, _error_response(429), _ok_response("recovered"))

    result = access_deepseek("hi", max_retry=3)

    assert result == "recovered"
    assert len(captured) == 2  # one retry after the rate limit


def test_pro_model_failure_falls_back_to_flash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    captured = _install(monkeypatch, _error_response(500), _ok_response("flash reply"))

    result = access_deepseek("hi", max_retry=1)

    assert result == "flash reply"
    models = [entry["body"]["model"] for entry in captured]
    assert models == [DEFAULT_MODEL, FALLBACK_MODEL]


def test_missing_key_raises_typed_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY_1", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="未找到 LLM API Key"):
        access_deepseek("hi", max_retry=1)
