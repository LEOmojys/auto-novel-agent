from __future__ import annotations

import json
import time

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

    def iter_content(self, chunk_size: int = 4096):
        body = self.text.encode("utf-8")
        for offset in range(0, len(body), chunk_size):
            yield body[offset : offset + chunk_size]

    def close(self) -> None:
        return None


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

    def fake_post(url, headers=None, data=None, timeout=None, stream=None):
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


def test_responses_api_style_builds_url_and_parses_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM_API_STYLE=responses → POST {base}/responses, body uses input, parses output[].content[].text."""
    from tools_call_deepseek import access_deepseek

    monkeypatch.setenv("LLM_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
    monkeypatch.setenv("LLM_API_KEY", "ark-test")
    monkeypatch.setenv("LLM_MODEL", "glm-4.7")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "glm-4.7")
    monkeypatch.setenv("LLM_API_STYLE", "responses")

    captured: list[dict] = []
    payload = {
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "火山回复"}]}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4},
    }

    def fake_post(url, headers=None, data=None, timeout=None, stream=None):
        captured.append({"url": url, "body": json.loads(data), "auth": headers.get("Authorization")})
        return _FakeResponse(200, payload)

    monkeypatch.setattr("tools_call_deepseek.requests.post", fake_post)

    result = access_deepseek("你好", max_retry=1)

    assert result == "火山回复"
    assert captured[0]["url"] == "https://ark.cn-beijing.volces.com/api/coding/v3/responses"
    assert captured[0]["body"]["model"] == "glm-4.7"
    assert captured[0]["body"]["input"] == [{"role": "user", "content": "你好"}]
    assert "max_output_tokens" in captured[0]["body"]
    assert captured[0]["auth"] == "Bearer ark-test"


def test_chat_style_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default (no LLM_API_STYLE) still uses chat completions."""
    from tools_call_deepseek import access_deepseek

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.delenv("LLM_API_STYLE", raising=False)
    captured: list[dict] = []

    def fake_post(url, headers=None, data=None, timeout=None, stream=None):
        captured.append({"url": url, "body": json.loads(data)})
        return _FakeResponse(200, {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    monkeypatch.setattr("tools_call_deepseek.requests.post", fake_post)

    result = access_deepseek("你好", max_retry=1)

    assert result == "ok"
    assert captured[0]["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert "messages" in captured[0]["body"]


class _HangingResponse(_FakeResponse):
    """A response whose body streams slowly, so a cancel can abort mid-read."""

    def __init__(self, status_code: int, payload: dict | str, token=None, hang_after: int = 1) -> None:
        super().__init__(status_code, payload)
        self._token = token
        self._hang_after = hang_after
        self.closed = False

    def iter_content(self, chunk_size: int = 4096):
        body = self.text.encode("utf-8")
        first = body[: self._hang_after]
        if first:
            yield first
        # Simulate the rest of the body arriving only after the cancel lands:
        while self._token is not None and not self._token.is_cancelled():
            time.sleep(0.01)
        # If cancelled, stop; otherwise yield the remainder.
        if self._token is not None and self._token.is_cancelled():
            self.closed = True
            return
        if len(body) > self._hang_after:
            yield body[self._hang_after :]

    def close(self) -> None:
        self.closed = True


class _Cancellable:
    def __init__(self) -> None:
        self._flag = False

    def cancel(self) -> None:
        self._flag = True

    def is_cancelled(self) -> bool:
        return self._flag


def test_access_deepseek_aborts_inflight_request_on_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    """A user cancel interrupts an in-flight HTTP body read instead of waiting."""
    from tools_call_deepseek import access_deepseek

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "glm-4.7")
    monkeypatch.setenv("LLM_API_STYLE", "chat")
    token = _Cancellable()
    payload = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    hung = _HangingResponse(200, payload, token=token, hang_after=1)

    def fake_post(url, headers=None, data=None, timeout=None, stream=None):
        return hung

    monkeypatch.setattr("tools_call_deepseek.requests.post", fake_post)

    import threading

    def run_call() -> list[object]:
        try:
            result = access_deepseek("你好", max_retry=1, cancellation=token)
            return [result]
        except Exception as error:  # noqa: BROAD_EXCEPT_OK
            return [error]

    out: list[object] = []
    t = threading.Thread(target=lambda: out.extend(run_call()), daemon=True)
    t.start()
    time.sleep(0.15)  # let the stream start and stall
    token.cancel()  # user cancels mid-request
    t.join(2.0)
    assert not t.is_alive(), "access_deepseek did not abort the cancelled request"
    assert hung.closed
