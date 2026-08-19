from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from novel_agent.contracts import ProviderOutcome, ProviderRequest
from novel_agent.provider import (
    CancellationToken,
    LegacyDeepSeekTransport,
    ProviderClient,
    ProviderDependencies,
    ProviderFailureKind,
    ProviderPolicy,
    ProviderTransport,
    ProviderTransportError,
    ProviderTransportResult,
)
from provider_test_support import (
    FIXED_TIME,
    PROMPT,
    PROMPT_HASH,
    CancellationAfterResponseTransport,
    CancellationSleeper,
    RecordingSleeper,
    SequenceMonotonic,
    SequenceTransport,
    completion,
    provider_client,
    provider_request,
    run_crash_after_response_scenario,
)


def _request() -> ProviderRequest:
    return provider_request()


def _completion() -> ProviderTransportResult:
    return completion()


def _client(
    transport: ProviderTransport,
    sleep: Callable[[float], None] | None = None,
) -> ProviderClient:
    return provider_client(transport, sleep)


def test_transient_failure_retries_then_succeeds() -> None:
    # Given: a transient provider failure followed by a valid structured response.
    transport = SequenceTransport(
        (
            ProviderTransportError(kind=ProviderFailureKind.TRANSIENT, detail="temporary upstream failure"),
            _completion(),
        )
    )
    sleeper = RecordingSleeper()
    client = _client(transport, sleeper)

    # When: the governed provider client generates one task result.
    result = client.generate(_request(), CancellationToken())

    # Then: it returns a typed success after the configured first retry delay.
    assert result.response.outcome == ProviderOutcome.SUCCEEDED
    assert result.content == "A safe, deterministic response."
    assert result.response.prompt_tokens == 11
    assert result.response.completion_tokens == 13
    assert result.response.response_sha256 == "3ba6a0f421b3061b281c6f39e5a4d03568495d43b7365f9af19dcf56b0406a7a"
    assert result.audit.request_sha256 == PROMPT_HASH
    assert len(result.attempts) == 2
    assert transport.calls == 2
    assert sleeper.delays == [1.0]
    assert [attempt.attempt_number for attempt in result.attempts] == [1, 2]
    assert {attempt.request_sha256 for attempt in result.attempts} == {PROMPT_HASH}
    assert {attempt.provider_call_id for attempt in result.attempts} == {"provider-call-001"}
    assert {attempt.adapter_version for attempt in result.attempts} == {"legacy-deepseek-v1"}


def test_default_policy_is_bounded_and_auditable() -> None:
    # Given: the provider boundary's default policy.
    policy = ProviderPolicy()

    # When: its external-call limits are inspected.
    limits = (
        policy.connect_timeout_seconds,
        policy.read_timeout_seconds,
        policy.logical_deadline_seconds,
        policy.maximum_attempts,
        policy.retry_delays_seconds,
    )

    # Then: the approved timeout, attempt, and backoff values are exact.
    assert limits == (10.0, 180.0, 600.0, 3, (1.0, 2.0))


def test_transient_failures_stop_after_three_attempts() -> None:
    # Given: three transient failures and a success that must remain unreachable.
    transport = SequenceTransport(
        (
            ProviderTransportError(kind=ProviderFailureKind.TRANSIENT, detail="first"),
            ProviderTransportError(kind=ProviderFailureKind.TRANSIENT, detail="second"),
            ProviderTransportError(kind=ProviderFailureKind.TRANSIENT, detail="third"),
            _completion(),
        )
    )
    sleeper = RecordingSleeper()
    client = _client(transport, sleeper)

    # When: the governed provider client exhausts its retry policy.
    result = client.generate(_request(), CancellationToken())

    # Then: it stops at three calls with the ordered 1s/2s backoff.
    assert result.response.outcome == ProviderOutcome.FAILED
    assert transport.calls == 3
    assert sleeper.delays == [1.0, 2.0]


def test_cancellation_before_request_discards_without_transport_call() -> None:
    # Given: a cancellation token already requested before the external boundary.
    transport = SequenceTransport((_completion(),))
    client = _client(transport)
    cancellation = CancellationToken()
    cancellation.cancel()

    # When: a caller attempts generation for the cancelled task.
    result = client.generate(_request(), cancellation)

    # Then: it is discarded before the transport can perform a side effect.
    assert result.response.outcome == ProviderOutcome.DISCARDED
    assert result.content is None
    assert transport.calls == 0


def test_cancellation_after_response_discards_late_result() -> None:
    # Given: a transport that receives a response after cancellation has been requested.
    cancellation = CancellationToken()
    transport = CancellationAfterResponseTransport(cancellation)
    client = _client(transport)

    # When: the provider boundary returns after the cancellation point.
    result = client.generate(_request(), cancellation)

    # Then: the late response cannot become a successful result for a coordinator to commit.
    assert result.response.outcome == ProviderOutcome.DISCARDED
    assert result.content is None
    assert len(result.attempts) == 1
    assert result.attempts[0].outcome == ProviderOutcome.DISCARDED


def test_provider_failure_is_typed_and_redacted() -> None:
    # Given: a configuration failure whose detail contains secret-like text.
    secret = "deepseek-key-value"
    transport = SequenceTransport(
        (ProviderTransportError(kind=ProviderFailureKind.CONFIGURATION, detail=secret),)
    )
    client = _client(transport)

    # When: the boundary maps the failure into its audited result.
    result = client.generate(_request(), CancellationToken())

    # Then: its category survives but the original error detail does not.
    assert result.response.outcome == ProviderOutcome.FAILED
    assert result.failure_kind == ProviderFailureKind.CONFIGURATION
    assert secret not in result.failure_summary


def test_legacy_deepseek_transport_parses_usage_without_governed_write() -> None:
    # Given: a compatible legacy callable that returns the existing token metadata shape.
    def legacy_call(
        text: str,
        *,
        max_retry: int,
        model_name: str,
        return_tokens: bool,
        caller: str,
    ) -> tuple[str, dict[str, int | str]]:
        assert text == PROMPT
        assert max_retry == 1
        assert model_name == "fake-model"
        assert return_tokens
        assert caller == "novel_agent"
        return "Legacy response.", {"model": "fake-model", "prompt_tokens": 5, "completion_tokens": 8}

    transport = LegacyDeepSeekTransport(call=legacy_call)

    # When: the governed boundary adapts the legacy result.
    result = transport.generate(_request())

    # Then: token data is typed without an artifact, memory, or run-state side effect.
    assert result.content == "Legacy response."
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 8


def test_legacy_deepseek_transport_accepts_real_access_deepseek_usage_payload() -> None:
    # Given: the exact usage payload ``access_deepseek(return_tokens=True)`` emits —
    # including ``total_tokens``, which previously failed the strict boundary and
    # made every real governed provider call degrade.
    def legacy_call(
        text: str,
        *,
        max_retry: int,
        model_name: str,
        return_tokens: bool,
        caller: str,
    ) -> tuple[str, dict[str, int | str]]:
        return "Real reply.", {
            "model": "deepseek-v4-flash",
            "prompt_tokens": 12,
            "completion_tokens": 34,
            "total_tokens": 46,
        }

    transport = LegacyDeepSeekTransport(call=legacy_call)

    result = transport.generate(_request())

    assert result.content == "Real reply."
    assert result.model_name == "deepseek-v4-flash"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 34


def test_legacy_runtime_error_returns_redacted_typed_failure() -> None:
    # Given: a legacy callable that exposes sensitive text through an untyped runtime error.
    secret = "secret-like-legacy-error"

    def legacy_call(
        text: str,
        *,
        max_retry: int,
        model_name: str,
        return_tokens: bool,
        caller: str,
    ) -> tuple[str, dict[str, int | str]]:
        raise RuntimeError(secret)

    client = _client(LegacyDeepSeekTransport(call=legacy_call))

    # When: the governed provider client reaches the untyped legacy boundary.
    result = client.generate(_request(), CancellationToken())

    # Then: the external error becomes a redacted, typed failed result.
    assert result.response.outcome == ProviderOutcome.FAILED
    assert result.failure_kind == ProviderFailureKind.TRANSIENT
    assert result.content is None
    assert secret not in result.failure_summary
    assert secret not in repr(result)


def test_malformed_legacy_output_returns_redacted_typed_failure() -> None:
    # Given: a legacy callback whose usage violates the typed provider boundary.
    def legacy_call(
        text: str,
        *,
        max_retry: int,
        model_name: str,
        return_tokens: bool,
        caller: str,
    ) -> tuple[str, dict[str, int | str]]:
        return "Malformed usage.", {"model": "fake-model", "prompt_tokens": "five", "completion_tokens": 8}

    client = _client(LegacyDeepSeekTransport(call=legacy_call))

    # When: the malformed external output crosses the legacy adapter.
    result = client.generate(_request(), CancellationToken())

    # Then: it is a redacted permanent failure, not an escaped validation error.
    assert result.response.outcome == ProviderOutcome.FAILED
    assert result.failure_kind == ProviderFailureKind.PERMANENT
    assert result.content is None


def test_cancellation_during_retry_delay_discards_without_second_attempt() -> None:
    # Given: a transient error whose retry wait is interrupted by cancellation.
    cancellation = CancellationToken()
    transport = SequenceTransport(
        (
            ProviderTransportError(kind=ProviderFailureKind.TRANSIENT, detail="temporary upstream failure"),
            _completion(),
        )
    )
    sleeper = CancellationSleeper(cancellation)
    client = _client(transport, sleeper)

    # When: the client reaches its first retry delay.
    result = client.generate(_request(), cancellation)

    # Then: cancellation discards the operation before the next external call.
    assert result.response.outcome == ProviderOutcome.DISCARDED
    assert transport.calls == 1
    assert sleeper.delays == [1.0]


def test_logical_deadline_fails_without_transport_call() -> None:
    # Given: an operation whose logical deadline has elapsed before the first attempt.
    transport = SequenceTransport((_completion(),))
    client = ProviderClient(
        dependencies=ProviderDependencies(
            transport=transport,
            now=lambda: FIXED_TIME,
            sleep=RecordingSleeper(),
            monotonic=SequenceMonotonic((0.0, 1.0)),
        ),
        policy=ProviderPolicy(logical_deadline_seconds=0.5),
    )

    # When: the provider client checks its deadline before requesting the transport.
    result = client.generate(_request(), CancellationToken())

    # Then: the attempt is typed as a timeout without an external side effect.
    assert result.response.outcome == ProviderOutcome.FAILED
    assert result.failure_kind == ProviderFailureKind.TIMEOUT
    assert transport.calls == 0


def test_crash_after_provider_response_retries_and_commits_once(tmp_path: Path) -> None:
    run_crash_after_response_scenario(tmp_path)
