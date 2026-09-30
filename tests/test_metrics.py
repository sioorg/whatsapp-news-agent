"""app.metrics: the small helpers, independent of any HTTP route (see
test_api.py's "Metrics endpoint" section for the GET /metrics contract)."""

import pytest

from app import metrics


def test_hash_user_is_short_and_deterministic():
    a = metrics.hash_user("919902245562")
    b = metrics.hash_user("919902245562")

    assert a == b
    assert len(a) == 8


def test_hash_user_does_not_leak_the_raw_identifier():
    sender = "919902245562"

    assert sender not in metrics.hash_user(sender)


def test_hash_user_differs_between_senders():
    assert metrics.hash_user("111") != metrics.hash_user("222")


def test_track_api_call_records_success():
    with metrics.track_api_call("test_metrics_success"):
        pass

    assert metrics.API_CALLS.labels(api="test_metrics_success", status="success")._value.get() == 1


def test_track_api_call_records_error_and_reraises():
    with pytest.raises(RuntimeError):
        with metrics.track_api_call("test_metrics_error"):
            raise RuntimeError("boom")

    assert metrics.API_CALLS.labels(api="test_metrics_error", status="error")._value.get() == 1


def test_timed_generator_yields_every_item():
    items = metrics.timed_generator("test_channel", iter(["a", "b", "c"]))

    assert list(items) == ["a", "b", "c"]


def test_timed_generator_records_duration_on_exhaustion():
    before = metrics.TURN_DURATION.labels(channel="test_channel_2")._sum.get()

    list(metrics.timed_generator("test_channel_2", iter(["x"])))

    after = metrics.TURN_DURATION.labels(channel="test_channel_2")._sum.get()
    assert after >= before
