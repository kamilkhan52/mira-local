import pytest
import requests

import mira.graph_ingest as gi


class _Resp:
    def __init__(self, status_code=200):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class _Session:
    """Fake session whose .post yields a scripted sequence of outcomes."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def post(self, url, json=None, timeout=None):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(gi.time, "sleep", lambda *_: None)


def test_retries_then_succeeds():
    session = _Session([requests.Timeout(), requests.ConnectionError(), _Resp(200)])
    resp = gi._post_with_retry(session, "http://x", {})
    assert resp.status_code == 200
    assert session.calls == 3


def test_raises_after_exhausting_retries():
    session = _Session([requests.Timeout()] * gi._POST_RETRIES)
    with pytest.raises(RuntimeError, match="failed after"):
        gi._post_with_retry(session, "http://x", {})
    assert session.calls == gi._POST_RETRIES


def test_does_not_retry_http_errors():
    # A 400/500 response is returned, not raised here — callers decide. No retry.
    session = _Session([_Resp(400)])
    resp = gi._post_with_retry(session, "http://x", {})
    assert resp.status_code == 400
    assert session.calls == 1


def test_post_entity_treats_400_as_already_exists():
    session = _Session([_Resp(400)])
    assert gi._post_entity(session, {
        "entity_name": "X", "description": "d", "entity_type": "Topic", "source_id": "s",
    }, "http://x/graph/entity/create") is False


def test_post_entity_timeout_recovers_as_already_exists():
    # Timeout on first attempt, server actually applied it -> retry sees 400.
    session = _Session([requests.Timeout(), _Resp(400)])
    assert gi._post_entity(session, {
        "entity_name": "X", "description": "d", "entity_type": "Topic", "source_id": "s",
    }, "http://x/graph/entity/create") is False
    assert session.calls == 2
