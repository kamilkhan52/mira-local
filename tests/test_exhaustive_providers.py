import httpx
import pytest

from chatbot.config import Settings
from mira.exhaustive.costs import PricingCatalog
import mira.exhaustive.providers as providers


class _FakeOpenAI:
    def __init__(self, **_kwargs):
        pass


@pytest.fixture(autouse=True)
def offline_pricing(monkeypatch):
    """No unit test may reach the live OpenRouter catalog."""
    def _unavailable(*_args, **_kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(providers.httpx, "get", _unavailable)


def test_build_engine_passes_configured_model_to_hypothesis_runner(
    monkeypatch,
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(providers, "OpenAI", _FakeOpenAI)
    settings = Settings(
        lightrag_api_key="test-key",
        exhaustive_synthesis_model="google/gemini-2.5-flash",
    )

    engine = providers.build_engine(settings)

    assert engine.hypothesis_runner.model == (
        settings.exhaustive_synthesis_model
    )


def test_build_engine_rejects_unpriced_configured_model_before_requests(
    monkeypatch,
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(providers, "OpenAI", _FakeOpenAI)
    settings = Settings(
        lightrag_api_key="test-key",
        exhaustive_map_model="vendor/unpriced-model",
    )

    with pytest.raises(
        RuntimeError,
        match="no price configured for exhaustive model",
    ):
        providers.build_engine(settings)


def test_live_pricing_is_fetched_at_startup(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(providers, "OpenAI", _FakeOpenAI)
    requested = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [
                {
                    "id": model,
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                }
                for model in (
                    providers.EMBEDDING_MODEL,
                    "google/gemini-2.5-flash",
                    "anthropic/claude-sonnet-5",
                )
            ]}

    def _get(url, **kwargs):
        requested.append(url)
        return _Response()

    monkeypatch.setattr(providers.httpx, "get", _get)
    settings = Settings(
        lightrag_api_key="test-key",
        exhaustive_map_model="google/gemini-2.5-flash",
        exhaustive_synthesis_model="anthropic/claude-sonnet-5",
    )

    engine = providers.build_engine(settings)

    assert requested == [f"{providers.OPENROUTER_BASE_URL}/models"]
    assert engine.estimator.catalog.price(
        "anthropic/claude-sonnet-5"
    ).input_per_million == pytest.approx(1.0)


def test_pricing_falls_back_with_a_startup_warning(monkeypatch, caplog):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(providers, "OpenAI", _FakeOpenAI)
    settings = Settings(lightrag_api_key="test-key")

    with caplog.at_level("WARNING", logger=providers.logger.name):
        engine = providers.build_engine(settings)

    assert "fallback pricing" in caplog.text
    fallback = PricingCatalog.fallback()
    assert engine.estimator.catalog.fetched_at == fallback.fetched_at


def test_map_stage_requests_the_validated_json_schema(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    captured = {}

    class _CapturingOpenAI:
        def __init__(self, **_kwargs):
            outer = self

            class _Completions:
                def create(self, **kwargs):
                    captured.update(kwargs)
                    raise RuntimeError("stop after capture")

            class _Chat:
                completions = _Completions()

            self.chat = _Chat()
            del outer

    monkeypatch.setattr(providers, "OpenAI", _CapturingOpenAI)
    engine = providers.build_engine(Settings(lightrag_api_key="test-key"))

    with pytest.raises(RuntimeError, match="stop after capture"):
        engine.compiler.model_call(
            model="google/gemini-2.5-flash",
            system="system",
            user="{}",
            stage="evidence_map",
            max_output_tokens=16_000,
        )

    schema = captured["extra_body"]["response_format"]["json_schema"]["schema"]
    assert captured["temperature"] == 0
    assert set(schema["properties"]) == {"processed_chunk_ids", "records"}
    assert "paper_name" in str(schema)
