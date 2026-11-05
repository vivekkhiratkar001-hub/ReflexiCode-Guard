"""Tests for GeminiProvider, Gemini config fields, and CLI provider selection.

All Gemini HTTP requests are mocked.  No real API key is required.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.github.cli import build_parser, build_pipeline
from src.github.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LLM_PROVIDER,
    GitHubConfigError,
    build_config,
)
from src.llm.context import ReviewContextBuilder
from src.llm.gemini_provider import GeminiProvider
from src.llm.mock_provider import MockProvider
from src.llm.ollama_provider import OllamaProvider
from src.llm.pipeline import ReviewPipeline
from src.llm.prompts import ReviewPromptBuilder
from src.llm.schemas import ReviewFinding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gemini_response(text: str, finish_reason: str = "STOP") -> MagicMock:
    """Build a mock requests.Response that looks like a Gemini success."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "candidates": [
            {
                "content": {"parts": [{"text": text}]},
                "finishReason": finish_reason,
            }
        ]
    }
    return resp


def _error_response(status: int, body: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    http_err = requests.exceptions.HTTPError(response=resp)
    resp.raise_for_status.side_effect = http_err
    resp.json.return_value = body or {}
    return resp


BASE_KWARGS = dict(owner="acme", repo="demo", pr_number=3)


# ---------------------------------------------------------------------------
# 1. GeminiProvider initialisation
# ---------------------------------------------------------------------------

class TestGeminiProviderInit(unittest.TestCase):
    def test_default_model_is_gemini_20_flash(self) -> None:
        p = GeminiProvider()
        self.assertEqual(p.model, "gemini-2.0-flash")

    def test_custom_model_is_stored(self) -> None:
        p = GeminiProvider(model="gemini-1.5-pro")
        self.assertEqual(p.model, "gemini-1.5-pro")

    def test_timeout_is_stored(self) -> None:
        p = GeminiProvider(timeout=60)
        self.assertEqual(p.timeout, 60)

    def test_empty_model_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            GeminiProvider(model="")

    def test_whitespace_model_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            GeminiProvider(model="   ")

    def test_model_too_long_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            GeminiProvider(model="x" * 201)

    def test_invalid_timeout_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            GeminiProvider(timeout=0)
        with self.assertRaises(ValueError):
            GeminiProvider(timeout=-5)

    def test_prompt_builder_default_created(self) -> None:
        p = GeminiProvider()
        self.assertIsInstance(p.prompt_builder, ReviewPromptBuilder)

    def test_custom_prompt_builder_is_stored(self) -> None:
        pb = ReviewPromptBuilder()
        p = GeminiProvider(prompt_builder=pb)
        self.assertIs(p.prompt_builder, pb)

    def test_url_construction_without_prefix(self) -> None:
        p = GeminiProvider(model="gemini-2.0-flash")
        self.assertEqual(
            p._endpoint_url(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        )

    def test_url_construction_strips_duplicate_models_prefix(self) -> None:
        p = GeminiProvider(model="models/gemini-2.0-flash")
        self.assertEqual(
            p._endpoint_url(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        )

    def test_url_construction_strips_slashes(self) -> None:
        p = GeminiProvider(model="/models/gemini-2.0-flash/")
        self.assertEqual(
            p._endpoint_url(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        )


# ---------------------------------------------------------------------------
# 2. Missing GEMINI_API_KEY
# ---------------------------------------------------------------------------

class TestGeminiMissingApiKey(unittest.TestCase):
    def test_missing_key_raises_runtime_error(self) -> None:
        p = GeminiProvider()
        with patch.dict(os.environ, {}, clear=True):
            # Ensure GEMINI_API_KEY is not set even if it was in the env.
            os.environ.pop("GEMINI_API_KEY", None)
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                p.review("some prompt")

    def test_empty_key_raises_runtime_error(self) -> None:
        p = GeminiProvider()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "   "}):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                p.review("some prompt")

    def test_api_key_is_not_in_exception_message(self) -> None:
        """The key must never appear in error text."""
        p = GeminiProvider()
        secret = "super-secret-key-12345"
        with patch.dict(os.environ, {"GEMINI_API_KEY": secret}):
            with patch(
                "src.llm.gemini_provider.requests.post",
                side_effect=requests.exceptions.ConnectionError("network down"),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    p.review("prompt")
        self.assertNotIn(secret, str(ctx.exception))


# ---------------------------------------------------------------------------
# 3. Successful Gemini JSON response
# ---------------------------------------------------------------------------

class TestGeminiSuccessfulResponse(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["GEMINI_API_KEY"] = "test-key"

    def tearDown(self) -> None:
        os.environ.pop("GEMINI_API_KEY", None)

    def test_returns_text_from_first_candidate(self) -> None:
        payload = '{"findings": []}'
        with patch(
            "src.llm.gemini_provider.requests.post",
            return_value=_gemini_response(payload),
        ):
            result = GeminiProvider().review("some prompt")
        self.assertEqual(result, payload)

    def test_multipart_text_is_joined(self) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": '{"findings"'}, {"text": ': []}'}]
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        with patch("src.llm.gemini_provider.requests.post", return_value=resp):
            result = GeminiProvider().review("prompt")
        self.assertIn("findings", result)

    def test_logs_completion_outcome(self) -> None:
        payload = '{"findings": []}'
        with patch(
            "src.llm.gemini_provider.requests.post",
            return_value=_gemini_response(payload),
        ):
            with self.assertLogs("src.llm.gemini_provider", level="INFO") as logs:
                GeminiProvider().review("prompt")
        self.assertTrue(any("outcome=completed" in line for line in logs.output))

    def test_api_key_not_in_log_output(self) -> None:
        secret = "must-not-appear-in-logs"
        os.environ["GEMINI_API_KEY"] = secret
        payload = '{"findings": []}'
        with patch(
            "src.llm.gemini_provider.requests.post",
            return_value=_gemini_response(payload),
        ):
            with self.assertLogs("src.llm.gemini_provider", level="INFO") as logs:
                GeminiProvider().review("prompt")
        for line in logs.output:
            self.assertNotIn(secret, line)


# ---------------------------------------------------------------------------
# 4. Empty or malformed Gemini responses
# ---------------------------------------------------------------------------

class TestGeminiMalformedResponses(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["GEMINI_API_KEY"] = "test-key"

    def tearDown(self) -> None:
        os.environ.pop("GEMINI_API_KEY", None)

    def test_empty_candidates_raises_runtime_error(self) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"candidates": []}
        with patch("src.llm.gemini_provider.requests.post", return_value=resp):
            with self.assertRaisesRegex(RuntimeError, "no candidates"):
                GeminiProvider().review("prompt")

    def test_missing_candidates_key_raises_runtime_error(self) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {}
        with patch("src.llm.gemini_provider.requests.post", return_value=resp):
            with self.assertRaisesRegex(RuntimeError, "no candidates"):
                GeminiProvider().review("prompt")

    def test_no_text_parts_raises_runtime_error(self) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]
        }
        with patch("src.llm.gemini_provider.requests.post", return_value=resp):
            with self.assertRaisesRegex(RuntimeError, "no usable text content"):
                GeminiProvider().review("prompt")

    def test_empty_text_raises_runtime_error(self) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": "   "}]}, "finishReason": "STOP"}
            ]
        }
        with patch("src.llm.gemini_provider.requests.post", return_value=resp):
            with self.assertRaisesRegex(RuntimeError, "empty review response"):
                GeminiProvider().review("prompt")

    def test_non_json_body_raises_runtime_error(self) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.side_effect = ValueError("no JSON")
        with patch("src.llm.gemini_provider.requests.post", return_value=resp):
            with self.assertRaisesRegex(RuntimeError, "non-JSON"):
                GeminiProvider().review("prompt")

    def test_safety_finish_reason_raises(self) -> None:
        with patch(
            "src.llm.gemini_provider.requests.post",
            return_value=_gemini_response("text", finish_reason="SAFETY"),
        ):
            with self.assertRaisesRegex(RuntimeError, "SAFETY"):
                GeminiProvider().review("prompt")

    def test_max_tokens_finish_reason_raises(self) -> None:
        with patch(
            "src.llm.gemini_provider.requests.post",
            return_value=_gemini_response("partial", finish_reason="MAX_TOKENS"),
        ):
            with self.assertRaisesRegex(RuntimeError, "truncated"):
                GeminiProvider().review("prompt")

    def test_blocked_prompt_raises_with_reason(self) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "promptFeedback": {"blockReason": "SAFETY"},
            "candidates": [],
        }
        with patch("src.llm.gemini_provider.requests.post", return_value=resp):
            with self.assertRaisesRegex(RuntimeError, "SAFETY"):
                GeminiProvider().review("prompt")


# ---------------------------------------------------------------------------
# 5. HTTP errors and rate limits
# ---------------------------------------------------------------------------

class TestGeminiHttpErrors(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["GEMINI_API_KEY"] = "test-key"

    def tearDown(self) -> None:
        os.environ.pop("GEMINI_API_KEY", None)

    def test_429_raises_rate_limit_error(self) -> None:
        resp = MagicMock()
        resp.status_code = 429
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)
        with patch("src.llm.gemini_provider.requests.post", return_value=resp):
            with self.assertRaisesRegex(RuntimeError, "rate limit"):
                GeminiProvider().review("prompt")

    def test_401_raises_auth_error(self) -> None:
        resp = MagicMock()
        resp.status_code = 401
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)
        with patch("src.llm.gemini_provider.requests.post", return_value=resp):
            with self.assertRaisesRegex(RuntimeError, "401"):
                GeminiProvider().review("prompt")

    def test_403_raises_auth_error(self) -> None:
        resp = MagicMock()
        resp.status_code = 403
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)
        with patch("src.llm.gemini_provider.requests.post", return_value=resp):
            with self.assertRaisesRegex(RuntimeError, "403"):
                GeminiProvider().review("prompt")

    def test_500_raises_http_error(self) -> None:
        resp = MagicMock()
        resp.status_code = 500
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)
        resp.json.return_value = {"error": {"code": 500, "message": "Internal error", "status": "INTERNAL"}}
        with patch("src.llm.gemini_provider.requests.post", return_value=resp):
            with self.assertRaisesRegex(RuntimeError, "500"):
                GeminiProvider().review("prompt")

    def test_404_raises_with_model_name_and_google_message(self) -> None:
        """HTTP 404 must include the model name AND Google's error message."""
        resp = MagicMock()
        resp.status_code = 404
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)
        resp.json.return_value = {
            "error": {
                "code": 404,
                "message": "models/bad-model is not found for API version v1beta",
                "status": "NOT_FOUND",
            }
        }
        with patch("src.llm.gemini_provider.requests.post", return_value=resp):
            with self.assertRaises(RuntimeError) as ctx:
                GeminiProvider(model="bad-model").review("prompt")
        msg = str(ctx.exception)
        self.assertIn("404", msg)
        self.assertIn("bad-model", msg)
        # Google's message must be surfaced
        self.assertIn("not found", msg.lower())

    def test_404_mentions_fix_suggestion(self) -> None:
        """404 error message must suggest a working model alternative."""
        resp = MagicMock()
        resp.status_code = 404
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)
        resp.json.return_value = {"error": {"code": 404, "message": "Model not found", "status": "NOT_FOUND"}}
        with patch("src.llm.gemini_provider.requests.post", return_value=resp):
            with self.assertRaises(RuntimeError) as ctx:
                GeminiProvider(model="gemini-2.5-flash").review("prompt")
        msg = str(ctx.exception)
        # Must suggest the broadly-available model
        self.assertIn("gemini-2.0-flash", msg)

    def test_404_api_key_not_in_error_message(self) -> None:
        """API key must never appear in any exception message."""
        secret = "secret-api-key-xyz"
        os.environ["GEMINI_API_KEY"] = secret
        resp = MagicMock()
        resp.status_code = 404
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)
        resp.json.return_value = {"error": {"code": 404, "message": "Model not found", "status": "NOT_FOUND"}}
        with patch("src.llm.gemini_provider.requests.post", return_value=resp):
            with self.assertRaises(RuntimeError) as ctx:
                GeminiProvider(model="missing-model").review("prompt")
        self.assertNotIn(secret, str(ctx.exception))

    def test_400_raises_with_google_message(self) -> None:
        """HTTP 400 must include Google's error message."""
        resp = MagicMock()
        resp.status_code = 400
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)
        resp.json.return_value = {
            "error": {
                "code": 400,
                "message": "responseMimeType is not supported for this model",
                "status": "INVALID_ARGUMENT",
            }
        }
        with patch("src.llm.gemini_provider.requests.post", return_value=resp):
            with self.assertRaises(RuntimeError) as ctx:
                GeminiProvider().review("prompt")
        msg = str(ctx.exception)
        self.assertIn("400", msg)
        self.assertIn("responseMimeType", msg)

    def test_timeout_raises_runtime_error(self) -> None:
        with patch(
            "src.llm.gemini_provider.requests.post",
            side_effect=requests.exceptions.Timeout("slow"),
        ):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                GeminiProvider().review("prompt")

    def test_connection_error_raises_runtime_error(self) -> None:
        with patch(
            "src.llm.gemini_provider.requests.post",
            side_effect=requests.exceptions.ConnectionError("offline"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Could not connect"):
                GeminiProvider().review("prompt")

    def test_generic_request_exception_raises_runtime_error(self) -> None:
        with patch(
            "src.llm.gemini_provider.requests.post",
            side_effect=requests.exceptions.RequestException("boom"),
        ):
            with self.assertRaisesRegex(RuntimeError, "RequestException"):
                GeminiProvider().review("prompt")


# ---------------------------------------------------------------------------
# 6. review_with_deadline
# ---------------------------------------------------------------------------

class TestGeminiReviewWithDeadline(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["GEMINI_API_KEY"] = "test-key"

    def tearDown(self) -> None:
        os.environ.pop("GEMINI_API_KEY", None)

    def test_deadline_uses_min_of_timeout_and_deadline(self) -> None:
        p = GeminiProvider(timeout=60)
        captured = {}
        payload = '{"findings": []}'

        def fake_post(url, *, params, json, timeout):
            captured["timeout"] = timeout
            return _gemini_response(payload)

        with patch("src.llm.gemini_provider.requests.post", side_effect=fake_post):
            p.review_with_deadline("prompt", timeout_seconds=10.0)
        self.assertAlmostEqual(captured["timeout"], 10.0, places=1)

    def test_deadline_larger_than_timeout_uses_provider_timeout(self) -> None:
        p = GeminiProvider(timeout=30)
        captured = {}

        def fake_post(url, *, params, json, timeout):
            captured["timeout"] = timeout
            return _gemini_response('{"findings": []}')

        with patch("src.llm.gemini_provider.requests.post", side_effect=fake_post):
            p.review_with_deadline("prompt", timeout_seconds=120.0)
        self.assertEqual(captured["timeout"], 30.0)

    def test_invalid_deadline_raises_value_error(self) -> None:
        p = GeminiProvider()
        with self.assertRaises(ValueError):
            p.review_with_deadline("prompt", timeout_seconds=-1.0)
        with self.assertRaises(ValueError):
            p.review_with_deadline("prompt", timeout_seconds=0.0)


# ---------------------------------------------------------------------------
# 7. Config: default provider is ollama
# ---------------------------------------------------------------------------

class TestConfigProviderDefault(unittest.TestCase):
    def test_default_provider_is_ollama(self) -> None:
        config = build_config(**BASE_KWARGS)
        self.assertEqual(config.llm_provider, "ollama")
        self.assertEqual(DEFAULT_LLM_PROVIDER, "ollama")

    def test_default_gemini_model_constant(self) -> None:
        self.assertEqual(DEFAULT_GEMINI_MODEL, "gemini-2.0-flash")

    def test_config_default_gemini_model_field(self) -> None:
        config = build_config(**BASE_KWARGS)
        self.assertEqual(config.gemini_model, DEFAULT_GEMINI_MODEL)


# ---------------------------------------------------------------------------
# 8. Config: provider from environment variables
# ---------------------------------------------------------------------------

class TestConfigProviderEnvVars(unittest.TestCase):
    def test_env_selects_gemini(self, monkeypatch=None) -> None:
        with patch.dict(os.environ, {"REFLEXICODE_LLM_PROVIDER": "gemini"}):
            config = build_config(**BASE_KWARGS)
        self.assertEqual(config.llm_provider, "gemini")

    def test_env_selects_ollama_explicitly(self) -> None:
        with patch.dict(os.environ, {"REFLEXICODE_LLM_PROVIDER": "ollama"}):
            config = build_config(**BASE_KWARGS)
        self.assertEqual(config.llm_provider, "ollama")

    def test_env_gemini_model_is_read(self) -> None:
        with patch.dict(
            os.environ,
            {"REFLEXICODE_LLM_PROVIDER": "gemini", "REFLEXICODE_GEMINI_MODEL": "gemini-1.5-pro"},
        ):
            config = build_config(**BASE_KWARGS)
        self.assertEqual(config.gemini_model, "gemini-1.5-pro")

    def test_explicit_arg_overrides_env(self) -> None:
        with patch.dict(os.environ, {"REFLEXICODE_LLM_PROVIDER": "gemini"}):
            config = build_config(**BASE_KWARGS, llm_provider="ollama")
        self.assertEqual(config.llm_provider, "ollama")


# ---------------------------------------------------------------------------
# 9. Config: invalid provider values
# ---------------------------------------------------------------------------

class TestConfigInvalidProvider(unittest.TestCase):
    def test_invalid_provider_value_raises(self) -> None:
        with pytest.raises(GitHubConfigError, match="REFLEXICODE_LLM_PROVIDER"):
            build_config(**BASE_KWARGS, llm_provider="anthropic")

    def test_invalid_provider_from_env_raises(self) -> None:
        with patch.dict(os.environ, {"REFLEXICODE_LLM_PROVIDER": "chatgpt"}):
            with pytest.raises(GitHubConfigError, match="REFLEXICODE_LLM_PROVIDER"):
                build_config(**BASE_KWARGS)

    def test_empty_gemini_model_raises(self) -> None:
        with pytest.raises(GitHubConfigError, match="Gemini model"):
            build_config(**BASE_KWARGS, llm_provider="gemini", gemini_model="  ")


# ---------------------------------------------------------------------------
# 10. CLI: --llm-provider and --gemini-model parsing
# ---------------------------------------------------------------------------

class TestCLIParsing(unittest.TestCase):
    def _parse(self, *args: str) -> object:
        return build_parser().parse_args(list(args))

    def test_llm_provider_gemini_parsed(self) -> None:
        args = self._parse("--llm-provider", "gemini", "--owner", "x", "--repo", "y", "--pr-number", "1")
        self.assertEqual(args.llm_provider, "gemini")

    def test_llm_provider_ollama_parsed(self) -> None:
        args = self._parse("--llm-provider", "ollama", "--owner", "x", "--repo", "y", "--pr-number", "1")
        self.assertEqual(args.llm_provider, "ollama")

    def test_llm_provider_defaults_to_none_when_omitted(self) -> None:
        """When not specified the arg is None; build_config reads the env or default."""
        args = self._parse("--owner", "x", "--repo", "y", "--pr-number", "1")
        self.assertIsNone(args.llm_provider)

    def test_gemini_model_parsed(self) -> None:
        args = self._parse(
            "--llm-provider", "gemini",
            "--gemini-model", "gemini-1.5-pro",
            "--owner", "x", "--repo", "y", "--pr-number", "1",
        )
        self.assertEqual(args.gemini_model, "gemini-1.5-pro")

    def test_invalid_provider_choice_causes_parser_error(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--llm-provider", "anthropic"])

    def test_ollama_args_still_present(self) -> None:
        """All existing Ollama CLI args must remain available."""
        args = self._parse(
            "--ollama-model", "qwen2.5-coder:7b",
            "--ollama-timeout", "90",
            "--ollama-num-predict", "512",
            "--ollama-num-ctx", "4096",
            "--owner", "x", "--repo", "y", "--pr-number", "1",
        )
        self.assertEqual(args.ollama_model, "qwen2.5-coder:7b")
        self.assertEqual(args.ollama_timeout, 90)
        self.assertEqual(args.ollama_num_predict, 512)
        self.assertEqual(args.ollama_num_ctx, 4096)


# ---------------------------------------------------------------------------
# 11. build_pipeline: Gemini provider selected
# ---------------------------------------------------------------------------

class TestBuildPipelineGemini(unittest.TestCase):
    def _gemini_config(self, **kwargs):
        return build_config(**BASE_KWARGS, llm_provider="gemini", **kwargs)

    def test_gemini_config_selects_gemini_provider(self) -> None:
        config = self._gemini_config()
        pipeline = build_pipeline(config)
        self.assertIsInstance(pipeline.provider, GeminiProvider)

    def test_gemini_model_is_forwarded(self) -> None:
        config = self._gemini_config(gemini_model="gemini-1.5-pro")
        pipeline = build_pipeline(config)
        self.assertEqual(pipeline.provider.model, "gemini-1.5-pro")

    def test_pipeline_uses_shared_prompt_builder(self) -> None:
        config = self._gemini_config()
        pipeline = build_pipeline(config)
        self.assertIs(pipeline.provider.prompt_builder, pipeline.prompt_builder)

    def test_pipeline_inherits_review_budget(self) -> None:
        config = self._gemini_config()
        pipeline = build_pipeline(config)
        self.assertEqual(pipeline.review_budget_seconds, config.review_budget_seconds)

    def test_pipeline_inherits_max_requests(self) -> None:
        config = self._gemini_config()
        pipeline = build_pipeline(config)
        self.assertEqual(pipeline.max_requests, config.review_max_requests)


# ---------------------------------------------------------------------------
# 12. build_pipeline: Ollama remains the default
# ---------------------------------------------------------------------------

class TestBuildPipelineOllamaDefault(unittest.TestCase):
    def test_no_provider_arg_uses_ollama(self) -> None:
        config = build_config(**BASE_KWARGS)
        pipeline = build_pipeline(config)
        self.assertIsInstance(pipeline.provider, OllamaProvider)

    def test_explicit_ollama_provider_uses_ollama(self) -> None:
        config = build_config(**BASE_KWARGS, llm_provider="ollama")
        pipeline = build_pipeline(config)
        self.assertIsInstance(pipeline.provider, OllamaProvider)

    def test_ollama_model_forwarded(self) -> None:
        config = build_config(**BASE_KWARGS, ollama_model="qwen2.5-coder:7b")
        pipeline = build_pipeline(config)
        self.assertEqual(pipeline.provider.model, "qwen2.5-coder:7b")


# ---------------------------------------------------------------------------
# 13. build_pipeline: mock-provider behavior unchanged
# ---------------------------------------------------------------------------

class TestBuildPipelineMock(unittest.TestCase):
    def test_use_mock_llm_overrides_provider_selection(self) -> None:
        config = build_config(**BASE_KWARGS, use_mock_llm=True)
        pipeline = build_pipeline(config)
        self.assertIsInstance(pipeline.provider, MockProvider)

    def test_mock_response_is_forwarded(self) -> None:
        config = build_config(
            **BASE_KWARGS,
            use_mock_llm=True,
            mock_llm_response='{"findings": []}',
        )
        pipeline = build_pipeline(config)
        self.assertEqual(pipeline.provider.response, '{"findings": []}')

    def test_use_mock_llm_with_gemini_provider_still_uses_mock(self) -> None:
        """--use-mock-llm always takes precedence over --llm-provider."""
        config = build_config(**BASE_KWARGS, use_mock_llm=True, llm_provider="gemini")
        pipeline = build_pipeline(config)
        self.assertIsInstance(pipeline.provider, MockProvider)


# ---------------------------------------------------------------------------
# 14. GeminiProvider: context types accepted
# ---------------------------------------------------------------------------

class TestGeminiContextTypes(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["GEMINI_API_KEY"] = "test-key"

    def tearDown(self) -> None:
        os.environ.pop("GEMINI_API_KEY", None)

    def test_string_prompt_is_sent_directly(self) -> None:
        captured = {}

        def fake_post(url, *, params, json, timeout):
            captured["text"] = json["contents"][0]["parts"][0]["text"]
            return _gemini_response('{"findings": []}')

        p = GeminiProvider()
        with patch("src.llm.gemini_provider.requests.post", side_effect=fake_post):
            p.review("direct prompt text")
        self.assertEqual(captured["text"], "direct prompt text")

    def test_unsupported_type_raises_type_error(self) -> None:
        p = GeminiProvider()
        with self.assertRaises(TypeError):
            p.review(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 15. Retry handling and backoff
# ---------------------------------------------------------------------------

class TestGeminiRetries(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["GEMINI_API_KEY"] = "test-key"

    def tearDown(self) -> None:
        os.environ.pop("GEMINI_API_KEY", None)

    def test_transient_503_recovers_on_retry(self) -> None:
        """Attempt 1 fails with 503, attempt 2 succeeds with 200."""
        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_503.json.return_value = {
            "error": {"code": 503, "message": "High demand", "status": "UNAVAILABLE"}
        }
        resp_200 = _gemini_response('{"findings": []}')

        sleeps = []
        p = GeminiProvider(
            max_retries=2,
            sleeper=lambda s: sleeps.append(s),
        )

        with patch("src.llm.gemini_provider.requests.post", side_effect=[resp_503, resp_200]):
            result = p.review("prompt")

        self.assertEqual(result, '{"findings": []}')
        self.assertEqual(len(sleeps), 1)
        self.assertEqual(p.retry_count, 1)

    def test_transient_503_exhausts_retries_and_raises(self) -> None:
        """When 503 persists across all attempts, raises GeminiHighDemandError."""
        from src.llm.gemini_provider import GeminiHighDemandError

        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_503.json.return_value = {
            "error": {
                "code": 503,
                "message": "This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.",
                "status": "UNAVAILABLE",
            }
        }

        sleeps = []
        p = GeminiProvider(
            max_retries=2,
            sleeper=lambda s: sleeps.append(s),
        )

        with patch("src.llm.gemini_provider.requests.post", return_value=resp_503):
            with self.assertRaises(GeminiHighDemandError) as ctx:
                p.review("prompt")

        # 1 initial attempt + 2 retries = 2 sleeps
        self.assertEqual(len(sleeps), 2)
        self.assertEqual(p.retry_count, 2)
        # Check no double periods in error message
        self.assertNotIn("..", str(ctx.exception))
        self.assertIn("503", str(ctx.exception))
        self.assertIn("Please try again later", str(ctx.exception))

    def test_retry_after_header_is_honored(self) -> None:
        """When Retry-After header is present, it overrides exponential backoff."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "4.5"}
        resp_429.json.return_value = {"error": {"code": 429, "message": "Rate limit exceeded"}}
        resp_200 = _gemini_response('{"findings": []}')

        sleeps = []
        p = GeminiProvider(
            max_retries=1,
            sleeper=lambda s: sleeps.append(s),
        )

        with patch("src.llm.gemini_provider.requests.post", side_effect=[resp_429, resp_200]):
            p.review("prompt")

        self.assertEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[0], 4.5, places=1)

    def test_permanent_errors_are_not_retried(self) -> None:
        """HTTP 400, 401, 403, and 404 should never trigger retries."""
        for status in (400, 401, 403, 404):
            resp = MagicMock()
            resp.status_code = status
            resp.json.return_value = {"error": {"code": status, "message": f"Error {status}"}}

            calls = []
            def fake_post(*args, **kwargs):
                calls.append(status)
                return resp

            sleeps = []
            p = GeminiProvider(
                max_retries=3,
                sleeper=lambda s: sleeps.append(s),
            )

            with patch("src.llm.gemini_provider.requests.post", side_effect=fake_post):
                with self.assertRaises(RuntimeError):
                    p.review("prompt")

            self.assertEqual(len(calls), 1, f"Status {status} was retried!")
            self.assertEqual(len(sleeps), 0)

    def test_retry_skipped_when_budget_exhausted(self) -> None:
        """Retries must stop when wall-clock deadline has expired."""
        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.json.return_value = {"error": {"code": 500, "message": "Server error"}}

        # Clock returns monotonically advancing values
        clock_vals = [0.0, 0.5, 9.5, 10.0]
        def fake_clock():
            return clock_vals.pop(0) if clock_vals else 10.0

        p = GeminiProvider(
            max_retries=3,
            clock=fake_clock,
            sleeper=lambda _: None,
        )

        with patch("src.llm.gemini_provider.requests.post", return_value=resp_500):
            with self.assertRaises(RuntimeError):
                p.review_with_deadline("prompt", timeout_seconds=1.0)


# ---------------------------------------------------------------------------
# 16. Fallback model execution
# ---------------------------------------------------------------------------

class TestGeminiFallback(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["GEMINI_API_KEY"] = "test-key"

    def tearDown(self) -> None:
        os.environ.pop("GEMINI_API_KEY", None)

    def test_fallback_model_called_when_primary_fails_503(self) -> None:
        """Primary model gets 503; fallback model succeeds with 200."""
        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_503.json.return_value = {"error": {"code": 503, "message": "High demand"}}

        resp_200 = _gemini_response('{"findings": []}')

        requested_urls = []
        def fake_post(url, *args, **kwargs):
            requested_urls.append(url)
            if "gemini-3.5-flash" in url:
                return resp_503
            return resp_200

        p = GeminiProvider(
            model="gemini-3.5-flash",
            fallback_model="gemini-2.0-flash",
            max_retries=1,
            sleeper=lambda _: None,
        )

        with patch("src.llm.gemini_provider.requests.post", side_effect=fake_post):
            result = p.review("prompt")

        self.assertEqual(result, '{"findings": []}')
        self.assertTrue(p.fallback_attempted)
        self.assertTrue(p.fallback_succeeded)
        self.assertEqual(p.last_used_model, "gemini-2.0-flash")
        # Check both primary and fallback were invoked
        self.assertTrue(any("gemini-3.5-flash" in u for u in requested_urls))
        self.assertTrue(any("gemini-2.0-flash" in u for u in requested_urls))

    def test_fallback_model_called_when_primary_404(self) -> None:
        """Primary model gets 404 (unavailable); fallback model succeeds."""
        resp_404 = MagicMock()
        resp_404.status_code = 404
        resp_404.json.return_value = {"error": {"code": 404, "message": "Model not found"}}
        resp_200 = _gemini_response('{"findings": []}')

        def fake_post(url, *args, **kwargs):
            if "bad-model" in url:
                return resp_404
            return resp_200

        p = GeminiProvider(
            model="bad-model",
            fallback_model="gemini-2.0-flash",
            sleeper=lambda _: None,
        )

        with patch("src.llm.gemini_provider.requests.post", side_effect=fake_post):
            result = p.review("prompt")

        self.assertEqual(result, '{"findings": []}')
        self.assertTrue(p.fallback_succeeded)
        self.assertEqual(p.last_used_model, "gemini-2.0-flash")

    def test_fallback_not_attempted_if_not_configured(self) -> None:
        """Without a fallback model, failure on primary model raises directly."""
        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_503.json.return_value = {"error": {"code": 503, "message": "High demand"}}

        p = GeminiProvider(
            model="gemini-3.5-flash",
            fallback_model=None,
            max_retries=0,
            sleeper=lambda _: None,
        )

        with patch("src.llm.gemini_provider.requests.post", return_value=resp_503):
            with self.assertRaises(RuntimeError):
                p.review("prompt")

        self.assertFalse(p.fallback_attempted)

    def test_fallback_failure_reports_failure(self) -> None:
        """When both primary and fallback fail, raises error with fallback_succeeded=False."""
        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_503.json.return_value = {"error": {"code": 503, "message": "High demand"}}

        p = GeminiProvider(
            model="gemini-3.5-flash",
            fallback_model="gemini-2.0-flash",
            max_retries=0,
            sleeper=lambda _: None,
        )

        with patch("src.llm.gemini_provider.requests.post", return_value=resp_503):
            with self.assertRaises(RuntimeError):
                p.review("prompt")

        self.assertTrue(p.fallback_attempted)
        self.assertFalse(p.fallback_succeeded)


# ---------------------------------------------------------------------------
# 17. Config and CLI for fallback and retries
# ---------------------------------------------------------------------------

class TestGeminiConfigAndCLI(unittest.TestCase):
    def test_cli_parses_fallback_model_and_retries(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--llm-provider", "gemini",
            "--gemini-model", "gemini-3.5-flash",
            "--gemini-fallback-model", "gemini-2.0-flash",
            "--gemini-max-retries", "2",
            "--owner", "o", "--repo", "r", "--pr-number", "1",
        ])
        self.assertEqual(args.gemini_fallback_model, "gemini-2.0-flash")
        self.assertEqual(args.gemini_max_retries, 2)

    def test_config_reads_env_for_fallback_and_retries(self) -> None:
        with patch.dict(os.environ, {
            "REFLEXICODE_GEMINI_FALLBACK_MODEL": "gemini-1.5-flash",
            "REFLEXICODE_GEMINI_MAX_RETRIES": "4",
        }):
            config = build_config(**BASE_KWARGS, llm_provider="gemini")
            self.assertEqual(config.gemini_fallback_model, "gemini-1.5-flash")
            self.assertEqual(config.gemini_max_retries, 4)

    def test_build_pipeline_passes_fallback_and_retries_to_provider(self) -> None:
        config = build_config(
            **BASE_KWARGS,
            llm_provider="gemini",
            gemini_model="gemini-3.5-flash",
            gemini_fallback_model="gemini-2.0-flash",
            gemini_max_retries=5,
        )
        pipeline = build_pipeline(config)
        self.assertEqual(pipeline.provider.model, "gemini-3.5-flash")
        self.assertEqual(pipeline.provider.fallback_model, "gemini-2.0-flash")
        self.assertEqual(pipeline.provider.max_retries, 5)


# ---------------------------------------------------------------------------
# 18. Message formatting: no duplicate periods or run-on words
# ---------------------------------------------------------------------------

class TestMessageFormatting(unittest.TestCase):
    def test_clean_diagnostic_message_normalizes_whitespace(self) -> None:
        from src.llm.gemini_provider import _clean_diagnostic_message
        raw = "High demand.\n\nPlease try again later."
        cleaned = _clean_diagnostic_message(raw)
        self.assertEqual(cleaned, "High demand. Please try again later")
        self.assertFalse(cleaned.endswith("."))

    def test_clean_diagnostic_message_fixes_highdemand_word(self) -> None:
        from src.llm.gemini_provider import _clean_diagnostic_message
        raw = "Model experiencing highdemand right now."
        cleaned = _clean_diagnostic_message(raw)
        self.assertIn("high demand", cleaned)
        self.assertNotIn("highdemand", cleaned)


# ---------------------------------------------------------------------------
# 19. Advanced End-to-End Gemini pipeline & reliability tests
# ---------------------------------------------------------------------------

class TestGeminiAdvancedPipelineReliability(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["GEMINI_API_KEY"] = "test-key"

    def tearDown(self) -> None:
        os.environ.pop("GEMINI_API_KEY", None)

    def test_partial_503_failure_allows_subsequent_batches_to_continue(self) -> None:
        """Batch 1 fails with 503, but Batch 2 succeeds within deadline."""
        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_503.json.return_value = {"error": {"code": 503, "message": "High demand"}}

        resp_200 = _gemini_response(
            '{"findings":[{"file":"src/b.py","line":1,"category":"Bug",'
            '"severity":"High","message":"Critical bug in module b",'
            '"evidence":"def b():","suggestion":null}]}'
        )

        provider = GeminiProvider(model="gemini-3.5-flash", max_retries=0, sleeper=lambda _: None)
        prompt_builder = ReviewPromptBuilder(max_files=1)
        pipeline = ReviewPipeline(
            provider,
            prompt_builder=prompt_builder,
            max_requests=2,
            review_budget_seconds=240,
        )

        context = ReviewContextBuilder().build(
            pr_title="Multi-file change",
            changed_code={"src/a.py": "def a():\n    return 1", "src/b.py": "def b():\n    return 2"},
            changed_lines={"src/a.py": [1], "src/b.py": [1]},
            changed_line_content={"src/a.py": {1: "def a():"}, "src/b.py": {1: "def b():"}},
        )

        with patch("src.llm.gemini_provider.requests.post", side_effect=[resp_503, resp_200]):
            findings = pipeline.review(context)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].file, "src/b.py")
        self.assertEqual(pipeline.last_batch_count, 2)
        self.assertEqual(pipeline.last_failed_batch_count, 1)
        self.assertEqual(pipeline.last_successful_batch_count, 1)
        self.assertIn("src/b.py", pipeline.last_reviewed_files)
        self.assertIn("src/a.py", pipeline.last_unreviewed_files)

    def test_503_with_fallback_model_switches_quickly(self) -> None:
        """On 503, provider switches quickly to fallback model which succeeds."""
        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_503.json.return_value = {"error": {"code": 503, "message": "High demand"}}

        resp_200 = _gemini_response('{"findings":[]}')

        calls = []

        def fake_post(url, *args, **kwargs):
            calls.append(url)
            if "gemini-3.5-flash" in url:
                return resp_503
            return resp_200

        provider = GeminiProvider(
            model="gemini-3.5-flash",
            fallback_model="gemini-2.0-flash",
            max_retries=3,
            sleeper=lambda _: None,
        )

        with patch("src.llm.gemini_provider.requests.post", side_effect=fake_post):
            result = provider.review("test prompt")

        self.assertEqual(result, '{"findings":[]}')
        self.assertTrue(provider.fallback_attempted)
        self.assertTrue(provider.fallback_succeeded)
        self.assertEqual(provider.last_used_model, "gemini-2.0-flash")
        primary_calls = [c for c in calls if "gemini-3.5-flash" in c]
        self.assertLessEqual(len(primary_calls), 2)

    def test_sticky_fallback_for_subsequent_requests(self) -> None:
        """Once fallback succeeds, subsequent requests prioritize the working fallback model."""
        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_503.json.return_value = {"error": {"code": 503, "message": "High demand"}}
        resp_200 = _gemini_response('{"findings":[]}')

        provider = GeminiProvider(
            model="gemini-3.5-flash",
            fallback_model="gemini-2.0-flash",
            max_retries=1,
            sleeper=lambda _: None,
        )

        calls = []
        def fake_post(url, *args, **kwargs):
            calls.append(url)
            if "gemini-3.5-flash" in url:
                return resp_503
            return resp_200

        with patch("src.llm.gemini_provider.requests.post", side_effect=fake_post):
            provider.review("batch 1")
            self.assertTrue(provider.fallback_succeeded)
            calls.clear()
            provider.review("batch 2")

        self.assertIn("gemini-2.0-flash", calls[0])

    def test_filter_unsupported_findings_removes_comment_evidence_and_short_messages(self) -> None:
        f1 = ReviewFinding(
            file="src/a.py",
            line=1,
            category="Bug",
            severity="High",
            message="This is a real bug in function call",
            evidence="value = compute(x)",
            suggestion=None,
        )
        f_comment = ReviewFinding(
            file="src/a.py",
            line=2,
            category="Bug",
            severity="High",
            message="Comment suggests a bug",
            evidence="# TODO: fix this bug later",
            suggestion=None,
        )
        f_short = ReviewFinding(
            file="src/a.py",
            line=3,
            category="Correctness",
            severity="Medium",
            message="bad",
            evidence="return None",
            suggestion=None,
        )
        filtered = ReviewPipeline._filter_unsupported_findings([f1, f_comment, f_short])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0], f1)

    def test_gemini_config_allows_large_review_max_requests_and_prompt(self) -> None:
        """Gemini allows review_max_requests up to 50 without Ollama context errors."""
        config = build_config(
            owner="o",
            repo="r",
            pr_number=1,
            llm_provider="gemini",
            review_max_requests=25,
            review_prompt_chars=18000,
            review_code_chars=8000,
        )
        self.assertEqual(config.review_max_requests, 25)
        self.assertEqual(config.review_prompt_chars, 18000)
        self.assertEqual(config.review_code_chars, 8000)


if __name__ == "__main__":
    unittest.main()

