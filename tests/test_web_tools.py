"""Fast unit tests for web_tools.py without real network access."""

import json
import os
import sys
from urllib.parse import parse_qs, urlparse

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import web_tools
from web_tools import (
    _DDGResultParser,
    _clean_ddg_url,
    build_web_context,
    duckduckgo_search,
    re_rank_snippets,
    wikipedia_search,
)


class DummyResponse:
    def __init__(self, body: str):
        self.body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.body


class TestDuckDuckGoParsing:
    def test_clean_ddg_url_unwraps_uddg(self):
        wrapped = "/l/?kh=-1&uddg=https%3A%2F%2Fexample.com%2Fa%3Fb%3D1"
        assert _clean_ddg_url(wrapped) == "https://example.com/a?b=1"

    def test_clean_ddg_url_returns_plain_url(self):
        assert _clean_ddg_url("https://example.com") == "https://example.com"

    def test_ddg_parser_extracts_title_url_and_snippet(self):
        html = """
        <html><body>
          <a class="result__a" href="/l/?uddg=https%3A%2F%2Fa.example"> First &amp; Result </a>
          <a class="result__snippet"> First snippet &amp; details. </a>
          <a class="result__a" href="https://b.example">Second</a>
          <div class="result__snippet">Second snippet.</div>
        </body></html>
        """
        parser = _DDGResultParser()
        parser.feed(html)
        assert parser.results == [
            {"title": " First & Result ", "url": "/l/?uddg=https%3A%2F%2Fa.example", "snippet": " First snippet & details. "},
            {"title": "Second", "url": "https://b.example", "snippet": "Second snippet."},
        ]

    def test_duckduckgo_search_posts_query_and_cleans_output(self, monkeypatch):
        captured = {}
        html = """
        <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fone">  One   Result </a>
        <a class="result__snippet"> Snippet&nbsp;one. </a>
        <a class="result__a" href="https://example.com/two">Two</a>
        <span class="result__snippet">Snippet two.</span>
        """

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            captured["data"] = req.data.decode("utf-8")
            captured["ua"] = req.headers.get("User-agent") or req.headers.get("User-Agent")
            captured["timeout"] = timeout
            return DummyResponse(html)

        monkeypatch.setattr(web_tools.urllib.request, "urlopen", fake_urlopen)
        results = duckduckgo_search("test query", max_results=1, timeout=3.5)

        assert captured["url"] == web_tools.DDG_HTML_URL
        assert parse_qs(captured["data"])["q"] == ["test query"]
        assert captured["timeout"] == 3.5
        assert results == [{
            "title": "One Result",
            "url": "https://example.com/one",
            "snippet": "Snippet one.",
        }]

    def test_duckduckgo_search_empty_query_returns_empty_list(self):
        assert duckduckgo_search("   ") == []

    def test_duckduckgo_search_wraps_network_errors(self, monkeypatch):
        def fake_urlopen(req, timeout):
            raise OSError("offline")

        monkeypatch.setattr(web_tools.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(RuntimeError, match="Web search failed"):
            duckduckgo_search("query")


class TestWikipediaSearch:
    def test_wikipedia_search_success(self, monkeypatch):
        payload = {
            "query": {
                "search": [
                    {"title": "Alan Turing", "snippet": "British <span>mathematician</span> &amp; pioneer"},
                    {"title": "Turing machine", "snippet": "Model of computation"},
                ]
            }
        }
        captured = {}

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            captured["ua"] = req.headers.get("User-agent") or req.headers.get("User-Agent")
            captured["timeout"] = timeout
            return DummyResponse(json.dumps(payload))

        monkeypatch.setattr(web_tools.urllib.request, "urlopen", fake_urlopen)
        results = wikipedia_search("Alan Turing", max_results=1, timeout=2)

        parsed = urlparse(captured["url"])
        assert parsed.netloc == "en.wikipedia.org"
        assert parse_qs(parsed.query)["srsearch"] == ["Alan Turing"]
        assert captured["timeout"] == 2
        assert results == [{
            "title": "Alan Turing",
            "url": "https://en.wikipedia.org/wiki/Alan_Turing",
            "snippet": "British mathematician & pioneer",
        }]

    def test_wikipedia_search_empty_query_returns_empty_list(self):
        assert wikipedia_search("\t") == []

    def test_wikipedia_search_failure_returns_empty_list(self, monkeypatch):
        def fake_urlopen(req, timeout):
            raise TimeoutError("timeout")

        monkeypatch.setattr(web_tools.urllib.request, "urlopen", fake_urlopen)
        assert wikipedia_search("anything") == []

    def test_wikipedia_search_skips_empty_snippets(self, monkeypatch):
        payload = {"query": {"search": [{"title": "No Snippet", "snippet": ""}]}}
        monkeypatch.setattr(
            web_tools.urllib.request,
            "urlopen",
            lambda req, timeout: DummyResponse(json.dumps(payload)),
        )
        assert wikipedia_search("No Snippet") == []


class TestRankingAndContext:
    def test_re_rank_snippets_prefers_query_terms_in_title(self):
        results = [
            {"title": "Cooking", "url": "1", "snippet": "a short note"},
            {"title": "Python tutorial", "url": "2", "snippet": "learn python testing"},
        ]
        ranked = re_rank_snippets("python testing", results)
        assert ranked[0]["title"] == "Python tutorial"

    def test_re_rank_snippets_keeps_empty_or_blank_query_unchanged(self):
        results = [{"title": "A", "url": "u", "snippet": "s"}]
        assert re_rank_snippets(" ", results) is results
        assert re_rank_snippets("anything", []) == []

    def test_build_web_context_uses_ddg_results_and_reranks(self, monkeypatch):
        monkeypatch.setattr(
            web_tools,
            "duckduckgo_search",
            lambda query, max_results, timeout: [
                {"title": "Irrelevant", "url": "1", "snippet": "nothing"},
                {"title": "Python testing", "url": "2", "snippet": "pytest tests"},
            ],
        )
        monkeypatch.setattr(web_tools, "wikipedia_search", lambda *a, **k: [])

        context = build_web_context("python pytest", max_results=2, timeout=1)
        assert context.splitlines()[0].startswith("[1] Python testing")
        assert "pytest tests" in context

    def test_build_web_context_falls_back_to_wikipedia(self, monkeypatch):
        def ddg_fail(*args, **kwargs):
            raise RuntimeError("blocked")

        monkeypatch.setattr(web_tools, "duckduckgo_search", ddg_fail)
        monkeypatch.setattr(
            web_tools,
            "wikipedia_search",
            lambda query, max_results, timeout: [
                {"title": "Fallback", "url": "u", "snippet": "from wikipedia"}
            ],
        )

        assert build_web_context("query") == "[1] Fallback: from wikipedia"

    def test_build_web_context_returns_empty_string_when_no_results(self, monkeypatch):
        monkeypatch.setattr(web_tools, "duckduckgo_search", lambda *a, **k: [])
        monkeypatch.setattr(web_tools, "wikipedia_search", lambda *a, **k: [])
        assert build_web_context("unknown") == ""

    def test_build_web_context_truncates_to_word_boundary(self, monkeypatch):
        monkeypatch.setattr(
            web_tools,
            "duckduckgo_search",
            lambda *a, **k: [{"title": "Title", "url": "u", "snippet": "word " * 100}],
        )
        context = build_web_context("word", max_results=1, max_chars=60)
        assert len(context) <= 61  # includes ellipsis
        assert context.endswith("…")
        assert not context.endswith(" …")
