"""
web_tools.py — Lightweight web search for AuraLite AI.

Uses the DuckDuckGo HTML endpoint (no API key, no extra dependencies —
only the Python standard library). Returns titles + snippets that can be
injected into the model's prompt as retrieval context (mini-RAG).
"""

from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0 Safari/537.36")

DDG_HTML_URL = "https://html.duckduckgo.com/html/"


class _DDGResultParser(HTMLParser):
    """Extracts result titles and snippets from DuckDuckGo HTML output."""

    def __init__(self):
        super().__init__()
        self.results: list[dict] = []
        self._in_title = False
        self._in_snippet = False
        self._current: dict | None = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        cls = attrs.get("class", "") or ""
        if tag == "a" and "result__a" in cls:
            self._in_title = True
            self._current = {"title": "", "url": attrs.get("href", ""),
                             "snippet": ""}
        elif "result__snippet" in cls:
            self._in_snippet = True

    def handle_endtag(self, tag):
        if self._in_title and tag == "a":
            self._in_title = False
            if self._current is not None:
                self.results.append(self._current)
        elif self._in_snippet and tag in ("a", "td", "div", "span"):
            self._in_snippet = False
            self._current = None

    def handle_data(self, data):
        if self._in_title and self._current is not None:
            self._current["title"] += data
        elif self._in_snippet and self.results:
            self.results[-1]["snippet"] += data


def _clean_ddg_url(raw: str) -> str:
    """DuckDuckGo wraps URLs as /l/?uddg=<encoded>. Unwrap them."""
    if "uddg=" in raw:
        try:
            qs = urllib.parse.urlparse(raw).query
            params = urllib.parse.parse_qs(qs)
            if "uddg" in params:
                return params["uddg"][0]
        except Exception:
            pass
    return raw


def duckduckgo_search(query: str, max_results: int = 5,
                      timeout: float = 10.0) -> list[dict]:
    """Search DuckDuckGo and return a list of result dicts.

    Each dict has keys: title, url, snippet.
    Raises RuntimeError on network/parse failure.
    """
    if not query.strip():
        return []

    data = urllib.parse.urlencode({"q": query}).encode("utf-8")
    req = urllib.request.Request(
        DDG_HTML_URL, data=data,
        headers={"User-Agent": USER_AGENT,
                 "Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        raise RuntimeError(f"Web search failed: {e}") from e

    parser = _DDGResultParser()
    parser.feed(raw)

    results = []
    for r in parser.results[:max_results]:
        title = html.unescape(re.sub(r"\s+", " ", r["title"])).strip()
        snippet = html.unescape(re.sub(r"\s+", " ", r["snippet"])).strip()
        if title:
            results.append({"title": title,
                            "url": _clean_ddg_url(r["url"]),
                            "snippet": snippet})
    return results


def build_web_context(query: str, max_results: int = 4,
                      max_chars: int = 1200,
                      timeout: float = 10.0) -> str:
    """Run a web search and format the results as plain-text context
    suitable for prepending to the model's prompt.

    Returns an empty string if nothing was found.
    """
    results = duckduckgo_search(query, max_results=max_results,
                                timeout=timeout)
    if not results:
        return ""

    lines = []
    for i, r in enumerate(results, 1):
        line = f"[{i}] {r['title']}: {r['snippet']}".strip().rstrip(":")
        lines.append(line)

    context = "\n".join(lines)
    if len(context) > max_chars:
        context = context[:max_chars].rsplit(" ", 1)[0] + "…"
    return context
