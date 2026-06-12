"""
web_tools.py — Lightweight web search for AuraLite AI.

Uses the DuckDuckGo HTML endpoint (no API key, no extra dependencies —
only the Python standard library). Returns titles + snippets that can be
injected into the model's prompt as retrieval context (mini-RAG).

IMPROVED (v2.2+):
- Wikipedia API Fallback: Automatically queries Wikipedia if DuckDuckGo fails or blocks.
- TF-IDF Snippet Re-ranking: Computes a semantic word-overlap/frequency score to bubble up the most relevant results.
- Advanced normalization and robust extraction.
"""

from __future__ import annotations

import html
import json
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


def wikipedia_search(query: str, max_results: int = 5,
                     timeout: float = 10.0) -> list[dict]:
    """Search Wikipedia using its stable, data-center friendly MediaWiki API.

    Each dict has keys: title, url, snippet.
    Returns an empty list on failure instead of raising to guarantee graceful fallback.
    """
    if not query.strip():
        return []

    search_url = (
        "https://en.wikipedia.org/w/api.php?action=query&list=search"
        f"&srsearch={urllib.parse.quote(query)}&utf8=&format=json"
    )
    req = urllib.request.Request(
        search_url,
        headers={"User-Agent": "AuraLiteAI/2.2 (https://github.com/AlexanderNyr/AuraLite-AI)"}
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        search_results = data.get("query", {}).get("search", [])
    except Exception as e:
        print(f"[Wikipedia Search] API request failed: {e}")
        return []

    results = []
    for r in search_results[:max_results]:
        title = r.get("title", "")
        raw_snippet = r.get("snippet", "")
        # Strip HTML tags Wikipedia API returns
        clean_snippet = re.sub(r"<[^>]+>", "", raw_snippet)
        clean_snippet = html.unescape(clean_snippet).strip()
        
        encoded_title = urllib.parse.quote(title.replace(" ", "_"))
        article_url = f"https://en.wikipedia.org/wiki/{encoded_title}"
        
        if title and clean_snippet:
            results.append({
                "title": title,
                "url": article_url,
                "snippet": clean_snippet
            })
    return results


def re_rank_snippets(query: str, results: list[dict]) -> list[dict]:
    """Re-rank retrieved snippets using a simple TF-IDF / term-frequency scoring

    relative to the query. Ensures high-quality relevant snippets bubble up.
    """
    if not results or not query.strip():
        return results

    # Tokenize query into lowercase words, skipping common English stopwords
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "with", "by", "of", "is", "was", "are", "were", "who", "what", "how",
        "where", "why", "which"
    }
    query_words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 1 and w.lower() not in stop_words]
    if not query_words:
        query_words = [w.lower() for w in re.findall(r"\w+", query)]

    scored_results = []
    for r in results:
        text = (r["title"] + " " + r["snippet"]).lower()
        words_in_text = re.findall(r"\w+", text)
        word_counts = {}
        for w in words_in_text:
            word_counts[w] = word_counts.get(w, 0) + 1

        score = 0.0
        for qw in query_words:
            if qw in word_counts:
                # Term frequency score with square root saturation
                score += (word_counts[qw] ** 0.5)
                # Extra weight for exact match in title
                if qw in r["title"].lower():
                    score += 1.5

        # Small decaying bonus for original search engine ranking order
        score += (1.0 / (len(scored_results) + 1)) * 0.1
        scored_results.append((score, r))

    # Sort descending by score
    scored_results.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored_results]


def build_web_context(query: str, max_results: int = 4,
                      max_chars: int = 1200,
                      timeout: float = 10.0) -> str:
    """Run a web search and format the results as plain-text context
    suitable for prepending to the model's prompt.

    Uses DuckDuckGo search as primary source, and falls back to Wikipedia
    search if DuckDuckGo fails, is blocked, or returns no results. Re-ranks
    snippets using a TF-IDF relevance metric.
    """
    results = []

    # 1. Try DuckDuckGo
    try:
        results = duckduckgo_search(query, max_results=max_results, timeout=timeout)
    except Exception as e:
        print(f"[AuraLite Web Search] DuckDuckGo search failed: {e}. Falling back to Wikipedia...")

    # 2. Wikipedia fallback if DDG returned nothing or failed
    if not results:
        try:
            results = wikipedia_search(query, max_results=max_results, timeout=timeout)
        except Exception as e:
            print(f"[AuraLite Web Search] Wikipedia fallback failed: {e}")

    if not results:
        return ""

    # 3. Apply TF-IDF-based re-ranking
    results = re_rank_snippets(query, results)

    # 4. Format context
    lines = []
    for i, r in enumerate(results[:max_results], 1):
        line = f"[{i}] {r['title']}: {r['snippet']}".strip().rstrip(":")
        lines.append(line)

    context = "\n".join(lines)
    if len(context) > max_chars:
        context = context[:max_chars].rsplit(" ", 1)[0] + "…"
    return context
