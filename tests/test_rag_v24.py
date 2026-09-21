"""Tests for the v2.4 RAG layer in web_tools.py:
SimpleVectorStore (hashed BoW fallback), semantic chunking, HyDE, citation context.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_tools import (
    SimpleVectorStore, semantic_chunks, hyde_query, build_rag_context,
)


class TestSemanticChunks:
    def test_respects_chunk_size(self):
        text = " ".join(f"Sentence number {i} ends here." for i in range(60))
        chunks = semantic_chunks(text, chunk_chars=120, overlap=20)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 120 + 60  # overlap slack, never a runaway chunk

    def test_overlap_preserves_continuity(self):
        text = " ".join(f"Sentence {i} is long enough." for i in range(60))
        chunks = semantic_chunks(text, chunk_chars=100, overlap=30)
        assert len(chunks) >= 2
        # consecutive chunks share some tail context
        tail = chunks[0][-30:]
        assert any(tok in chunks[1] for tok in tail.split() if tok)

    def test_single_long_sentence_hard_split(self):
        # No sentence boundaries at all -> character fallback kicks in.
        text = " ".join(f"word{i}" for i in range(200))
        chunks = semantic_chunks(text, chunk_chars=100, overlap=20)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) <= 100

    def test_short_text_single_chunk(self):
        assert semantic_chunks("just one short sentence.") == ["just one short sentence."]

    def test_no_empty_chunks(self):
        text = "a. " * 500
        for c in semantic_chunks(text, chunk_chars=80, overlap=10):
            assert c.strip()


class TestHyDE:
    def test_hyde_wraps_query(self):
        out = hyde_query("what is RoPE?")
        assert "what is RoPE?" in out
        assert len(out) > len("what is RoPE?")


class TestSimpleVectorStore:
    def test_add_and_search_returns_most_similar(self, tmp_path):
        store = SimpleVectorStore(str(tmp_path / "s.json"))
        store.add("cats are fluffy animals", source="pets")
        store.add("quantum mechanics studies particles", source="physics")
        store.add("dogs are loyal animals", source="pets")
        hits = store.search("animals", k=2)
        assert len(hits) == 2
        assert hits[0]["source"] == "pets"
        assert hits[0]["score"] > 0

    def test_persist_and_reload(self, tmp_path):
        path = str(tmp_path / "s.json")
        store = SimpleVectorStore(path)
        store.add("persistent chunk", source="local")
        store.persist()
        assert os.path.exists(path)
        json.loads(open(path, encoding="utf-8").read())  # valid JSON
        store2 = SimpleVectorStore(path)
        assert len(store2.items) == 1
        assert store2.items[0]["text"] == "persistent chunk"

    def test_corrupt_store_file_recovers_empty(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text("{not json", encoding="utf-8")
        store = SimpleVectorStore(str(path))
        assert store.items == []

    def test_search_empty_store(self, tmp_path):
        store = SimpleVectorStore(str(tmp_path / "s.json"))
        assert store.search("anything") == []

    def test_cosine_normalized(self):
        store = SimpleVectorStore.__new__(SimpleVectorStore)
        store.dim = 16
        store._model = None
        a = store._embed("hello world")
        a2 = store._embed("hello world")  # deterministic
        assert a == a2
        norm = sum(v * v for v in a) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    def test_metadata_roundtrip(self, tmp_path):
        store = SimpleVectorStore(str(tmp_path / "s.json"))
        store.add("text", source="web", metadata={"url": "http://x"})
        assert store.items[0]["metadata"]["url"] == "http://x"


class TestBuildRagContext:
    def test_citations_present(self, tmp_path, monkeypatch):
        import web_tools
        monkeypatch.setattr(web_tools, "build_web_context", lambda *a, **k: "")
        docs = [("notes.txt", "RoPE rotates pairs of embedding dimensions. " * 10)]
        ctx = build_rag_context("what is RoPE?", local_docs=docs,
                                store_path=str(tmp_path / "s.json"), k=3,
                                use_hyde=False)
        assert "[source: notes.txt]" in ctx

    def test_web_snippets_appended(self, tmp_path, monkeypatch):
        import web_tools
        monkeypatch.setattr(web_tools, "build_web_context",
                            lambda *a, **k: "[1] Web: some snippet")
        docs = [("local.md", "local content about transformers " * 8)]
        ctx = build_rag_context("transformers", local_docs=docs,
                                store_path=str(tmp_path / "s.json"), k=2)
        assert "Web: some snippet" in ctx

    def test_web_failure_is_tolerated(self, tmp_path, monkeypatch):
        import web_tools

        def boom(*a, **k):
            raise RuntimeError("offline")

        monkeypatch.setattr(web_tools, "build_web_context", boom)
        docs = [("d.md", "tolerable content " * 8)]
        ctx = build_rag_context("content", local_docs=docs,
                                store_path=str(tmp_path / "s.json"), k=2)
        assert "tolerable content" in ctx
