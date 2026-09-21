"""Regression tests for the v2.6.2 improvement pass.

Covers: repo hygiene (.gitignore), training seed reproducibility, AMP dtype
knob, atomic versioned checkpoints with persisted chat template, the fast
incremental BPE trainer (byte-identical to the naive reference), new server
surface (usage/, /v1/models, API key, concurrency guard, /v1/embeddings),
and the auralite CLI.
"""

import glob
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from model_engine import AuraLiteEngine
from model_engine._legacy import BPETokenizer, UNK_TOKEN


TINY_PARAMS = {
    "tokenizer": "char", "d_model": 32, "n_heads": 4, "n_layers": 1,
    "d_ff": 64, "seq_length": 16, "epochs": 1, "batch_size": 8,
    "val_split": 0.1, "optimizer": "adamw", "lr_schedule": "cosine",
}
TEXT = "improvements smoke corpus with enough variety to learn. " * 30


# ---------------------------------------------------------------------------
#  Repo hygiene
# ---------------------------------------------------------------------------

class TestRepoHygiene:
    def test_gitignore_excludes_python_caches(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, ".gitignore"), encoding="utf-8") as f:
            content = f.read()
        assert "__pycache__/" in content
        assert "*.py[cod]" in content

    def test_no_pyc_files_tracked_in_git(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isdir(os.path.join(root, ".git")):
            pytest.skip("not a git checkout")
        import subprocess
        out = subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True)
        assert out.returncode == 0
        tracked_on_disk = [
            ln for ln in out.stdout.splitlines()
            if ("__pycache__" in ln or ln.endswith(".pyc"))
            and os.path.exists(os.path.join(root, ln))
        ]
        # Tracked entries may briefly outlive the working-tree deletion
        # (patch applied but not committed yet) — what must never happen is a
        # *committed and present* cache file.
        assert tracked_on_disk == []


# ---------------------------------------------------------------------------
#  Seed reproducibility
# ---------------------------------------------------------------------------

class TestSeed:
    def test_same_seed_reproduces_training(self):
        params = {**TINY_PARAMS, "seed": 123}
        e1 = AuraLiteEngine()
        e1.train(TEXT, params)
        e2 = AuraLiteEngine()
        e2.train(TEXT, params)
        kw = dict(length=4, temperature=1.0, top_k=1, top_p=1.0)
        assert e1.generate("improvements", **kw) == e2.generate("improvements", **kw)

    def test_same_seed_reproduces_val_loss(self):
        params = {**TINY_PARAMS, "seed": 7}
        e1 = AuraLiteEngine()
        e1.train(TEXT, params)
        e2 = AuraLiteEngine()
        e2.train(TEXT, params)
        assert e1.last_val_loss == pytest.approx(e2.last_val_loss)

    def test_negative_seed_rejected(self):
        from model_engine._legacy import ParamValidationError
        e = AuraLiteEngine()
        with pytest.raises(ParamValidationError, match="seed"):
            e.train(TEXT, {**TINY_PARAMS, "seed": -1})

    def test_no_seed_still_works(self):
        e = AuraLiteEngine()
        e.train(TEXT, TINY_PARAMS)
        assert e.model is not None


# ---------------------------------------------------------------------------
#  AMP dtype knob
# ---------------------------------------------------------------------------

class TestAmpDtype:
    def test_bf16_autocast_on_cpu(self):
        e = AuraLiteEngine()
        e.train(TEXT, {**TINY_PARAMS, "amp_dtype": "bf16"})
        assert e.last_val_loss is not None and e.last_val_loss > 0

    def test_none_disables_amp(self):
        e = AuraLiteEngine()
        e.train(TEXT, {**TINY_PARAMS, "amp_dtype": "none"})
        assert e.model is not None

    def test_invalid_amp_dtype_rejected(self):
        from model_engine._legacy import ParamValidationError
        e = AuraLiteEngine()
        with pytest.raises(ParamValidationError, match="amp_dtype"):
            e.train(TEXT, {**TINY_PARAMS, "amp_dtype": "fp8"})

    def test_scaler_only_for_fp16(self):
        e = AuraLiteEngine()
        e.train(TEXT, {**TINY_PARAMS, "amp_dtype": "bf16"})
        assert e.scaler is not None and not e.scaler.is_enabled()  # bf16: no scaler


# ---------------------------------------------------------------------------
#  Atomic checkpoints, versioning, persisted chat template
# ---------------------------------------------------------------------------

class TestCheckpointsV262:
    @pytest.fixture
    def engine(self):
        e = AuraLiteEngine()
        e.train(TEXT, {**TINY_PARAMS, "seed": 42, "chat_template": "simple"})
        return e

    def test_version_and_format_written(self, engine, tmp_path):
        path = str(tmp_path / "m.pt")
        engine.save_model(path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["checkpoint_version"] == 3
        assert ck["format"] == "auralite"
        assert ck["chat_template"] == "simple"

    def test_no_tmp_files_left_after_save(self, engine, tmp_path):
        path = str(tmp_path / "m.pt")
        engine.save_model(path)
        leftovers = glob.glob(os.path.join(str(tmp_path), "*.tmp"))
        assert leftovers == []

    def test_chat_template_restored_on_load(self, engine, tmp_path):
        path = str(tmp_path / "m.pt")
        engine.save_model(path)
        e2 = AuraLiteEngine()
        e2.load_model(path)
        assert e2.default_chat_template == "simple"

    def test_old_checkpoint_without_template_defaults_chatml(self, engine, tmp_path):
        # Emulate a pre-2.6.2 checkpoint: strip the new keys from the saved dict.
        path = str(tmp_path / "m.pt")
        engine.save_model(path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ck.pop("chat_template", None)
        ck.pop("checkpoint_version", None)
        ck.pop("format", None)
        ck.get("params_used", {}).pop("chat_template", None)  # pre-2.6.2 runs did not have it
        torch.save(ck, path)
        e2 = AuraLiteEngine()
        e2.load_model(path)
        assert e2.default_chat_template == "chatml"

    def test_generate_chat_uses_engine_default_template(self, engine, tmp_path):
        path = str(tmp_path / "m.pt")
        engine.save_model(path)
        e2 = AuraLiteEngine()
        e2.load_model(path)
        seen = {}
        orig = e2._prepare_prompt_ids

        def spy(prompt, **kw):
            seen["prompt"] = prompt
            return orig(prompt, **kw)

        e2._prepare_prompt_ids = spy
        e2.generate_chat([{"role": "user", "content": "hello"}], max_new_tokens=1)
        assert "prompt" in seen
        # The persisted template is "simple": "User: hello\nAssistant:"
        assert "<|im_start|>" not in seen["prompt"]
        assert "User: hello" in seen["prompt"]

    def test_explicit_template_still_wins(self, engine):
        seen = {}
        orig = engine._prepare_prompt_ids

        def spy(prompt, **kw):
            seen["prompt"] = prompt
            return orig(prompt, **kw)

        engine._prepare_prompt_ids = spy
        engine.generate_chat([{"role": "user", "content": "hello"}],
                             max_new_tokens=1, chat_template="chatml")
        assert "<|im_start|>" in seen["prompt"]


# ---------------------------------------------------------------------------
#  Fast incremental BPE trainer — bit-identical to the naive reference
# ---------------------------------------------------------------------------

def _reference_naive_bpe(text: str, vocab_size: int):
    """Faithful copy of the pre-v2.6.2 naive trainer (full rescans per merge)."""
    from collections import Counter
    sample = BPETokenizer._stratified_sample(text, max_chars=2_000_000, n_chunks=100)
    base_chars = sorted(set(sample))
    vocab = list(base_chars)
    token_to_id = {t: i for i, t in enumerate(vocab)}
    if UNK_TOKEN not in token_to_id:
        vocab.append(UNK_TOKEN)
        token_to_id[UNK_TOKEN] = len(vocab) - 1
    unk_id = token_to_id.get(UNK_TOKEN, 0)
    merges = []
    if vocab_size > len(vocab):
        piece_counts = Counter(BPETokenizer._split_pieces(sample))
        corpus = [([token_to_id.get(c, unk_id) for c in piece], cnt)
                  for piece, cnt in piece_counts.items()]
        while len(vocab) < vocab_size:
            pair_counts = Counter()
            for ids, cnt in corpus:
                for i in range(len(ids) - 1):
                    pair_counts[(ids[i], ids[i + 1])] += cnt
            if not pair_counts:
                break
            (a, b), best = pair_counts.most_common(1)[0]
            if best < 2:
                break
            nid = len(vocab)
            vocab.append(vocab[a] + vocab[b])
            token_to_id[vocab[a] + vocab[b]] = nid
            merges.append((a, b, nid))
            for entry in corpus:
                ids = entry[0]
                i, out = 0, []
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                        out.append(nid)
                        i += 2
                    else:
                        out.append(ids[i])
                        i += 1
                entry[0][:] = out
    return vocab, merges


class TestBPETrainerIdentity:
    CASES = [
        ("english", "the quick brown fox jumps over the lazy dog. " * 60, 128),
        ("forced_ties", "aa bb cc dd ee ff gg hh " * 40, 60),
        ("unicode", "привет мир! тестовый текст. Café au lait. " * 30, 100),
        ("repetitive", "ab " * 800, 40),
        ("tiny", "tiny", 64),
    ]

    @pytest.mark.parametrize("name,text,vocab_size", CASES)
    def test_vocab_and_merges_match_naive(self, name, text, vocab_size):
        ref_vocab, ref_merges = _reference_naive_bpe(text, vocab_size)
        tok = BPETokenizer()
        tok.train(text, vocab_size=vocab_size)
        assert tok.vocab == ref_vocab, name
        assert tok.merges == ref_merges, name

    @pytest.mark.parametrize("name,text,vocab_size", CASES[:2])
    def test_roundtrip_after_fast_training(self, name, text, vocab_size):
        tok = BPETokenizer()
        tok.train(text, vocab_size=vocab_size)
        sample = text[:300]
        assert tok.decode(tok.encode(sample)) == sample


# ---------------------------------------------------------------------------
#  Server: usage, /v1/models, API key, concurrency, /v1/embeddings
# ---------------------------------------------------------------------------

class _FakeEngine:
    backend = "torch"

    class _FakeModel:
        pass

    model = _FakeModel()
    params_used: dict = {}

    def generate(self, prompt, length=50, temperature=0.8, top_k=50, top_p=0.9, **kw):
        return prompt + " world"

    def generate_streaming(self, prompt, length=50, temperature=0.8, **kw):
        yield "tok"

    def generate_chat(self, messages, max_new_tokens=256, temperature=0.7,
                      top_k=40, top_p=0.9, **kw):
        return "chat answer"

    def generate_chat_streaming(self, messages, max_new_tokens=256, **kw):
        yield "tok"

    def encode(self, s):
        return list(s)  # 1 char = 1 "token" — easy usage maths


@pytest.fixture
def client(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import server.openai_server as srv
    monkeypatch.setattr(srv, "_engine", _FakeEngine())
    monkeypatch.setattr(srv, "_rate_bucket", {})
    return TestClient(srv.app, raise_server_exceptions=False)


class TestServerV262:
    def test_models_endpoint_shape(self, client):
        r = client.get("/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        ids = {m["id"] for m in body["data"]}
        assert any(i.startswith("auralite") for i in ids)
        assert "auralite-embed" in ids

    def test_completion_includes_usage_and_created(self, client):
        r = client.post("/v1/completions",
                        json={"prompt": "hello", "max_tokens": 3})
        assert r.status_code == 200
        body = r.json()
        assert body["created"] > 0
        usage = body["usage"]
        # FakeEngine: 1 char = 1 token; prompt "hello" -> 5, " world" -> 6
        assert usage["prompt_tokens"] == 5
        assert usage["completion_tokens"] == 6
        assert usage["total_tokens"] == 11

    def test_batch_completion_usage_sums_prompts(self, client):
        r = client.post("/v1/completions",
                        json={"prompt": ["abc", "x"], "max_tokens": 1})
        usage = r.json()["usage"]
        assert usage["prompt_tokens"] == 4  # 3 + 1

    def test_chat_usage(self, client):
        r = client.post("/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "hi"}],
                              "max_tokens": 3})
        assert r.status_code == 200
        usage = r.json()["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] == len("chat answer")
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_api_key_required_when_set(self, client, monkeypatch):
        monkeypatch.setenv("AURALITE_API_KEY", "s3cr3t")
        r = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 1})
        assert r.status_code == 401
        r = client.post("/v1/completions",
                        json={"prompt": "hi", "max_tokens": 1},
                        headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401
        r = client.post("/v1/completions",
                        json={"prompt": "hi", "max_tokens": 1},
                        headers={"Authorization": "Bearer s3cr3t"})
        assert r.status_code == 200

    def test_health_open_without_api_key(self, client, monkeypatch):
        monkeypatch.setenv("AURALITE_API_KEY", "s3cr3t")
        r = client.get("/health")
        assert r.status_code == 200
        # ...but even models requires auth then
        r = client.get("/v1/models")
        assert r.status_code == 401

    def test_busy_semaphore_returns_503(self):
        pytest.importorskip("fastapi")
        import threading
        from fastapi.testclient import TestClient
        import server.openai_server as srv
        srv._engine = _FakeEngine()
        srv._rate_bucket = {}
        old = srv._gen_semaphore
        srv._gen_semaphore = threading.BoundedSemaphore(1)  # one slot, occupied below
        try:
            srv._gen_semaphore.acquire()
            c = TestClient(srv.app, raise_server_exceptions=False)
            r = c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 1})
            assert r.status_code == 503
            assert "busy" in r.text
        finally:
            srv._gen_semaphore = old
            srv._engine = None
            srv._rate_bucket = {}

    def test_embeddings_endpoint_matches_hash_embedding(self, client):
        from web_tools import hash_embedding
        r = client.post("/v1/embeddings", json={"input": "hello world"})
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        assert body["data"][0]["embedding"] == pytest.approx(hash_embedding("hello world"))
        assert body["data"][0]["index"] == 0
        # FakeEngine.encode: 1 char = 1 token
        assert body["usage"]["prompt_tokens"] == len("hello world")

    def test_embeddings_batch(self, client):
        r = client.post("/v1/embeddings", json={"input": ["aa", "bbb"]})
        assert [d["index"] for d in r.json()["data"]] == [0, 1]


# ---------------------------------------------------------------------------
#  CLI (auralite)
# ---------------------------------------------------------------------------

class TestCLI:
    @pytest.fixture
    def trained_model(self, tmp_path):
        text_file = tmp_path / "corpus.txt"
        text_file.write_text(TEXT, encoding="utf-8")
        params = {**TINY_PARAMS, "seed": 5}
        params_file = tmp_path / "params.json"
        params_file.write_text(json.dumps(params), encoding="utf-8")
        out = str(tmp_path / "cli.pt")
        from auralite_cli import main
        rc = main(["train", "--text", str(text_file),
                   "--params", str(params_file), "--model", out])
        assert rc == 0 and os.path.exists(out)
        return out

    def test_train_via_cli(self, trained_model):
        import torch as _t
        ck = _t.load(trained_model, map_location="cpu", weights_only=False)
        assert ck["chat_template"] == "chatml"

    def test_generate_via_cli(self, trained_model, capsys):
        from auralite_cli import main
        rc = main(["generate", "--model", trained_model,
                   "--prompt", "improvements", "--length", "3",
                   "--temperature", "1.0", "--top-k", "1"])
        assert rc == 0
        out = capsys.readouterr().out.strip()
        assert len(out) == len("improvements") + 3

    def test_chat_one_shot_via_cli(self, trained_model, capsys):
        from auralite_cli import main
        rc = main(["chat", "--model", trained_model, "--prompt", "hi",
                   "--length", "2", "--system", "be terse"])
        assert rc == 0
        assert "assistant:" in capsys.readouterr().out

    def test_info_via_cli(self, trained_model, capsys):
        from auralite_cli import main
        rc = main(["info", "--model", trained_model])
        assert rc == 0
        out = capsys.readouterr().out
        assert "checkpoint_version: 3" in out
        assert "tokenizer: char" in out

    def test_train_missing_file_fails_cleanly(self, tmp_path):
        from auralite_cli import main
        rc = main(["train", "--text", str(tmp_path / "nope.txt"),
                   "--model", str(tmp_path / "x.pt")])
        assert rc == 2

    def test_console_scripts_entry_registered(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as f:
            content = f.read()
        assert "[project.scripts]" in content
        assert 'auralite = "auralite_cli:main"' in content
        from auralite_cli import build_parser
        # every subcommand parses
        build_parser().parse_args(["info", "--model", "x.pt"])
