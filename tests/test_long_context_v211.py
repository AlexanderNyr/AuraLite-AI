"""2.11 "long context": the generation window, chunked prefill, context
extension, bits per byte and the Ollama `num_predict` mapping.

A RoPE model trained on N-token windows has never seen an attention distance
of N or more. 2.10 let generation attend to the whole history, so everything
past the trained length turned into noise (docs/engine.md, "Context length").
2.11 attends to the newest `max_seq_len` positions by default.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auralite.engine import AuraLiteEngine, ModernTransformer  # noqa: E402
from auralite.engine.kv_cache import KVCache  # noqa: E402


def tiny(max_seq_len: int = 16, **kw) -> ModernTransformer:
    torch.manual_seed(0)
    return ModernTransformer(vocab_size=40, d_model=32, n_heads=4, n_layers=2, d_ff=64,
                             max_seq_len=max_seq_len, **kw).eval()


def full_causal_reference(model: ModernTransformer, x):
    """What 2.10 computed: every query attends to every earlier key."""
    saved = model.context_window
    model.set_context_window(None)
    try:
        with torch.no_grad():
            return model(x)
    finally:
        model.set_context_window(saved)


# ----------------------------------------------------------------------
#  The window in the model
# ----------------------------------------------------------------------
class TestModelWindow:
    def test_default_window_is_the_trained_length(self):
        model = tiny(16)
        assert model.context_window == 16
        assert all(a.context_window == 16 for a in model._attention_layers())
        assert model._attention_layers()[0].attention_window() == 16

    def test_alibi_models_get_the_window_too(self):
        # Measured (docs/engine.md): ALiBi degrades less past its trained
        # length than plain RoPE, but the window is still better.
        assert tiny(16, use_alibi=True).context_window == 16

    def test_a_trained_sliding_window_is_kept(self):
        model = tiny(16, sliding_window=8)
        assert model.context_window is None
        assert model._attention_layers()[0].attention_window() == 8

    def test_nothing_changes_inside_the_trained_length(self):
        model = tiny(16)
        x = torch.randint(0, 40, (2, 16))
        with torch.no_grad():
            windowed = model(x)
        assert torch.equal(windowed, full_causal_reference(model, x))

    def test_past_the_trained_length_queries_see_the_window_only(self):
        model = tiny(16)
        x = torch.randint(0, 40, (1, 60))
        with torch.no_grad():
            windowed = model(x)
            # Two layers of 16-key windows: the last output depends on the last
            # 2 x 15 + 1 tokens only, wherever the sequence starts (RoPE is
            # relative), so a forward over those 31 tokens gives the same logits.
            alone = model(x[:, -31:])
        reference = full_causal_reference(model, x)
        assert torch.allclose(windowed[:, :16], reference[:, :16], atol=1e-6)
        assert not torch.allclose(windowed[:, 20:], reference[:, 20:], atol=1e-3)
        assert torch.allclose(windowed[:, -1], alone[:, -1], atol=1e-4)

    def test_training_mode_ignores_the_window(self):
        model = tiny(16)
        x = torch.randint(0, 40, (1, 40))
        model.train()
        with torch.no_grad():
            trained_view = model(x)
        model.eval()
        assert torch.allclose(trained_view, full_causal_reference(model, x), atol=1e-6)

    def test_cached_decode_and_chunked_prefill_match_the_windowed_forward(self):
        model = tiny(16)
        x = torch.randint(0, 40, (1, 70))
        with torch.no_grad():
            reference = model(x)
            model.reset_cache()
            steps = [model(x[:, :1], start_pos=0, use_cache=True)]
            steps += [model(x[:, p:p + 1], start_pos=p, use_cache=True) for p in range(1, 70)]
            model.reset_cache()
            chunks = [model(x[:, s:s + 25], start_pos=s, use_cache=True) for s in range(0, 70, 25)]
            model.reset_cache()
        assert torch.allclose(torch.cat(steps, 1), reference, atol=1e-5)
        assert torch.allclose(torch.cat(chunks, 1), reference, atol=1e-5)

    def test_kv_cache_stays_bounded_past_the_window(self):
        def cache_bytes_after(model, n):
            with torch.no_grad():
                model.reset_cache()
                model(torch.zeros(1, 1, dtype=torch.long), start_pos=0, use_cache=True)
                for p in range(1, n):
                    model(torch.zeros(1, 1, dtype=torch.long), start_pos=p, use_cache=True)
            size = model.kv_cache_bytes()
            model.reset_cache()
            return size

        model = tiny(16)
        bounded = cache_bytes_after(model, 600)
        model.set_context_window(None)
        unbounded = cache_bytes_after(model, 600)
        assert bounded < unbounded / 2

    def test_set_context_window_validates_and_drops_caches(self):
        model = tiny(16)
        with torch.no_grad():
            model(torch.zeros(1, 4, dtype=torch.long), use_cache=True)
        assert model.cache_length == 4
        model.set_context_window(8)
        assert model.cache_length == 0 and model.context_window == 8
        model.set_context_window(0)
        assert model.context_window is None
        with pytest.raises(ValueError):
            model.set_context_window(-3)

    def test_rope_tables_grow_geometrically(self, monkeypatch):
        model = tiny(16)
        attn = model._attention_layers()[0]
        sizes = []
        real = type(attn)._build_rope_buffers

        def spy(self, seq_len):
            if self is attn:
                sizes.append(seq_len)
            return real(self, seq_len)

        monkeypatch.setattr(type(attn), "_build_rope_buffers", spy)
        with torch.no_grad():
            model.reset_cache()
            for p in range(300):
                model(torch.zeros(1, 1, dtype=torch.long), start_pos=p, use_cache=True)
        # 2.10 rebuilt the table for every position past 16 (284 rebuilds)
        assert len(sizes) <= 6 and attn.rope_cos.shape[0] >= 300


class TestExtendContext:
    def test_extension_raises_the_window_with_the_length(self):
        model = tiny(16)
        assert model.extend_context(64) is True
        assert model.max_seq_len == 64 and model.context_window == 64
        assert all(a.max_seq_len == 64 for a in model._attention_layers())
        assert model.extend_context(32) is False      # never shrinks

    def test_an_explicit_window_is_left_alone(self):
        model = tiny(16)
        model.set_context_window(8)
        model.extend_context(64)
        assert model.context_window == 8

    @pytest.mark.parametrize("scaling", [
        {"type": "ntk", "factor": 4.0},
        {"type": "dynamic_ntk", "factor": 2.0, "original_max_position_embeddings": 8},
        {"type": "yarn", "factor": 4.0},
        {"type": "linear", "factor": 2.0},
    ])
    def test_rope_frequencies_do_not_move(self, scaling):
        model = tiny(16, rope_scaling=dict(scaling))
        attn = model._attention_layers()[0]
        before = attn._rope_inv_freq_and_scale()
        model.extend_context(64)
        after = attn._rope_inv_freq_and_scale()
        assert torch.equal(before[0], after[0]) and before[1] == after[1]
        assert model.rope_scaling.get("original_max_position_embeddings") is not None

    def test_dynamic_ntk_is_static_and_equals_ntk(self):
        a = tiny(16, rope_scaling={"type": "ntk", "factor": 4.0})._attention_layers()[0]
        b = tiny(16, rope_scaling={"type": "dynamic_ntk", "factor": 4.0})._attention_layers()[0]
        assert torch.equal(a._rope_inv_freq_and_scale()[0], b._rope_inv_freq_and_scale()[0])


# ----------------------------------------------------------------------
#  Acceptance: a tiny deterministic model trained with seq_length=16
# ----------------------------------------------------------------------
PATTERN_WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
                 "nu xi omicron pi rho sigma tau upsilon phi chi psi omega").split()


def _corpus(n_words: int, seed: int) -> str:
    gen = torch.Generator().manual_seed(seed)
    picks = torch.randint(0, len(PATTERN_WORDS), (n_words,), generator=gen).tolist()
    return " ".join(PATTERN_WORDS[i] for i in picks) + " "


@pytest.fixture(scope="module")
def window16_engine():
    engine = AuraLiteEngine()
    engine.train(_corpus(6000, seed=1), {
        "tokenizer": "char", "d_model": 48, "d_ff": 128, "n_heads": 4, "n_layers": 2,
        "n_kv_heads": 4, "seq_length": 16, "batch_size": 16, "lr": 1e-2, "epochs": 2,
        "val_split": 0.0, "seed": 0, "optimizer": "adamw", "lr_schedule": "cosine",
        "amp_dtype": "none", "use_compile": False,
    })
    return engine


def _incremental_nll(engine: AuraLiteEngine, ids: list[int]):
    """Teacher-forced NLL per position through the KV cache, as generation runs."""
    model = engine.model
    model.eval()
    engine._sync_context_window()
    losses = []
    with torch.inference_mode():
        model.reset_cache()
        for pos in range(len(ids) - 1):
            logits = model(torch.tensor([[ids[pos]]]), start_pos=pos, use_cache=True)[0, -1]
            losses.append(float(torch.nn.functional.cross_entropy(
                logits.float().unsqueeze(0), torch.tensor([ids[pos + 1]]))))
        model.reset_cache()
    return torch.tensor(losses)


class TestWindowAcceptance:
    def test_nll_past_four_windows_stays_within_ten_percent(self, window16_engine):
        engine = window16_engine
        text = _corpus(400, seed=7)
        curves = []
        for start in range(0, 8 * 150, 150):
            ids = engine.encode(text[start:start + 129])
            curves.append(_incremental_nll(engine, ids))
        mean = torch.stack(curves).mean(0)
        in_window = float(mean[1:16].mean())
        beyond = float(mean[64:128].mean())        # positions past 4 x window
        assert beyond <= 1.10 * in_window, (in_window, beyond)

        engine.set_context_window("full")
        try:
            full_curves = [_incremental_nll(engine, engine.encode(text[s:s + 129]))
                           for s in range(0, 8 * 150, 150)]
        finally:
            engine.set_context_window("auto")
        full_beyond = float(torch.stack(full_curves).mean(0)[64:128].mean())
        # the 2.10 behaviour this release fixes
        assert full_beyond > 1.10 * beyond, (beyond, full_beyond)


# ----------------------------------------------------------------------
#  Every generation path follows the engine setting
# ----------------------------------------------------------------------
@pytest.fixture()
def cache_windows(monkeypatch):
    """Record the window of every KV cache the model creates."""
    seen: list[int | None] = []
    real_init = KVCache.__init__

    def spy(self, spec, window=None, *args, **kwargs):
        seen.append(window)
        real_init(self, spec, window, *args, **kwargs)

    monkeypatch.setattr(KVCache, "__init__", spy)
    return seen


class TestEnginePaths:
    @pytest.mark.parametrize("path", ["generate", "stream", "batch", "speculative", "chat",
                                      "chat_stream", "thinking"])
    def test_every_path_generates_with_the_window(self, fresh_trained_engine, cache_windows,
                                                  path):
        engine = fresh_trained_engine
        window = engine.model.max_seq_len
        assert engine.context_window == window

        def run():
            if path == "generate":
                engine.generate("The quick", 40, temperature=0.0, top_k=1)
            elif path == "stream":
                list(engine.generate_streaming("The quick", 40, temperature=0.0, top_k=1))
            elif path == "batch":
                engine.generate_batch(["The quick", "The quack"], 40, temperature=0.0, top_k=1)
            elif path == "speculative":
                engine.generate_speculative("The quick", 40, temperature=0.0, top_k=1)
            elif path == "thinking":
                engine.generate_with_thinking("The quick", 40, temperature=0.0)
            else:
                from auralite.engine import _legacy
                if not _legacy.HAS_CHAT_SUPPORT:
                    pytest.skip("chat support unavailable")
                messages = [{"role": "user", "content": "hi"}]
                if path == "chat":
                    engine.generate_chat(messages, max_new_tokens=40, temperature=0.0, top_k=1)
                else:
                    list(engine.generate_chat_streaming(messages, max_new_tokens=40,
                                                        temperature=0.0, top_k=1))

        run()
        assert cache_windows and set(cache_windows) == {window}
        cache_windows.clear()
        engine.set_context_window("full")
        run()
        assert cache_windows and set(cache_windows) == {None}

    def test_stream_batch_and_generate_agree_past_the_window(self, fresh_trained_engine):
        engine = fresh_trained_engine
        prompt, n = "The quick", 60                    # far past the 16-token window
        torch.manual_seed(5)
        plain = engine.generate(prompt, n, temperature=0.9, top_k=20)
        torch.manual_seed(5)
        streamed = prompt + "".join(engine.generate_streaming(prompt, n, temperature=0.9,
                                                              top_k=20))
        torch.manual_seed(5)
        batched = engine.generate_batch([prompt], n, temperature=0.9, top_k=20)[0]
        assert plain == streamed == batched

    def test_full_context_reproduces_the_2_10_computation(self, fresh_trained_engine):
        engine = fresh_trained_engine
        engine.set_context_window("full")
        ids = engine.encode("The quick")
        out = engine._generate_ids(ids, 40, temperature=0.0, top_k=1)
        # greedy reference: full causal attention recomputed from scratch
        model = engine.model
        reference = list(ids)
        with torch.no_grad():
            for _ in range(40):
                logits = full_causal_reference(model, torch.tensor([reference]))[0, -1]
                reference.append(int(torch.argmax(logits)))
        assert out == reference

    def test_long_prompts_are_prefilled_in_exact_chunks(self, fresh_trained_engine, monkeypatch):
        engine = fresh_trained_engine
        ids = (engine.encode("The quick brown fox. ") * 40)[:700]
        prompt = torch.tensor([ids])
        widths = []
        real_infer = engine._infer

        def spy(t, **kwargs):
            widths.append(int(t.shape[1]))
            return real_infer(t, **kwargs)

        monkeypatch.setattr(engine, "_infer", spy)
        engine.model.reset_cache()
        with torch.inference_mode():
            last = engine._prefill(prompt)[0, -1]
            engine.model.reset_cache()
            reference = engine.model(prompt)[0, -1]
        assert widths == [256, 256, 188]
        assert torch.allclose(last, reference, atol=1e-4)

    def test_a_capped_long_prompt_still_gets_every_requested_token(self, fresh_trained_engine):
        engine = fresh_trained_engine
        prompt = "abcd " * 1800                        # 9,000 tokens > the 8,192 cap
        ids = engine._generate_ids(engine._prepare_prompt_ids(prompt, max_gen_tokens=12), 12,
                                   temperature=0.0, top_k=1)
        assert len(ids) == 8191 + 12                   # 2.10 stopped after 1 token

    def test_the_setting_survives_loading_another_model(self, fresh_trained_engine,
                                                        trained_ckpt_path):
        engine = fresh_trained_engine
        engine.set_context_window("full")
        engine.load_model(trained_ckpt_path)
        assert engine.full_context and engine.context_window is None
        engine.generate("The", 3, temperature=0.0, top_k=1)
        assert engine.model.context_window is None
        engine.full_context = False
        assert engine.context_window == engine.model.max_seq_len

    @pytest.mark.parametrize("value, expected", [
        ("auto", "auto"), (None, "auto"), ("", "auto"), ("full", 0), (0, 0), ("none", 0),
        ("24", 24), (24, 24),
    ])
    def test_set_context_window_parsing(self, fresh_trained_engine, value, expected):
        engine = fresh_trained_engine
        engine.set_context_window(value)
        assert engine._context_window_setting == expected
        want = {"auto": engine.model.max_seq_len, 0: None}.get(expected, expected)
        assert engine.context_window == want == engine.model.context_window

    @pytest.mark.parametrize("value, error", [(-1, ValueError), ("-5", ValueError),
                                              ("wide", ValueError), (True, TypeError)])
    def test_set_context_window_rejects_nonsense(self, value, error):
        with pytest.raises(error):
            AuraLiteEngine().set_context_window(value)

    def test_no_model_means_no_window(self):
        engine = AuraLiteEngine()
        engine.set_context_window(32)            # stored for the next model
        assert engine.context_window is None


class TestContinueTrainingExtendsContext:
    def test_longer_windows_extend_the_saved_model(self, tmp_path):
        text = _corpus(1500, seed=3)
        params = {"tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_layers": 1,
                  "seq_length": 16, "batch_size": 8, "lr": 1e-2, "epochs": 1, "val_split": 0.0,
                  "seed": 0, "amp_dtype": "none", "use_compile": False}
        engine = AuraLiteEngine()
        engine.train(text, params)
        assert engine.context_window == 16
        engine.train(text, dict(params, seq_length=48, continue_training=True))
        assert engine.model.max_seq_len == 48 and engine.context_window == 48
        path = str(tmp_path / "extended.pt")
        engine.save_model(path)
        loaded = AuraLiteEngine()
        loaded.load_model(path)
        assert loaded.model.max_seq_len == 48 and loaded.context_window == 48


# ----------------------------------------------------------------------
#  Bits per byte
# ----------------------------------------------------------------------
class TestBitsPerByte:
    def test_char_training_reports_bits_per_byte(self):
        engine = AuraLiteEngine()
        summary = engine.train(_corpus(1200, seed=4), {
            "tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_layers": 1,
            "seq_length": 16, "batch_size": 8, "lr": 1e-2, "epochs": 1, "val_split": 0.1,
            "seed": 0, "amp_dtype": "none"})
        # ASCII: one byte per token, so bits per byte = loss / ln 2
        assert summary["val_bpb"] == pytest.approx(summary["val_loss"] / math.log(2), rel=1e-9)
        assert summary["best_val_bpb"] == pytest.approx(
            summary["best_val_loss"] / math.log(2), rel=1e-9)
        assert engine.last_val_bpb == summary["val_bpb"]

    def test_multibyte_tokens_are_counted_in_bytes(self):
        engine = AuraLiteEngine()
        text = "".join(("añb€c " if i % 2 else "xyz ") for i in range(800))
        engine.train(text, {
            "tokenizer": "bpe", "bpe_vocab_size": 64, "d_model": 32, "d_ff": 64, "n_heads": 4,
            "n_layers": 1, "seq_length": 16, "batch_size": 8, "lr": 1e-2, "epochs": 1,
            "val_split": 0.2, "seed": 0, "amp_dtype": "none"})
        table = engine._token_byte_lengths()
        assert table is not None
        for token_id in range(int(engine.vocab_size)):
            assert table[token_id] == len(engine.decode([token_id]).encode("utf-8"))
        summary = engine.last_train_summary
        assert summary is not None and summary["val_bpb"] < summary["val_loss"] / math.log(2)

    def test_eval_compute_bpb(self, trained_engine):
        from auralite import eval as auralite_eval

        text = "AuraLite trains small language models on CPU. " * 4   # in-vocabulary
        avg, n_pred = auralite_eval._compute_avg_token_loss(trained_engine, text)
        expected = avg * n_pred / math.log(2) / len(text[1:].encode("utf-8"))
        assert auralite_eval.compute_bpb(trained_engine, text) == pytest.approx(expected)
        assert auralite_eval.compute_bpb(trained_engine, "x") == float("inf")

    def test_eval_main_prints_every_metric_from_one_pass(self, trained_ckpt_path, tmp_path,
                                                         capsys, monkeypatch):
        from auralite import eval as auralite_eval

        test_file = tmp_path / "test.txt"
        test_file.write_text("The quick brown fox. " * 5, encoding="utf-8")
        passes = []
        real = auralite_eval._compute_avg_token_loss
        monkeypatch.setattr(auralite_eval, "_compute_avg_token_loss",
                            lambda *a, **k: passes.append(1) or real(*a, **k))
        assert auralite_eval.main(["--model", trained_ckpt_path, "--test", str(test_file),
                                   "--gen-length", "5"]) == 0
        out = capsys.readouterr().out
        assert "Bits-per-byte:" in out and "Perplexity:" in out
        assert len(passes) == 1


# ----------------------------------------------------------------------
#  Server: Ollama num_predict (P0.2) and the window setting
# ----------------------------------------------------------------------
fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")


@pytest.fixture()
def server(monkeypatch):
    from fastapi.testclient import TestClient

    import auralite.server.openai_server as srv
    from tests.test_server_openai import FakeEngine

    engine = FakeEngine()
    monkeypatch.setattr(srv, "_engine", engine)
    monkeypatch.setattr(srv, "_rate_bucket", {})
    monkeypatch.delenv("AURALITE_MAX_TOKENS", raising=False)
    with TestClient(srv.app, raise_server_exceptions=False) as client:
        yield client, engine


def _budget(engine, endpoint: str) -> int:
    if endpoint == "/api/generate":
        return engine.generation_kwargs[-1]["length"]
    return engine.chat_kwargs[-1]["max_new_tokens"]


def _ollama_body(endpoint: str, options: dict) -> dict:
    body: dict = {"stream": False, "options": options}
    if endpoint == "/api/generate":
        body["prompt"] = "hi"
    else:
        body["messages"] = [{"role": "user", "content": "hi"}]
    return body


@pytest.mark.parametrize("endpoint", ["/api/generate", "/api/chat"])
class TestOllamaNumPredict:
    @pytest.mark.parametrize("value, expected", [
        (-1, 8192), (-2, 8192), (10_000_000, 8192), (5, 5), (0, 0), (10.0, 10), (None, 128),
    ])
    def test_token_budget(self, server, endpoint, value, expected):
        client, engine = server
        r = client.post(endpoint, json=_ollama_body(endpoint, {"num_predict": value}))
        assert r.status_code == 200, r.text
        assert _budget(engine, endpoint) == expected

    def test_missing_num_predict_keeps_the_default(self, server, endpoint):
        client, engine = server
        assert client.post(endpoint, json=_ollama_body(endpoint, {})).status_code == 200
        assert _budget(engine, endpoint) == 128

    @pytest.mark.parametrize("value", ["abc", 1.5, True, -3, [4], "12"])
    def test_invalid_values_are_a_400_in_ollama_shape(self, server, endpoint, value):
        client, _ = server
        r = client.post(endpoint, json=_ollama_body(endpoint, {"num_predict": value}))
        assert r.status_code == 400
        body = r.json()
        assert isinstance(body["error"], str) and "num_predict" in body["error"]

    def test_the_cap_is_configurable(self, server, endpoint, monkeypatch):
        client, engine = server
        monkeypatch.setenv("AURALITE_MAX_TOKENS", "100")
        client.post(endpoint, json=_ollama_body(endpoint, {"num_predict": -1}))
        assert _budget(engine, endpoint) == 100
        client.post(endpoint, json=_ollama_body(endpoint, {"num_predict": 5000}))
        assert _budget(engine, endpoint) == 100


class TestOpenAIMaxTokensCap:
    @pytest.mark.parametrize("endpoint, body", [
        ("/v1/completions", {"prompt": "hi"}),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
    ])
    def test_cap_rejects_and_follows_the_env(self, server, monkeypatch, endpoint, body):
        client, _ = server
        assert client.post(endpoint, json={**body, "max_tokens": 8192}).status_code == 200
        assert client.post(endpoint, json={**body, "max_tokens": 8193}).status_code == 422
        monkeypatch.setenv("AURALITE_MAX_TOKENS", "100")
        assert client.post(endpoint, json={**body, "max_tokens": 100}).status_code == 200
        assert client.post(endpoint, json={**body, "max_tokens": 101}).status_code == 422
        monkeypatch.setenv("AURALITE_MAX_TOKENS", "not a number")
        assert client.post(endpoint, json={**body, "max_tokens": 8192}).status_code == 200


def test_server_engine_follows_auralite_context_window(monkeypatch):
    import auralite.server.openai_server as srv

    monkeypatch.setattr(srv, "_engine", None)
    monkeypatch.delenv("AURALITE_MODEL", raising=False)
    monkeypatch.setenv("AURALITE_CONTEXT_WINDOW", "full")
    assert srv.get_engine().full_context is True
    monkeypatch.setattr(srv, "_engine", None)
    monkeypatch.setenv("AURALITE_CONTEXT_WINDOW", "48")
    assert srv.get_engine()._context_window_setting == 48


# ----------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------
class TestCli:
    def _record(self, monkeypatch):
        calls = []
        real = AuraLiteEngine.set_context_window

        def spy(self, window="auto"):
            calls.append(window)
            return real(self, window)

        monkeypatch.setattr(AuraLiteEngine, "set_context_window", spy)
        return calls

    @pytest.mark.parametrize("flags, expected", [
        ([], "auto"), (["--full-context"], "full"), (["--context-window", "24"], "24"),
    ])
    def test_generate_applies_the_flag(self, trained_ckpt_path, monkeypatch, capsys,
                                       flags, expected):
        from auralite import cli

        calls = self._record(monkeypatch)
        assert cli.main(["generate", "--model", trained_ckpt_path, "--prompt", "The",
                         "--length", "20", *flags]) == 0
        assert calls == [expected]
        assert capsys.readouterr().out.strip()

    def test_chat_applies_the_flag(self, trained_ckpt_path, monkeypatch, capsys):
        from auralite import cli
        from auralite.engine import _legacy

        if not _legacy.HAS_CHAT_SUPPORT:
            pytest.skip("chat support unavailable")
        calls = self._record(monkeypatch)
        assert cli.main(["chat", "--model", trained_ckpt_path, "--prompt", "hi",
                         "--length", "8", "--full-context"]) == 0
        assert calls == ["full"]

    def test_flags_are_mutually_exclusive(self, trained_ckpt_path):
        from auralite import cli

        with pytest.raises(SystemExit):
            cli.main(["generate", "--model", trained_ckpt_path, "--prompt", "x",
                      "--full-context", "--context-window", "8"])

    def test_serve_hands_the_window_to_the_server(self, monkeypatch, trained_ckpt_path):
        from auralite import cli

        uvicorn = pytest.importorskip("uvicorn")
        monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
        # set first so that monkeypatch restores both variables afterwards
        monkeypatch.setenv("AURALITE_MODEL", "placeholder")
        monkeypatch.setenv("AURALITE_CONTEXT_WINDOW", "placeholder")
        monkeypatch.delenv("AURALITE_CONTEXT_WINDOW")
        assert cli.main(["serve", "--model", trained_ckpt_path, "--full-context"]) == 0
        assert os.environ["AURALITE_CONTEXT_WINDOW"] == "full"
        monkeypatch.setenv("AURALITE_CONTEXT_WINDOW", "32")       # env wins over "auto"
        assert cli.main(["serve", "--model", trained_ckpt_path]) == 0
        assert os.environ["AURALITE_CONTEXT_WINDOW"] == "32"

    def test_info_reports_the_generation_window(self, trained_ckpt_path, capsys):
        from auralite import cli

        assert cli.main(["info", "--model", trained_ckpt_path]) == 0
        assert "generation_window: 16 tokens" in capsys.readouterr().out

    def test_train_resume_extends_the_context(self, tmp_path, capsys):
        import json

        from auralite import cli

        corpus = tmp_path / "corpus.txt"
        corpus.write_text(_corpus(1500, seed=9), encoding="utf-8")
        params = {"tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_layers": 1,
                  "seq_length": 16, "batch_size": 8, "lr": 0.01, "epochs": 1,
                  "val_split": 0.1, "seed": 0}
        short = tmp_path / "short.json"
        short.write_text(json.dumps(params), encoding="utf-8")
        long = tmp_path / "long.json"
        long.write_text(json.dumps(dict(params, seq_length=48)), encoding="utf-8")
        first, second = tmp_path / "short.pt", tmp_path / "long.pt"
        assert cli.main(["train", "--quiet", "--text", str(corpus), "--params", str(short),
                         "--model", str(first)]) == 0
        assert "bits/byte" in capsys.readouterr().out
        assert cli.main(["train", "--quiet", "--text", str(corpus), "--params", str(long),
                         "--model", str(second), "--resume", str(first)]) == 0
        capsys.readouterr()
        assert cli.main(["info", "--model", str(second)]) == 0
        out = capsys.readouterr().out
        assert "max_seq_len: 48" in out and "generation_window: 48 tokens" in out
        assert cli.main(["train", "--quiet", "--text", str(corpus), "--params", str(long),
                         "--model", str(second), "--resume", str(tmp_path / "nope.pt")]) == 2


# ----------------------------------------------------------------------
#  The benchmark behind docs/engine.md section 9 keeps running
# ----------------------------------------------------------------------
def test_context_bench_runs(capsys, repo_root, tmp_path):
    import importlib.util
    import json

    path = os.path.join(repo_root, "benchmarks", "context_bench.py")
    spec = importlib.util.spec_from_file_location("context_bench_under_test", path)
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(_corpus(900, seed=11), encoding="utf-8")
    assert bench.main(["loss", "--json", "--text", str(corpus), "--seq", "8", "--epochs", "1",
                       "--d-model", "32", "--d-ff", "64", "--layers", "1", "--windows", "4",
                       "--batch-size", "8"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    losses = {row["variant"]: row for row in rows if "0-7" in row}
    assert set(losses) == {"window", "full", "ntk8", "yarn8", "alibi"}
    # inside the trained length the window changes nothing
    assert losses["window"]["0-7"] == losses["full"]["0-7"]
    assert set(losses["window"]) >= {"0-7", "8-15", "16-31", "32-63"}

    bench.ROWS.clear()
    assert bench.main(["speed", "--json", "--tokens", "24", "--repeats", "1"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert [row["mode"] for row in rows] == ["window", "full"]
    assert all(row["tokens_per_s"] > 0 and row["kv_cache_kib"] > 0 for row in rows)
    with pytest.raises(SystemExit):
        bench.main(["warp-drive"])
    with pytest.raises(SystemExit):
        bench.main(["loss"])                     # needs --text
