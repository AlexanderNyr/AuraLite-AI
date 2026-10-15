"""Concurrent generations on one engine (2.12).

Up to 2.11 the KV cache lived in the attention layers and was shared by
everything that generated with the model at the same time: four threads
decoding four greedy prompts returned 20 of 20 outputs that differed from the
same prompts decoded one after another, through the engine and through
``/v1/completions`` alike. Each generation now owns its cache (a
``DecodeSlot``) and the engine interleaves decode steps under its model lock.
"""
from __future__ import annotations

import concurrent.futures as cf
import sys
import threading
import time
import types

import pytest

torch = pytest.importorskip("torch")

from auralite.engine import AuraLiteEngine, GGUFModelProxy  # noqa: E402
from auralite.engine.decode_slots import DecodeSlot, attention_layers  # noqa: E402

TEXT = (
    "The quick brown fox jumps over the lazy dog near the river bank.\n"
    "A small grey cat sleeps on the warm window sill every afternoon.\n"
    "Seven old sailors sing loud songs about the distant northern sea.\n"
    "My neighbour bakes fresh bread with honey and walnuts on Sundays.\n"
) * 12
PROMPTS = ["The quick", "A small grey", "Seven old", "My neighbour"]
LENGTH = 48


@pytest.fixture(scope="module")
def race_engine():
    torch.manual_seed(0)
    engine = AuraLiteEngine()
    engine.train(TEXT, {
        "tokenizer": "char", "d_model": 48, "d_ff": 96, "n_heads": 4, "n_layers": 2,
        "seq_length": 32, "batch_size": 16, "lr": 5e-3, "epochs": 100, "max_steps": 80,
        "val_split": 0.0, "seed": 0, "use_qk_norm": False,
    })
    return engine


def _greedy(engine, prompt):
    return engine.generate(prompt, LENGTH, temperature=0.0, top_k=1, top_p=1.0)


def _mismatches(run_one, refs, rounds=5, workers=4):
    bad = []
    with cf.ThreadPoolExecutor(workers) as pool:
        for _ in range(rounds):
            futures = {pool.submit(run_one, p): p for p in PROMPTS}
            for future in cf.as_completed(futures):
                prompt = futures[future]
                out = future.result()
                if out != refs[prompt]:
                    bad.append((prompt, out))
    return bad


def test_references_are_distinct_and_reproducible(race_engine):
    refs = [_greedy(race_engine, p) for p in PROMPTS]
    assert refs == [_greedy(race_engine, p) for p in PROMPTS]
    # Different prompts continue differently, so a mixed-up cache shows.
    assert len({r[len(p):] for r, p in zip(refs, PROMPTS, strict=True)}) == len(PROMPTS)


def test_four_threads_match_the_sequential_results(race_engine):
    refs = {p: _greedy(race_engine, p) for p in PROMPTS}
    assert _mismatches(lambda p: _greedy(race_engine, p), refs) == []


def test_streams_and_batches_interleave_with_plain_generations(race_engine):
    refs = {p: _greedy(race_engine, p) for p in PROMPTS}

    def mixed(prompt):
        if prompt == PROMPTS[0]:
            return prompt + "".join(race_engine.generate_streaming(
                prompt, LENGTH, temperature=0.0, top_k=1, top_p=1.0))
        if prompt == PROMPTS[1]:
            return race_engine.generate_batch([prompt, PROMPTS[2]], LENGTH,
                                              temperature=0.0, top_k=1, top_p=1.0)[0]
        return _greedy(race_engine, prompt)

    assert _mismatches(mixed, refs, rounds=3) == []


def test_two_streams_advanced_alternately_in_one_thread(race_engine):
    a_ref = _greedy(race_engine, PROMPTS[0])[len(PROMPTS[0]):]
    b_ref = _greedy(race_engine, PROMPTS[1])[len(PROMPTS[1]):]
    a = race_engine.generate_streaming(PROMPTS[0], LENGTH, temperature=0.0, top_k=1, top_p=1.0)
    b = race_engine.generate_streaming(PROMPTS[1], LENGTH, temperature=0.0, top_k=1, top_p=1.0)
    got_a, got_b = [], []
    for x, y in zip(a, b, strict=False):
        got_a.append(x)
        got_b.append(y)
    got_a += list(a)
    got_b += list(b)
    assert "".join(got_a) == a_ref
    assert "".join(got_b) == b_ref


def test_an_abandoned_stream_blocks_nobody(race_engine):
    ref = _greedy(race_engine, PROMPTS[2])
    stale = race_engine.generate_streaming(PROMPTS[0], LENGTH, temperature=0.0, top_k=1)
    next(stale)                                  # suspended mid-generation, never closed
    result = {}
    worker = threading.Thread(target=lambda: result.setdefault("out", _greedy(race_engine,
                                                                                PROMPTS[2])))
    worker.start()
    worker.join(timeout=60)
    assert not worker.is_alive(), "a suspended stream kept the model locked"
    assert result["out"] == ref
    assert race_engine.model_lock.acquire(timeout=5)
    race_engine.model_lock.release()
    stale.close()


def test_the_model_holds_no_generation_state_between_steps(race_engine):
    stream = race_engine.generate_streaming(PROMPTS[0], 8, temperature=0.0, top_k=1)
    next(stream)
    layers = attention_layers(race_engine.model)
    assert layers and all(attn._cache is None for attn in layers)
    stream.close()


def test_a_slot_refuses_a_replaced_model(race_engine):
    slot = DecodeSlot(16)
    slot.bind(race_engine.model)
    other = AuraLiteEngine()
    other.train(TEXT[:400], {"tokenizer": "char", "d_model": 16, "d_ff": 32, "n_heads": 2,
                             "n_layers": 1, "seq_length": 16, "epochs": 1, "val_split": 0.0})
    with pytest.raises(RuntimeError, match="replaced"):
        slot.bind(other.model)


def test_concurrent_completions_over_http(race_engine):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import auralite.server.openai_server as srv

    previous = srv._engine
    srv._engine = race_engine
    try:
        client = TestClient(srv.app)

        def via_http(prompt):
            r = client.post("/v1/completions", json={
                "model": "auralite", "prompt": prompt, "max_tokens": LENGTH,
                "temperature": 0.0, "top_p": 1.0})
            r.raise_for_status()
            return r.json()["choices"][0]["text"]

        refs = {p: via_http(p) for p in PROMPTS}
        assert _mismatches(via_http, refs) == []
    finally:
        srv._engine = previous


# ----------------------------------------------------------------------
#  GGUF: one generation at a time on the llama.cpp context
# ----------------------------------------------------------------------
class _SlowLlama:
    """Records how many calls run inside llama.cpp at the same time."""

    def __init__(self, **kwargs):
        self.metadata = {}
        self.active = 0
        self.peak = 0
        self.calls = []
        self.lock = threading.Lock()

    def n_vocab(self):
        return 10

    def tokenize(self, data, add_bos=False, special=True):
        return list(data)

    def detokenize(self, ids):
        return bytes(ids)

    def _enter(self, kwargs):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls.append(kwargs)

    def _leave(self):
        with self.lock:
            self.active -= 1

    def create_completion(self, **kwargs):
        if kwargs.get("stream"):
            return self._stream(kwargs)
        self._enter(kwargs)
        time.sleep(0.02)
        self._leave()
        return {"choices": [{"text": " ok"}]}

    def _stream(self, kwargs):
        self._enter(kwargs)
        try:
            for piece in ("a", "b", "c"):
                time.sleep(0.01)
                yield {"choices": [{"text": piece}]}
        finally:
            self._leave()

    create_chat_completion = create_completion


@pytest.fixture()
def slow_gguf(monkeypatch):
    module = types.ModuleType("llama_cpp")
    module.Llama = _SlowLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    return GGUFModelProxy("model.gguf")


def test_gguf_generations_take_turns(slow_gguf):
    def one(i):
        if i % 2:
            return "".join(c["choices"][0]["text"]
                           for c in slow_gguf.create_completion("p", stream=True))
        return slow_gguf.create_completion("p")["choices"][0]["text"]

    with cf.ThreadPoolExecutor(4) as pool:
        outs = list(pool.map(one, range(12)))
    assert outs == [" ok", "abc"] * 6
    assert slow_gguf.llama.peak == 1


def test_an_abandoned_gguf_stream_releases_the_context(slow_gguf):
    stream = slow_gguf.create_completion("p", stream=True)
    next(stream)                                 # the stream holds the context now
    assert not slow_gguf.generation_lock.acquire(blocking=False)
    stream.close()                               # abandoned: closing releases it
    assert slow_gguf.create_completion("p")["choices"][0]["text"] == " ok"


def test_gguf_passes_presence_and_frequency_penalties(slow_gguf):
    slow_gguf.create_completion("p", presence_penalty=0.5, frequency_penalty=0.25)
    assert slow_gguf.llama.calls[-1]["presence_penalty"] == 0.5
    assert slow_gguf.llama.calls[-1]["frequency_penalty"] == 0.25
    slow_gguf.create_completion("p")
    assert "presence_penalty" not in slow_gguf.llama.calls[-1]


def test_engine_gguf_generation_with_penalties(slow_gguf):
    # 2.11 raised TypeError: the proxy did not accept the penalties.
    engine = AuraLiteEngine()
    engine.model = slow_gguf
    engine.tokenizer = slow_gguf.tokenizer
    out = engine.generate("hi", 5, presence_penalty=0.5, frequency_penalty=0.5)
    assert out == "hi ok"
