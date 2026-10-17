"""Batched decoding (2.14): concurrent generations share one forward pass.

A lone generation must compute exactly what 2.13 computed (bit for bit), and
generations decoding together must return what each returns alone (greedy, on
the test model). Rows are released when a generation ends, a full batch
falls back to the 2.13 path, and a replaced model is refused.
"""
from __future__ import annotations

import copy
import queue
import threading
import time

import pytest

torch = pytest.importorskip("torch")

from auralite.engine import AuraLiteEngine  # noqa: E402
from auralite.engine import batching  # noqa: E402
from auralite.engine.batching import DecodeBatch, batchable_model, max_batch_rows  # noqa: E402
from auralite.engine.compute_thread import ComputeThread, run_on_compute_thread  # noqa: E402
from auralite.engine.decode_slots import DecodeSlot  # noqa: E402

TEXT = (
    "The quick brown fox jumps over the lazy dog near the river bank.\n"
    "A small grey cat sleeps on the warm window sill every afternoon.\n"
    "Seven old sailors sing loud songs about the distant northern sea.\n"
    "My neighbour bakes fresh bread with honey and walnuts on Sundays.\n"
) * 12
PROMPTS = ["The quick", "A small grey", "Seven old sailors", "My"]
GREEDY = dict(temperature=0.0, top_k=1, top_p=1.0)
BASE = {"tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_layers": 2,
        "seq_length": 32, "batch_size": 16, "lr": 5e-3, "epochs": 100, "max_steps": 40,
        "val_split": 0.0, "seed": 0}

#: (name, extra train params): every attention variant the batched step repeats.
VARIANTS = [
    ("plain", {}),
    ("gqa", {"n_kv_heads": 2}),
    ("sliding_window", {"sliding_window": 12}),
    ("alibi", {"use_alibi": True}),
    ("int8_cache", {"kv_cache_dtype": "int8"}),
    ("fp16_cache", {"kv_cache_dtype": "fp16"}),
    ("qk_norm", {"use_qk_norm": True}),
    ("moe", {"use_moe": True, "num_experts": 2}),
    ("alibi_window_gqa", {"use_alibi": True, "sliding_window": 12, "n_kv_heads": 2}),
]

_ENGINES: dict[str, AuraLiteEngine] = {}


def _engine(name: str = "plain") -> AuraLiteEngine:
    if name not in _ENGINES:
        torch.manual_seed(0)
        engine = AuraLiteEngine()
        engine.train(TEXT, {**BASE, **dict(VARIANTS)[name]})
        _ENGINES[name] = engine
    return _ENGINES[name]


@pytest.fixture()
def engine() -> AuraLiteEngine:
    return _engine()


class _Counter:
    """Counts batched forward passes and the rows they decoded."""

    def __init__(self, monkeypatch) -> None:
        self.passes: list[int] = []
        original = DecodeBatch.forward

        def counting(batch, items):
            self.passes.append(len(items))
            return original(batch, items)

        monkeypatch.setattr(DecodeBatch, "forward", counting)


def _recorded_logits(engine, prompt, length):
    seen: list = []
    original = engine._sample_token

    def recording(logits, *args, **kwargs):
        seen.append(logits.detach().clone())
        return original(logits, *args, **kwargs)

    engine._sample_token = recording
    try:
        out = engine.generate(prompt, length, **GREEDY)
    finally:
        del engine._sample_token
    return out, seen


# ----------------------------------------------------------------------
#  One row is the 2.13 computation
# ----------------------------------------------------------------------
@pytest.mark.parametrize("variant", [name for name, _ in VARIANTS])
def test_one_row_matches_the_unbatched_path_bit_for_bit(variant, monkeypatch):
    engine = _engine(variant)
    # long enough to run past the context window (32) and the batch capacity
    monkeypatch.setenv("AURALITE_MAX_BATCH", "1")
    ref_text, ref_logits = _recorded_logits(engine, PROMPTS[1], 150)
    monkeypatch.setenv("AURALITE_MAX_BATCH", "4")
    counter = _Counter(monkeypatch)
    text, logits = _recorded_logits(engine, PROMPTS[1], 150)
    assert counter.passes and set(counter.passes) == {1}, "the batch was not used"
    assert text == ref_text
    assert len(logits) == len(ref_logits)
    assert all(torch.equal(a, b) for a, b in zip(logits, ref_logits, strict=True))


# ----------------------------------------------------------------------
#  Several rows: the same greedy output as alone
# ----------------------------------------------------------------------
def _together(engine, prompts, length):
    """Run `prompts` on threads that start their decode steps together."""
    results: dict[str, str] = {}
    start = threading.Barrier(len(prompts))

    def run(prompt):
        start.wait()
        results[prompt] = engine.generate(prompt, length, **GREEDY)

    threads = [threading.Thread(target=run, args=(p,)) for p in prompts]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads)
    return results


@pytest.mark.parametrize("variant", ["plain", "alibi_window_gqa", "int8_cache", "moe"])
def test_concurrent_generations_match_their_lone_results(variant, monkeypatch):
    engine = _engine(variant)
    refs = {p: engine.generate(p, 90, **GREEDY) for p in PROMPTS}
    counter = _Counter(monkeypatch)
    for _ in range(3):
        assert _together(engine, PROMPTS, 90) == refs
    assert max(counter.passes) > 1, "the generations never shared a forward pass"


def test_rows_with_gaps_and_any_order_give_the_same_logits(engine):
    """Rows 0 and 2 step while row 1 idles; items in any order."""
    engine.model.eval()
    engine._sync_context_window()

    def run():
        batch = None
        rows, lengths = [], [5, 9, 13]
        for n in lengths:
            slot = DecodeSlot(n + 20)
            with engine._decode_step(slot), torch.inference_mode():
                engine._prefill(torch.arange(1, n + 1).view(1, n) % 20 + 1)
            batch = batch or engine._new_decode_batch(engine.model, 4, slot.caches)
            rows.append(batch.admit(slot.caches, slot.capacity))
        assert batch is not None and rows == [0, 1, 2]
        both = batch.forward([(2, 3, 13), (0, 4, 5)])           # unsorted, row 1 idle
        return batch, both

    batch, both = run_on_compute_thread(run)
    assert batch.length == [6, 9, 14, 0]
    # the same tokens one row at a time, on fresh rows
    def alone():
        out = []
        for n, token in ((13, 3), (5, 4)):
            slot = DecodeSlot(n + 20)
            with engine._decode_step(slot), torch.inference_mode():
                engine._prefill(torch.arange(1, n + 1).view(1, n) % 20 + 1)
            single = engine._new_decode_batch(engine.model, 1, slot.caches)
            row = single.admit(slot.caches, slot.capacity)
            out.append(single.forward([(row, token, n)])[0])
        return out

    ref = run_on_compute_thread(alone)
    assert torch.allclose(both[0], ref[0], atol=1e-5)
    assert torch.allclose(both[1], ref[1], atol=1e-5)


# ----------------------------------------------------------------------
#  Rows: release, fallback, replaced model
# ----------------------------------------------------------------------
def test_an_ended_or_abandoned_generation_releases_its_row(engine):
    stream = engine.generate_streaming(PROMPTS[0], 30, **GREEDY)
    next(stream)
    next(stream)                                   # now decoding in a row
    batch = engine.__dict__["_decode_batch"]
    assert batch.active_rows() == 1 and batch.storage is not None
    stream.close()                                 # abandoned
    assert batch.active_rows() == 0
    assert batch.storage is None and batch.model is None   # memory and model let go
    engine.generate(PROMPTS[0], 10, **GREEDY)      # a new generation, a new batch
    assert engine.__dict__["_decode_batch"].active_rows() == 0


def test_a_full_batch_falls_back_to_the_unbatched_path(engine, monkeypatch):
    monkeypatch.setenv("AURALITE_MAX_BATCH", "2")
    refs = [engine.generate(p, 40, **GREEDY)[len(p):] for p in PROMPTS[:3]]
    streams = [engine.generate_streaming(p, 40, **GREEDY) for p in PROMPTS[:3]]
    got = [[next(s)] for s in streams]             # three prompts in
    got = [g + [next(s)] for g, s in zip(got, streams, strict=True)]
    batch = engine.__dict__["_decode_batch"]
    assert batch.rows == 2 and batch.active_rows() == 2   # the third decodes alone
    for g, s in zip(got, streams, strict=True):
        g.extend(s)
    assert ["".join(g) for g in got] == refs
    assert batch.active_rows() == 0


def test_a_replaced_model_stops_a_batched_generation():
    eng = AuraLiteEngine()
    eng.train(TEXT, BASE)
    other = AuraLiteEngine()
    other.train(TEXT[:400], {"tokenizer": "char", "d_model": 16, "d_ff": 32, "n_heads": 2,
                             "n_layers": 1, "seq_length": 16, "epochs": 1, "val_split": 0.0})
    stream = eng.generate_streaming(PROMPTS[0], 30, **GREEDY)
    next(stream)
    next(stream)
    batch = eng.__dict__["_decode_batch"]
    assert batch.active_rows() == 1
    eng.model = other.model
    with pytest.raises(RuntimeError, match="replaced"):
        next(stream)
    assert batch.active_rows() == 0


def test_without_the_compute_thread_there_is_no_batch(engine, monkeypatch):
    ref = engine.generate(PROMPTS[2], 40, **GREEDY)
    engine.__dict__.pop("_decode_batch", None)
    monkeypatch.setenv("AURALITE_COMPUTE_THREAD", "0")
    assert engine.generate(PROMPTS[2], 40, **GREEDY) == ref
    assert "_decode_batch" not in engine.__dict__


def test_generate_batch_is_one_batch_with_the_same_greedy_output(engine, monkeypatch):
    prompts = ["The", "A small grey", "The", "Seven old sailors sing"]
    refs = [engine.generate(p, 40, **GREEDY) for p in prompts]
    counter = _Counter(monkeypatch)
    assert engine.generate_batch(prompts, 40, **GREEDY) == refs
    # 3 unique prompts of 3 lengths decode together: 39 passes of 3 rows
    assert counter.passes == [3] * 39


def test_generate_batch_of_equal_lengths_is_the_213_batch(engine):
    prompts = ["The quick", "A small g", "Seven old", "My neighb"]
    ids = [engine._prepare_prompt_ids(p, max_gen_tokens=40) for p in prompts]
    sampled = dict(temperature=0.9, top_k=10, top_p=0.9)
    torch.manual_seed(3)
    new = engine.generate_batch(prompts, 40, **sampled)
    torch.manual_seed(3)
    old = engine._generate_equal_length(ids, 40, engine._sampling_params(**sampled))
    assert new == [p + engine.decode(o)[len(engine.decode(i)):]
                   for p, i, o in zip(prompts, ids, old, strict=True)]


# ----------------------------------------------------------------------
#  Which models batch
# ----------------------------------------------------------------------
def test_max_batch_rows_reads_the_environment(monkeypatch):
    monkeypatch.delenv("AURALITE_MAX_BATCH", raising=False)
    assert max_batch_rows() == 4
    for raw, rows in (("1", 1), ("0", 1), ("8", 8), ("1000", 64), ("many", 4), (" 2 ", 2)):
        monkeypatch.setenv("AURALITE_MAX_BATCH", raw)
        assert max_batch_rows() == rows


class _ForeignLinear(torch.nn.Linear):
    pass


def test_batchable_model_refuses_what_it_cannot_run(engine):
    model = engine.model.eval()
    assert batchable_model(model)
    assert not batchable_model(None)
    model.train()
    assert not batchable_model(model)
    model.eval()
    handle = model.register_forward_hook(lambda *_: None)
    assert not batchable_model(model)
    handle.remove()
    handle = model.layers[0].attn.register_forward_pre_hook(lambda *_: None)
    assert not batchable_model(model)
    handle.remove()
    # A module from outside torch.nn and AuraLite (a quantized linear layer,
    # say) may mix the rows: refused.
    foreign = copy.deepcopy(model)
    old = foreign.layers[0].attn.W_q
    foreign.layers[0].attn.W_q = _ForeignLinear(old.in_features, old.out_features, bias=False)
    assert not batchable_model(foreign)

    class Custom(type(model)):
        def forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs)

    custom = Custom(**engine._model_arch_kwargs(model)).eval()
    assert not batchable_model(custom)
    assert batchable_model(model)


# ----------------------------------------------------------------------
#  The compute thread gathers the steps of a group
# ----------------------------------------------------------------------
class _Group:
    """A stand-in DecodeBatch: records what it is asked to run."""

    def __init__(self, wait: float = 0.0, fail: bool = False) -> None:
        self.calls: list[list] = []
        self.wait = wait
        self.fail = fail
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()

    def batch_wait(self, jobs):
        return self.wait

    def run_jobs(self, jobs):
        self.entered.set()
        self.gate.wait(10)
        self.calls.append([job.item for job in jobs])
        if self.fail:
            raise ValueError("step failed")
        for job in jobs:
            job.result = ("done", job.item)


def _submit_all(local, group, items):
    out: dict = {}

    def run(item):
        try:
            out[item] = local.run_in_group(group, item)
        except BaseException as exc:  # noqa: BLE001
            out[item] = exc

    threads = [threading.Thread(target=run, args=(i,)) for i in items]
    for t in threads:
        t.start()
    return out, threads


def test_queued_steps_of_a_group_run_as_one_call():
    local = ComputeThread("test-group")
    group = _Group()
    try:
        blocker = threading.Event()
        local.run(lambda: None)                         # thread started
        hold = threading.Thread(target=lambda: local.run(blocker.wait, 10))
        hold.start()                                    # keeps the thread busy...
        time.sleep(0.05)
        out, threads = _submit_all(local, group, ["a", "b", "c"])
        time.sleep(0.1)                                 # ...while three steps queue up
        blocker.set()
        for t in threads + [hold]:
            t.join(10)
        assert sorted(group.calls[0]) == ["a", "b", "c"] and len(group.calls) == 1
        assert out == {i: ("done", i) for i in "abc"}
    finally:
        local.stop()


def test_the_group_waits_for_late_steps_but_not_for_other_work():
    local = ComputeThread("test-wait")
    group = _Group(wait=0.5)
    try:
        local.run(lambda: None)
        out, threads = _submit_all(local, group, ["early"])
        time.sleep(0.05)
        out2, threads2 = _submit_all(local, group, ["late"])
        for t in threads + threads2:
            t.join(10)
        assert group.calls == [["early", "late"]]
        # other work ends the wait at once
        started = time.monotonic()
        out3, threads3 = _submit_all(local, group, ["x"])
        time.sleep(0.05)
        assert local.run(lambda: "other") == "other"
        for t in threads3:
            t.join(10)
        assert time.monotonic() - started < 0.4
        assert out3 == {"x": ("done", "x")}
    finally:
        local.stop()


def test_an_error_reaches_every_step_of_the_call():
    local = ComputeThread("test-fail")
    group = _Group(fail=True)
    try:
        local.run(lambda: None)
        out, threads = _submit_all(local, group, ["a"])
        for t in threads:
            t.join(10)
        assert isinstance(out["a"], ValueError)
        assert local.run(lambda: 7) == 7                # the thread survived
    finally:
        local.stop()


class _InterruptAfterPut:
    """A job queue whose put() is followed by Ctrl+C (see test_compute_thread_v213)."""

    def __init__(self, jobs, before_raise=lambda: None) -> None:
        self.jobs = jobs
        self.before_raise = before_raise

    def put(self, job):
        self.jobs.put(job)
        self.before_raise()
        raise KeyboardInterrupt

    def __getattr__(self, name):
        return getattr(self.jobs, name)


def test_ctrl_c_drops_a_step_that_has_not_started():
    local = ComputeThread("test-ctrl-c")
    group = _Group()
    try:
        local.run(lambda: None)
        group.gate.clear()
        busy = threading.Thread(target=lambda: local.run_in_group(group, "first"))
        busy.start()                                    # the thread sits in run_jobs
        time.sleep(0.05)
        jobs = local._jobs
        local._jobs = _InterruptAfterPut(jobs)          # type: ignore[assignment]
        try:
            with pytest.raises(KeyboardInterrupt):
                local.run_in_group(group, "dropped")
        finally:
            local._jobs = jobs
        group.gate.set()
        busy.join(10)
        assert local.run(lambda: 1) == 1
        assert group.calls == [["first"]]               # "dropped" never ran
    finally:
        local.stop()


def test_ctrl_c_waits_for_a_step_in_flight():
    local = ComputeThread("test-ctrl-c-wait")
    group = _Group()
    try:
        local.run(lambda: None)
        group.gate.clear()
        jobs = local._jobs
        local._jobs = _InterruptAfterPut(                # type: ignore[assignment]
            jobs, lambda: group.entered.wait(10))       # Ctrl+C once the step runs
        released = threading.Timer(0.2, group.gate.set)
        released.start()
        started = time.monotonic()
        try:
            with pytest.raises(KeyboardInterrupt):
                local.run_in_group(group, "running")
        finally:
            local._jobs = jobs
        # the step had started: the caller waited for it before raising
        assert group.calls == [["running"]]
        assert time.monotonic() - started >= 0.15
    finally:
        local.stop()


def test_the_engine_batch_gathers_steps_on_the_real_thread(engine, monkeypatch):
    """Four streams advanced by four threads: most passes carry several rows."""
    counter = _Counter(monkeypatch)
    done: queue.Queue = queue.Queue()

    def run(prompt):
        done.put(engine.generate(prompt, 60, **GREEDY))

    threads = [threading.Thread(target=run, args=(p,)) for p in PROMPTS]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert done.qsize() == 4
    rows = sum(counter.passes)
    assert rows == 4 * 59
    assert len(counter.passes) < rows                  # some passes were shared
    assert batching.BATCH_WAIT_S <= 0.005
