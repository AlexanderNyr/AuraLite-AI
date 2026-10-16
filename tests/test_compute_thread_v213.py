"""The compute thread of 2.13 (``auralite/engine/compute_thread.py``).

Every generation forward pass runs on one thread, ``auralite-compute``,
whichever thread calls the engine, so the process keeps one OpenMP team.
These tests pin the contract: same outputs, exceptions and grad modes come
through, calls on the thread itself run inline, Ctrl+C waits for the step in
flight and drops a call that has not started, a forked child starts a thread
of its own, and ``AURALITE_COMPUTE_THREAD=0`` turns it off.
"""
from __future__ import annotations

import _thread
import os
import signal
import sys
import threading
import time
import traceback
import warnings

import pytest

torch = pytest.importorskip("torch")

from auralite.engine import AuraLiteEngine  # noqa: E402
from auralite.engine import compute_thread as ct  # noqa: E402

TEXT = ("the quick brown fox jumps over the lazy dog near the river bank.\n"
        "a small grey cat sleeps on the warm window sill every afternoon.\n") * 12
PARAMS = {"tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 2, "n_layers": 1,
          "seq_length": 32, "batch_size": 8, "epochs": 1, "max_steps": 8, "val_split": 0.0,
          "seed": 0, "progress_every_seconds": 1e9}


@pytest.fixture(scope="module")
def engine():
    eng = AuraLiteEngine()
    eng.train(TEXT, PARAMS)
    return eng


def _on_thread(fn):
    box = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc
    t = threading.Thread(target=target)
    t.start()
    t.join(60)
    if "error" in box:
        raise box["error"]
    return box["value"]


# ----------------------------------------------------------------------
#  The thread itself
# ----------------------------------------------------------------------
def test_calls_run_on_the_compute_thread_and_return_their_value():
    assert ct.run_on_compute_thread(lambda: threading.current_thread().name) == "auralite-compute"
    assert ct.run_on_compute_thread(lambda a, b=0: a + b, 2, b=3) == 5
    assert ct.compute_thread.alive


def test_exceptions_keep_their_type_and_traceback():
    def boom():
        raise KeyError("missing")

    with pytest.raises(KeyError, match="missing") as info:
        ct.run_on_compute_thread(boom)
    frames = [f.name for f in traceback.extract_tb(info.value.__traceback__)]
    assert "boom" in frames


def test_grad_and_inference_modes_go_along():
    def modes():
        return torch.is_inference_mode_enabled(), torch.is_grad_enabled()

    assert ct.run_on_compute_thread(modes) == (False, True)
    with torch.no_grad():
        assert ct.run_on_compute_thread(modes) == (False, False)
    with torch.inference_mode():
        assert ct.run_on_compute_thread(modes) == (True, False)
        out = ct.run_on_compute_thread(lambda: torch.ones(2) * 2)
    assert out.is_inference()


def test_a_call_from_the_compute_thread_runs_inline():
    def outer():
        return ct.run_on_compute_thread(lambda: threading.current_thread().name)

    assert ct.run_on_compute_thread(outer) == "auralite-compute"


def test_calls_from_many_threads_are_served_in_turn():
    names = []
    lock = threading.Lock()

    def work(i):
        name = ct.run_on_compute_thread(lambda: threading.current_thread().name)
        with lock:
            names.append(name)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert names == ["auralite-compute"] * 8


def test_run_inline_keeps_calls_on_the_calling_thread():
    with ct.run_inline():
        assert ct.run_on_compute_thread(threading.current_thread) is threading.current_thread()
    assert ct.run_on_compute_thread(threading.current_thread) is not threading.current_thread()


def test_opt_out_runs_everything_inline(monkeypatch):
    monkeypatch.setenv("AURALITE_COMPUTE_THREAD", "0")
    assert not ct.compute_thread_enabled()
    assert ct.run_on_compute_thread(threading.current_thread) is threading.current_thread()
    assert ct.release_caller_pool() is False


@pytest.mark.skipif(threading.current_thread() is not threading.main_thread(),
                    reason="needs the main thread")
def test_ctrl_c_waits_for_the_step_in_flight():
    started, seen, finished = threading.Event(), threading.Event(), threading.Event()

    def on_sigint(signum, frame):        # the default handler, plus a note that it ran
        seen.set()
        raise KeyboardInterrupt

    def slow_step():
        started.set()
        seen.wait(10)                    # the caller has its Ctrl+C...
        time.sleep(0.2)                  # ...while this step goes on for a while
        finished.set()

    def press_ctrl_c():
        started.wait(10)
        _thread.interrupt_main()

    previous = signal.signal(signal.SIGINT, on_sigint)
    try:
        threading.Thread(target=press_ctrl_c, daemon=True).start()
        with pytest.raises(KeyboardInterrupt):
            ct.run_on_compute_thread(slow_step)
    finally:
        signal.signal(signal.SIGINT, previous)
    assert seen.is_set() and finished.is_set()
    assert ct.run_on_compute_thread(lambda: 7) == 7      # still serving


class _InterruptAfterPut:
    """Stands in for the job queue: Ctrl+C lands right after a job is handed over."""

    def __init__(self, jobs, before_raise):
        self.jobs, self.before_raise = jobs, before_raise

    def put(self, job):
        self.jobs.put(job)
        self.before_raise()
        raise KeyboardInterrupt


def test_ctrl_c_right_after_the_hand_off_still_waits_for_the_step():
    local = ct.ComputeThread(name="auralite-test")
    local.run(lambda: None)
    started, finished = threading.Event(), threading.Event()

    def step():
        started.set()
        time.sleep(0.2)
        finished.set()

    local._jobs = _InterruptAfterPut(local._jobs, lambda: started.wait(10))
    try:
        with pytest.raises(KeyboardInterrupt):
            local.run(step)
        assert finished.is_set()
    finally:
        local._jobs = local._jobs.jobs
        local.stop()


def test_ctrl_c_drops_a_call_that_has_not_started():
    local = ct.ComputeThread(name="auralite-test")
    busy, release, ran = threading.Event(), threading.Event(), []

    def occupy():
        busy.set()
        release.wait(10)

    blocker = threading.Thread(target=local.run, args=(occupy,), daemon=True)
    blocker.start()
    assert busy.wait(10)
    local._jobs = _InterruptAfterPut(local._jobs, lambda: None)
    try:
        with pytest.raises(KeyboardInterrupt):     # at once: the thread is still busy
            local.run(lambda: ran.append(1))
    finally:
        local._jobs = local._jobs.jobs
        release.set()
    blocker.join(10)
    assert local.run(lambda: 5) == 5               # later calls are served...
    assert ran == []                               # ...and the dropped one never ran
    local.stop()


@pytest.mark.skipif(not hasattr(os, "fork") or sys.platform == "darwin",
                    reason="needs fork()")
def test_a_forked_child_starts_its_own_thread():
    ct.run_on_compute_thread(lambda: None)
    parent_thread = ct.compute_thread._thread
    read_fd, write_fd = os.pipe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)   # "multi-threaded fork"
        pid = os.fork()
    if pid == 0:                                           # pragma: no cover - child
        code = 1
        try:
            fresh = not ct.compute_thread.alive
            name = ct.run_on_compute_thread(lambda: threading.current_thread().name)
            other = ct.compute_thread._thread is not parent_thread
            code = 0 if (fresh and other and name == "auralite-compute") else 2
        finally:
            os.write(write_fd, bytes([code]))
            os._exit(code)
    os.close(write_fd)
    result = os.read(read_fd, 1)
    os.close(read_fd)
    os.waitpid(pid, 0)
    assert result == b"\x00"


def test_a_stopped_thread_starts_again():
    local = ct.ComputeThread(name="auralite-test")
    assert local.run(lambda: threading.current_thread().name) == "auralite-test"
    local.stop()
    assert not local.alive
    assert local.run(lambda: 3) == 3


def test_release_openmp_pool_is_best_effort():
    assert isinstance(ct.release_openmp_pool(), bool)
    assert isinstance(ct.release_compute_pool(), bool)
    if sys.platform.startswith("linux"):
        torch.ones(1 << 16).sum()                          # use this thread's team
        assert ct.release_openmp_pool() is True


# ----------------------------------------------------------------------
#  The engine
# ----------------------------------------------------------------------
def test_generation_forwards_run_on_the_compute_thread(engine):
    seen = set()
    handle = engine.model.register_forward_hook(
        lambda *_: seen.add(threading.current_thread().name))
    try:
        engine.generate("the quick", 5, temperature=0.0, top_k=1)
        _on_thread(lambda: engine.generate("the quick", 5, temperature=0.0, top_k=1))
    finally:
        handle.remove()
    assert seen == {"auralite-compute"}


def test_outputs_match_the_opt_out(engine, monkeypatch):
    greedy = dict(temperature=0.0, top_k=1)
    ref_greedy = engine.generate("a small", 40, **greedy)
    torch.manual_seed(5)
    ref_seeded = engine.generate("a small", 40, temperature=0.9, top_k=20)
    assert _on_thread(lambda: engine.generate("a small", 40, **greedy)) == ref_greedy

    def seeded():
        torch.manual_seed(5)
        return engine.generate("a small", 40, temperature=0.9, top_k=20)
    assert _on_thread(seeded) == ref_seeded
    monkeypatch.setenv("AURALITE_COMPUTE_THREAD", "0")
    assert engine.generate("a small", 40, **greedy) == ref_greedy
    assert seeded() == ref_seeded


def test_an_error_in_the_forward_reaches_the_caller(engine, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("forward failed")

    monkeypatch.setattr(engine.model, "forward", broken)
    with pytest.raises(RuntimeError, match="forward failed"):
        _on_thread(lambda: engine.generate("the", 3, temperature=0.0, top_k=1))


def test_training_stays_on_the_calling_thread():
    eng = AuraLiteEngine()
    eng.train(TEXT, PARAMS)
    seen = set()
    handle = eng.model.register_forward_hook(
        lambda *_: seen.add(threading.current_thread().name))
    try:
        _on_thread(lambda: eng.train(TEXT, dict(PARAMS, continue_training=True,
                                                val_split=0.1)))
    finally:
        handle.remove()
    assert seen and "auralite-compute" not in seen


def test_attention_maps_run_on_the_compute_thread(engine, monkeypatch):
    seen = set()
    handle = engine.model.register_forward_hook(
        lambda *_: seen.add(threading.current_thread().name))
    try:
        maps = _on_thread(lambda: engine.attention_maps("the quick brown", layer=0))
    finally:
        handle.remove()
    assert maps.shape[-1] == len(engine.encode("the quick brown"))
    assert seen == {"auralite-compute"}


# ----------------------------------------------------------------------
#  The benchmark behind docs/engine.md § 10 keeps running
# ----------------------------------------------------------------------
def test_thread_bench_runs(capsys, repo_root):
    import importlib.util
    import json

    path = os.path.join(repo_root, "benchmarks", "thread_bench.py")
    spec = importlib.util.spec_from_file_location("thread_bench_under_test", path)
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    bench.MODEL = {k: v for k, v in PARAMS.items() if k != "max_steps"}
    bench.ROUNDS = 1
    decode = bench.child("decode", 4, 4)
    assert decode["same"] is True and decode["worker"] > 0 and decode["four"] > 0
    assert set(bench.child("load", 4, 4)) == {"main", "worker"}
    assert bench.main(["--child", "train", "--tokens", "4", "--steps", "3"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["steps_per_s"] > 0
