"""Training on several computers over a local network (2.14, cluster.py).

The computers are played by processes on this machine: the main computer is
the test process, every worker a ``python -c run_worker(...)`` subprocess,
connected over 127.0.0.1 with the same gloo transport a LAN uses. A cluster
run must train the model one computer trains (same windows per step, the
batch split between the computers), stop when the main computer says so, and
refuse workers that do not belong to the run.
"""
from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import sys
import threading
import time

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")

from auralite.engine import AuraLiteEngine, validate_params  # noqa: E402
from auralite.engine import cluster  # noqa: E402

pytestmark = pytest.mark.skipif(not (dist.is_available() and dist.is_gloo_available()),
                                reason="this torch build has no gloo backend")

TEXT = ("The quick brown fox jumps over the lazy dog near the river bank.\n"
        "A small grey cat sleeps on the warm window sill every afternoon.\n"
        "Seven old sailors sing loud songs about the distant northern sea.\n") * 30
PARAMS = {"tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_layers": 2,
          "seq_length": 32, "batch_size": 16, "epochs": 3, "seed": 0, "val_split": 0.1,
          "progress_every_seconds": 1e9, "early_stopping_patience": 0}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _start_worker(port: int, *, once: bool = True, token: str | None = None) -> subprocess.Popen:
    code = (f"import sys; sys.path.insert(0, {REPO!r}); "
            f"from auralite.engine.cluster import run_worker; "
            f"print('served', run_worker('127.0.0.1:{port}', once={once}, timeout_s=90, "
            f"token={token!r}, on_status=lambda m: print('status:', m, flush=True)), flush=True)")
    env = dict(os.environ, AURALITE_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    return subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env, encoding="utf-8")


def _finish(proc: subprocess.Popen, timeout: float = 180) -> str:
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        raise AssertionError(f"worker did not finish:\n{out}") from None
    return out


def _cluster_params(port: int, workers: int = 1, **extra) -> dict:
    return {**PARAMS, "cluster_workers": workers, "cluster_port": port,
            "cluster_host": "127.0.0.1", "cluster_timeout_s": 90, **extra}


# ----------------------------------------------------------------------
#  Pieces
# ----------------------------------------------------------------------
def test_addresses():
    assert cluster.parse_address("192.168.1.10") == ("192.168.1.10", 29500)
    assert cluster.parse_address(" host:1234 ") == ("host", 1234)
    assert cluster.parse_address("[fe80::1]:29501") == ("fe80::1", 29501)
    assert cluster.parse_address("[fe80::1]") == ("fe80::1", 29500)
    with pytest.raises(ValueError):
        cluster.parse_address("")
    assert cluster.local_ip_towards("127.0.0.1") == "127.0.0.1"
    assert isinstance(cluster.lan_address(), str)


def test_job_params_leave_out_cluster_settings_and_paths():
    params = {**PARAMS, "cluster_workers": 2, "cluster_token": "x",
              "corpus_cache_path": "C:\\\\corpus.bin", "autosave_path": "/tmp/a.pt",
              "callback": object()}
    sent = cluster.job_params(params)
    assert sent == PARAMS
    json.dumps(sent)


def test_validation_of_cluster_parameters():
    assert validate_params({"cluster_workers": 2, "cluster_port": 29500}) == []
    for bad in ({"cluster_workers": -1}, {"cluster_workers": 1.5}, {"cluster_workers": True},
                {"cluster_port": 0}, {"cluster_port": 70000}, {"cluster_timeout_s": 0},
                {"cluster_host": 5}, {"cluster_token": 7}):
        assert validate_params(bad), bad


def test_continue_training_is_refused_on_a_cluster():
    engine = AuraLiteEngine()
    engine._cluster = type("Session", (), {"world_size": 2, "rank": 0, "is_main": True})()
    with pytest.raises(ValueError, match="continue a model"):
        engine.train(TEXT, {**PARAMS, "continue_training": True})


def test_the_rendezvous_refuses_extra_workers_and_wrong_tokens():
    port = _free_port()
    main = cluster._make_store("127.0.0.1", port, is_master=True, timeout_s=30)
    job = cluster._open_run(main, 1, token="secret")

    def client():
        return cluster._make_store("127.0.0.1", port, is_master=False, timeout_s=30)

    with pytest.raises(cluster.ClusterError, match="another cluster token"):
        cluster._worker_rendezvous(client(), token="wrong")
    with pytest.raises(cluster.ClusterError, match="already served"):
        cluster._worker_rendezvous(client(), token="secret", skip_job=job)
    result: dict = {}
    first = threading.Thread(target=lambda: result.setdefault(
        "first", cluster._worker_rendezvous(client(), token="secret")))
    first.start()
    deadline = time.monotonic() + 20
    while int(main.add("auralite/joined", 0)) < 1 and time.monotonic() < deadline:
        time.sleep(0.05)
    with pytest.raises(cluster.ClusterError, match="already has its 1 worker"):
        cluster._worker_rendezvous(client(), token="secret")
    main.set("auralite/start", json.dumps({"world_size": 2}))
    first.join(20)
    assert result["first"] == (1, 2, job)


def test_a_stop_event_ends_the_wait_for_workers():
    stop = threading.Event()
    threading.Timer(0.5, stop.set).start()
    with pytest.raises(cluster.ClusterError, match="stopped"):
        cluster.host_cluster(1, port=_free_port(), host="127.0.0.1", timeout_s=30,
                             stop_event=stop)


def test_too_few_workers_time_out():
    with pytest.raises(cluster.ClusterError, match="only 0 of 1 workers"):
        cluster.host_cluster(1, port=_free_port(), host="127.0.0.1", timeout_s=1.5)


# ----------------------------------------------------------------------
#  Runs
# ----------------------------------------------------------------------
def test_two_computers_train_the_model_one_computer_trains():
    single = AuraLiteEngine()
    alone = single.train(TEXT, PARAMS)
    port = _free_port()
    worker = _start_worker(port)
    statuses: list[str] = []

    def progress(epoch, total, loss, val, info=None):
        if info and info.get("phase") == "cluster":
            statuses.append(info["message"])

    main = AuraLiteEngine()
    together = main.train(TEXT, _cluster_params(port), progress_callback=progress)
    out = _finish(worker)
    assert worker.returncode == 0, out
    assert "served 1" in out and "Training as worker 1 of 1" in out
    assert together["cluster_size"] == 2 and alone["cluster_size"] == 1
    assert together["optimizer_steps"] == alone["optimizer_steps"]
    assert together["val_loss"] == pytest.approx(alone["val_loss"], rel=0.01)
    assert any("1/1 workers joined" in s for s in statuses)
    assert not any("note -" in s for s in statuses)     # the connection checks passed
    # the engine is an ordinary single-computer engine again
    assert (main.is_distributed, main.world_size) == (False, 1)
    assert not dist.is_initialized()
    assert main.generate("The quick", 12, temperature=0.0, top_k=1).startswith("The quick")


def test_the_main_computer_decides_when_the_run_ends():
    """A time limit on the main computer ends the run on every computer."""
    port = _free_port()
    worker = _start_worker(port)
    finish = threading.Event()
    summary = AuraLiteEngine().train(
        TEXT, _cluster_params(port, epochs=500, time_budget_s=6.0), finish_event=finish)
    out = _finish(worker)
    assert worker.returncode == 0, out
    assert summary["stop_reason"] == "time_budget"
    assert summary["epochs_completed"] < 500


def test_a_stop_on_the_main_computer_stops_every_worker():
    port = _free_port()
    worker = _start_worker(port)
    stop = threading.Event()

    def progress(epoch, total, loss, val, info=None):
        if epoch >= 2:                     # two epochs in: press Stop
            stop.set()

    summary = AuraLiteEngine().train(TEXT, _cluster_params(port, epochs=500),
                                     progress_callback=progress, stop_event=stop)
    out = _finish(worker)
    assert worker.returncode == 0, out
    assert summary["stop_reason"] == "stopped"


def test_three_computers_with_early_stopping_and_bpe():
    # Random letters: the validation loss stops improving within a few epochs.
    rng = random.Random(0)
    noise = "".join(rng.choice("abcdefgh \n") for _ in range(6000))
    port = _free_port()
    workers = [_start_worker(port), _start_worker(port)]
    summary = AuraLiteEngine().train(noise, _cluster_params(
        port, workers=2, tokenizer="bpe", bpe_vocab_size=280, epochs=200,
        early_stopping_patience=1, batch_size=24))
    for proc in workers:
        out = _finish(proc)
        assert proc.returncode == 0, out
    assert summary["cluster_size"] == 3
    assert summary["stop_reason"] == "early_stopping"


def test_a_worker_serves_one_run_after_another():
    port = _free_port()
    worker = _start_worker(port, once=False)
    try:
        engine = AuraLiteEngine()
        first = engine.train(TEXT, _cluster_params(port, epochs=1))
        second = engine.train(TEXT, _cluster_params(port, epochs=1, seed=1))
        assert first["cluster_size"] == second["cluster_size"] == 2
    finally:
        worker.terminate()
        out = _finish(worker, timeout=30)
    assert out.count("Run finished") == 2, out


def test_a_worker_of_another_version_is_refused(monkeypatch):
    real = cluster._versions
    monkeypatch.setattr(cluster, "_versions", lambda: {**real(), "auralite": "0.0.1"})
    port = _free_port()
    worker = _start_worker(port)
    with pytest.raises(cluster.ClusterError, match="only 0 of 1 workers"):
        AuraLiteEngine().train(TEXT, _cluster_params(port, cluster_timeout_s=8))
    out = _finish(worker, timeout=60)
    assert worker.returncode != 0 and "version mismatch" in out, out


def test_the_cli_flags():
    from auralite.cli import build_parser, cmd_cluster_worker

    args = build_parser().parse_args(["train", "--text", "t.txt", "--cluster-workers", "2",
                                      "--cluster-port", "29501", "--cluster-token", "abc",
                                      "--cluster-timeout", "2m"])
    assert (args.cluster_workers, args.cluster_port, args.cluster_token) == (2, 29501, "abc")
    assert args.cluster_timeout == 120
    args = build_parser().parse_args(["cluster-worker", "192.168.1.10:29501", "--once"])
    assert args.func is cmd_cluster_worker and args.main == "192.168.1.10:29501" and args.once


# ----------------------------------------------------------------------
#  2.14.0 CI: a computer that exits right after its last collective
# ----------------------------------------------------------------------
def _bare_session() -> cluster.ClusterSession:
    """A session whose collectives the test replaces (no process group)."""
    return cluster.ClusterSession(None, rank=0, world_size=2, bind_ip="127.0.0.1",
                                  timeout_s=5, job_id="test")


def test_a_collective_returns_once_gloo_has_let_go_of_its_tensor(monkeypatch):
    """gloo's worker thread drops its reference to a collective's tensor a
    moment after it has woken the caller, and with torch 2.14 that release
    takes the GIL. A timer plays the worker thread: the collective may only
    return once the timer has let go."""
    held = []

    def late_release(tensor, *args, **kwargs):
        held.append(tensor)
        threading.Timer(0.2, held.clear).start()

    monkeypatch.setattr(cluster, "_settle_enabled", True)
    monkeypatch.setattr(cluster.dist, "all_reduce", late_release)
    monkeypatch.setattr(cluster.dist, "broadcast", late_release)
    session = _bare_session()
    start = time.monotonic()
    assert session.allreduce_sum([1.0, 2.0]) == [1.0, 2.0]
    assert held == [] and time.monotonic() - start >= 0.15
    assert session.broadcast_floats([3.0]) == [3.0]
    assert held == []
    assert session.broadcast_json({"a": 1}) == {"a": 1}       # the size and the chunk
    assert held == []


def test_a_failed_collective_raises_at_once(monkeypatch):
    """Its traceback holds the tensor: waiting would only run into the time limit."""
    def fails(tensor, *args, **kwargs):
        raise RuntimeError("Connection closed by peer")

    monkeypatch.setattr(cluster, "_settle_enabled", True)
    monkeypatch.setattr(cluster.dist, "all_reduce", fails)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="closed by peer"):
        _bare_session().allreduce_sum([1.0])
    assert time.monotonic() - start < 1.0
    assert cluster._settle_enabled is True


def test_a_tensor_that_stays_held_turns_the_waiting_off(monkeypatch):
    """Something else keeping the tensor (a future torch) costs one wait, not one per step."""
    held = []
    monkeypatch.setattr(cluster, "_settle_enabled", True)
    monkeypatch.setattr(cluster, "_SETTLE_TIMEOUT_S", 0.2)
    monkeypatch.setattr(cluster.dist, "all_reduce", lambda t, *a, **k: held.append(t))
    session = _bare_session()
    start = time.monotonic()
    session.allreduce_sum([1.0])
    assert 0.15 <= time.monotonic() - start < 5
    assert cluster._settle_enabled is False
    start = time.monotonic()
    session.allreduce_sum([1.0])
    assert time.monotonic() - start < 0.1


_EXITING_COMPUTER = """
import sys
sys.path.insert(0, {repo!r})
from auralite.engine import cluster
if {main!r}:
    session = cluster.host_cluster(1, port={port}, host="127.0.0.1", timeout_s=60)
else:
    session = cluster.join_cluster("127.0.0.1:{port}", timeout_s=60)
for i in range(10):
    session.allreduce_sum([float(i), 1.0])
    session.broadcast_floats([float(i)])
print("done", flush=True)
"""


def test_computers_that_exit_right_after_their_last_collective_exit_cleanly():
    """The 2.14.0 CI failure on purpose: two processes run collectives and
    exit at once, without closing the session. Before the fix about one
    process in five aborted on Linux with Python 3.11 and 3.12 ("terminate
    called without an active exception", exit code -6)."""
    env = dict(os.environ, AURALITE_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    for _ in range(3):
        port = _free_port()
        procs = [subprocess.Popen(
            [sys.executable, "-c", _EXITING_COMPUTER.format(repo=REPO, main=main, port=port)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
            encoding="utf-8") for main in (True, False)]
        for proc in procs:
            out = _finish(proc, timeout=120)
            assert proc.returncode == 0, out
            assert "done" in out, out
