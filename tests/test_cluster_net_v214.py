"""Why a computer of a cluster cannot connect, in words (2.14, cluster.py).

A worker that cannot reach the main computer used to hand the problem to
torch's TCPStore client, which tells no reason apart and logs every failed
try with a C++ stack trace. A plain socket now asks first and the worker says
what to check; the main computer says what to check when no worker comes, and
the computers try the connections gloo will open before it does, so a
firewall gets named at once instead of after the cluster timeout. Two
"computers" on one machine use two loopback addresses (127.0.0.1, 127.0.0.2)
because gloo picks the direction of each connection by the addresses.
"""
from __future__ import annotations

import errno
import json
import os
import socket
import subprocess
import sys
import threading
import time

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")

from auralite.engine import cluster  # noqa: E402

pytestmark = pytest.mark.skipif(not (dist.is_available() and dist.is_gloo_available()),
                                reason="this torch build has no gloo backend")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _client(port: int):
    """A worker's store client once the main computer's port answers."""
    deadline = time.monotonic() + 20
    while cluster.connection_problem("127.0.0.1", port, timeout_s=1.0) is not None:
        assert time.monotonic() < deadline, "the main computer's store never came up"
        time.sleep(0.05)
    return cluster._make_store("127.0.0.1", port, is_master=False, timeout_s=30)


def _second_loopback() -> None:
    """Skip unless 127.0.0.2 works as a second address of this machine (not macOS)."""
    try:
        with socket.socket() as s:
            s.bind(("127.0.0.2", 0))
    except OSError:
        pytest.skip("127.0.0.2 is not an address of this machine")


def _raise(exc: BaseException):
    def fake(*_args, **_kwargs):
        raise exc
    return fake


# ----------------------------------------------------------------------
#  The connection check
# ----------------------------------------------------------------------
def test_the_connection_check_tells_the_reasons_apart(monkeypatch):
    listening = cluster._ProbePort("127.0.0.1")
    try:
        assert cluster.connection_problem("127.0.0.1", listening.port) is None
    finally:
        listening.close()
    problem = cluster.connection_problem("127.0.0.1", _free_port())
    assert problem is not None
    # Windows repeats a refused request for about 2 s, close to the time limit
    assert problem[0] in (("refused", "timeout") if sys.platform == "win32" else ("refused",))
    for exc, expected in [
            (TimeoutError("timed out"), "timeout"),         # a firewall drops the packets
            (socket.gaierror(-2, "Name or service not known"), "name"),
            (OSError(errno.EHOSTUNREACH, "No route to host"), "unreachable")]:
        monkeypatch.setattr(cluster.socket, "create_connection", _raise(exc))
        problem = cluster.connection_problem("192.168.1.25", 29500)
        assert problem is not None and problem[0] == expected


def test_every_reason_has_words_for_the_worker():
    hints = {kind: cluster._main_computer_hint("192.168.1.25", 29500, (kind, "No route to host"))
             for kind in ("refused", "timeout", "name", "unreachable")}
    assert "nothing listens on port 29500" in hints["refused"]
    assert "firewall" in hints["timeout"] and "private" in hints["timeout"]
    assert "does not resolve" in hints["name"]
    assert "(No route to host)" in hints["unreachable"] and "same network" in hints["unreachable"]


def test_the_firewall_rule_names_the_interpreter_that_listens(monkeypatch):
    monkeypatch.setattr(cluster.sys, "platform", "linux")
    assert cluster._python_program() == ""
    monkeypatch.setattr(cluster.sys, "platform", "win32")
    monkeypatch.setattr(cluster.sys, "_base_executable", r"C:\Python313\python.exe",
                        raising=False)
    assert cluster._python_program() == r"C:\Python313\python.exe"


def test_lan_addresses_put_the_shown_one_first(monkeypatch):
    monkeypatch.setattr(cluster, "lan_address", lambda: "192.168.1.25")
    infos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))
             for ip in ("127.0.1.1", "10.8.0.2", "192.168.1.25", "169.254.3.4")]
    monkeypatch.setattr(cluster.socket, "getaddrinfo", lambda *a, **k: infos)
    assert cluster.lan_addresses() == ["192.168.1.25", "10.8.0.2"]


# ----------------------------------------------------------------------
#  A worker waiting for the main computer
# ----------------------------------------------------------------------
def test_a_waiting_worker_says_why_once_and_leaves_torch_out_of_it(monkeypatch):
    stop = threading.Event()
    reasons = iter([("timeout", "timed out")] * 3 + [("refused", "refused")] * 2)

    def problem(host, port, timeout_s=3.0):
        try:
            return next(reasons)
        except StopIteration:
            stop.set()
            return "refused", "refused"

    stores = []
    monkeypatch.setattr(cluster, "connection_problem", problem)
    monkeypatch.setattr(cluster, "_make_store", lambda *a, **k: stores.append(a))
    monkeypatch.setattr(cluster.time, "sleep", lambda _s: None)
    statuses: list[str] = []
    with pytest.raises(cluster.ClusterError, match="stopped"):
        cluster.join_cluster("192.168.1.25", on_status=statuses.append, stop_event=stop)
    assert stores == []                           # no TCPStore client while nothing answers
    assert len(statuses) == 2                     # once per reason, not once per try
    assert statuses[0].startswith("Waiting for the main computer at 192.168.1.25:29500")
    assert "firewall" in statuses[0] and "nothing listens" in statuses[1]


def test_a_worker_without_a_main_computer_prints_no_cpp_traces():
    """The worker's output while the main computer is missing (the old one
    printed a TCPStore warning with a C++ stack trace every 5 seconds)."""
    code = (f"import sys; sys.path.insert(0, {REPO!r}); "
            f"from auralite.engine.cluster import run_worker; "
            f"run_worker('127.0.0.1:{_free_port()}', once=True, timeout_s=90, "
            f"on_status=lambda m: print('status:', m, flush=True))")
    env = dict(os.environ, AURALITE_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env, encoding="utf-8")
    try:
        out, _ = proc.communicate(timeout=7.5)    # beyond torch's 5 s connect timeout
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    assert "nothing listens on port" in out, out
    assert "TCPStore" not in out and "socket.cpp" not in out and "frame #" not in out, out


# ----------------------------------------------------------------------
#  The main computer
# ----------------------------------------------------------------------
def test_the_main_computer_says_what_to_check_when_no_worker_comes(monkeypatch):
    monkeypatch.setattr(cluster, "_NO_WORKER_HINT_S", 0.3)
    statuses: list[str] = []
    with pytest.raises(cluster.ClusterError, match="only 0 of 1 workers"):
        cluster.host_cluster(1, port=_free_port(), host="127.0.0.1", timeout_s=1.5,
                             on_status=statuses.append)
    hints = [s for s in statuses if s.startswith("Cluster: still waiting")]
    assert len(hints) == 1 and "firewall" in hints[0] and "127.0.0.1:" in hints[0]


def test_gloo_connects_from_the_larger_address_to_the_smaller():
    assert cluster._gloo_dials("192.168.1.30", "192.168.1.25") is True
    assert cluster._gloo_dials("192.168.1.25", "192.168.1.30") is False
    assert cluster._gloo_dials("192.168.1.100", "192.168.1.25") is True   # numbers, not text
    assert cluster._gloo_dials("10.0.0.5", "10.0.0.5") is None            # the ports decide
    assert cluster._gloo_dials("fe80::2", "fe80::1") is True
    assert cluster._gloo_dials("192.168.1.30", "fe80::1") is None
    assert cluster._gloo_dials("desktop-6pcifob", "192.168.1.25") is None


def test_the_connection_checks_stop_only_a_run_that_needs_the_connection(monkeypatch):
    port = _free_port()
    store = cluster._make_store("127.0.0.1", port, is_master=True, timeout_s=30)
    store.set("auralite/ip/1", "127.0.0.1")
    assert cluster._check_connections(store, 1, "127.0.0.2", port) == ("", [])  # an old worker
    listening = cluster._ProbePort("127.0.0.1")
    try:
        store.set("auralite/probe/1", str(listening.port))
        store.set("auralite/reach/1", "ok")
        assert cluster._check_connections(store, 1, "127.0.0.2", port) == ("", [])
    finally:
        listening.close()
    # The worker cannot be connected to: fatal when the main computer has the
    # larger address (gloo connects from it), a note otherwise.
    store.set("auralite/probe/1", str(_free_port()))
    error, notes = cluster._check_connections(store, 1, "127.0.0.2", port)
    assert error.startswith("the computers cannot open the connections training needs: "
                            "the main computer (127.0.0.2) cannot connect to worker 1 "
                            "(127.0.0.1, ")
    assert "firewall of 127.0.0.1" in error and "start the training again" in error
    assert notes == []
    error, notes = cluster._check_connections(store, 1, "126.0.0.1", port)
    assert error == "" and len(notes) == 1 and "this run does not need" in notes[0]
    # The main computer cannot be connected to (beyond the cluster port).
    monkeypatch.setattr(cluster, "_try_connect", lambda host, port: None)
    store.set("auralite/reach/1", "timeout")
    error, notes = cluster._check_connections(store, 1, "126.0.0.1", port)
    assert "worker 1 (127.0.0.1) reaches port" in error and "(timeout)" in error
    assert "not only port" in error and notes == []
    error, notes = cluster._check_connections(store, 1, "127.0.0.2", port)
    assert error == "" and "main computer's firewall" in notes[0]


def test_a_worker_hears_why_a_run_was_cancelled():
    port = _free_port()
    main = cluster._make_store("127.0.0.1", port, is_master=True, timeout_s=30)
    cluster._open_run(main, 1, token=None)
    main.set("auralite/start", json.dumps({"cancelled": True,
                                            "reason": "only 0 of 1 workers joined within 9 s"}))
    with pytest.raises(cluster.ClusterError) as caught:
        cluster._worker_rendezvous(_client(port))
    assert str(caught.value) == ("the main computer cancelled the run "
                                 "(only 0 of 1 workers joined within 9 s)")


class _NoGloo:
    """A ClusterSession that starts no process group: the rendezvous alone."""
    started: list = []

    def __init__(self, store, **kwargs):
        self.store, self.kwargs = store, kwargs       # the store serves while it lives

    def start(self):
        _NoGloo.started.append(self.kwargs)


class _SlowStore:
    """A worker's store client that looks at the store only every half second."""

    def __init__(self, store):
        self._store = store

    def __getattr__(self, name):
        return getattr(self._store, name)

    def check(self, keys):
        time.sleep(0.5)
        return self._store.check(keys)


def _rendezvous(monkeypatch, worker_ip: str, *, worker_port_open: bool = False,
                main_host: str = "127.0.0.2"):
    """host_cluster on 127.0.0.2 and one worker at `worker_ip`, without gloo.

    The worker's own port is closed unless `worker_port_open`; it looks for
    the main computer's second port at `main_host`. Returns the main
    computer's session or its error message, the worker's result and the
    main computer's status lines.
    """
    _second_loopback()
    monkeypatch.setattr(cluster, "ClusterSession", _NoGloo)
    monkeypatch.setattr(_NoGloo, "started", [])
    port = _free_port()
    own = cluster._ProbePort(worker_ip) if worker_port_open else None
    worker: dict = {}

    def join():
        try:
            worker["rank"] = cluster._worker_rendezvous(
                _SlowStore(_client(port)), my_ip=worker_ip,
                probe_port=own.port if own is not None else _free_port(), main_host=main_host)
        except cluster.ClusterError as exc:
            worker["error"] = str(exc)

    thread = threading.Thread(target=join)
    thread.start()
    statuses: list[str] = []
    try:
        main: object = cluster.host_cluster(1, port=port, host="127.0.0.2", timeout_s=30,
                                            on_status=statuses.append)
    except cluster.ClusterError as exc:
        main = str(exc)                   # no traceback keeps the main computer's port open
    finally:
        thread.join(20)
        if own is not None:
            own.close()
    return main, worker, statuses


def test_a_worker_gloo_must_connect_to_but_cannot_stops_the_run_at_once(monkeypatch):
    t0 = time.monotonic()
    main, worker, _statuses = _rendezvous(monkeypatch, "127.0.0.1")
    assert isinstance(main, str) and "cannot connect to worker 1 (127.0.0.1, " in main, main
    # the main computer kept its port open until the (slow) worker had read why
    assert worker == {"error": f"the main computer cancelled the run ({main})"}
    assert _NoGloo.started == []                       # gloo never ran
    assert time.monotonic() - t0 < 20                  # not the 30 s cluster timeout


def test_a_connection_the_run_does_not_need_is_only_a_note(monkeypatch):
    main, worker, statuses = _rendezvous(monkeypatch, "127.0.0.3")
    assert isinstance(main, _NoGloo), main
    assert worker["rank"][:2] == (1, 2) and _NoGloo.started[0]["world_size"] == 2
    notes = [s for s in statuses if s.startswith("Cluster: note - ")]
    assert len(notes) == 1 and "cannot connect to worker 1 (127.0.0.3, " in notes[0]
    assert not any("reaches port" in s for s in statuses)   # the worker reached the main one


def test_a_worker_that_reaches_only_the_cluster_port_stops_the_run(monkeypatch):
    """The main computer's second port does not answer the worker (at
    127.0.0.3 nothing listens), and gloo will connect from the worker's
    larger address to the main computer."""
    main, worker, _statuses = _rendezvous(monkeypatch, "127.0.0.3", worker_port_open=True,
                                          main_host="127.0.0.3")
    assert isinstance(main, str), main
    assert "worker 1 (127.0.0.3) reaches port" in main and "not only port" in main
    assert "cannot connect to worker" not in main          # its own port answered
    assert worker == {"error": f"the main computer cancelled the run ({main})"}


# ----------------------------------------------------------------------
#  The main computer's own store (Windows)
# ----------------------------------------------------------------------
def test_the_main_computers_store_connects_to_itself_over_the_loopback(monkeypatch):
    """torch's server listens on every interface and then connects to the
    given address; Windows refuses a connection to 0.0.0.0 (10049)."""
    seen = []

    class Recorder:
        def __init__(self, **kwargs):
            seen.append((kwargs["host_name"], kwargs["is_master"]))

    monkeypatch.setattr(cluster.dist, "TCPStore", Recorder)
    for host in ("0.0.0.0", "::", ""):
        cluster._make_store(host, 29500, is_master=True, timeout_s=5)
    cluster._make_store("192.168.1.25", 29500, is_master=True, timeout_s=5)
    cluster._make_store("192.168.1.25", 29500, is_master=False, timeout_s=5)
    assert seen == [("127.0.0.1", True)] * 3 + [("192.168.1.25", True),
                                                ("192.168.1.25", False)]


def test_a_main_computer_on_the_default_address_opens_its_run_as_on_windows(monkeypatch):
    """A TCPStore that refuses connections to 0.0.0.0 as Windows does: the
    run must open (and then time out without workers), not fail to open."""
    real = cluster.dist.TCPStore

    def windows_like(**kwargs):
        if kwargs["host_name"] in ("", "0.0.0.0", "::"):
            raise RuntimeError("The client socket has failed to connect to any network address "
                               "of (0.0.0.0, 29500). (system error: 10049 - The requested "
                               "address is not valid in its context.)")
        return real(**kwargs)

    monkeypatch.setattr(cluster.dist, "TCPStore", windows_like)
    statuses: list[str] = []
    with pytest.raises(cluster.ClusterError, match="only 0 of 1 workers"):
        cluster.host_cluster(1, port=_free_port(), host="0.0.0.0", timeout_s=1.5,
                             on_status=statuses.append)
    assert statuses and "0/1 workers joined" in statuses[0]


def test_a_connected_worker_says_so_while_the_run_does_not_open(monkeypatch):
    monkeypatch.setattr(cluster, "_NO_RUN_HINT_S", 0.2)
    port = _free_port()
    main = cluster._make_store("127.0.0.1", port, is_master=True, timeout_s=30)

    def open_late():
        cluster._open_run(main, 1, token=None)
        main.set("auralite/start", json.dumps({"cancelled": True, "reason": "a test"}))

    timer = threading.Timer(1.0, open_late)
    timer.start()
    statuses: list[str] = []
    try:
        with pytest.raises(cluster.ClusterError, match=r"cancelled the run \(a test\)"):
            cluster._worker_rendezvous(_client(port), on_status=statuses.append)
    finally:
        timer.join()
    connected = [s for s in statuses if s.startswith("Connected to the main computer")]
    assert len(connected) == 1 and "main computer's window" in connected[0]
