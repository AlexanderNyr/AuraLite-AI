"""Training on several computers over a local network (2.14).

One computer has the text and starts the run: ``train(text, {...,
"cluster_workers": 2})``, ``auralite train --cluster-workers 2`` or the GUI's
*Cluster* box. The others join it: ``auralite cluster-worker 192.168.1.10``
(or *Join* in the GUI). Windows and Linux computers can be mixed. Every
worker gets the text, the parameters and the tokenizer from the main
computer, so it needs nothing but AuraLite (the same version, with the same
torch major.minor version).

How the run is split:

* **Data parallel.** Every computer holds the whole model. Each optimizer
  step, every computer computes the gradients of its part of the batch and
  the gradients are averaged over the network (PyTorch
  ``DistributedDataParallel`` over gloo), so every computer takes the same
  step. ``batch_size`` stays the batch of the whole step: with two computers
  each one computes 16 of the 32 windows. A cluster run therefore trains the
  same model as one computer would, step for step (up to rounding), in less
  time when the computers are fast enough compared to the network.
* **One decides.** Validation is split between the computers and its losses
  summed, so every computer sees the same validation loss. Time limits, the
  Stop button and Ctrl+C act on the main computer; their state goes to the
  others before every step, so all computers stop at the same step.
* **Only the main computer** reports progress and saves checkpoints.

What travels: the rendezvous runs over a ``torch.distributed.TCPStore`` on
the main computer's port (default 29500); gloo then connects every pair of
computers on free ports. Only JSON and raw tensors are exchanged, never
pickles. There is no encryption and no real authentication (an optional
``cluster_token`` keeps unrelated runs apart): use it on a trusted local
network, never on a port reachable from the internet. Firewalls must let
Python accept connections on private networks.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import socket
import sys
import threading
import time
import uuid
from datetime import timedelta
from typing import Any, Callable

import torch
import torch.distributed as dist

__all__ = ["DEFAULT_PORT", "ClusterError", "ClusterSession", "cluster_params",
           "connection_problem", "host_cluster", "join_cluster", "lan_address",
           "lan_addresses", "parse_address", "run_worker"]

DEFAULT_PORT = 29500
#: Bumped whenever the rendezvous or the job format changes.
PROTOCOL = 1
_KEY = "auralite/"
_BACKEND = "auralite_gloo"
#: Train parameters that only concern the cluster, or paths of the main
#: computer: never sent to workers (a worker keeps its corpus memmap in its
#: temp directory).
CLUSTER_KEYS = ("cluster_workers", "cluster_port", "cluster_host", "cluster_timeout_s",
                "cluster_token", "corpus_cache_path", "autosave_path")
_CHUNK = 32 * 1024 * 1024                 # bytes per broadcast of a large payload

_registered = False
_bind_ip = ""                             # the address this process's gloo device uses
_register_lock = threading.Lock()
#: How long a collective waits for gloo's worker thread to let go of its
#: tensor (`_collective`); one wait that long turns the waiting off.
_SETTLE_TIMEOUT_S = 2.0
_settle_enabled = True
#: Seconds one connection check waits for an answer (`connection_problem`).
_PROBE_TIMEOUT_S = 3.0
#: Seconds without any worker after which the main computer says what to check.
_NO_WORKER_HINT_S = 20.0


class ClusterError(RuntimeError):
    """The cluster could not be formed (or a worker should not join this run)."""


# ----------------------------------------------------------------------
#  Addresses
# ----------------------------------------------------------------------
def parse_address(text: str, default_port: int = DEFAULT_PORT) -> tuple[str, int]:
    """``"host"``, ``"host:port"`` or ``"[v6]:port"`` -> ``(host, port)``."""
    text = str(text).strip()
    if not text:
        raise ValueError("empty address")
    if text.startswith("["):
        host, _, rest = text[1:].partition("]")
        port = rest[1:] if rest.startswith(":") else ""
    elif text.count(":") == 1:
        host, port = text.split(":")
    else:
        host, port = text, ""
    return host, int(port) if port else int(default_port)


def local_ip_towards(host: str, port: int = DEFAULT_PORT) -> str:
    """The address of this computer's interface that leads to `host`.

    A connected UDP socket picks the route without sending anything. gloo
    binds to this address: left to itself it uses whatever the host name
    resolves to, which on many Linux machines is 127.0.1.1, and the other
    computers then cannot reach it.
    """
    try:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        with socket.socket(family, socket.SOCK_DGRAM) as probe:
            probe.connect((host, int(port) or DEFAULT_PORT))
            return str(probe.getsockname()[0])
    except OSError:
        return "127.0.0.1"


def lan_address() -> str:
    """This computer's address on its main network (what workers type in)."""
    for probe in ("10.255.255.255", "192.168.255.255", "8.8.8.8"):
        ip = local_ip_towards(probe, 9)
        if not ip.startswith("127."):
            return ip
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def lan_addresses() -> list[str]:
    """Every IPv4 address of this computer, :func:`lan_address` first (2.14.0).

    Loopback and link-local addresses are left out. A computer with a VPN or
    a virtual machine adapter has several; the one the workers must type is
    the one on the network they share.
    """
    main = lan_address()
    found = [] if main.startswith("127.") else [main]
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        infos = []
    for info in infos:
        ip = str(info[4][0])
        if not ip.startswith(("127.", "169.254.")) and ip not in found:
            found.append(ip)
    return found


def connection_problem(host: str, port: int,
                       timeout_s: float = _PROBE_TIMEOUT_S) -> tuple[str, str] | None:
    """None when ``host:port`` accepts a TCP connection, else ``(kind, detail)`` (2.14.0).

    `kind` is ``"refused"`` (the computer answers, nothing listens on the
    port), ``"timeout"`` (no answer at all: a firewall drops the packets, or
    nothing is at that address), ``"name"`` (the host name does not resolve)
    or ``"unreachable"`` (no route). A plain socket, because torch's TCPStore
    client tells none of these apart: it retries for its whole timeout and
    logs every failed try with a C++ stack trace.
    """
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return None
    except socket.gaierror as exc:
        return "name", str(exc)
    except ConnectionRefusedError as exc:
        return "refused", str(exc)
    except TimeoutError as exc:
        return "timeout", str(exc) or "timed out"
    except OSError as exc:
        return "unreachable", str(exc.strerror or exc)


def _python_program() -> str:
    """The program a Windows firewall rule must name, else "" (2.14.0).

    A venv's python.exe only starts the base interpreter, and that one
    listens, so the rule must name `sys._base_executable`.
    """
    if sys.platform != "win32":
        return ""
    return str(getattr(sys, "_base_executable", "") or sys.executable)


def _main_computer_hint(host: str, port: int, problem: tuple[str, str]) -> str:
    """What a worker that cannot reach the main computer should check."""
    kind, detail = problem
    if kind == "refused":
        return (f"{host} answers, but nothing listens on port {port} yet: start the "
                f"training there with cluster workers, or check the port")
    if kind == "name":
        return f"the name {host!r} does not resolve: use the address the main computer shows"
    if kind == "unreachable":
        return (f"{host} cannot be reached ({detail}): check the address, and that both "
                f"computers are on the same network")
    return (f"no answer. Start the training there first, check that {host} is the address "
            f"it shows, and let Python accept connections in its firewall (on Windows: allow "
            f"python.exe on private networks, and make the network a private one)")


class _ProbePort:
    """A listening socket the other computers try before the run (2.14.0).

    gloo connects every pair of computers on free ports. A firewall that
    drops those connections leaves gloo waiting for the whole cluster
    timeout without a word; trying a port of the same program first names
    the computer to fix. The kernel completes the connections in the backlog,
    so nothing has to accept them.
    """

    def __init__(self, ip: str) -> None:
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        self._sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            if family == socket.AF_INET6:  # IPv4 too, as torch's own server does
                self._sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            self._sock.bind((ip, 0))
            self._sock.listen(16)
        except OSError:
            self._sock.close()
            raise
        self.port = int(self._sock.getsockname()[1])

    def close(self) -> None:
        self._sock.close()


def _open_probe_port(ip: str) -> _ProbePort | None:
    try:
        return _ProbePort(ip)
    except OSError:
        return None                        # no check then; gloo will tell


# ----------------------------------------------------------------------
#  torch.distributed plumbing
# ----------------------------------------------------------------------
def _versions() -> dict[str, str]:
    from auralite import __version__
    return {"auralite": __version__, "torch": ".".join(torch.__version__.split(".")[:2]),
            "protocol": str(PROTOCOL)}


def _token_hash(token: str | None) -> str:
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()


def _make_store(host: str, port: int, *, is_master: bool, timeout_s: float) -> Any:
    kwargs: dict[str, Any] = dict(host_name=host, port=int(port), world_size=None,
                                  is_master=is_master, timeout=timedelta(seconds=timeout_s),
                                  wait_for_workers=False)
    tcp_store: Any = dist.TCPStore  # pyright: ignore[reportPrivateImportUsage]
    try:
        return tcp_store(**kwargs)
    except (RuntimeError, ValueError) as exc:
        if "libuv" not in str(exc).lower():
            raise
        kwargs["use_libuv"] = False       # Windows builds without libuv
        return tcp_store(**kwargs)


def _register_backend() -> str:
    """A gloo backend bound to :data:`_bind_ip` (falls back to plain gloo)."""
    global _registered
    with _register_lock:
        if _registered:
            return _BACKEND
        try:
            gloo: Any = dist.ProcessGroupGloo  # pyright: ignore[reportPrivateImportUsage]
            gloo._Options                  # noqa: B018 - private API probe

            def create(store: Any, rank: int, size: int, timeout: timedelta) -> Any:
                options = gloo._Options()
                options._timeout = timeout
                options._devices = [gloo.create_device(hostname=_bind_ip)]
                return gloo(store, rank, size, options)

            dist.Backend.register_backend(_BACKEND, create, devices=["cpu"])
        except (AttributeError, RuntimeError, ValueError, TypeError):
            return "gloo"                  # this torch has another API: default device
        _registered = True
        return _BACKEND


def _collective(op: Callable[[Any], Any], tensor: Any) -> None:
    """Run the blocking collective ``op(tensor)``, then wait until gloo let go of `tensor`.

    torch 2.14 keeps a tensor's Python object alive while C++ code holds the
    tensor as well; whoever drops the last C++ reference releases the Python
    object, and that takes the GIL. After a collective that is gloo's worker
    thread, a moment *after* it has woken the caller. When the caller's
    process exits in that moment (a worker right after its last run), Python
    can end the thread that waits for the GIL with ``pthread_exit``, inside
    C++ code that must not be unwound, and the process aborts: "terminate
    called without an active exception" (2.14.0 CI: Linux, Python 3.11 and
    3.12). Returning only once `tensor` is back to its own references
    (`sys.getrefcount`) leaves the worker thread nothing to do in Python.

    A collective that raises returns at once: its traceback holds the tensor,
    and the session ends anyway.
    """
    global _settle_enabled
    before = sys.getrefcount(tensor)
    op(tensor)
    if _settle_enabled:
        deadline = time.monotonic() + _SETTLE_TIMEOUT_S
        while sys.getrefcount(tensor) > before:
            if time.monotonic() > deadline:
                _settle_enabled = False        # held elsewhere: stop waiting for it
                break
            time.sleep(0.0002)                 # the worker thread needs the GIL


def _broadcast_from_main(tensor: Any) -> None:
    dist.broadcast(tensor, 0)


class ClusterSession:
    """One computer's side of a training run on several computers.

    Rank 0 is the main computer. :meth:`start` joins the process group;
    the other methods are collectives that every computer calls at the same
    point of the run.
    """

    def __init__(self, store: Any, *, rank: int, world_size: int, bind_ip: str,
                 timeout_s: float, job_id: str) -> None:
        self.store = store
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.bind_ip = bind_ip
        self.timeout_s = float(timeout_s)
        self.job_id = job_id
        self.started = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def start(self) -> None:
        global _bind_ip
        if not dist.is_available() or not dist.is_gloo_available():
            raise ClusterError("this PyTorch build has no gloo backend")
        if dist.is_initialized():
            raise ClusterError("a torch.distributed process group is already running "
                               "in this process (torchrun?); a cluster run needs its own")
        _bind_ip = self.bind_ip
        backend = _register_backend()
        prefix_store: Any = dist.PrefixStore  # pyright: ignore[reportPrivateImportUsage]
        dist.init_process_group(backend=backend,
                                store=prefix_store(f"{_KEY}pg/{self.job_id}/", self.store),
                                rank=self.rank, world_size=self.world_size,
                                timeout=timedelta(seconds=self.timeout_s))
        self.started = True

    def close(self) -> None:
        if self.started and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:              # noqa: BLE001 - best effort on shutdown
                pass
        self.started = False
        self.store = None

    # ---- collectives -------------------------------------------------
    def broadcast_bytes(self, data: bytes | None) -> bytes:
        """Rank 0's `data` on every computer (sent in chunks)."""
        size = torch.tensor([len(data) if self.is_main and data is not None else 0],
                            dtype=torch.int64)
        _collective(_broadcast_from_main, size)
        total = int(size.item())
        out = bytearray(total) if not self.is_main else bytearray(data or b"")
        for start in range(0, total, _CHUNK):
            end = min(total, start + _CHUNK)
            if self.is_main:
                chunk = torch.frombuffer(out, dtype=torch.uint8, offset=start,
                                         count=end - start).clone()
            else:
                chunk = torch.empty(end - start, dtype=torch.uint8)
            _collective(_broadcast_from_main, chunk)
            if not self.is_main:
                out[start:end] = chunk.numpy().tobytes()
        return bytes(out)

    def broadcast_json(self, obj: Any = None) -> Any:
        data = json.dumps(obj).encode("utf-8") if self.is_main else None
        return json.loads(self.broadcast_bytes(data).decode("utf-8"))

    def broadcast_floats(self, values: list[float]) -> list[float]:
        t = torch.tensor([float(v) for v in values], dtype=torch.float64)
        _collective(_broadcast_from_main, t)
        return [float(v) for v in t.tolist()]

    def allreduce_sum(self, values: list[float]) -> list[float]:
        t = torch.tensor([float(v) for v in values], dtype=torch.float64)
        _collective(dist.all_reduce, t)
        return [float(v) for v in t.tolist()]

    # ---- the job -------------------------------------------------------
    def send_job(self, text: str, params: dict) -> None:
        """Main computer: the parameters and the training text to every worker."""
        self.broadcast_json({"params": job_params(params), "versions": _versions()})
        self.broadcast_bytes(text.encode("utf-8"))

    def receive_job(self) -> tuple[str, dict]:
        job = self.broadcast_json(None)
        text = self.broadcast_bytes(None).decode("utf-8")
        return text, dict(job["params"])


def job_params(params: dict) -> dict:
    """`params` without the cluster settings and anything JSON cannot carry."""
    out = {}
    for key, value in params.items():
        if key in CLUSTER_KEYS:
            continue
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        out[key] = value
    return out


def cluster_params(params: dict) -> dict[str, Any]:
    """The cluster settings in train parameters, with defaults."""
    return {
        "workers": int(params.get("cluster_workers") or 0),
        "port": int(params.get("cluster_port") or DEFAULT_PORT),
        "host": str(params.get("cluster_host") or "0.0.0.0"),
        "timeout_s": float(params.get("cluster_timeout_s") or 600.0),
        "token": params.get("cluster_token") or None,
    }


# ----------------------------------------------------------------------
#  Forming the cluster
# ----------------------------------------------------------------------
def _open_run(store: Any, workers: int, token: str | None) -> str:
    """Publish a new run on the main computer's store; returns its id."""
    job_id = uuid.uuid4().hex
    store.set(_KEY + "expected", str(int(workers)))
    store.set(_KEY + "versions", json.dumps(_versions()))
    store.set(_KEY + "token", _token_hash(token))
    store.set(_KEY + "job_id", job_id)     # last: workers wait for it
    return job_id


def host_cluster(workers: int, *, port: int = DEFAULT_PORT, host: str = "0.0.0.0",
                 timeout_s: float = 600.0, token: str | None = None,
                 on_status: Callable[[str], None] | None = None,
                 stop_event: Any = None) -> ClusterSession:
    """Main computer: open the port, wait for `workers` workers, start the group."""
    workers = int(workers)
    if workers < 1:
        raise ValueError("a cluster needs at least one worker")
    status = on_status or (lambda _msg: None)
    store = None
    deadline = time.monotonic() + 10.0
    while store is None:                  # the previous run's port may still close
        try:
            store = _make_store(host, port, is_master=True, timeout_s=max(timeout_s, 60.0))
        except (RuntimeError, OSError) as exc:
            if time.monotonic() > deadline:
                raise ClusterError(f"cannot open port {port} for the cluster: {exc}") from exc
            time.sleep(0.5)
    probe_port = _open_probe_port(host or "0.0.0.0")
    if probe_port is not None:            # before the run opens: workers look for it then
        store.set(_KEY + "probe/0", str(probe_port.port))
    job_id = _open_run(store, workers, token)
    shown = lan_address() if host in ("", "0.0.0.0", "::") else host
    started = time.monotonic()
    deadline = started + float(timeout_s)
    joined = -1
    hint_shown = False
    try:
        while True:
            now_joined = int(store.add(_KEY + "joined", 0))
            if now_joined != joined:
                joined = now_joined
                status(f"Cluster: {min(joined, workers)}/{workers} workers joined - on the "
                       f"other computers run: auralite cluster-worker {shown}:{port}")
            if joined >= workers:
                break
            if stop_event is not None and stop_event.is_set():
                raise ClusterError("stopped while waiting for workers")
            if time.monotonic() > deadline:
                raise ClusterError(f"only {joined} of {workers} workers joined within "
                                   f"{timeout_s:g} s")
            if not hint_shown and time.monotonic() - started > _NO_WORKER_HINT_S:
                hint_shown = True
                status(_no_worker_hint(shown, port))
            time.sleep(0.2)
        first_worker = store.get(_KEY + "ip/1").decode("utf-8")
        bind_ip = host if host not in ("", "0.0.0.0", "::") else local_ip_towards(first_worker)
        error, notes = _check_connections(store, workers, bind_ip, port)
        for note in notes:
            status(f"Cluster: note - {note}")
        if error:
            raise ClusterError(error)
        store.set(_KEY + "start", json.dumps({"world_size": workers + 1}))
        session = ClusterSession(store, rank=0, world_size=workers + 1, bind_ip=bind_ip,
                                 timeout_s=timeout_s, job_id=job_id)
        session.start()
    except BaseException as exc:
        reason = (str(exc) if isinstance(exc, ClusterError)
                  else _first_line(exc) if isinstance(exc, Exception) else "stopped")
        try:
            store.set(_KEY + "start", json.dumps({"cancelled": True, "reason": reason}))
            # The port closes with this function: let the workers read why first.
            waiting = min(max(joined, 0), workers)
            give_up = time.monotonic() + 3.0
            while (waiting and int(store.add(_KEY + "start_seen", 0)) < waiting
                   and time.monotonic() < give_up):
                time.sleep(0.05)
        except Exception:                  # noqa: BLE001
            pass
        raise
    finally:
        if probe_port is not None:
            probe_port.close()
    status(f"Cluster: training on {workers + 1} computers")
    return session


def _first_line(exc: BaseException) -> str:
    """The first line of an error, without torch's C++ stack trace."""
    lines = str(exc).strip().splitlines()
    return (lines[0] if lines else type(exc).__name__)[:300]


def _no_worker_hint(shown: str, port: int) -> str:
    """Main computer: what to check when the workers do not come (2.14.0)."""
    others = [ip for ip in lan_addresses() if ip != shown]
    also = f" (this computer also has {', '.join(others)})" if others else ""
    program = _python_program()
    python = f"Python ({program})" if program else "Python"
    return (f"Cluster: still waiting. Workers must reach {shown}:{port}{also}. A worker "
            f"that gets no answer is blocked by this computer's firewall: let {python} accept "
            f"connections on private networks (docs/training.md, section 12)")


def _gloo_dials(a: str, b: str) -> bool | None:
    """Whether gloo makes the computer at address `a` connect to the one at `b`.

    gloo opens one connection per pair of computers, from the larger address
    to the smaller one (``Device::isInitiator`` in gloo/transport/tcp: the
    addresses compared byte for byte, then the ports). None when the
    addresses are equal or not IP addresses: the ports decide then.
    """
    try:
        x, y = ipaddress.ip_address(a), ipaddress.ip_address(b)
    except ValueError:
        return None
    if x.version != y.version or x == y:
        return None
    return x.packed > y.packed            # memcmp, as gloo compares them


def _try_connect(host: str, port: int) -> tuple[str, str] | None:
    """:func:`connection_problem`, asked twice when there was no answer.

    Windows sends the first repeat of a connection request after 3 s, so one
    lost packet on Wi-Fi would otherwise read as a firewall.
    """
    problem = connection_problem(host, port)
    if problem is not None and problem[0] == "timeout":
        problem = connection_problem(host, port)
    return problem


def _check_connections(store: Any, workers: int, main_ip: str,
                       port: int) -> tuple[str, list[str]]:
    """Main computer: try the connections of the run before gloo does (2.14.0).

    Each worker tried a second port of this computer (``reach/{rank}``) and
    offers one of its own (``probe/{rank}``); workers of a build without
    these keys are not checked. A computer that cannot be connected to stops
    the run when gloo will connect to it (from a computer with a larger
    address): gloo would wait for the whole cluster timeout and then fail
    without naming anyone. Returns ``(error, notes)``: the error ("" when
    none), and the failures this run does not need, which another address
    (DHCP) would.
    """
    ips = {0: main_ip}
    for rank in range(1, workers + 1):
        ips[rank] = store.get(f"{_KEY}ip/{rank}").decode("utf-8")
    failed: dict[int, list[str]] = {}     # rank that could not be reached -> what failed
    program = _python_program()
    python = f"Python ({program})" if program else "Python"
    for rank in range(1, workers + 1):
        if store.check([f"{_KEY}reach/{rank}"]):
            kind = store.get(f"{_KEY}reach/{rank}").decode("utf-8")
            if kind != "ok":
                failed.setdefault(0, []).append(
                    f"worker {rank} ({ips[rank]}) reaches port {port} of the main computer "
                    f"({main_ip}) but not its other ports ({kind}): let {python} accept "
                    f"connections in the main computer's firewall on private networks, not "
                    f"only port {port}")
        if store.check([f"{_KEY}probe/{rank}"]):
            problem = _try_connect(ips[rank], int(store.get(f"{_KEY}probe/{rank}").decode("utf-8")))
            if problem is not None:
                failed.setdefault(rank, []).append(
                    f"the main computer ({main_ip}) cannot connect to worker {rank} "
                    f"({ips[rank]}, {problem[0]}): let Python accept connections in the "
                    f"firewall of {ips[rank]} on private networks")
    errors: list[str] = []
    notes: list[str] = []
    for target, texts in failed.items():
        needed = any(_gloo_dials(ips[other], ips[target]) for other in ips if other != target)
        (errors if needed else notes).extend(texts)
    if not errors:
        return "", [f"{text} (this run does not need that connection, a computer with "
                    f"another address would)" for text in notes]
    return ("the computers cannot open the connections training needs: " + "; ".join(errors)
            + ". Then start the training again; the workers wait for it."), notes


def join_cluster(main: str, *, timeout_s: float = 600.0, token: str | None = None,
                 skip_job: str | None = None,
                 on_status: Callable[[str], None] | None = None,
                 stop_event: Any = None) -> ClusterSession:
    """Worker: connect to the main computer and join its next run.

    Waits (polling, so `stop_event` can end it) for a main computer that
    offers a run other than `skip_job`, saying why it cannot connect yet.
    Raises :class:`ClusterError` when this worker cannot take part (versions,
    token, enough workers already).
    """
    host, port = parse_address(main)
    status = on_status or (lambda _msg: None)
    store = None
    shown = ""
    while store is None:                  # short tries, so `stop_event` can end the wait
        if stop_event is not None and stop_event.is_set():
            raise ClusterError("stopped")
        problem = connection_problem(host, port)
        if problem is None:               # only then torch's client: it logs every failure
            try:
                store = _make_store(host, port, is_master=False, timeout_s=5.0)
                break
            except (RuntimeError, OSError) as exc:
                message = f"{host}:{port} answers but does not serve a run: {_first_line(exc)}"
        else:
            message = _main_computer_hint(host, port, problem)
        if message != shown:              # once per reason, not once per try
            status(f"Waiting for the main computer at {host}:{port} - {message}")
            shown = message
        time.sleep(1.0)
    store.set_timeout(timedelta(seconds=max(timeout_s, 60.0)))
    my_ip = local_ip_towards(host, port)
    probe_port = _open_probe_port(my_ip)
    try:
        rank, world_size, job_id = _worker_rendezvous(
            store, token=token, skip_job=skip_job, on_status=status, stop_event=stop_event,
            my_ip=my_ip, probe_port=probe_port.port if probe_port is not None else 0,
            main_host=host)
    finally:
        if probe_port is not None:
            probe_port.close()
    session = ClusterSession(store, rank=rank, world_size=world_size, bind_ip=my_ip,
                             timeout_s=timeout_s, job_id=job_id)
    session.start()
    return session


def _worker_rendezvous(store: Any, *, token: str | None = None, skip_job: str | None = None,
                       on_status: Callable[[str], None] | None = None,
                       stop_event: Any = None, my_ip: str = "", probe_port: int = 0,
                       main_host: str = "") -> tuple[int, int, str]:
    """A worker's side of the rendezvous: ``(rank, world_size, job_id)``.

    With `probe_port` the main computer tries to connect to this worker
    before the run, with `main_host` this worker tries a second port of the
    main computer (see :func:`_check_connections`).
    """
    status = on_status or (lambda _msg: None)

    def wait_for(key: str) -> bytes:
        while not store.check([_KEY + key]):
            if stop_event is not None and stop_event.is_set():
                raise ClusterError("stopped")
            time.sleep(0.2)
        return store.get(_KEY + key)

    job_id = wait_for("job_id").decode("utf-8")
    if job_id == skip_job:
        raise ClusterError("already served this run")
    versions = json.loads(wait_for("versions").decode("utf-8"))
    mine = _versions()
    if versions != mine:
        raise ClusterError(f"version mismatch: the main computer runs AuraLite "
                           f"{versions.get('auralite')} with torch {versions.get('torch')} "
                           f"(protocol {versions.get('protocol')}), this one AuraLite "
                           f"{mine['auralite']} with torch {mine['torch']}")
    if store.get(_KEY + "token").decode("utf-8") != _token_hash(token):
        raise ClusterError("the main computer uses another cluster token")
    expected = int(store.get(_KEY + "expected").decode("utf-8"))
    rank = int(store.add(_KEY + "joined", 1))
    if rank > expected:
        raise ClusterError(f"the main computer already has its {expected} worker(s)")
    # before ip/{rank}: the main computer reads these once that key is there
    if probe_port:
        store.set(f"{_KEY}probe/{rank}", str(int(probe_port)))
    if main_host and store.check([_KEY + "probe/0"]):
        problem = _try_connect(main_host, int(store.get(_KEY + "probe/0").decode("utf-8")))
        store.set(f"{_KEY}reach/{rank}", "ok" if problem is None else problem[0])
    store.set(f"{_KEY}ip/{rank}", my_ip or "127.0.0.1")
    status(f"Joined as worker {rank}/{expected}; waiting for the main computer")
    start = json.loads(wait_for("start").decode("utf-8"))
    store.add(_KEY + "start_seen", 1)     # the main computer may close the port now
    if start.get("cancelled"):
        reason = start.get("reason")
        raise ClusterError("the main computer cancelled the run"
                           + (f" ({reason})" if reason else ""))
    return rank, int(start["world_size"]), job_id


def run_worker(main: str, *, once: bool = False, timeout_s: float = 600.0,
               token: str | None = None, on_status: Callable[[str], None] | None = None,
               stop_event: Any = None) -> int:
    """Serve training runs of the main computer at `main` (``host[:port]``).

    Joins a run, trains its part and joins the next run, until `stop_event`
    is set (or after one run with `once`). Returns the number of runs served.
    Connection problems are reported through `on_status` and retried.
    """
    from auralite.engine import AuraLiteEngine

    status = on_status or (lambda _msg: None)
    served = 0
    last_job: str | None = None
    while stop_event is None or not stop_event.is_set():
        session = None
        try:
            status(f"Connecting to {main}...")
            session = join_cluster(main, timeout_s=timeout_s, token=token, skip_job=last_job,
                                   on_status=status, stop_event=stop_event)
            last_job = session.job_id
            text, params = session.receive_job()
            status(f"Training as worker {session.rank} of {session.world_size - 1} "
                   f"({len(text):,} characters)")
            engine = AuraLiteEngine()
            engine._cluster = session
            try:
                engine.train(text, params)
            finally:
                engine._cluster = None
            served += 1
            status(f"Run finished ({served} served)")
        except ClusterError as exc:
            if stop_event is not None and stop_event.is_set():
                break
            status(f"{exc}; retrying")
            if once and served == 0 and "version" in str(exc):
                raise
            time.sleep(2.0)
        except (RuntimeError, OSError) as exc:
            if stop_event is not None and stop_event.is_set():
                break
            status(f"The run ended with an error: {_first_line(exc)}; retrying")
            time.sleep(2.0)
        finally:
            if session is not None:
                session.close()
        if once and served:
            break
    return served
