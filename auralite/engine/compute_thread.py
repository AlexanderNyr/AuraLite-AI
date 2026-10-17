"""One thread for the engine's tensor work (2.13).

PyTorch's CPU kernels run their parallel loops on an OpenMP thread team, and
every Python thread that runs such a loop gets a team of its own. Once there
are more team threads than CPUs, libgomp stops letting idle team threads spin
and puts them to sleep after every loop, and each loop then pays for waking
them up. In 2.12 a model trained or loaded on the main thread and used from a
worker thread (the API server, the GUI's generation thread) decoded at 39 %
of the speed of the main thread: 242 against 618 tokens per second on two
CPUs, and four concurrent requests got no further.

2.13 runs every generation forward pass of `AuraLiteEngine` on one
long-lived thread, ``auralite-compute``, so the process keeps a single team
however many threads call the engine, and the thread that trained or loaded
a model releases its own team (``omp_pause_resource_all``) when it is done:

* :func:`run_on_compute_thread` / :func:`on_compute_thread` hand a call to the
  thread and wait for it. The caller's ``inference_mode`` and grad mode go
  along, exceptions come back with their type and traceback, and a call made
  on the compute thread itself runs inline.
* :func:`release_openmp_pool` frees the calling thread's OpenMP team;
  :func:`release_compute_pool` frees the compute thread's (``train()`` does
  that first, so its own team does not compete with an idle one).
* :func:`run_inline` marks a block in which calls stay on the current thread:
  training validates and samples on its own thread.
* 2.14: :meth:`ComputeThread.run_in_group` queues a decode step of a
  :class:`~auralite.engine.batching.DecodeBatch`. The thread runs every queued
  step of one batch as a single forward pass (see ``batching.py``).

``AURALITE_COMPUTE_THREAD=0`` turns all of this off (2.12 behaviour). The
thread never takes the engine's locks: callers hold them while they wait.
"""
from __future__ import annotations

import collections
import contextlib
import ctypes
import functools
import glob
import os
import queue
import threading
import time
from typing import Any, Callable, Iterator, TypeVar

__all__ = ["ComputeThread", "compute_thread", "compute_thread_enabled", "offloading",
           "on_compute_thread", "release_caller_pool", "release_compute_pool",
           "release_openmp_pool", "run_in_group", "run_inline", "run_on_compute_thread"]

F = TypeVar("F", bound=Callable[..., Any])

ENV_VAR = "AURALITE_COMPUTE_THREAD"

#: The OpenMP runtime whose pause frees only the *calling* thread's team: GNU
#: libgomp, which PyTorch's Linux wheels use. 2.14: Intel and LLVM OpenMP
#: (Windows and macOS wheels) are no longer paused. Their pause acts on every
#: team of the process at once, and the 2.13 Windows test run hung (the job
#: never finished; Intel OpenMP has had hangs in the very suspend code a soft
#: pause drives, llvm/llvm-project#55929). There `release_openmp_pool` is a
#: no-op returning False, as when no runtime is found.
_OMP_LIBRARIES = ("libgomp*.so*",)
_OMP_PAUSE_SOFT = 1                       # omp_pause_soft: free threads, keep state

_pause_fn: Any = None                     # None = not looked up yet, False = absent
_pause_lock = threading.Lock()
_local = threading.local()
#: The real Thread class, taken at import: code (tests, frameworks) that swaps
#: `threading.Thread` for an inline stand-in must not turn the compute thread
#: into a call that blocks its caller forever (2.14).
_Thread = threading.Thread


def compute_thread_enabled() -> bool:
    """False when ``AURALITE_COMPUTE_THREAD`` is ``0`` / ``false`` / ``no`` / ``off``."""
    return os.environ.get(ENV_VAR, "1").strip().lower() not in ("0", "false", "no", "off")


def _omp_pause_resource_all() -> Any:
    """``omp_pause_resource_all`` of the OpenMP runtime torch loaded, or None."""
    global _pause_fn
    if _pause_fn is None:
        with _pause_lock:
            if _pause_fn is None:
                _pause_fn = _find_pause_function() or False
    return _pause_fn or None


def _find_pause_function() -> Any:
    try:
        import torch
    except ImportError:                    # pragma: no cover - torch is a dependency
        return None
    torch_file = torch.__file__
    if not torch_file:                     # pragma: no cover - namespace package
        return None
    lib_dir = os.path.join(os.path.dirname(torch_file), "lib")
    for pattern in _OMP_LIBRARIES:
        for path in sorted(glob.glob(os.path.join(lib_dir, pattern))):
            try:
                fn = ctypes.CDLL(path).omp_pause_resource_all
            except (OSError, AttributeError):
                continue
            fn.argtypes = [ctypes.c_int]
            fn.restype = ctypes.c_int
            return fn
    return None


def release_openmp_pool() -> bool:
    """Free the calling thread's OpenMP thread team; True when it was released.

    The next parallel loop on this thread builds a new team (tens of
    microseconds). A no-op returning False when torch's OpenMP runtime is
    not GNU libgomp (2.14: Windows and macOS), or when called inside a
    parallel region.
    """
    fn = _omp_pause_resource_all()
    if fn is None:
        return False
    try:
        return int(fn(_OMP_PAUSE_SOFT)) == 0
    except Exception:                      # noqa: BLE001 - best effort
        return False


class _Job:
    __slots__ = ("call", "inference", "grad", "claim", "done", "result", "error",
                 "group", "item")

    def __init__(self, fn: Callable[..., Any] | None, args: tuple = (), kwargs: dict | None = None,
                 *, group: Any = None, item: Any = None) -> None:
        import torch
        self.call: tuple[Callable[..., Any], tuple, dict] | None = (
            None if fn is None else (fn, args, kwargs or {}))
        # 2.14: a job of a group (a DecodeBatch) carries an item instead of a
        # call; the thread hands every queued job of the group to
        # `group.run_jobs(jobs)` at once.
        self.group = group
        self.item = item
        self.inference = torch.is_inference_mode_enabled()
        self.grad = torch.is_grad_enabled()
        # Taken once, either by the compute thread (to run the call) or by the
        # caller (to cancel it): whoever gets it first decides.
        self.claim = threading.Lock()
        self.done = threading.Lock()
        self.done.acquire()
        self.result: Any = None
        self.error: BaseException | None = None

    def cancel(self) -> bool:
        """Drop the call if it has not started; False when it has (or is done)."""
        if not self.claim.acquire(blocking=False):
            return False
        self.call = None
        self.item = None
        return True

    def execute(self) -> None:
        import torch
        call, self.call = self.call, None          # drop the references when done
        try:
            assert call is not None
            fn, args, kwargs = call
            if self.inference:
                with torch.inference_mode():
                    self.result = fn(*args, **kwargs)
            else:
                with torch.inference_mode(False), torch.set_grad_enabled(self.grad):
                    self.result = fn(*args, **kwargs)
        except BaseException as exc:       # noqa: BLE001 - handed to the caller
            self.error = exc
        finally:
            call = None
            self.done.release()


class ComputeThread:
    """A daemon thread that runs submitted calls one at a time, in order."""

    def __init__(self, name: str = "auralite-compute") -> None:
        self.name = name
        self._jobs: queue.SimpleQueue = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()

    @property
    def alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def in_thread(self) -> bool:
        """Is the caller the compute thread?"""
        return threading.current_thread() is self._thread

    def _ensure_started(self) -> None:
        if self.alive:
            return
        with self._start_lock:
            if not self.alive:
                self._jobs = queue.SimpleQueue()
                thread = _Thread(target=self._loop, args=(self._jobs,),
                                 name=self.name, daemon=True)
                thread.start()
                self._thread = thread

    @staticmethod
    def _loop(jobs: queue.SimpleQueue) -> None:
        backlog: collections.deque = collections.deque()
        while True:
            job = backlog.popleft() if backlog else jobs.get()
            if job is None:
                if not backlog and jobs.empty():
                    return
                continue                   # submitted after stop(): keep serving
            if job.group is None:
                if job.claim.acquire(blocking=False):
                    job.execute()          # (otherwise cancelled by its caller)
                continue
            batch = [job]
            ComputeThread._gather(jobs, backlog, batch)
            _run_group(job.group, batch)

    @staticmethod
    def _gather(jobs: queue.SimpleQueue, backlog: collections.deque, batch: list) -> None:
        """Add every queued job of ``batch[0].group`` to `batch` (2.14).

        Takes what is queued now and, while the group says so
        (``group.batch_wait``), waits a moment for steps still on their way.
        Other jobs go to `backlog` and run right after the batch; one arriving
        during the wait ends it, so a new prompt is not held up.
        """
        group = batch[0].group
        if backlog:
            rest = [j for j in backlog if j is None or j.group is not group]
            batch.extend(j for j in backlog if j is not None and j.group is group)
            backlog.clear()
            backlog.extend(rest)
        deadline = None
        while True:
            try:
                job = jobs.get_nowait()
            except queue.Empty:
                wait = group.batch_wait(batch)
                if wait <= 0:
                    return
                now = time.monotonic()
                if deadline is None:
                    deadline = now + wait
                if now >= deadline:
                    return
                try:
                    job = jobs.get(timeout=deadline - now)
                except queue.Empty:
                    return
            if job is not None and job.group is group:
                batch.append(job)
                continue
            backlog.append(job)
            if deadline is not None:
                return                     # someone else is waiting: run now

    def _submit(self, job: _Job) -> Any:
        try:
            # put() is inside the try: Ctrl+C can land right after it returns.
            self._jobs.put(job)
            # A timeout keeps the wait interruptible by Ctrl+C on Windows too.
            while not job.done.acquire(timeout=0.25):
                pass
        except BaseException:
            if not job.cancel():
                while not job.done.acquire(timeout=0.25):
                    pass
            raise
        error = job.error
        if error is None:
            return job.result
        job.error = None
        try:
            raise error
        finally:
            error = None

    def run_in_group(self, group: Any, item: Any) -> Any:
        """Queue `item` for ``group.run_jobs`` and return the job's result (2.14).

        Jobs of the same group that are queued together run as one call;
        cancellation and Ctrl+C behave as in :meth:`run`. Inline (a group of
        one) on the compute thread itself.
        """
        if self.in_thread():
            return _run_group_inline(group, item)
        try:
            self._ensure_started()
        except RuntimeError:               # no new threads at interpreter shutdown
            return _run_group_inline(group, item)
        return self._submit(_Job(None, group=group, item=item))

    def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``fn(*args, **kwargs)`` on the compute thread and return its result.

        Inline when called on the compute thread. Ctrl+C while waiting drops
        the call if it has not started yet and otherwise lets it finish first,
        so the model is never used by two threads at once, and then raises
        KeyboardInterrupt.
        """
        if self.in_thread():
            return fn(*args, **kwargs)
        try:
            self._ensure_started()
        except RuntimeError:               # no new threads at interpreter shutdown
            return fn(*args, **kwargs)
        return self._submit(_Job(fn, args, kwargs))

    def stop(self, timeout: float | None = 5.0) -> None:
        """End the thread after the calls already submitted (tests, shutdown)."""
        thread = self._thread
        if thread is None or not thread.is_alive() or self.in_thread():
            return
        self._jobs.put(None)
        thread.join(timeout)

    def _after_fork_in_child(self) -> None:
        # Threads do not survive fork(): start a new one on first use.
        self._jobs = queue.SimpleQueue()
        self._thread = None
        self._start_lock = threading.Lock()


def _run_group(group: Any, jobs: list) -> None:
    """Run the jobs of one group that are still wanted, as one call (2.14)."""
    claimed = [job for job in jobs if job.claim.acquire(blocking=False)]
    if not claimed:
        return
    try:
        group.run_jobs(claimed)
    except BaseException as exc:           # noqa: BLE001 - handed to every caller
        for job in claimed:
            job.result = None
            job.error = exc
    finally:
        for job in claimed:
            job.item = None
            job.done.release()


def _run_group_inline(group: Any, item: Any) -> Any:
    job = _Job(None, group=group, item=item)
    group.run_jobs([job])
    return job.result


compute_thread = ComputeThread()


def _after_fork_in_child() -> None:
    global _pause_lock
    _pause_lock = threading.Lock()
    compute_thread._after_fork_in_child()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def _inline() -> bool:
    return (getattr(_local, "inline", 0) > 0 or compute_thread.in_thread()
            or not compute_thread_enabled())


def run_on_compute_thread(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """``fn(*args, **kwargs)`` on the compute thread (inline when that is off)."""
    if _inline():
        return fn(*args, **kwargs)
    return compute_thread.run(fn, *args, **kwargs)


def run_in_group(group: Any, item: Any) -> Any:
    """Queue a step of `group` (a :class:`DecodeBatch`) on the compute thread (2.14).

    Inline, as a group of one, when the compute thread is off or the caller
    is inside :func:`run_inline`.
    """
    if _inline():
        return _run_group_inline(group, item)
    return compute_thread.run_in_group(group, item)


def offloading() -> bool:
    """Do this thread's engine calls go to the compute thread (2.14)?"""
    return not _inline()


def on_compute_thread(fn: F) -> F:
    """Decorator: every call of ``fn`` runs through :func:`run_on_compute_thread`."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _inline():
            return fn(*args, **kwargs)
        return compute_thread.run(fn, *args, **kwargs)
    return wrapper  # type: ignore[return-value]


@contextlib.contextmanager
def run_inline() -> Iterator[None]:
    """Calls made by this thread inside the block stay on this thread."""
    _local.inline = getattr(_local, "inline", 0) + 1
    try:
        yield
    finally:
        _local.inline -= 1


def release_compute_pool() -> bool:
    """Free the compute thread's OpenMP team (False when the thread is not running)."""
    if not compute_thread.alive or not compute_thread_enabled():
        return False
    return bool(compute_thread.run(release_openmp_pool))


def release_caller_pool() -> bool:
    """Free the calling thread's team once its work is done (2.13 engine paths).

    Only with the compute thread on: then this thread's later engine calls run
    on the compute thread, and an idle team here would only slow that one down.
    """
    if not compute_thread_enabled() or compute_thread.in_thread():
        return False
    return release_openmp_pool()
