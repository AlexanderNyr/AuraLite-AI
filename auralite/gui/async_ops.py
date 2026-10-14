"""gui/async_ops.py — Tk-safe bridge to background work (v2.6.5).

Tkinter is single-threaded: widgets may only be touched from the main thread.
The GUI already runs long jobs (training, generation, agent loops) in worker
threads and schedules UI updates with ``root.after``. :class:`TkAsyncBridge`
centralises that pattern:

* :meth:`TkAsyncBridge.call` runs a *blocking* function in the executor and
  hands its result to a callback on the Tk thread (the case the tabs need);
* :meth:`TkAsyncBridge.submit` does the same for coroutines on a dedicated
  asyncio loop;
* :meth:`TkAsyncBridge.poll` drains finished callbacks from the Tk event loop
  and is robust: a callback that raises is logged, not fatal;
* :meth:`TkAsyncBridge.shutdown` stops the loop and the executor, and the class
  can be used as a context manager.

The previous version of this module unconditionally started a thread and an
event loop and left both running after the window was closed.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

__all__ = ["TkAsyncBridge"]

logger = logging.getLogger("auralite.gui")


class TkAsyncBridge:
    """Run background work and deliver the results on the Tk main thread."""

    def __init__(self, root, interval_ms: int = 50, max_workers: int = 2):
        self.root = root
        self.interval_ms = int(interval_ms)
        self.queue: queue.Queue[tuple[Callable, tuple, dict]] = queue.Queue()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever,
                                       name="auralite-tk-async", daemon=True)
        self.thread.start()
        self.executor = ThreadPoolExecutor(max_workers=max_workers,
                                           thread_name_prefix="auralite-gui")
        self._polling = True
        self._closed = False
        self._scheduling = False
        self.poll()

    # ---- background work ---------------------------------------------
    def submit(self, coro, callback: Callable | None = None):
        """Schedule a coroutine on the bridge's loop; result → ``callback``."""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        if callback is not None:
            future.add_done_callback(
                lambda fut: self.queue.put((callback, (fut,), {})))
        return future

    def call(self, func: Callable, *args, callback: Callable | None = None,
             **kwargs):
        """Run a blocking ``func`` in the executor.

        ``callback(result)`` — or ``callback(result, None)``, see below — is
        invoked later on the Tk thread via :meth:`poll`. Errors are delivered
        to ``callback(result=None, error=<exception>)`` so the tab can show a
        message box instead of silently losing the failure.
        """
        def _run():
            try:
                result = func(*args, **kwargs)
                self.queue.put((_deliver, (callback, result, None), {}))
                return result
            except Exception as exc:  # noqa: BLE001 - reported to the UI thread
                logger.exception("background task failed: %s", exc)
                self.queue.put((_deliver, (callback, None, exc), {}))
                return None

        return self.executor.submit(_run)

    # ---- Tk main thread ----------------------------------------------
    def poll(self) -> None:
        """Drain queued callbacks. Called from Tk's event loop.

        Re-entrancy guard: test doubles (and a few real Tk wrappers) implement
        ``after()`` by calling the callback *immediately*, which would make the
        "schedule the next poll" step recurse forever. While we are inside that
        scheduling call the guard simply returns.
        """
        if self._closed or self._scheduling:
            return
        while True:
            try:
                func, args, kwargs = self.queue.get_nowait()
            except queue.Empty:
                break
            try:
                func(*args, **kwargs)
            except Exception:  # a broken callback must not kill the loop
                logger.exception("Tk callback raised; continuing")
        if self._polling:
            self._scheduling = True
            try:
                self.root.after(self.interval_ms, self.poll)
            except Exception:      # window already destroyed
                self._polling = False
            finally:
                self._scheduling = False

    def stop_polling(self) -> None:
        self._polling = False

    # ---- lifecycle ----------------------------------------------------
    def shutdown(self, timeout: float = 1.0) -> None:
        """Stop polling, drain the loop and the executor. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._polling = False
        self.executor.shutdown(wait=False)
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except RuntimeError:
            pass
        self.thread.join(timeout=timeout)
        if not self.thread.is_alive():
            try:
                self.loop.close()
            except Exception:       # pragma: no cover - already closed
                pass

    def __enter__(self) -> "TkAsyncBridge":
        return self

    def __exit__(self, *exc_info) -> None:
        self.shutdown()


def _deliver(callback: Callable | None, result: Any, error: BaseException | None) -> None:
    """Call ``callback`` with the bridge's supported signatures."""
    if callback is None:
        if error is not None:
            logger.error("background task failed with no callback: %s", error)
        return
    if error is None:
        callback(result)
        return
    try:
        callback(None, error)
    except TypeError:
        callback(None)
