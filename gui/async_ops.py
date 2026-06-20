"""Tkinter asyncio/root.after queue bridge."""
from __future__ import annotations
import asyncio
import queue
import threading
from typing import Callable

class TkAsyncBridge:
    def __init__(self, root, interval_ms: int = 50):
        self.root = root
        self.interval_ms = interval_ms
        self.queue: queue.Queue[tuple[Callable, tuple, dict]] = queue.Queue()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def submit(self, coro, callback: Callable | None = None):
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        if callback:
            fut.add_done_callback(lambda f: self.queue.put((callback, (f,), {})))
        return fut

    def poll(self):
        while not self.queue.empty():
            fn, args, kwargs = self.queue.get_nowait()
            fn(*args, **kwargs)
        self.root.after(self.interval_ms, self.poll)
