"""Datasets for in-memory and memory-mapped token corpora."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

def validation_split_point(text: str, val_split: float, search: int = 2000) -> int:
    """Index of the first validation character: the last `val_split` of `text` (2.13).

    The cut moves forward to just after the next line break (or space) within
    `search` characters, so no word is cut in two. ``len(text)`` means there
    is nothing to hold out. `train()` fits a BPE tokenizer on the text before
    this point only: fitted on the whole text, it learned merges for words that
    occur only in the held-out tail, and the validation loss came out too low
    (docs/training.md, section 10).
    """
    n = len(text)
    if n == 0 or not 0.0 < float(val_split) < 1.0:
        return n
    cut = int(n * (1.0 - float(val_split)))
    for sep in ("\n", " "):
        found = text.find(sep, cut, min(n, cut + int(search)))
        if found != -1:
            return found + 1
    return cut


def _window_count(span: int, stride: int) -> int:
    """Windows per pass for `span` stride-1 windows and the given stride.

    Constant for every window offset in ``[0, min(stride, span))``, so the
    length of an epoch does not depend on the offset drawn for it. A corpus
    shorter than one stride still yields one window.
    """
    if span <= 0:
        return 0
    return max(1, span // stride)


class CharDataset(Dataset):
    """Token windows over one in-memory LongTensor.

    The corpus is tokenised once and stored as a single `torch.LongTensor`;
    windows are *views* created with `unfold()`, so nothing is copied per item.

    ``stride=1`` (the default when the class is used directly) yields every
    window — consecutive samples overlap by ``seq_length - 1`` tokens.
    ``stride=seq_length`` yields non-overlapping windows: one pass over the
    dataset then sees every token once, which is what the engine uses for an
    epoch (2.9). `set_window_offset` shifts the window grid; the engine draws a
    fresh offset per epoch so different epochs see different windows.
    """

    def __init__(self, encoded: torch.Tensor, seq_length: int, stride: int = 1):
        self.data = encoded.contiguous()
        self.seq_length = int(seq_length)
        self.stride = max(1, int(stride))
        self._span = max(0, len(self.data) - self.seq_length)   # stride-1 windows
        self._len = _window_count(self._span, self.stride)
        self.window_offset = 0
        self._arange = torch.arange(self.seq_length + 1)
        self._build_views()

    @property
    def max_window_offset(self) -> int:
        """Largest valid argument of :meth:`set_window_offset`."""
        return max(0, min(self.stride, self._span) - 1)

    def set_window_offset(self, offset: int) -> None:
        offset = int(offset)
        if not 0 <= offset <= self.max_window_offset:
            raise ValueError(f"window offset must be in [0, {self.max_window_offset}], "
                             f"got {offset}")
        if offset != self.window_offset:
            self.window_offset = offset
            self._build_views()

    def _build_views(self) -> None:
        if self._len == 0:
            self.x = self.data.new_empty((0, self.seq_length))
            self.y = self.data.new_empty((0, self.seq_length))
            return
        windows = self.data[self.window_offset:].unfold(
            0, self.seq_length + 1, self.stride)[:self._len]
        self.x = windows[:, :-1]
        self.y = windows[:, 1:]

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def get_batch(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Windows `indices` as one ``(len(indices), seq_length)`` batch (one gather)."""
        starts = indices * self.stride + self.window_offset
        windows = self.data[starts[:, None] + self._arange]
        return windows[:, :-1], windows[:, 1:]


class PagedDataset(Dataset):
    """Sliding-window dataset backed by a NumPy memory map.

    Use this for tokenized corpora that are too large to hold as a single RAM
    tensor.  The file is expected to contain a flat array of integer token ids
    (`uint16`, `uint32`, or `int64`). Each item reads only the requested window.

    v2.6.5: `offset` / `limit` allow the train and validation slices of one
    memory-mapped corpus to share a single file (training uses the mmap path
    automatically when `corpus_cache="mmap"`, see `AuraLiteEngine.train`).
    """

    def __init__(self, path: str | Path, seq_length: int, *, dtype: str = "uint32",
                 mode: str = "r", offset: int = 0, limit: int | None = None,
                 stride: int = 1):
        self.path = Path(path)
        self.seq_length = int(seq_length)
        self.dtype = np.dtype(dtype)
        self.data = np.memmap(self.path, dtype=self.dtype, mode=mode)
        total = int(self.data.shape[0])
        self.offset = max(0, min(int(offset), total))          # start token id
        end = total if limit is None else min(total, self.offset + max(0, int(limit)))
        self._end = end
        # 2.9: `stride` / `set_window_offset` work as in CharDataset.
        self.stride = max(1, int(stride))
        self._span = max(0, end - self.offset - self.seq_length)
        self._len = _window_count(self._span, self.stride)
        self.window_offset = 0
        self._arange = np.arange(self.seq_length + 1, dtype=np.int64)

    @property
    def max_window_offset(self) -> int:
        """Largest valid argument of :meth:`set_window_offset`."""
        return max(0, min(self.stride, self._span) - 1)

    def set_window_offset(self, offset: int) -> None:
        offset = int(offset)
        if not 0 <= offset <= self.max_window_offset:
            raise ValueError(f"window offset must be in [0, {self.max_window_offset}], "
                             f"got {offset}")
        self.window_offset = offset

    @property
    def n_tokens(self) -> int:
        """How many token ids this view covers (without the window slack)."""
        return self._end - self.offset

    @classmethod
    def from_tokens(cls, path: str | Path, tokens: Iterable[int], seq_length: int,
                    *, dtype: str = "uint32") -> "PagedDataset":
        arr = np.asarray(list(tokens), dtype=np.dtype(dtype))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mm = np.memmap(path, dtype=arr.dtype, mode="w+", shape=arr.shape)
        mm[:] = arr[:]
        mm.flush()
        del mm
        return cls(path, seq_length, dtype=dtype)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self._len:
            raise IndexError(idx)
        start = self.offset + self.window_offset + idx * self.stride
        window = np.asarray(self.data[start: start + self.seq_length + 1], dtype=np.int64)
        x = torch.from_numpy(window[:-1].copy()).long()
        y = torch.from_numpy(window[1:].copy()).long()
        return x, y

    def get_batch(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Windows `indices` as one ``(len(indices), seq_length)`` batch.

        One fancy-index into the memmap: only the pages of these windows are read.
        """
        starts = (indices.numpy().astype(np.int64) * self.stride
                  + (self.offset + self.window_offset))
        windows = torch.from_numpy(
            np.asarray(self.data[starts[:, None] + self._arange], dtype=np.int64))
        return windows[:, :-1], windows[:, 1:]


def write_tokens_memmap(path: str | Path, chunks: Iterable, *,
                        dtype: str = "uint32") -> int:
    """Write token-id chunks to a flat memory-mapped file; return the count.

    Chunks are written as they arrive, so the corpus never exists as one Python
    list — that is the whole point of the mmap training path. The mapping grows
    geometrically and the file is truncated to the exact token count afterwards.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dt = np.dtype(dtype)
    total = 0
    capacity = 0
    mm = None
    for chunk in chunks:
        arr = np.asarray(list(chunk), dtype=dt)
        if arr.size == 0:
            continue
        if mm is None:
            capacity = max(int(arr.size), 1 << 20)
            mm = np.memmap(path, dtype=dt, mode="w+", shape=(capacity,))
        elif total + arr.size > capacity:
            new_capacity = max(capacity * 2, total + arr.size)
            mm.flush()
            mm = np.memmap(path, dtype=dt, mode="r+", shape=(new_capacity,))
            capacity = new_capacity
        mm[total:total + arr.size] = arr
        total += arr.size
    if mm is None:                            # empty corpus
        mm = np.memmap(path, dtype=dt, mode="w+", shape=(0,))
    mm.flush()
    mm = None                                 # close the mapping before truncating
    with open(path, "r+b") as fh:
        fh.truncate(total * dt.itemsize)
    return total


class WindowBatches:
    """In-process batches for `CharDataset` / `PagedDataset` (2.9).

    Replaces a `DataLoader` in `AuraLiteEngine.train`. A batch is one gather
    from the token tensor (≈0.02 ms for 32×64 tokens), so there is nothing to
    parallelise; worker processes only cost start-up time and IPC — on Windows
    every worker is a fresh interpreter that re-imports torch and receives a
    pickled copy of the whole corpus.

    * ``set_epoch(e)`` picks the epoch's permutation and, when the dataset has
      a stride > 1 and ``random_offsets`` is on, a fresh window offset. Both
      come from ``seed`` and ``e``, so every DDP rank agrees on them.
    * ``rank`` / ``world_size`` shard each epoch like `DistributedSampler`
      (``drop_last=False``): the permutation is padded by wrapping around so
      that every rank runs the same number of batches.
    """

    def __init__(self, dataset, batch_size: int, *, shuffle: bool = True,
                 seed: int = 0, random_offsets: bool = True, rank: int = 0,
                 world_size: int = 1, pin_memory: bool = False,
                 drop_last: bool = False):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not 0 <= rank < max(1, world_size):
            raise ValueError(f"rank {rank} is outside world_size {world_size}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.random_offsets = bool(random_offsets)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.pin_memory = bool(pin_memory)
        # A short last batch is a new input shape — torch.compile would build a
        # second graph for it, so the compiled training path drops it.
        self.drop_last = bool(drop_last) and self._per_rank() >= self.batch_size
        self.epoch = 0

    def _generator(self, salt: int) -> torch.Generator:
        return torch.Generator().manual_seed(
            (self.seed * 1_000_003 + self.epoch * 7_919 + salt) % (2 ** 63))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        max_offset = int(getattr(self.dataset, "max_window_offset", 0))
        if self.random_offsets and max_offset > 0:
            offset = int(torch.randint(0, max_offset + 1, (1,),
                                       generator=self._generator(1)))
            self.dataset.set_window_offset(offset)

    def _per_rank(self) -> int:
        return math.ceil(len(self.dataset) / self.world_size)

    def __len__(self) -> int:
        if self.drop_last:
            return self._per_rank() // self.batch_size
        return math.ceil(self._per_rank() / self.batch_size)

    def __iter__(self):
        n = len(self.dataset)
        if n == 0:
            return
        if self.shuffle:
            order = torch.randperm(n, generator=self._generator(2))
        else:
            order = torch.arange(n)
        if self.world_size > 1:
            total = self._per_rank() * self.world_size
            if total > n:
                order = torch.cat([order, order[:total - n]])
            order = order[self.rank::self.world_size]
        stop = len(self) * self.batch_size if self.drop_last else len(order)
        for start in range(0, stop, self.batch_size):
            xb, yb = self.dataset.get_batch(order[start:start + self.batch_size])
            if self.pin_memory:
                xb, yb = xb.pin_memory(), yb.pin_memory()
            yield xb, yb


def spread_indices(n: int, k: int) -> torch.Tensor:
    """`k` indices spread evenly over ``range(n)`` (all of them when ``k >= n``)."""
    if k >= n:
        return torch.arange(n)
    return torch.linspace(0, n - 1, k).round().long().unique()


class _Subset:
    """The windows `indices` of a dataset, with the dataset's `get_batch`."""

    def __init__(self, dataset, indices: torch.Tensor):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def get_batch(self, positions: torch.Tensor):
        return self.dataset.get_batch(self.indices[positions])


__all__ = ["CharDataset", "PagedDataset", "WindowBatches", "spread_indices",
           "validation_split_point", "write_tokens_memmap"]
