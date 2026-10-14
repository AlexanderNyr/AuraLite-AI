"""Datasets for in-memory and memory-mapped token corpora."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    """Sliding-window token-level dataset backed by one in-memory LongTensor.

    The full corpus is tokenised exactly once before training and stored as a
    single `torch.LongTensor`. We then create window *views* via `unfold()` so
    the DataLoader reads pre-tokenised tensors instead of repeatedly slicing raw
    Python text/list data on the hot path.
    """

    def __init__(self, encoded: torch.Tensor, seq_length: int):
        self.data = encoded.contiguous()
        self.seq_length = seq_length

        if len(self.data) <= self.seq_length:
            self.x = self.data.new_empty((0, self.seq_length))
            self.y = self.data.new_empty((0, self.seq_length))
        else:
            windows = self.data.unfold(0, self.seq_length + 1, 1)
            self.x = windows[:, :-1]
            self.y = windows[:, 1:]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]



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
                 mode: str = "r", offset: int = 0, limit: int | None = None):
        self.path = Path(path)
        self.seq_length = int(seq_length)
        self.dtype = np.dtype(dtype)
        self.data = np.memmap(self.path, dtype=self.dtype, mode=mode)
        total = int(self.data.shape[0])
        self.offset = max(0, min(int(offset), total))          # start token id
        end = total if limit is None else min(total, self.offset + max(0, int(limit)))
        self._end = end
        self._len = max(0, end - self.offset - self.seq_length)

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
        start = self.offset + idx
        window = np.asarray(self.data[start: start + self.seq_length + 1], dtype=np.int64)
        x = torch.from_numpy(window[:-1].copy()).long()
        y = torch.from_numpy(window[1:].copy()).long()
        return x, y


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


__all__ = ["CharDataset", "PagedDataset", "write_tokens_memmap"]
