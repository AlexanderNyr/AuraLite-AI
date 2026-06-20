"""Datasets for in-memory and memory-mapped token corpora."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from . import CharDataset


class PagedDataset(Dataset):
    """Sliding-window dataset backed by a NumPy memory map.

    Use this for tokenized corpora that are too large to hold as a single RAM
    tensor.  The file is expected to contain a flat array of integer token ids
    (`uint16`, `uint32`, or `int64`). Each item reads only the requested window.
    """

    def __init__(self, path: str | Path, seq_length: int, *, dtype: str = "uint32", mode: str = "r"):
        self.path = Path(path)
        self.seq_length = int(seq_length)
        self.dtype = np.dtype(dtype)
        self.data = np.memmap(self.path, dtype=self.dtype, mode=mode)
        self._len = max(0, int(self.data.shape[0]) - self.seq_length)

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
        window = np.asarray(self.data[idx: idx + self.seq_length + 1], dtype=np.int64)
        x = torch.from_numpy(window[:-1].copy()).long()
        y = torch.from_numpy(window[1:].copy()).long()
        return x, y


__all__ = ["CharDataset", "PagedDataset"]
