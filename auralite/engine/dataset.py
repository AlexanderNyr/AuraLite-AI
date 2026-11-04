"""Datasets for in-memory and memory-mapped token corpora."""
from __future__ import annotations

import bisect
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

#: Target id that the loss skips (``nn.CrossEntropyLoss``'s ``ignore_index``, 2.15).
IGNORE_INDEX = -100

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

    `targets` (2.15, optional) is a tensor as long as `encoded` that the
    windows' targets ``y`` come from instead: the token ids where the model
    must learn the next token and ``IGNORE_INDEX`` (-100, what
    ``nn.CrossEntropyLoss`` skips) elsewhere. Chat fine-tuning trains only on
    the assistant's replies this way.
    """

    def __init__(self, encoded: torch.Tensor, seq_length: int, stride: int = 1,
                 targets: torch.Tensor | None = None):
        self.data = encoded.contiguous()
        self.targets = targets.contiguous() if targets is not None else None
        if self.targets is not None and self.targets.shape != self.data.shape:
            raise ValueError(f"targets must have the shape of the tokens "
                             f"({tuple(self.data.shape)}), got {tuple(self.targets.shape)}")
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
        if self.targets is not None:
            self.y = self.targets[self.window_offset:].unfold(
                0, self.seq_length + 1, self.stride)[:self._len][:, 1:]

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def get_batch(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Windows `indices` as one ``(len(indices), seq_length)`` batch (one gather)."""
        starts = indices * self.stride + self.window_offset
        positions = starts[:, None] + self._arange
        windows = self.data[positions]
        if self.targets is not None:
            return windows[:, :-1], self.targets[positions][:, 1:]
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


class ShardedDataset(Dataset):
    """Windows of token shards: flat files of token ids on disk (2.18, big datasets).

    `PagedDataset` over several files, as ``auralite data prepare`` writes
    them (:class:`~auralite.engine.shards.TokenShards`). `shards` are
    ``(path, tokens)`` pairs, or ``(path, tokens, start)`` for the tokens of
    a file from token `start` on; every file is memory-mapped and a batch reads
    only the pages of its windows, so a dataset far larger than the memory
    trains. A window never spans two files: a file shorter than
    ``seq_length + 1`` tokens gives none. `stride` and ``set_window_offset``
    work as in `PagedDataset`; a file shorter than the offset uses the
    largest offset it can. At most `open_files` files are mapped at a time
    (a terabyte of shards is thousands of files).
    """

    def __init__(self, shards: Iterable[tuple], seq_length: int, *,
                 dtype: str = "uint16", stride: int = 1, open_files: int = 128):
        if seq_length < 1:
            raise ValueError(f"seq_length must be >= 1, got {seq_length}")
        self.seq_length = int(seq_length)
        self.stride = max(1, int(stride))
        self.dtype = np.dtype(dtype)
        self.n_tokens = 0
        self._files: list[tuple[str, int, int]] = []
        self._spans: list[int] = []
        self._open: dict[int, np.memmap] = {}
        self._open_files = max(1, int(open_files))
        counts: list[int] = []
        for entry in shards:
            path, tokens = entry[0], int(entry[1])
            start = int(entry[2]) if len(entry) > 2 else 0
            self.n_tokens += tokens
            span = tokens - self.seq_length
            if span < 1:
                continue
            self._files.append((str(path), tokens, start))
            self._spans.append(span)
            counts.append(_window_count(span, self.stride))
        self._starts = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]).astype(np.int64)
        self._len = int(self._starts[-1])
        self._arange = np.arange(self.seq_length + 1, dtype=np.int64)
        self.window_offset = 0

    @property
    def shards(self) -> int:
        """How many files have at least one window."""
        return len(self._files)

    def _map(self, s: int) -> np.memmap:
        found = self._open.pop(s, None)
        if found is None:
            if len(self._open) >= self._open_files:
                self._open.pop(next(iter(self._open)))       # the least recently used
            path, tokens, start = self._files[s]
            found = np.memmap(path, dtype=self.dtype, mode="r", shape=(tokens,),
                              offset=start * self.dtype.itemsize)
        self._open[s] = found
        return found

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_open"] = {}
        return state

    @property
    def max_window_offset(self) -> int:
        """Largest valid argument of :meth:`set_window_offset`."""
        return max((min(self.stride, span) - 1 for span in self._spans), default=0)

    def set_window_offset(self, offset: int) -> None:
        offset = int(offset)
        if not 0 <= offset <= self.max_window_offset:
            raise ValueError(f"window offset must be in [0, {self.max_window_offset}], "
                             f"got {offset}")
        self.window_offset = offset

    def __len__(self) -> int:
        return self._len

    def _windows(self, idx: np.ndarray) -> np.ndarray:
        if len(idx) and (int(idx.min()) < 0 or int(idx.max()) >= self._len):
            raise IndexError(f"window index out of range [0, {self._len})")
        shard = np.searchsorted(self._starts, idx, side="right") - 1
        local = idx - self._starts[shard]
        out = np.empty((len(idx), self.seq_length + 1), dtype=np.int64)
        for s in np.unique(shard):
            rows = np.nonzero(shard == s)[0]
            offset = min(self.window_offset, max(0, min(self.stride, self._spans[s]) - 1))
            starts = local[rows] * self.stride + offset
            out[rows] = self._map(int(s))[starts[:, None] + self._arange]
        return out

    def __getitem__(self, idx: int):
        window = torch.from_numpy(self._windows(np.asarray([int(idx)], dtype=np.int64))[0])
        return window[:-1].clone(), window[1:].clone()

    def get_batch(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Windows `indices` as one ``(len(indices), seq_length)`` batch."""
        idx = np.asarray(indices.numpy() if isinstance(indices, torch.Tensor) else indices,
                         dtype=np.int64)
        windows = torch.from_numpy(self._windows(idx))
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


#: Above this many windows an epoch's order is not a `torch.randperm` (8 bytes
#: per window) but a keyed pseudo-random permutation computed batch by batch
#: (2.18): a 100 GB dataset has billions of windows.
LAZY_SHUFFLE_WINDOWS = 1 << 24

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64's finalizer on uint64 arrays (wraps modulo 2**64)."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def permute_indices(positions: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Element `positions` of a pseudo-random permutation of ``range(n)`` (2.18).

    A keyed 4-round Feistel network on the smallest even number of bits that
    covers `n`, with cycle walking for the values it maps past ``n - 1``: a
    bijection of ``range(n)`` that needs no memory, so any batch of an epoch
    (and the first batch after a resume) is computed on its own. `seed`
    picks the permutation.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    bits = max(2, (int(n) - 1).bit_length())
    bits += bits & 1
    half = np.uint64(bits // 2)
    mask = np.uint64((1 << (bits // 2)) - 1)
    keys = [_mix64(np.asarray([(int(seed) * 4 + r + 1) & 0xFFFFFFFFFFFFFFFF], dtype=np.uint64))[0]
            for r in range(4)]

    def rounds(x: np.ndarray) -> np.ndarray:
        left, right = x >> half, x & mask
        for key in keys:
            left, right = right, left ^ (_mix64(right ^ key) & mask)
        return (left << half) | right

    with np.errstate(over="ignore"):
        y = rounds(np.asarray(positions, dtype=np.uint64) & _MASK64)
        outside = y >= np.uint64(n)
        while outside.any():
            y[outside] = rounds(y[outside])
            outside = y >= np.uint64(n)
    return y.astype(np.int64)


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
    * 2.18: ``skip_batches(k)`` starts the next pass at batch `k` (a resumed
      run continues its epoch exactly). Above `LAZY_SHUFFLE_WINDOWS` windows
      the order comes from :func:`permute_indices`, one batch at a time.
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
        self._skip = 0

    def _generator(self, salt: int) -> torch.Generator:
        return torch.Generator().manual_seed(
            (self.seed * 1_000_003 + self.epoch * 7_919 + salt) % (2 ** 63))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if hasattr(self.dataset, "set_epoch"):        # 2.16: packed dialogs
            self.dataset.set_epoch(self.epoch, self._generator(3))
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

    def skip_batches(self, count: int) -> None:
        """Start the next pass over the data at batch `count` (2.18, resumed runs)."""
        self._skip = max(0, int(count))

    def __iter__(self):
        n = len(self.dataset)
        first, self._skip = self._skip, 0
        if n == 0:
            return
        if self.shuffle and n > LAZY_SHUFFLE_WINDOWS:
            yield from self._lazy_batches(n, first)
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
        for start in range(first * self.batch_size, stop, self.batch_size):
            xb, yb = self.dataset.get_batch(order[start:start + self.batch_size])
            if self.pin_memory:
                xb, yb = xb.pin_memory(), yb.pin_memory()
            yield xb, yb

    def _lazy_batches(self, n: int, first: int):
        """The batches of an epoch of `n` windows, ordered by `permute_indices`."""
        seed = int(torch.randint(0, 2 ** 62, (1,), generator=self._generator(2)))
        stop = len(self) * self.batch_size if self.drop_last else self._per_rank()
        for start in range(first * self.batch_size, stop, self.batch_size):
            local = np.arange(start, min(start + self.batch_size, stop), dtype=np.int64)
            positions = (local * self.world_size + self.rank) % n
            order = torch.from_numpy(permute_indices(positions, n, seed))
            xb, yb = self.dataset.get_batch(order)
            if self.pin_memory:
                xb, yb = xb.pin_memory(), yb.pin_memory()
            yield xb, yb


class PackedDataset(Dataset):
    """Windows that each hold whole pieces of text: packed dialogs (2.16).

    `ids` and `targets` are ``(n, seq_length + 1)`` tensors from
    :func:`pack_sequences`; window ``i`` gives ``x = ids[i, :-1]`` and ``y =
    targets[i, 1:]``. A window starts where a dialog starts and ends after the
    last dialog that fits in it (the rest is padding with an ``IGNORE_INDEX``
    target), so no reply is learned without the question before it. There is
    no stride and no window offset.

    Given the `pieces` the windows were packed from, ``set_epoch`` (which
    `WindowBatches` calls) packs them again for each epoch with ties between
    pieces of one length broken at random: the same number of windows, each as
    full as before, but different pieces side by side in every epoch.
    """

    def __init__(self, ids: torch.Tensor, targets: torch.Tensor,
                 pieces: list[tuple[list[int], list[int]]] | None = None):
        if ids.dim() != 2 or ids.shape != targets.shape or ids.shape[1] < 2:
            raise ValueError(f"ids and targets must be (n, seq_length + 1) tensors of one "
                             f"shape, got {tuple(ids.shape)} and {tuple(targets.shape)}")
        self.ids = ids.contiguous()
        self.targets = targets.contiguous()
        self.seq_length = int(ids.shape[1]) - 1
        self.pieces = pieces

    def set_epoch(self, epoch: int, generator: torch.Generator) -> None:
        """Pack the pieces again for `epoch` (see the class docs); no-op without them."""
        if self.pieces:
            ids, targets = pack_sequences(self.pieces, self.seq_length + 1,
                                          generator=generator)
            if ids.shape == self.ids.shape:
                self.ids, self.targets = ids, targets

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def __getitem__(self, idx: int):
        return self.ids[idx, :-1], self.targets[idx, 1:]

    def get_batch(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Windows `indices` as one ``(len(indices), seq_length)`` batch."""
        return self.ids[indices][:, :-1], self.targets[indices][:, 1:]


def pack_sequences(sequences: list[tuple[list[int], list[int]]], width: int,
                   pad_id: int = 0, *, generator: torch.Generator | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(ids, targets)`` tensors of shape ``(n, width)`` holding `sequences` (2.16).

    Each sequence is ``(ids, targets)`` of at most `width` tokens and is kept
    whole. Best-fit decreasing: the longest sequence goes first, each into the
    window with the least room left that it fits in (a new window if none
    does). In a window the sequences keep their order; the room after them is
    `pad_id` with an ``IGNORE_INDEX`` target. With a `generator`, sequences of
    one length go in a random order (and keep it in their windows): the
    windows are as many and as full, with other sequences side by side.
    """
    width = int(width)
    rank = (torch.randperm(len(sequences), generator=generator).tolist()
            if generator is not None else list(range(len(sequences))))
    order = sorted(range(len(sequences)), key=lambda i: (-len(sequences[i][0]), rank[i]))
    room: list[tuple[int, int]] = []          # (free tokens, window), sorted
    windows: list[list[int]] = []
    for i in order:
        n = len(sequences[i][0])
        if n > width:
            raise ValueError(f"a sequence of {n} tokens does not fit a window of {width}")
        if n == 0:
            continue
        k = bisect.bisect_left(room, (n, -1))
        if k < len(room):
            free, w = room.pop(k)
        else:
            free, w = width, len(windows)
            windows.append([])
        windows[w].append(i)
        if free - n > 0:
            bisect.insort(room, (free - n, w))
    ids = torch.full((len(windows), width), int(pad_id), dtype=torch.long)
    targets = torch.full((len(windows), width), IGNORE_INDEX, dtype=torch.long)
    for w, members in enumerate(windows):
        at = 0
        for i in sorted(members, key=rank.__getitem__):
            seq_ids, seq_targets = sequences[i]
            n = len(seq_ids)
            ids[w, at:at + n] = torch.as_tensor(seq_ids, dtype=torch.long)
            targets[w, at:at + n] = torch.as_tensor(seq_targets, dtype=torch.long)
            at += n
    return ids, targets


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


__all__ = ["IGNORE_INDEX", "LAZY_SHUFFLE_WINDOWS", "CharDataset", "PackedDataset", "PagedDataset",
           "ShardedDataset", "WindowBatches", "pack_sequences", "permute_indices", "spread_indices",
           "validation_split_point", "write_tokens_memmap"]
