"""Read and write ``.safetensors`` files without the safetensors package (2.18.1).

The format is simple: an 8-byte little-endian header length, a JSON header
(name -> dtype, shape, byte range; plus an optional ``__metadata__`` dict of
strings) and the raw little-endian tensor bytes. Hugging Face models ship
their weights this way, so importing and exporting them needs no extra
package. Tensors are read one at a time (a 7B model is never in memory twice)
and written one at a time from callables, for the same reason.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Callable, Iterable, Iterator

import torch

__all__ = ["DTYPES", "SafetensorsFile", "TensorSpec", "write_safetensors", "dtype_name"]

#: safetensors dtype names -> torch dtypes (the ones torch has)
DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool,
}
for _name, _attr in (("F8_E4M3", "float8_e4m3fn"), ("F8_E5M2", "float8_e5m2")):
    if hasattr(torch, _attr):
        DTYPES[_name] = getattr(torch, _attr)
_NAMES = {dtype: name for name, dtype in DTYPES.items()}
_MAX_HEADER = 100 * 1024 * 1024


def dtype_name(dtype: torch.dtype) -> str:
    try:
        return _NAMES[dtype]
    except KeyError:
        raise ValueError(f"safetensors cannot store {dtype}") from None


class SafetensorsFile:
    """One ``.safetensors`` file: its header, and its tensors on demand."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError(f"{self.path}: not a safetensors file (too short)")
            (length,) = struct.unpack("<Q", raw)
            if length > _MAX_HEADER:
                raise ValueError(f"{self.path}: not a safetensors file (header of {length} bytes)")
            try:
                header = json.loads(f.read(length))
            except ValueError as err:
                raise ValueError(f"{self.path}: not a safetensors file ({err})") from err
        if not isinstance(header, dict):
            raise ValueError(f"{self.path}: not a safetensors file")
        self.metadata: dict[str, str] = dict(header.pop("__metadata__", None) or {})
        self.entries: dict[str, dict] = header
        self.data_start = 8 + length
        size = self.path.stat().st_size
        for name, entry in self.entries.items():
            start, end = entry.get("data_offsets", (0, -1))
            if entry.get("dtype") not in DTYPES:
                continue                              # refused only when read
            if not 0 <= start <= end or self.data_start + end > size:
                raise ValueError(f"{self.path}: tensor {name!r} lies outside the file")

    def keys(self) -> list[str]:
        return list(self.entries)

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(int(s) for s in self.entries[name]["shape"])

    def dtype(self, name: str) -> str:
        return str(self.entries[name]["dtype"])

    def get(self, name: str) -> torch.Tensor:
        """Tensor `name`, read from the file (its own memory, not a mapping)."""
        entry = self.entries[name]
        kind = str(entry["dtype"])
        if kind not in DTYPES:
            raise ValueError(f"{self.path}: tensor {name!r} has the dtype {kind}, which this "
                             f"torch cannot hold")
        dtype = DTYPES[kind]
        shape = [int(s) for s in entry["shape"]]
        start, end = (int(v) for v in entry["data_offsets"])
        n = end - start
        if n == 0:
            return torch.empty(shape, dtype=dtype)
        buffer = bytearray(n)
        with open(self.path, "rb") as f:
            f.seek(self.data_start + start)
            if f.readinto(buffer) != n:
                raise ValueError(f"{self.path}: tensor {name!r} is cut short")
        return torch.frombuffer(buffer, dtype=dtype).reshape(shape)


class TensorSpec:
    """A tensor to write: its name, shape and dtype now, its values on demand."""

    __slots__ = ("name", "shape", "dtype", "produce")

    def __init__(self, name: str, shape: Iterable[int], dtype: torch.dtype,
                 produce: Callable[[], torch.Tensor]):
        self.name = str(name)
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.produce = produce

    @property
    def nbytes(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n * torch.empty((), dtype=self.dtype).element_size()


def _header(specs: list[TensorSpec], metadata: dict | None) -> bytes:
    header: dict = {}
    offset = 0
    for spec in specs:
        header[spec.name] = {"dtype": dtype_name(spec.dtype), "shape": list(spec.shape),
                             "data_offsets": [offset, offset + spec.nbytes]}
        offset += spec.nbytes
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    blob = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return blob + b" " * ((-len(blob)) % 8)


def _tensor_bytes(tensor: torch.Tensor, spec: TensorSpec) -> memoryview:
    tensor = tensor.detach()
    if tuple(tensor.shape) != spec.shape or tensor.dtype != spec.dtype:
        raise ValueError(f"tensor {spec.name!r}: got {tuple(tensor.shape)} {tensor.dtype}, "
                         f"announced {spec.shape} {spec.dtype}")
    flat = tensor.to("cpu").contiguous().reshape(-1)
    if flat.numel() == 0:
        return memoryview(b"")
    return flat.view(torch.uint8).numpy().data


def write_safetensors(path: str | os.PathLike, specs: Iterable[TensorSpec],
                      metadata: dict | None = None) -> int:
    """Write `specs` to `path` (atomically); returns the file size in bytes.

    Each tensor is produced, written and released before the next one.
    """
    specs = list(specs)
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("two tensors with one name")
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        with open(temporary, "wb") as f:
            blob = _header(specs, metadata)
            f.write(struct.pack("<Q", len(blob)))
            f.write(blob)
            for spec in specs:
                f.write(_tensor_bytes(spec.produce(), spec))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path.stat().st_size


def iter_folder_tensors(folder: str | os.PathLike) -> Iterator[tuple[str, SafetensorsFile]]:
    """``(tensor name, file)`` of every tensor of a model folder (sharded or not)."""
    folder = Path(folder)
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map") or {}
        files: dict[str, SafetensorsFile] = {}
        for name, shard in weight_map.items():
            if shard not in files:
                files[shard] = SafetensorsFile(folder / shard)
            yield name, files[shard]
        return
    single = folder / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"{folder}: no model.safetensors (or model.safetensors.index.json)")
    file = SafetensorsFile(single)
    for name in file.keys():
        yield name, file
