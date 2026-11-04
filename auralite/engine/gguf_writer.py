"""Write native models as GGUF files for llama.cpp, Ollama and LM Studio (2.18.1).

A dependency-free GGUF v3 writer for the models :mod:`hf_models` exports:
Llama and Mistral (llama.cpp's ``llama`` architecture), Qwen2 and Qwen3, with
their byte-level BPE tokenizer (merges, special tokens, the pre-tokenizer
name llama.cpp knows it by) and chat template. Weights are written as F32,
F16, BF16 or Q8_0; smaller types (Q4_K_M, ...) come from llama.cpp's own
``llama-quantize`` run on the F16 or Q8_0 file.

What llama.cpp expects and this file does:

* tensor names ``token_embd``, ``blk.N.attn_q`` ... ``output``; dimensions
  innermost first (the reverse of torch's shape);
* the ``llama`` architecture rotates query/key pairs the GPT-J way, so the
  rows of W_q and W_k are permuted (as llama.cpp's converter does); Qwen
  models use the NeoX rotation AuraLite also uses and are written as they are;
* Llama 3.1's RoPE scaling travels as a ``rope_freqs`` tensor;
* 1-D tensors (norms, biases) stay F32.
"""
from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .hf_bpe import HFBPETokenizer
from .hf_models import (HFModelError, _native_model, export_family, export_tensors,
                        jinja_chat_template, token_id)

__all__ = ["GGUF_TYPES", "export_gguf", "read_gguf", "pre_tokenizer_name"]

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32

#: ggml tensor types this writer produces
GGUF_TYPES = {"f32": 0, "f16": 1, "q8_0": 8, "bf16": 30}
_FILE_TYPES = {"f32": 0, "f16": 1, "q8_0": 7, "bf16": 32}

# GGUF metadata value types
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL, _STRING, _ARRAY, \
    _UINT64, _INT64, _FLOAT64 = range(13)

# llama.cpp token types
_NORMAL, _UNKNOWN, _CONTROL, _USER_DEFINED, _UNUSED, _BYTE = range(1, 7)

_ARCH = {"llama": "llama", "mistral": "llama", "qwen2": "qwen2", "qwen3": "qwen3"}

#: The text llama.cpp tokenizes to tell BPE pre-tokenizers apart, and the
#: sha256 of the token ids of some it knows (convert_hf_to_gguf.py).
CHECK_TEXT = ("\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n\U0001f680 (normal) "
              "\U0001f636\u200d\U0001f32b\ufe0f (multiple emojis concatenated) \u2705 "
              "\U0001f999\U0001f999 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 "
              "\u1780\u17b6\u1793\u17cb\u178f\u17c2\u1796\u17b7\u179f\u17c1\u179f\u17a2\u17b6"
              "\u1785\U0001f601 ?\u6211\u60f3\u5728apple\u5de5\u4f5c1314151\u5929\uff5e "
              "------======= \u043d\u0435\u0449\u043e \u043d\u0430 "
              "\u0411\u044a\u043b\u0433\u0430\u0440\u0441\u043a\u0438 ''''''```````\"\"\"\""
              "......!!!!!!?????? I've been 'told he's there, 'RE you sure? 'M not sure I'll "
              "make it, 'D you like some tea? We'Ve a'lL")
KNOWN_PRE_TOKENIZERS = {
    "0ef9807a4087ebef797fc749390439009c3b9eda9ad1a097abbe738f486c01e5": "llama-bpe",
    "c136ed14d01c2745d4f60a9596ae66800e2b61fa45643e72436041855ad4089d": "smaug-bpe",
    "e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea": "qwen2",
    "d4540891389ea895b53b399da6ac824becc30f2fba0e9ddbb98f92e55ca0e97c": "qwen2",
    "6e040dfe72e4b85855588c53acf4909ac4e98e55bc3a33cd7db499180dc42a78": "qwen2",
    "b3f499bb4255f8ca19fccd664443283318f2fd2414d5e0b040fbdd0cc195d6c5": "deepseek-r1-qwen",
    "855059429035d75a914d1eda9f10a876752e281a054a7a3d421ef0533e5b6249": "smollm",
    "3ce83efda5659b07b1ad37ca97ca5797ea4285d9b9ab0dc679e4a720c9da7454": "gpt-2",
    "63b97e4253352e6f357cc59ea5b583e3a680eaeaf2632188c2b952de2588485e": "tekken",
    "9d032fcbd5501f4a38150912590928bfb36091efb5df11b8e2124b0390e3fb1e": "falcon3",
}


# ----------------------------------------------------------------------
#  Tokenizer
# ----------------------------------------------------------------------
def _encode_like_transformers(tokenizer: HFBPETokenizer, text: str) -> list[int]:
    """``tokenizer.encode(text)`` of transformers: with the post-processor's specials."""
    ids = list(tokenizer.encode(text))
    processor = tokenizer.data.get("post_processor")
    if not isinstance(processor, dict):
        return ids
    for step in processor.get("processors", [processor]):
        if not isinstance(step, dict) or step.get("type") != "TemplateProcessing":
            continue
        out: list[int] = []
        for item in step.get("single") or []:
            if "SpecialToken" in item:
                special = token_id(tokenizer, item["SpecialToken"].get("id"))
                if special is not None:
                    out.append(special)
            elif "Sequence" in item:
                out.extend(ids)
        return out
    return ids


def pre_tokenizer_name(tokenizer: HFBPETokenizer) -> str | None:
    """llama.cpp's name for the tokenizer's pre-tokenizer, or None if unknown."""
    ids = _encode_like_transformers(tokenizer, CHECK_TEXT)
    digest = hashlib.sha256(str(ids).encode()).hexdigest()
    return KNOWN_PRE_TOKENIZERS.get(digest)


def _merges(tokenizer: HFBPETokenizer) -> list[str]:
    merges = (tokenizer.data.get("model") or {}).get("merges") or []
    out = []
    for merge in merges:
        if isinstance(merge, str):
            out.append(merge)
        else:
            left, right = (str(part).replace(" ", chr(ord(" ") + 256)) for part in merge)
            out.append(f"{left} {right}")
    return out


def _vocab(tokenizer: HFBPETokenizer, rows: int) -> tuple[list[str], list[int]]:
    tokens = [f"[PAD{i}]" for i in range(rows)]
    types = [_UNUSED] * rows
    for token, index in tokenizer.token_to_id.items():
        if index < rows:
            tokens[index], types[index] = token, _NORMAL
    for entry in tokenizer.data.get("added_tokens") or []:
        if not isinstance(entry, dict) or int(entry.get("id", rows)) >= rows:
            continue
        index = int(entry["id"])
        tokens[index] = str(entry.get("content", ""))
        content = tokens[index]
        special = bool(entry.get("special")) or (content.startswith("<|")
                                                 and content.endswith("|>"))
        types[index] = _CONTROL if special else _USER_DEFINED
    return tokens, types


# ----------------------------------------------------------------------
#  Low-level writing
# ----------------------------------------------------------------------
def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _value(value: Any, kind: int | None = None) -> tuple[int, bytes]:
    """(GGUF type, encoded bytes) of a metadata value."""
    if kind is None:
        if isinstance(value, bool):
            kind = _BOOL
        elif isinstance(value, int):
            kind = _UINT32 if 0 <= value < 2 ** 32 else _INT64
        elif isinstance(value, float):
            kind = _FLOAT32
        elif isinstance(value, str):
            kind = _STRING
        elif isinstance(value, (list, tuple)):
            kind = _ARRAY
        else:
            raise TypeError(f"no GGUF type for {type(value).__name__}")
    if kind == _STRING:
        return kind, _string(str(value))
    if kind == _BOOL:
        return kind, struct.pack("<?", bool(value))
    if kind == _ARRAY:
        items = list(value) if isinstance(value, (list, tuple)) else [value]
        if items and all(isinstance(v, str) for v in items):
            body = b"".join(_string(str(v)) for v in items)
            return kind, struct.pack("<IQ", _STRING, len(items)) + body
        if items and all(isinstance(v, float) for v in items):
            return kind, struct.pack("<IQ", _FLOAT32, len(items)) + np.asarray(
                items, dtype="<f4").tobytes()
        return kind, struct.pack("<IQ", _INT32, len(items)) + np.asarray(
            items, dtype="<i4").tobytes()
    formats = {_UINT8: "<B", _INT8: "<b", _UINT16: "<H", _INT16: "<h", _UINT32: "<I",
               _INT32: "<i", _FLOAT32: "<f", _UINT64: "<Q", _INT64: "<q", _FLOAT64: "<d"}
    return kind, struct.pack(formats[kind], value)


def _q8_0(tensor: torch.Tensor) -> bytes:
    """ggml's Q8_0: blocks of 32 values, an fp16 scale and 32 int8 each."""
    flat = tensor.detach().float().reshape(-1, 32)
    scale = flat.abs().amax(dim=1, keepdim=True) / 127.0
    inverse = torch.where(scale > 0, 1.0 / scale, torch.zeros_like(scale))
    codes = torch.round(flat * inverse).clamp_(-127, 127).to(torch.int8)
    blocks = np.empty(flat.shape[0], dtype=[("d", "<f2"), ("qs", "i1", (32,))])
    blocks["d"] = scale.squeeze(1).to(torch.float16).numpy()
    blocks["qs"] = codes.numpy()
    return blocks.tobytes()


def _tensor_bytes(tensor: torch.Tensor, kind: str) -> bytes:
    if kind == "q8_0":
        return _q8_0(tensor)
    dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[kind]
    flat = tensor.detach().to(dtype).contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


def _nbytes(shape: tuple[int, ...], kind: str) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    if kind == "q8_0":
        return n // 32 * 34
    return n * {"f32": 4, "f16": 2, "bf16": 2}[kind]


def _llama_permute(weight: torch.Tensor, heads: int) -> torch.Tensor:
    """llama.cpp's q/k row order for the ``llama`` architecture (GPT-J rotation)."""
    rows = weight.shape[0]
    return (weight.reshape(heads, 2, rows // heads // 2, *weight.shape[1:])
            .swapaxes(1, 2).reshape(weight.shape))


_GGUF_NAMES = {
    "attn_norm.weight": "attn_norm.weight", "ffn_norm.weight": "ffn_norm.weight",
    "attn.W_q.weight": "attn_q.weight", "attn.W_q.bias": "attn_q.bias",
    "attn.W_k.weight": "attn_k.weight", "attn.W_k.bias": "attn_k.bias",
    "attn.W_v.weight": "attn_v.weight", "attn.W_v.bias": "attn_v.bias",
    "attn.W_o.weight": "attn_output.weight",
    "attn.q_norm.weight": "attn_q_norm.weight", "attn.k_norm.weight": "attn_k_norm.weight",
    "ffn.gate.weight": "ffn_gate.weight", "ffn.up.weight": "ffn_up.weight",
    "ffn.down.weight": "ffn_down.weight",
}


def _gguf_name(native: str) -> str:
    if native == "embedding.weight":
        return "token_embd.weight"
    if native == "final_norm.weight":
        return "output_norm.weight"
    if native == "head.weight":
        return "output.weight"
    _layers, index, rest = native.split(".", 2)
    return f"blk.{index}.{_GGUF_NAMES[rest]}"


def _llama3_rope_factors(scaling: dict, head_dim: int) -> torch.Tensor:
    base = float(scaling.get("base", 10000.0))
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    factor = float(scaling.get("factor", 8.0))
    low = float(scaling.get("low_freq_factor", 1.0))
    high = float(scaling.get("high_freq_factor", 4.0))
    original = float(scaling.get("original_max_position_embeddings", 8192))
    out = []
    for freq in freqs.tolist():
        wavelen = 2 * np.pi / freq
        if wavelen < original / high:
            out.append(1.0)
        elif wavelen > original / low:
            out.append(factor)
        else:
            smooth = (original / wavelen - low) / (high - low)
            out.append(1 / ((1 - smooth) / factor + smooth))
    return torch.tensor(out, dtype=torch.float32)


# ----------------------------------------------------------------------
#  Export
# ----------------------------------------------------------------------
def export_gguf(model: Any, tokenizer: Any, path: str | os.PathLike, *,
                quant: str = "f16", chat_template: str = "chatml",
                source: dict | None = None, name: str | None = None,
                progress_callback: Callable[[dict], None] | None = None) -> dict:
    """Write a native model as a GGUF file; returns a summary (with warnings)."""
    kind = str(quant or "f16").lower()
    if kind not in GGUF_TYPES:
        raise HFModelError(f"GGUF type must be one of {', '.join(GGUF_TYPES)}, got {quant!r}")
    model = _native_model(model)
    family = export_family(model, source)
    arch = _ARCH[family]
    if not isinstance(tokenizer, HFBPETokenizer) or not tokenizer.byte_level:
        raise HFModelError("GGUF export needs a byte-level BPE tokenizer (Llama 3, Qwen, "
                           "SmolLM, GPT-2 style); for SentencePiece-style ones export --format "
                           "hf and convert with llama.cpp's convert_hf_to_gguf.py")
    warnings: list[str] = []
    heads, kv_heads = int(model.n_heads), int(model.n_kv_heads or model.n_heads)
    head_dim = int(model.head_dim)
    rows = int(model.embedding.num_embeddings)
    scaling = dict(model.rope_scaling or {})
    rope_kind = str(scaling.get("type", "none")).lower()
    theta = float(scaling.get("base", 10000.0))
    if rope_kind in ("ntk", "dynamic_ntk") and float(scaling.get("factor", 1.0) or 1.0) > 1.0:
        theta = float(model._attention_layers()[0]._ntk_base())
    pre = pre_tokenizer_name(tokenizer)
    if pre is None:
        pre = "default"
        warnings.append("llama.cpp does not know this tokenizer's pre-tokenizer: written as "
                        "'default', which may split some text differently")
    if family == "mistral" and model.sliding_window:
        warnings.append(f"llama.cpp's llama architecture has no sliding window: prompts longer "
                        f"than {model.sliding_window} tokens attend further than in training")

    meta: list[tuple[str, Any, int | None]] = [
        ("general.architecture", arch, None),
        ("general.name", name or (source or {}).get("repo") or "auralite-model", None),
        ("general.file_type", _FILE_TYPES[kind], _UINT32),
        ("general.quantization_version", 2, _UINT32),
        (f"{arch}.context_length", int(model.max_seq_len), _UINT32),
        (f"{arch}.embedding_length", int(model.d_model), _UINT32),
        (f"{arch}.block_count", int(model.n_layers), _UINT32),
        (f"{arch}.feed_forward_length", int(model.d_ff), _UINT32),
        (f"{arch}.attention.head_count", heads, _UINT32),
        (f"{arch}.attention.head_count_kv", kv_heads, _UINT32),
        (f"{arch}.attention.layer_norm_rms_epsilon", float(model.norm_eps), _FLOAT32),
        (f"{arch}.rope.freq_base", theta, _FLOAT32),
        (f"{arch}.rope.dimension_count", head_dim, _UINT32),
        (f"{arch}.attention.key_length", head_dim, _UINT32),
        (f"{arch}.attention.value_length", head_dim, _UINT32),
        (f"{arch}.vocab_size", rows, _UINT32),
    ]
    factor = float(scaling.get("factor", 1.0) or 1.0)
    if rope_kind == "linear" and factor > 1.0:
        meta += [(f"{arch}.rope.scaling.type", "linear", None),
                 (f"{arch}.rope.scaling.factor", factor, _FLOAT32)]
    elif rope_kind == "yarn" and factor > 1.0:
        meta += [(f"{arch}.rope.scaling.type", "yarn", None),
                 (f"{arch}.rope.scaling.factor", factor, _FLOAT32),
                 (f"{arch}.rope.scaling.original_context_length",
                  int(scaling.get("original_max_position_embeddings", model.max_seq_len)),
                  _UINT32)]
    tokens, types = _vocab(tokenizer, rows)
    meta += [("tokenizer.ggml.model", "gpt2", None), ("tokenizer.ggml.pre", pre, None),
             ("tokenizer.ggml.tokens", tokens, None), ("tokenizer.ggml.token_type", types, None),
             ("tokenizer.ggml.merges", _merges(tokenizer), None)]
    bos = (source or {}).get("bos_token_id")
    eos = (source or {}).get("eos_token_id")
    if eos is None:
        from .shards import _eos_id
        found = _eos_id(tokenizer)
        eos = found[0] if found else None
    if bos is not None:
        meta.append(("tokenizer.ggml.bos_token_id", int(bos), _UINT32))
    if eos is not None:
        meta.append(("tokenizer.ggml.eos_token_id", int(eos), _UINT32))
    end = {"chatml": "<|im_end|>", "llama3": "<|eot_id|>", "gemma": "<end_of_turn>",
           "phi": "<|end|>"}.get(chat_template)
    eot = token_id(tokenizer, end) if end else None
    if eot is not None and eot != eos:
        meta.append(("tokenizer.ggml.eot_token_id", int(eot), _UINT32))
    meta.append(("tokenizer.ggml.add_bos_token", bool((source or {}).get("add_bos_token")),
                 None))
    jinja = None
    if (source or {}).get("chat_template") == chat_template:
        jinja = (source or {}).get("chat_template_jinja")
    jinja = jinja or jinja_chat_template(chat_template)
    if jinja:
        meta.append(("tokenizer.chat_template", jinja, None))

    tensors: list[tuple[str, tuple[int, ...], str, Callable[[], torch.Tensor]]] = []
    if rope_kind == "llama3" and factor > 1.0:
        factors = _llama3_rope_factors(scaling, head_dim)
        tensors.append(("rope_freqs.weight", tuple(factors.shape), "f32", lambda f=factors: f))
    for native, shape, produce in export_tensors(model):
        gguf_name = _gguf_name(native)
        tensor_kind = kind if len(shape) == 2 else "f32"
        if tensor_kind == "q8_0" and shape[-1] % 32:
            tensor_kind = "f16"
        transform = produce
        if arch == "llama" and native.endswith(("attn.W_q.weight", "attn.W_q.bias")):
            transform = (lambda p=produce: _llama_permute(p(), heads))
        elif arch == "llama" and native.endswith(("attn.W_k.weight", "attn.W_k.bias")):
            transform = (lambda p=produce: _llama_permute(p(), kv_heads))
        tensors.append((gguf_name, tuple(shape), tensor_kind, transform))

    out_path = Path(path)
    temporary = out_path.with_name(out_path.name + f".tmp{os.getpid()}")
    try:
        with open(temporary, "wb") as f:
            f.write(GGUF_MAGIC + struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(meta)))
            for key, value, value_kind in meta:
                kind_id, body = _value(value, value_kind)
                f.write(_string(key) + struct.pack("<I", kind_id) + body)
            offset = 0
            for tensor_name, shape, tensor_kind, _produce in tensors:
                dims = list(reversed(shape))
                f.write(_string(tensor_name) + struct.pack("<I", len(dims))
                        + struct.pack(f"<{len(dims)}Q", *dims)
                        + struct.pack("<IQ", GGUF_TYPES[tensor_kind], offset))
                offset += _nbytes(shape, tensor_kind)
                offset += (-offset) % ALIGNMENT
            f.write(b"\0" * ((-f.tell()) % ALIGNMENT))
            for index, (tensor_name, shape, tensor_kind, produce) in enumerate(tensors):
                data = _tensor_bytes(produce(), tensor_kind)
                if len(data) != _nbytes(shape, tensor_kind):
                    raise HFModelError(f"{tensor_name}: {len(data)} bytes, expected "
                                       f"{_nbytes(shape, tensor_kind)}")
                f.write(data)
                f.write(b"\0" * ((-len(data)) % ALIGNMENT))
                if progress_callback is not None and (index % 16 == 0 or
                                                      index == len(tensors) - 1):
                    progress_callback({"stage": "write", "done": index + 1,
                                       "total": len(tensors)})
        os.replace(temporary, out_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": str(out_path), "architecture": arch, "family": family, "type": kind,
            "tensors": len(tensors), "bytes": out_path.stat().st_size,
            "pre_tokenizer": pre, "chat_template": chat_template, "warnings": warnings}


# ----------------------------------------------------------------------
#  Reading (tests, diagnostics)
# ----------------------------------------------------------------------
def read_gguf(path: str | os.PathLike) -> tuple[dict, list[dict]]:
    """The metadata and tensor list of a GGUF file (not the tensor data).

    ``meta["__version__"]`` is the format version and ``meta["__data_offset__"]``
    where the tensor data starts (each tensor's ``offset`` counts from there).
    """
    with open(path, "rb") as f:
        def unpack(fmt: str) -> Any:
            size = struct.calcsize(fmt)
            values = struct.unpack(fmt, f.read(size))
            return values[0] if len(values) == 1 else values

        def string() -> str:
            return f.read(unpack("<Q")).decode("utf-8")

        def value(kind: int) -> Any:
            if kind == _STRING:
                return string()
            if kind == _ARRAY:
                inner, count = unpack("<IQ")
                return [value(inner) for _ in range(count)]
            formats = {_UINT8: "<B", _INT8: "<b", _UINT16: "<H", _INT16: "<h",
                       _UINT32: "<I", _INT32: "<i", _FLOAT32: "<f", _BOOL: "<?",
                       _UINT64: "<Q", _INT64: "<q", _FLOAT64: "<d"}
            return unpack(formats[kind])

        if f.read(4) != GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        version, n_tensors, n_kv = unpack("<IQQ")
        meta: dict = {"__version__": version}
        for _ in range(n_kv):
            key = string()
            meta[key] = value(unpack("<I"))
        tensors = []
        for _ in range(n_tensors):
            tensor_name = string()
            n_dims = unpack("<I")
            dims = [unpack("<Q") for _ in range(n_dims)]
            kind, offset = unpack("<IQ")
            tensors.append({"name": tensor_name, "dims": dims, "type": kind, "offset": offset})
        alignment = int(meta.get("general.alignment", ALIGNMENT))
        meta["__data_offset__"] = f.tell() + (-f.tell()) % alignment
    return meta, tensors
