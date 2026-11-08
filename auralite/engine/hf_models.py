"""Hugging Face models in AuraLite's own engine: import and export (2.18.1).

Llama, Mistral, Qwen2 and Qwen3 checkpoints are the same decoder AuraLite
builds (RMSNorm, RoPE, grouped-query attention, SwiGLU); only the names of
the tensors and a few settings differ. This module maps one onto the other,
without transformers or the safetensors package:

* :func:`import_hf` reads a model folder (or downloads ``hf:REPO[@REV]`` with
  resume) and builds a :class:`~auralite.engine.model.ModernTransformer`
  with exactly its weights - on the "meta" device first, so no random
  weights are made, and optionally with the block linears in 4 bits (NF4,
  the base of QLoRA). Logits match transformers to float32 rounding.
* :func:`export_hf` writes a native model as a Hugging Face folder
  (``config.json``, sharded safetensors, ``tokenizer.json``,
  ``tokenizer_config.json`` with a chat template); LoRA adapters are merged
  on the way, 4-bit layers dequantized.

GGUF export (llama.cpp, Ollama, LM Studio) lives in ``gguf_writer.py`` and
uses the same tensor walk.
"""
from __future__ import annotations

import copy
import hashlib
import functools
import json
import logging
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, cast

import torch
import torch.nn as nn

from .hf_bpe import HFBPETokenizer, TokenizerFormatError
from .layers import HeadwiseRMSNorm, RMSNorm, TransformerBlock
from .model import ModernTransformer
from .nf4 import QUANTIZABLE_LINEARS, NF4Linear
from .safetensors_lite import TensorSpec, iter_folder_tensors, write_safetensors

logger = logging.getLogger("auralite")

__all__ = ["HFModelError", "HFImport", "SUPPORTED_ARCHITECTURES", "export_hf", "export_tensors",
           "hf_model_dir", "import_hf", "is_hf_model_spec", "native_config", "parse_hf_spec",
           "detect_chat_template", "jinja_chat_template", "token_id"]

#: Hugging Face architectures AuraLite imports, and the family each belongs to.
SUPPORTED_ARCHITECTURES = {
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "mistral",
    "Qwen2ForCausalLM": "qwen2",
    "Qwen3ForCausalLM": "qwen3",
    # 2.23: Qwen3.5 (0.8B to 9B) - hybrid linear / full attention; its text part
    "Qwen3_5ForConditionalGeneration": "qwen3_5",
    "Qwen3_5ForCausalLM": "qwen3_5",
}
_MODEL_TYPES = {"llama": "llama", "mistral": "mistral", "qwen2": "qwen2", "qwen3": "qwen3",
                "qwen3_5": "qwen3_5", "qwen3_5_text": "qwen3_5"}
_FAMILY_ARCH = {family: arch for arch, family in SUPPORTED_ARCHITECTURES.items()}
_FAMILY_ARCH["qwen3_5"] = "Qwen3_5ForCausalLM"      # what an export of the text part says

#: Files of a model repository the import needs (the weights come on top).
_MODEL_FILES = ("config.json", "tokenizer.json")
_OPTIONAL_FILES = ("generation_config.json", "tokenizer_config.json",
                   "special_tokens_map.json", "chat_template.jinja")


class HFModelError(ValueError):
    """A Hugging Face model AuraLite cannot import or export, and why."""


# ----------------------------------------------------------------------
#  Where a model comes from
# ----------------------------------------------------------------------
def is_hf_model_spec(spec: Any) -> bool:
    """True for ``hf:REPO[@REV]`` and for a local Hugging Face model folder."""
    if not isinstance(spec, (str, os.PathLike)):
        return False
    text = os.fspath(spec)
    if text.lower().startswith("hf:"):
        return True
    return os.path.isdir(text) and os.path.isfile(os.path.join(text, "config.json"))


def parse_hf_spec(spec: str) -> tuple[str, str]:
    """``hf:org/name@rev`` -> ("org/name", "rev"); the revision defaults to main."""
    text = str(spec)
    if text.lower().startswith("hf:"):
        text = text[3:]
    repo, _, revision = text.partition("@")
    repo = repo.strip().strip("/")
    if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", repo):
        raise HFModelError(f"not a Hugging Face model id: {spec!r} (expected hf:ORG/NAME, "
                           f"for example hf:HuggingFaceTB/SmolLM2-135M)")
    return repo, (revision.strip() or "main")


def _hf_token() -> str | None:
    from .shards import _hf_token as token
    return token()


def _cache_root(cache_dir: str | os.PathLike | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    from .shards import auralite_home
    return auralite_home() / "hf"


class _StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but never send the Hub token to another host (the CDN)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != \
                urllib.parse.urlsplit(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None                       # the redirect itself carries the file's metadata


def _headers(start: int = 0) -> dict:
    headers = {"User-Agent": "auralite"}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if start:
        headers["Range"] = f"bytes={start}-"
    return headers


def _request(url: str, *, start: int = 0) -> Any:
    opener = urllib.request.build_opener(_StripAuthOnRedirect)
    return opener.open(urllib.request.Request(url, headers=_headers(start)), timeout=60)


def _http_error(err: urllib.error.HTTPError, repo: str, name: str, url: str) -> Exception:
    if err.code in (401, 403):
        return HFModelError(f"{repo} is gated or private: accept its license on "
                            f"huggingface.co and set HF_TOKEN (HTTP {err.code})")
    if err.code == 404:
        return FileNotFoundError(f"{repo} has no {name}")
    return HFModelError(f"could not download {url}: HTTP {err.code}")


def _probe(url: str, repo: str, name: str) -> dict:
    """Size and sha256 of a file, from the Hub's own answer (before the CDN).

    The Hub answers a large (LFS) file with a redirect whose ``X-Linked-Size``
    and ``X-Linked-Etag`` are the file's size and sha256; the CDN's own ETag
    after the redirect is another hash (Xet's). A server that does not answer
    HEAD just gives nothing to check.
    """
    request = urllib.request.Request(url, headers=_headers(), method="HEAD")
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=60) as response:
            headers = response.headers
    except urllib.error.HTTPError as err:
        if err.code in (301, 302, 303, 307, 308):
            headers = err.headers
        elif err.code in (405, 501):
            return {}
        else:
            raise _http_error(err, repo, name, url) from err
    except OSError:
        return {}
    size = str(headers.get("X-Linked-Size") or headers.get("Content-Length") or "0")
    etag = str(headers.get("X-Linked-Etag") or "").strip().removeprefix("W/").strip('"')
    return {"size": int(size) if size.isdigit() else 0,
            "sha256": etag if re.fullmatch(r"[0-9a-f]{64}", etag) else None}


def _download_file(url: str, target: Path, *, repo: str, name: str,
                   progress: Callable[[dict], None] | None = None,
                   stop_event: Any = None, done_before: int = 0, total_all: int = 0,
                   expected: dict | None = None) -> int:
    """Download `url` to `target`, continuing a ``.part`` file; returns its size."""
    expected = expected if expected is not None else _probe(url, repo, name)
    part = target.with_name(target.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    response: Any = None
    try:
        response = _request(url, start=have)
    except urllib.error.HTTPError as err:
        if not (err.code == 416 and have):        # 416: the part file is complete already
            raise _http_error(err, repo, name, url) from err
    except OSError as err:
        raise HFModelError(f"could not download {url}: {err}") from err
    total = int(expected.get("size") or 0)
    if response is not None:
        with response:
            if getattr(response, "status", 200) != 206:
                have = 0                           # no resume: start over
            length = int(response.headers.get("Content-Length") or 0)
            total = total or (have + length if length else 0)
            target.parent.mkdir(parents=True, exist_ok=True)
            last = 0.0
            with open(part, "ab" if have else "wb") as out:
                while True:
                    if stop_event is not None and stop_event.is_set():
                        raise InterruptedError(f"download of {name} stopped (run it again to "
                                               f"continue where it stopped)")
                    block = response.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    have += len(block)
                    now = time.monotonic()
                    if progress is not None and now - last > 0.5:
                        last = now
                        progress({"stage": "download", "file": name, "bytes": have,
                                  "total": total, "all_bytes": done_before + have,
                                  "all_total": total_all})
    if total and have != total:
        raise HFModelError(f"download of {name} ended early ({have:,} of {total:,} bytes; "
                           f"run it again to continue)")
    if expected.get("sha256"):
        digest = hashlib.sha256()
        with open(part, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                digest.update(block)
        if digest.hexdigest() != expected["sha256"]:
            part.unlink()
            raise HFModelError(f"{name} of {repo} is corrupt (sha256 mismatch); it was "
                               f"deleted, run the import again")
    os.replace(part, target)
    return target.stat().st_size


def hf_model_dir(spec: str | os.PathLike, *, cache_dir: str | os.PathLike | None = None,
                 progress_callback: Callable[[dict], None] | None = None,
                 stop_event: Any = None) -> Path:
    """The local folder of a model: `spec` itself, or ``hf:REPO[@REV]`` downloaded.

    Downloads go to ``$AURALITE_HOME/hf/ORG__NAME/REV`` (``~/.auralite``);
    an interrupted download continues where it stopped (``.part`` files and
    HTTP ranges), a finished file is never fetched twice, and files whose
    server gives a sha256 are checked. ``HF_TOKEN`` (or ``huggingface-cli
    login``) opens gated models such as Llama; ``HF_ENDPOINT`` points at a
    mirror.
    """
    text = os.fspath(spec)
    if not text.lower().startswith("hf:"):
        folder = Path(text)
        if not (folder / "config.json").is_file():
            raise HFModelError(f"{folder} is not a Hugging Face model folder (no config.json)")
        return folder
    repo, revision = parse_hf_spec(text)
    folder = _cache_root(cache_dir) / repo.replace("/", "__") / revision
    marker = folder / ".auralite-complete.json"
    if marker.is_file():
        # downloaded before: no network at all (works offline)
        try:
            files = json.loads(marker.read_text(encoding="utf-8")).get("files") or []
        except (OSError, ValueError):
            files = []
        if files and all((folder / name).is_file() for name in files):
            return folder
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

    def url(name: str) -> str:
        return f"{endpoint}/{repo}/resolve/{revision}/{name}"

    def fetch(name: str, *, optional: bool = False, done: int = 0, total: int = 0,
              expected: dict | None = None) -> int:
        target = folder / name
        if target.is_file():
            return target.stat().st_size
        try:
            return _download_file(url(name), target, repo=repo, name=name,
                                  progress=progress_callback, stop_event=stop_event,
                                  done_before=done, total_all=total, expected=expected)
        except FileNotFoundError:
            if optional:
                return 0
            raise

    for name in _MODEL_FILES:
        try:
            fetch(name)
        except FileNotFoundError as err:
            if name == "tokenizer.json":
                raise HFModelError(f"{repo} has no tokenizer.json: AuraLite reads the "
                                   f"tokenizers of the `tokenizers` library (BPE)") from err
            raise HFModelError(f"{repo} does not exist or has no {name}") from err
    for name in _OPTIONAL_FILES:
        fetch(name, optional=True)
    if fetch("model.safetensors.index.json", optional=True):
        index = json.loads((folder / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shards = sorted(set((index.get("weight_map") or {}).values()))
    else:
        shards = ["model.safetensors"]
    missing = [name for name in shards if not (folder / name).is_file()]
    probes = {name: _probe(url(name), repo, name) for name in missing}
    total = sum(int(p.get("size") or 0) for p in probes.values())
    done = 0
    for name in shards:
        try:
            done += fetch(name, done=done, total=total, expected=probes.get(name))
        except FileNotFoundError as err:
            raise HFModelError(f"{repo} has no safetensors weights ({name}); AuraLite does not "
                               f"read pickled .bin weights") from err
    present = [name for name in (*_MODEL_FILES, *_OPTIONAL_FILES,
                                 "model.safetensors.index.json", *shards)
               if (folder / name).is_file()]
    marker.write_text(json.dumps({"repo": repo, "revision": revision, "files": present}),
                      encoding="utf-8")
    return folder


# ----------------------------------------------------------------------
#  Configuration
# ----------------------------------------------------------------------
def _family(hf: dict) -> str:
    archs = hf.get("architectures") or []
    for arch in archs:
        if arch in SUPPORTED_ARCHITECTURES:
            return SUPPORTED_ARCHITECTURES[arch]
    family = _MODEL_TYPES.get(str(hf.get("model_type", "")).lower())
    if family is not None and not archs:
        return family
    shown = ", ".join(archs) or str(hf.get("model_type") or "an unknown architecture")
    raise HFModelError(f"{shown}: AuraLite imports {', '.join(SUPPORTED_ARCHITECTURES)} "
                       f"models (other architectures can be used through the transformers "
                       f"backend, or as GGUF files)")


def _rope_settings(hf: dict) -> tuple[float, dict]:
    """(theta, the rope scaling dict as transformers 4 or 5 writes it)."""
    params = hf.get("rope_parameters")
    if isinstance(params, dict) and "full_attention" in params and isinstance(
            params["full_attention"], dict):
        params = params["full_attention"]
    scaling = hf.get("rope_scaling")
    merged: dict = {}
    if isinstance(params, dict):
        merged.update(params)
    if isinstance(scaling, dict):
        merged.update(scaling)
    theta = merged.get("rope_theta", hf.get("rope_theta", 10000.0))
    return float(theta if theta is not None else 10000.0), merged


def text_config(hf: dict) -> dict:
    """The language model's part of a config: Qwen3.5's ``text_config`` (2.23), else all."""
    inner = hf.get("text_config")
    if isinstance(inner, dict):
        merged = dict(inner)
        if merged.get("tie_word_embeddings") is None and "tie_word_embeddings" in hf:
            merged["tie_word_embeddings"] = hf["tie_word_embeddings"]
        return merged
    return hf


def _qwen35_config(hf: dict) -> dict:
    """The hybrid fields of a Qwen3.5 config (2.23)."""
    if hf.get("mlp_only_layers"):
        raise HFModelError("mlp_only_layers: this Qwen3.5 variant is not supported")
    if hf.get("num_experts") or hf.get("num_experts_per_tok"):
        raise HFModelError("Qwen3.5 mixture-of-experts models (35B-A3B and up) are not "
                           "supported: AuraLite imports the dense ones (0.8B to 9B)")
    hidden_layers = int(hf["num_hidden_layers"])
    types = hf.get("layer_types")
    if not types:
        interval = int(hf.get("full_attention_interval") or 4)
        types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                 for i in range(hidden_layers)]
    types = [str(t) for t in types]
    if len(types) != hidden_layers or set(types) - {"full_attention", "linear_attention"}:
        raise HFModelError(f"layer_types {sorted(set(types))!r}: AuraLite knows "
                           f"full_attention and linear_attention")
    act = str(hf.get("hidden_act", "silu")).lower()
    if act not in ("silu", "swish"):
        raise HFModelError(f"hidden_act={act!r}: AuraLite's linear attention uses silu")
    return {"layer_types": types,
            "linear_attention": {
                "num_k_heads": int(hf.get("linear_num_key_heads") or 16),
                "num_v_heads": int(hf.get("linear_num_value_heads") or 16),
                "k_head_dim": int(hf.get("linear_key_head_dim") or 128),
                "v_head_dim": int(hf.get("linear_value_head_dim") or 128),
                "conv_kernel": int(hf.get("linear_conv_kernel_dim") or 4)},
            "attn_output_gate": bool(hf.get("attn_output_gate", True))}


def native_config(hf: dict) -> dict:
    """The :class:`ModernTransformer` arguments of a Hugging Face ``config.json``.

    Raises :class:`HFModelError` naming the setting for anything that would
    not compute the same function (another activation, MLP or output biases,
    a RoPE scaling AuraLite does not reproduce, per-layer sliding windows).
    """
    family = _family(hf)
    hf = text_config(hf)
    act = str(hf.get("hidden_act", "silu")).lower()
    if act not in ("silu", "swish"):
        raise HFModelError(f"hidden_act={act!r}: AuraLite's feed-forward is SwiGLU (silu)")
    if hf.get("mlp_bias"):
        raise HFModelError("mlp_bias=true: AuraLite's feed-forward has no biases")
    if family in ("llama", "qwen3", "mistral", "qwen3_5") and hf.get("attention_bias"):
        raise HFModelError("attention_bias=true: AuraLite has no bias on the attention's "
                           "output projection")
    if hf.get("pretraining_tp", 1) not in (None, 1):
        raise HFModelError("pretraining_tp > 1 is not supported")
    if hf.get("rope_interleaved"):
        raise HFModelError("rope_interleaved=true: AuraLite rotates the two halves of a head "
                           "(rotate_half), not interleaved pairs")
    hidden = int(hf["hidden_size"])
    heads = int(hf["num_attention_heads"])
    kv_heads = int(hf.get("num_key_value_heads") or heads)
    head_dim = int(hf.get("head_dim") or hidden // heads)
    theta, rope = _rope_settings(hf)
    kind = str(rope.get("rope_type", rope.get("type", "default")) or "default").lower()
    scaling: dict = {"base": theta}
    # 2.23: Qwen3.5 rotates a share of each head; its multimodal RoPE (mrope) gives text
    # tokens the same position on its three axes, which is plain RoPE
    rotary_dim = None
    factor = rope.get("partial_rotary_factor", hf.get("partial_rotary_factor"))
    if factor not in (None, 1, 1.0):
        rotary_dim = int(head_dim * float(factor))
        if rotary_dim % 2 or not 0 < rotary_dim <= head_dim:
            raise HFModelError(f"partial_rotary_factor={factor}: {rotary_dim} rotary dims")
    if kind == "linear":
        scaling.update(type="linear", factor=float(rope["factor"]))
    elif kind == "llama3":
        scaling.update(type="llama3", factor=float(rope["factor"]),
                       low_freq_factor=float(rope.get("low_freq_factor", 1.0)),
                       high_freq_factor=float(rope.get("high_freq_factor", 4.0)),
                       original_max_position_embeddings=int(
                           rope.get("original_max_position_embeddings", 8192)))
    elif kind == "yarn":
        if rope.get("mscale") or rope.get("mscale_all_dim") or rope.get("truncate") is False:
            raise HFModelError("this YaRN variant (mscale / truncate settings) is not supported")
        original = rope.get("original_max_position_embeddings") or hf.get(
            "max_position_embeddings")
        if original is None:
            raise HFModelError("YaRN RoPE scaling without original_max_position_embeddings")
        scaling.update(type="yarn", factor=float(rope["factor"]),
                       original_max_position_embeddings=int(original),
                       beta_fast=float(rope.get("beta_fast", 32.0)),
                       beta_slow=float(rope.get("beta_slow", 1.0)), yarn_blend="paper")
        if rope.get("attention_factor") is not None:
            scaling["attention_factor"] = float(rope["attention_factor"])
    elif kind != "default":
        raise HFModelError(f"RoPE scaling {kind!r} is not supported (AuraLite reproduces "
                           f"default, linear, llama3 and yarn)")
    sliding = None
    if family == "mistral":
        sliding = hf.get("sliding_window") or None
    elif hf.get("use_sliding_window"):
        raise HFModelError("use_sliding_window=true: per-layer sliding windows are not "
                           "supported")
    vocab = int(hf["vocab_size"])
    tie = bool(hf.get("tie_word_embeddings", False))
    explicit_head = head_dim if (hidden % heads or head_dim != hidden // heads) else None
    out = {
        "vocab_size": vocab, "d_model": hidden, "n_heads": heads,
        "n_layers": int(hf["num_hidden_layers"]), "d_ff": int(hf["intermediate_size"]),
        "max_seq_len": int(hf.get("max_position_embeddings") or 4096),
        "n_kv_heads": kv_heads if kv_heads != heads else None,
        "rope_scaling": scaling, "sliding_window": int(sliding) if sliding else None,
        "tie_word_embeddings": tie, "use_qk_norm": family in ("qwen3", "qwen3_5"),
        "qk_norm_shared": family in ("qwen3", "qwen3_5"), "head_dim": explicit_head,
        "qkv_bias": family == "qwen2", "norm_eps": float(hf.get("rms_norm_eps", 1e-6)),
    }
    if rotary_dim is not None:
        out["rotary_dim"] = rotary_dim
    if family == "qwen3_5":
        out.update(_qwen35_config(hf))
    return out


# ----------------------------------------------------------------------
#  Tensor names
# ----------------------------------------------------------------------
_LAYER_NAMES = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn.W_q.weight", "self_attn.q_proj.bias": "attn.W_q.bias",
    "self_attn.k_proj.weight": "attn.W_k.weight", "self_attn.k_proj.bias": "attn.W_k.bias",
    "self_attn.v_proj.weight": "attn.W_v.weight", "self_attn.v_proj.bias": "attn.W_v.bias",
    "self_attn.o_proj.weight": "attn.W_o.weight",
    "self_attn.q_norm.weight": "attn.q_norm.weight",
    "self_attn.k_norm.weight": "attn.k_norm.weight",
    "mlp.gate_proj.weight": "ffn.gate.weight", "mlp.up_proj.weight": "ffn.up.weight",
    "mlp.down_proj.weight": "ffn.down.weight",
}
_NATIVE_LAYER_NAMES = {native: hf for hf, native in _LAYER_NAMES.items()}
_GLOBAL_NAMES = {"model.embed_tokens.weight": "embedding.weight",
                 "model.norm.weight": "final_norm.weight", "lm_head.weight": "head.weight"}
_NATIVE_GLOBAL_NAMES = {native: hf for hf, native in _GLOBAL_NAMES.items()}
_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.(.+)")
_IGNORED = re.compile(r".*\.rotary_emb\.inv_freq$")


# 2.23: Qwen3.5 - its text part sits under model.language_model (model. in a text-only
# checkpoint); RMSNorms other than the gated one keep weight - 1 ("zero-centered"); q_proj
# holds each head's query and its output gate one after the other
_QWEN35_LINEAR = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "conv1d", "norm",
                  "out_proj", "A_log", "dt_bias")
_QWEN35_ATTN = {"W_q": "q_proj", "W_g": "q_proj", "W_k": "k_proj", "W_v": "v_proj",
                "W_o": "o_proj", "q_norm": "q_norm", "k_norm": "k_norm"}
_QWEN35_ZERO_CENTERED = re.compile(r"(final_norm|layers\.\d+\.(attn_norm|ffn_norm|attn\.q_norm|"
                                   r"attn\.k_norm))\.weight")
_QWEN35_SKIP = re.compile(r"(model\.visual\.|model\.vision_tower\.|mtp\.).*")


def qwen35_hf_name(native: str, prefix: str = "model.language_model.") -> str:
    """The Qwen3.5 checkpoint name of a native parameter (2.23)."""
    if native == "embedding.weight":
        return prefix + "embed_tokens.weight"
    if native == "final_norm.weight":
        return prefix + "norm.weight"
    if native == "head.weight":
        return "lm_head.weight"
    match = re.fullmatch(r"layers\.(\d+)\.(.+)", native)
    if match is None:
        raise HFModelError(f"{native}: no Qwen3.5 equivalent")
    layer, rest = match.groups()
    base = f"{prefix}layers.{layer}."
    if rest in ("attn_norm.weight", "ffn_norm.weight"):
        return base + ("input_layernorm.weight" if rest.startswith("attn")
                       else "post_attention_layernorm.weight")
    if rest.startswith("ffn."):
        return base + "mlp." + {"gate": "gate_proj", "up": "up_proj",
                                "down": "down_proj"}[rest.split(".")[1]] + ".weight"
    if rest.startswith("attn."):
        part = rest[len("attn."):]
        head = part.split(".")[0]
        if head in _QWEN35_LINEAR:
            return base + "linear_attn." + part
        if head in _QWEN35_ATTN:
            return base + "self_attn." + _QWEN35_ATTN[head] + part[len(head):]
    raise HFModelError(f"{native}: no Qwen3.5 equivalent")


def hf_to_native(name: str) -> str | None:
    if name in _GLOBAL_NAMES:
        return _GLOBAL_NAMES[name]
    match = _LAYER_RE.fullmatch(name)
    if match and match.group(2) in _LAYER_NAMES:
        return f"layers.{match.group(1)}.{_LAYER_NAMES[match.group(2)]}"
    return None


def native_to_hf(name: str) -> str:
    if name in _NATIVE_GLOBAL_NAMES:
        return _NATIVE_GLOBAL_NAMES[name]
    match = re.fullmatch(r"layers\.(\d+)\.(.+)", name)
    if match and match.group(2) in _NATIVE_LAYER_NAMES:
        return f"model.layers.{match.group(1)}.{_NATIVE_LAYER_NAMES[match.group(2)]}"
    raise HFModelError(f"{name}: no Hugging Face equivalent")


# ----------------------------------------------------------------------
#  Tokenizer and chat template
# ----------------------------------------------------------------------
def token_id(tokenizer: Any, token: Any) -> int | None:
    """The id of `token` (an added/special token or a vocabulary entry), or None."""
    if not token:
        return None
    if isinstance(token, dict):
        token = token.get("content")
    data = getattr(tokenizer, "data", None)
    if isinstance(data, dict):
        for entry in data.get("added_tokens") or []:
            if isinstance(entry, dict) and entry.get("content") == token:
                return int(entry["id"])
    lookup = getattr(tokenizer, "token_to_id", None) or {}
    value = lookup.get(str(token))
    return int(value) if value is not None else None


def detect_chat_template(tokenizer_config: dict | None, tokenizer: Any = None,
                         jinja: str | None = None) -> str:
    """AuraLite's chat template for a model, from its Jinja chat template.

    ``<|im_start|>`` -> chatml (Qwen, SmolLM2), ``<|start_header_id|>`` ->
    llama3, ``<start_of_turn>`` -> gemma, ``<|assistant|>`` -> phi,
    ``[INST]`` -> mistral. A model without a template (a base model) gets
    chatml when its tokenizer has the ChatML markers, else llama3 when it has
    Llama 3's, else chatml.
    """
    template = jinja
    if not template and isinstance(tokenizer_config, dict):
        template = tokenizer_config.get("chat_template")
    if isinstance(template, list):                 # [{"name": ..., "template": ...}]
        named = {str(t.get("name")): str(t.get("template", "")) for t in template
                 if isinstance(t, dict)}
        template = named.get("default") or next(iter(named.values()), "")
    text = str(template or "")
    for marker, name in (("<|im_start|>", "chatml"), ("<|start_header_id|>", "llama3"),
                         ("<start_of_turn>", "gemma"), ("<|assistant|>", "phi"),
                         ("[INST]", "mistral")):
        if marker in text:
            return name
    if tokenizer is not None:
        if token_id(tokenizer, "<|im_start|>") is not None:
            return "chatml"
        if token_id(tokenizer, "<|start_header_id|>") is not None:
            return "llama3"
    return "chatml"


_JINJA = {
    "qwen3_5": (
        "{% for message in messages %}{% if message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + message['content'] + "
        "'<|im_end|>' + '\\n' }}{% else %}{{ '<|im_start|>' + message['role'] + '\\n' + "
        "message['content'] + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n<think>\\n\\n</think>"
        "\\n\\n' }}{% endif %}"),
    "chatml": (
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + "
        "message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"),
    "llama3": (
        "{{ bos_token }}{% for message in messages %}{{ '<|start_header_id|>' + "
        "message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
        "{% endfor %}{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"),
    "gemma": (
        "{% for message in messages %}{% set role = 'model' if message['role'] == "
        "'assistant' else message['role'] %}{{ '<start_of_turn>' + role + '\\n' + "
        "message['content'] + '<end_of_turn>\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}"),
    "phi": (
        "{% for message in messages %}{{ '<|' + message['role'] + '|>\\n' + "
        "message['content'] + '<|end|>\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"),
}


def jinja_chat_template(name: str) -> str | None:
    """The Jinja (``tokenizer_config.json``) form of AuraLite's template `name`."""
    return _JINJA.get(str(name or "").lower())


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# ----------------------------------------------------------------------
#  Import
# ----------------------------------------------------------------------
@dataclass
class HFImport:
    """What :func:`import_hf` built."""

    model: ModernTransformer
    tokenizer: HFBPETokenizer
    chat_template: str
    source: dict
    folder: Path
    info: dict = field(default_factory=dict)


_DTYPES = {"float32": torch.float32, "fp32": torch.float32, "bfloat16": torch.bfloat16,
           "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}


def _dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    try:
        return _DTYPES[str(value or "float32").lower()]
    except KeyError:
        raise HFModelError(f"dtype must be one of float32, bfloat16, float16; got {value!r}"
                           ) from None


def _parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    path, _, attr = name.rpartition(".")
    return model.get_submodule(path), attr


def import_hf(source: str | os.PathLike, *, quantize: str | None = None,
              dtype: Any = "float32", device: Any = "cpu", block_size: int = 64,
              double_quant: bool = True, quantize_targets=QUANTIZABLE_LINEARS,
              cache_dir: str | os.PathLike | None = None,
              progress_callback: Callable[[dict], None] | None = None,
              stop_event: Any = None) -> HFImport:
    """Build the native model of a Hugging Face Llama / Mistral / Qwen2 / Qwen3.

    Args:
        source: a model folder or ``hf:REPO[@REV]`` (downloaded, see
            :func:`hf_model_dir`).
        quantize: ``"nf4"`` stores the attention and feed-forward weights in
            4 bits (QLoRA); embeddings, head and norms keep `dtype`.
        dtype: of the weights that are not quantized (default float32, what
            CPU training needs; bfloat16 halves the memory of GPU inference).
        device: where the weights go (one tensor at a time).
    """
    started = time.time()
    if quantize not in (None, "", "none", "nf4"):
        raise HFModelError(f"quantize must be 'nf4' or None, got {quantize!r}")
    quantize = "nf4" if quantize == "nf4" else None
    folder = hf_model_dir(source, cache_dir=cache_dir, progress_callback=progress_callback,
                          stop_event=stop_event)
    hf_config = _read_json(folder / "config.json")
    kwargs = native_config(hf_config)
    family = _family(hf_config)
    tokenizer_file = folder / "tokenizer.json"
    if not tokenizer_file.is_file():
        raise HFModelError(f"{folder} has no tokenizer.json")
    try:
        tokenizer = HFBPETokenizer(json.loads(tokenizer_file.read_text(encoding="utf-8")),
                                   source=str(source))
    except TokenizerFormatError as err:
        raise HFModelError(f"the tokenizer of {source} cannot be used: {err}") from err
    tokenizer_config = _read_json(folder / "tokenizer_config.json")
    jinja_file = folder / "chat_template.jinja"
    jinja = jinja_file.read_text(encoding="utf-8") if jinja_file.is_file() else None
    if jinja is None and tokenizer_config.get("chat_template"):
        jinja = tokenizer_config["chat_template"]
    if not isinstance(jinja, str):
        jinja = None
    template = detect_chat_template(tokenizer_config, tokenizer, jinja)
    if family == "qwen3_5" and template == "chatml" and token_id(tokenizer, "</think>") is not None:
        template = "qwen3_5"           # 2.23: its replies open with an empty thinking block
    torch_dtype = _dtype(dtype)
    device = torch.device(device)

    with torch.device("meta"):
        model = ModernTransformer(**kwargs)
    tensors = dict(iter_folder_tensors(folder))
    used: set[str] = set()
    hf_name: Callable[[str], str] = native_to_hf
    if family == "qwen3_5":
        # 2.23: the multimodal checkpoints keep the text part under model.language_model
        prefix = ("model.language_model." if any(n.startswith("model.language_model.")
                                                 for n in tensors) else "model.")
        hf_name = functools.partial(qwen35_hf_name, prefix=prefix)

    def read(native: str) -> torch.Tensor | None:
        name = hf_name(native)
        file = tensors.get(name)
        if file is None:
            return None
        used.add(name)
        tensor = file.get(name)
        if family == "qwen3_5":
            tensor = _qwen35_tensor(native, tensor, kwargs)
        return tensor

    def need(native: str) -> torch.Tensor:
        tensor = read(native)
        if tensor is None:
            raise HFModelError(f"{source}: the weights have no {hf_name(native)}")
        return tensor

    def check(native: str, tensor: torch.Tensor, shape: tuple) -> torch.Tensor:
        if tuple(tensor.shape) != tuple(shape):
            raise HFModelError(f"{hf_name(native)} has the shape {tuple(tensor.shape)}, "
                               f"the config says {tuple(shape)}")
        return tensor

    targets = set(quantize_targets or ())
    quantized = 0
    modules = [(name, module) for name, module in model.named_modules()
               if isinstance(module, (nn.Linear, nn.Embedding, RMSNorm, HeadwiseRMSNorm))]
    total = len(modules)
    for done, (name, module) in enumerate(modules, start=1):
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("import stopped")
        if name == "head" and kwargs["tie_word_embeddings"]:
            continue
        weight = check(name + ".weight", need(name + ".weight"), tuple(module.weight.shape))
        parent, attr = _parent(model, name) if "." in name else (model, name)
        if isinstance(module, nn.Linear):
            bias = None
            if module.bias is not None:
                bias = check(name + ".bias", need(name + ".bias"), tuple(module.bias.shape))
            if quantize and attr in targets and name.startswith("layers."):
                setattr(parent, attr, NF4Linear.from_weight(
                    weight.to(device), None if bias is None else bias.to(device),
                    block_size=block_size, double_quant=double_quant))
                quantized += 1
            else:
                module.weight = nn.Parameter(weight.to(device=device, dtype=torch_dtype))
                if bias is not None:
                    module.bias = nn.Parameter(bias.to(device=device, dtype=torch_dtype))
        else:
            module.weight = nn.Parameter(weight.to(device=device, dtype=torch_dtype))
        del weight
        if progress_callback is not None and (done % 16 == 0 or done == total):
            progress_callback({"stage": "convert", "done": done, "total": total})
    # 2.23: what the loop above does not reach - a linear-attention layer's convolution,
    # its decay (A_log) and time-step bias (dt_bias)
    for name, param in list(model.named_parameters()):
        if not param.is_meta:
            continue
        tensor = read(name)
        if tensor is None:
            continue
        check(name, tensor, tuple(param.shape))
        owner, attr = _parent(model, name)
        setattr(owner, attr, nn.Parameter(tensor.to(device=device, dtype=torch_dtype)))
    if kwargs["tie_word_embeddings"]:
        model.tie_weights()
    skipped = sorted(n for n in tensors if family == "qwen3_5" and _QWEN35_SKIP.fullmatch(n))
    unused = sorted(n for n in tensors if n not in used and not _IGNORED.fullmatch(n)
                    and not (n == "lm_head.weight" and kwargs["tie_word_embeddings"])
                    and n not in skipped)
    if unused:
        raise HFModelError(f"{source}: tensors AuraLite has no place for: "
                           f"{', '.join(unused[:6])}{' ...' if len(unused) > 6 else ''}")
    model.rebuild_rope_tables(device)
    left = [n for n, p in list(model.named_parameters()) + list(model.named_buffers())
            if p.is_meta]
    if left:
        raise HFModelError(f"{source}: no weights for {', '.join(left[:6])}")
    model.eval()

    generation_config = _read_json(folder / "generation_config.json")
    bos = tokenizer_config.get("bos_token") if tokenizer_config else None
    eos = tokenizer_config.get("eos_token") if tokenizer_config else None
    add_bos = tokenizer_config.get("add_bos_token")
    if add_bos is None:
        add_bos = _post_processor_adds_bos(tokenizer.data, bos)
    repo, revision = (parse_hf_spec(str(source)) if str(source).lower().startswith("hf:")
                      else (str(source), ""))
    architecture = next((a for a in hf_config.get("architectures") or ()
                         if a in SUPPORTED_ARCHITECTURES), _FAMILY_ARCH[family])
    source_info = {
        "family": family, "architecture": architecture, "repo": repo,
        "revision": revision, "config": hf_config, "generation_config": generation_config,
        "tokenizer_config": {k: v for k, v in tokenizer_config.items()
                             if k != "added_tokens_decoder"},
        "chat_template_jinja": jinja, "chat_template": template,
        "bos_token_id": token_id(tokenizer, bos), "eos_token_id": token_id(tokenizer, eos),
        "add_bos_token": bool(add_bos),
    }
    params = model.count_parameters()
    info = {"source": str(source), "folder": str(folder), "architecture": architecture,
            "params": params, "quantized_layers": quantized, "quantize": quantize,
            "left_out": _left_out(skipped),
            "dtype": str(torch_dtype).replace("torch.", ""), "chat_template": template,
            "vocab_size": kwargs["vocab_size"], "context": kwargs["max_seq_len"],
            "seconds": time.time() - started, "memory_bytes": model_bytes(model)}
    logger.info("[AuraLite] Imported %s: %s params, %s%s", source, f"{params:,}",
                info["dtype"], f", {quantized} layers in 4 bits (NF4)" if quantized else "")
    return HFImport(model, tokenizer, template, source_info, folder, info)


def _qwen35_tensor(native: str, tensor: torch.Tensor, kwargs: dict) -> torch.Tensor:
    """A Qwen3.5 tensor as the native model holds it (2.23): the query or the output gate
    of q_proj, an RMSNorm weight + 1."""
    if native.endswith((".attn.W_q.weight", ".attn.W_g.weight")):
        heads = int(kwargs["n_heads"])
        head_dim = int(kwargs.get("head_dim") or kwargs["d_model"] // heads)
        if tensor.shape[0] != 2 * heads * head_dim:
            raise HFModelError(f"q_proj has {tensor.shape[0]} rows, not query and gate for "
                               f"{heads} heads of {head_dim}")
        pairs = tensor.reshape(heads, 2, head_dim, tensor.shape[-1])
        part = pairs[:, 0] if native.endswith("W_q.weight") else pairs[:, 1]
        return part.reshape(heads * head_dim, tensor.shape[-1]).contiguous()
    if _QWEN35_ZERO_CENTERED.fullmatch(native):
        return tensor.to(torch.float32) + 1.0
    return tensor


def _left_out(skipped: list[str]) -> list[str]:
    """What an import left out, in words (2.23: Qwen3.5's vision tower and MTP head)."""
    parts = []
    if any(n.startswith(("model.visual.", "model.vision_tower.")) for n in skipped):
        parts.append("the vision encoder (images and video)")
    if any(n.startswith("mtp.") for n in skipped):
        parts.append("the multi-token prediction head")
    return parts


def _post_processor_adds_bos(data: dict, bos: Any) -> bool:
    processor = data.get("post_processor") if isinstance(data, dict) else None
    if not isinstance(processor, dict):
        return False
    for step in processor.get("processors", [processor]):
        if isinstance(step, dict) and step.get("type") == "TemplateProcessing":
            single = step.get("single") or []
            first = single[0].get("SpecialToken", {}).get("id") if single and isinstance(
                single[0], dict) else None
            if first and (bos is None or first == (bos.get("content") if isinstance(bos, dict)
                                                   else bos)):
                return True
    return False


def model_bytes(model: nn.Module) -> int:
    """Bytes of every weight of `model` (parameters and saved buffers, shared ones once).

    The RoPE tables (derived, rebuilt on load) are not counted.
    """
    seen: set[int] = set()
    total = 0
    buffers = [b for name, b in model.named_buffers()
               if not name.endswith((".rope_cos", ".rope_sin", "._probe"))]
    for tensor in list(model.parameters()) + buffers:
        key = id(tensor)
        if key in seen or tensor.is_meta:
            continue
        seen.add(key)
        total += tensor.numel() * tensor.element_size()
    return total


# ----------------------------------------------------------------------
#  Export
# ----------------------------------------------------------------------
def _native_model(model: Any) -> ModernTransformer:
    model = getattr(model, "module", model)
    model = getattr(model, "_orig_mod", model)
    if not isinstance(model, ModernTransformer):
        raise HFModelError("export needs a native AuraLite model (a .pt / .safetensors "
                           "checkpoint, or an imported model)")
    return model


def export_family(model: ModernTransformer, source: dict | None = None) -> str:
    """The Hugging Face family a native model is written as, or HFModelError."""
    if model.use_alibi:
        raise HFModelError("ALiBi models have no Hugging Face Llama/Qwen equivalent")
    if model.use_moe:
        raise HFModelError("mixture-of-experts models have no Hugging Face Llama/Qwen "
                           "equivalent in AuraLite's export")
    if str((model.rope_scaling or {}).get("yarn_blend")) == "legacy":
        raise HFModelError("this model was trained with AuraLite's YaRN before 2.18.1, whose "
                           "frequencies no other runtime computes")
    family = (source or {}).get("family")
    if getattr(model, "is_hybrid", False):
        # 2.23: linear and full attention - Qwen3.5's text model
        if not model.use_qk_norm or model.qkv_bias or model.sliding_window:
            raise HFModelError("a hybrid model is written as Qwen3.5, which has a QK-norm and "
                               "no q/k/v biases or sliding window")
        if not model.qk_norm_shared and not _qk_norm_heads_equal(model):
            raise HFModelError("this hybrid model has a QK-norm scale per head; Qwen3.5 shares "
                               "one scale over the heads")
        return "qwen3_5"
    if getattr(model, "attn_output_gate", False) or getattr(model, "rotary_dim", None):
        raise HFModelError("an attention output gate or partial RoPE without linear-attention "
                           "layers has no Hugging Face equivalent here")
    if model.use_qk_norm:
        if not model.qk_norm_shared and not _qk_norm_heads_equal(model):
            raise HFModelError("this model has a QK-norm scale per head; Qwen3 (the only "
                               "family with QK-norm here) shares one scale over the heads")
        if model.qkv_bias:
            raise HFModelError("QK-norm together with q/k/v biases has no equivalent")
        return "qwen3"
    if model.qkv_bias:
        return "qwen2"
    return "mistral" if (model.sliding_window or family == "mistral") else "llama"


def _qk_norm_heads_equal(model: ModernTransformer) -> bool:
    for attn in model._rope_layers():
        for norm in (attn.q_norm, attn.k_norm):
            weight = norm.weight.detach()
            if weight.dim() == 2 and not torch.equal(weight, weight[:1].expand_as(weight)):
                return False
    return True


def _rope_config(model: ModernTransformer, attn_head_dim: int) -> tuple[float, dict | None]:
    """(rope_theta, rope_scaling) of the Hugging Face config."""
    scaling = dict(model.rope_scaling or {})
    kind = str(scaling.get("type", "none")).lower()
    theta = float(scaling.get("base", 10000.0))
    factor = float(scaling.get("factor", 1.0) or 1.0)
    if kind in ("none", "") or factor <= 1.0:
        return theta, None
    if kind in ("ntk", "dynamic_ntk"):
        # AuraLite's NTK is static: plain RoPE with a larger base
        return float(model._rope_layers()[0]._ntk_base()), None
    if kind == "linear":
        return theta, {"rope_type": "linear", "factor": factor}
    if kind == "llama3":
        return theta, {"rope_type": "llama3", "factor": factor,
                       "low_freq_factor": float(scaling.get("low_freq_factor", 1.0)),
                       "high_freq_factor": float(scaling.get("high_freq_factor", 4.0)),
                       "original_max_position_embeddings": int(
                           scaling.get("original_max_position_embeddings", 8192))}
    if kind == "yarn":
        out = {"rope_type": "yarn", "factor": factor,
               "original_max_position_embeddings": int(
                   scaling.get("original_max_position_embeddings", model.max_seq_len)),
               "beta_fast": float(scaling.get("beta_fast", 32.0)),
               "beta_slow": float(scaling.get("beta_slow", 1.0))}
        if scaling.get("attention_factor") is not None:
            out["attention_factor"] = float(scaling["attention_factor"])
        return theta, out
    raise HFModelError(f"RoPE scaling {kind!r} has no Hugging Face equivalent")


def hf_config_for(model: ModernTransformer, family: str, tokenizer: Any,
                  source: dict | None = None, dtype: torch.dtype = torch.bfloat16) -> dict:
    """The ``config.json`` of a native model written as `family`."""
    theta, rope_scaling = _rope_config(model, model.head_dim)
    base = dict((source or {}).get("config") or {}) if (source or {}).get("family") == family \
        else {}
    for key in ("rope_parameters", "rope_scaling", "rope_theta", "torch_dtype", "dtype",
                "transformers_version", "_name_or_path", "layer_types", "auto_map"):
        base.pop(key, None)
    bos = (source or {}).get("bos_token_id")
    eos = (source or {}).get("eos_token_id")
    if eos is None:
        from .shards import _eos_id
        found = _eos_id(tokenizer)
        eos = found[0] if found else None
    config = dict(base)
    config.update({
        "architectures": [_FAMILY_ARCH[family]], "model_type": family,
        "vocab_size": int(model.embedding.num_embeddings), "hidden_size": int(model.d_model),
        "intermediate_size": int(model.d_ff), "num_hidden_layers": int(model.n_layers),
        "num_attention_heads": int(model.n_heads),
        "num_key_value_heads": int(model.n_kv_heads or model.n_heads),
        "head_dim": int(model.head_dim), "hidden_act": "silu",
        "max_position_embeddings": int(model.max_seq_len),
        "rms_norm_eps": float(model.norm_eps), "rope_theta": theta,
        "rope_scaling": rope_scaling,
        "tie_word_embeddings": bool(model.head.weight is model.embedding.weight),
        "torch_dtype": str(dtype).replace("torch.", ""), "use_cache": True,
        "attention_dropout": 0.0, "bos_token_id": bos, "eos_token_id": eos,
    })
    # transformers 5 reads rope_parameters; 4.x reads rope_theta / rope_scaling
    config["rope_parameters"] = dict(rope_scaling or {"rope_type": "default"},
                                     rope_theta=theta)
    if family == "qwen3_5":
        return _qwen35_config_for(model, config, base, theta)
    if family == "llama":
        config.update(attention_bias=False, mlp_bias=False, pretraining_tp=1)
    elif family == "mistral":
        config["sliding_window"] = int(model.sliding_window) if model.sliding_window else None
    elif family in ("qwen2", "qwen3"):
        config.update(use_sliding_window=False, sliding_window=None,
                      max_window_layers=int(model.n_layers))
        if family == "qwen3":
            config["attention_bias"] = False
    return config


def _qwen35_config_for(model: ModernTransformer, config: dict, base: dict,
                       theta: float) -> dict:
    """The text-only ``config.json`` of a hybrid model (2.23): ``Qwen3_5ForCausalLM``."""
    source_text = text_config(base) if base else {}
    linear = dict(model.linear_attention or {})
    rotary = int(model.rotary_dim or model.head_dim)
    rope = {"rope_type": "default", "rope_theta": theta,
            "partial_rotary_factor": rotary / float(model.head_dim)}
    old_rope = source_text.get("rope_parameters") or {}
    section = old_rope.get("mrope_section")
    if not (isinstance(section, list) and sum(int(x) for x in section) == rotary // 2):
        third = rotary // 2 // 3
        section = [third, third, rotary // 2 - 2 * third]
    rope.update(mrope_section=[int(x) for x in section],
                mrope_interleaved=bool(old_rope.get("mrope_interleaved", True)))
    types = list(model.layer_types or [])
    full = [i for i, t in enumerate(types) if t == "full_attention"]
    interval = (full[1] - full[0]) if len(full) > 1 else (full[0] + 1 if full else len(types))
    out = {key: config[key] for key in (
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "hidden_act",
        "max_position_embeddings", "rms_norm_eps", "tie_word_embeddings", "torch_dtype",
        "use_cache", "attention_dropout", "bos_token_id", "eos_token_id")}
    out.update({
        "architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text",
        "attention_bias": False, "attn_output_gate": bool(model.attn_output_gate),
        "layer_types": types, "full_attention_interval": int(interval),
        "linear_num_key_heads": int(linear.get("num_k_heads", 16)),
        "linear_num_value_heads": int(linear.get("num_v_heads", 16)),
        "linear_key_head_dim": int(linear.get("k_head_dim", 128)),
        "linear_value_head_dim": int(linear.get("v_head_dim", 128)),
        "linear_conv_kernel_dim": int(linear.get("conv_kernel", 4)),
        "mlp_only_layers": [], "mtp_num_hidden_layers": 0, "rope_parameters": rope,
        "dtype": config.get("torch_dtype")})
    return out


def qwen35_export_tensors(model: Any, prefix: str = "model."
                          ) -> Iterator[tuple[str, tuple[int, ...], Callable[[], torch.Tensor]]]:
    """``(Qwen3.5 name, shape, produce float32 tensor)`` of a hybrid model (2.23): the
    query and the output gate interleaved per head into q_proj, the zero-centered norms
    as weight - 1, adapters merged, 4-bit layers dequantized."""
    from .linear_attention import GatedDeltaNet
    model = _native_model(model)
    adapters = {name: adapter for name, _linear, adapter in model.lora_pairs()}
    rows, width = int(model.embedding.num_embeddings), int(model.d_model)

    def norm(module: Any) -> Callable[[], torch.Tensor]:
        return lambda: module.weight.detach().float().reshape(-1, module.weight.shape[-1])[0] - 1.0

    yield (prefix + "embed_tokens.weight", (rows, width),
           lambda: model.embedding.weight.detach().float())
    for i, block in enumerate(model.layers):
        layer = cast(TransformerBlock, block)
        native = f"layers.{i}."
        base = f"{prefix}layers.{i}."
        yield (base + "input_layernorm.weight", (width,), norm(layer.attn_norm))
        yield (base + "post_attention_layernorm.weight", (width,), norm(layer.ffn_norm))
        attn: Any = layer.attn              # an Attention or (2.23) a GatedDeltaNet
        if isinstance(attn, GatedDeltaNet):
            for proj in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                module = getattr(attn, proj)
                yield (f"{base}linear_attn.{proj}.weight",
                       (int(module.out_features), int(module.in_features)),
                       lambda m=module, a=adapters.get(f"{native}attn.{proj}"):
                       _merged_weight(m, a))
            yield (base + "linear_attn.conv1d.weight", tuple(attn.conv1d.weight.shape),
                   lambda a=attn: a.conv1d.weight.detach().float())
            for name in ("A_log", "dt_bias"):
                yield (f"{base}linear_attn.{name}", tuple(getattr(attn, name).shape),
                       lambda a=attn, n=name: getattr(a, n).detach().float())
            yield (base + "linear_attn.norm.weight", tuple(attn.norm.weight.shape),
                   lambda a=attn: a.norm.weight.detach().float())
        else:
            heads, head_dim = int(attn.n_heads), int(attn.head_dim)

            def q_proj(a: Any = attn, i: int = i, heads: int = heads,
                       head_dim: int = head_dim) -> torch.Tensor:
                q = _merged_weight(a.W_q, adapters.get(f"layers.{i}.attn.W_q"))
                g = _merged_weight(a.W_g, adapters.get(f"layers.{i}.attn.W_g"))
                return torch.stack([q.reshape(heads, head_dim, -1),
                                    g.reshape(heads, head_dim, -1)], dim=1).reshape(
                    2 * heads * head_dim, -1)

            yield (base + "self_attn.q_proj.weight", (2 * heads * head_dim, width), q_proj)
            for proj, hf in (("W_k", "k_proj"), ("W_v", "v_proj"), ("W_o", "o_proj")):
                module = getattr(attn, proj)
                yield (f"{base}self_attn.{hf}.weight",
                       (int(module.out_features), int(module.in_features)),
                       lambda m=module, a=adapters.get(f"{native}attn.{proj}"):
                       _merged_weight(m, a))
            yield (base + "self_attn.q_norm.weight", (head_dim,), norm(attn.q_norm))
            yield (base + "self_attn.k_norm.weight", (head_dim,), norm(attn.k_norm))
        for proj, hf in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
            module = getattr(layer.ffn, proj)
            yield (f"{base}mlp.{hf}.weight", (int(module.out_features), int(module.in_features)),
                   lambda m=module, a=adapters.get(f"{native}ffn.{proj}"): _merged_weight(m, a))
    yield (prefix + "norm.weight", (width,), norm(model.final_norm))
    if model.head.weight is not model.embedding.weight:
        yield ("lm_head.weight", (rows, width), lambda: model.head.weight.detach().float())


def _merged_weight(module: nn.Module, adapter: Any) -> torch.Tensor:
    weight = module.dequantize(torch.float32) if isinstance(module, NF4Linear) \
        else cast(torch.Tensor, module.weight).detach().float()
    if adapter is not None:
        weight = weight + adapter.delta_weight().detach().float().to(weight.device)
    return weight


def export_tensors(model: Any) -> Iterator[tuple[str, tuple[int, ...], Callable[[], torch.Tensor]]]:
    """``(native name, shape, produce float32 tensor)`` of every weight to export.

    LoRA adapters are merged into the weights they adapt and 4-bit layers
    dequantized - one tensor at a time, without changing `model`. A tied
    output head is not listed.
    """
    model = _native_model(model)
    adapters: dict[str, Any] = {}
    for name, _linear, adapter in model.lora_pairs():
        adapters[name] = adapter
    tied = model.head.weight is model.embedding.weight
    rows = int(model.embedding.num_embeddings)
    yield ("embedding.weight", (rows, int(model.d_model)),
           lambda: model.embedding.weight.detach().float())
    for i, block in enumerate(model.layers):
        prefix = f"layers.{i}."
        layer = cast(TransformerBlock, block)
        attn: Any = layer.attn              # an Attention or (2.23) a GatedDeltaNet
        ffn = layer.ffn
        yield (prefix + "attn_norm.weight", tuple(layer.attn_norm.weight.shape),
               lambda m=layer.attn_norm: m.weight.detach().float())
        for part, block, names in (("attn", attn, ("W_q", "W_k", "W_v", "W_o")),
                                   ("ffn", ffn, ("gate", "up", "down"))):
            for proj in names:
                module = getattr(block, proj)
                key = f"{prefix}{part}.{proj}"
                shape = (int(module.out_features), int(module.in_features))
                yield (key + ".weight", shape,
                       lambda m=module, a=adapters.get(key): _merged_weight(m, a))
                if getattr(module, "bias", None) is not None:
                    yield (key + ".bias", (int(module.out_features),),
                           lambda m=module: m.bias.detach().float())
            if part == "attn" and model.use_qk_norm:
                for norm_name in ("q_norm", "k_norm"):
                    norm = getattr(attn, norm_name)
                    head_dim = int(attn.head_dim)
                    yield (f"{prefix}attn.{norm_name}.weight", (head_dim,),
                           lambda n=norm: n.weight.detach().float().reshape(
                               -1, n.weight.shape[-1])[0])
        yield (prefix + "ffn_norm.weight", tuple(layer.ffn_norm.weight.shape),
               lambda m=layer.ffn_norm: m.weight.detach().float())
    yield ("final_norm.weight", tuple(model.final_norm.weight.shape),
           lambda: model.final_norm.weight.detach().float())
    if not tied:
        yield ("head.weight", (rows, int(model.d_model)), lambda: model.head.weight.detach().float())


def _tokenizer_files(tokenizer: Any, template: str, source: dict | None,
                     context: int) -> tuple[dict, dict]:
    """(tokenizer.json, tokenizer_config.json) contents."""
    if not isinstance(tokenizer, HFBPETokenizer):
        raise HFModelError("only models with a Hugging Face tokenizer (an imported model, or "
                           "one trained with --tokenizer hf:... / a tokenizer.json) can be "
                           "exported: AuraLite's own tokenizers have no tokenizer.json form")
    data = copy.deepcopy(tokenizer.data)
    config = dict((source or {}).get("tokenizer_config") or {})
    jinja = None
    if (source or {}).get("chat_template") == template:
        jinja = (source or {}).get("chat_template_jinja")
    if not jinja:
        jinja = jinja_chat_template(template)
    if jinja:
        config["chat_template"] = jinja
    config["added_tokens_decoder"] = {
        str(int(entry["id"])): {key: entry.get(key, False) for key in
                                ("content", "lstrip", "normalized", "rstrip", "single_word",
                                 "special")}
        for entry in data.get("added_tokens") or [] if isinstance(entry, dict)}
    config.setdefault("model_max_length", int(context))
    config.setdefault("clean_up_tokenization_spaces", False)
    config.setdefault("tokenizer_class", "PreTrainedTokenizerFast")
    if "eos_token" not in config:
        from .shards import _eos_id
        found = _eos_id(tokenizer)
        if found:
            config["eos_token"] = found[1]
    if template == "chatml" and token_id(tokenizer, "<|im_end|>") is not None:
        config.setdefault("pad_token", config.get("eos_token"))
    return data, config


def export_hf(model: Any, tokenizer: Any, out_dir: str | os.PathLike, *,
              chat_template: str = "chatml", source: dict | None = None,
              dtype: Any = "bfloat16", max_shard_bytes: int = 4 * 1024 ** 3,
              progress_callback: Callable[[dict], None] | None = None) -> dict:
    """Write a native model as a Hugging Face model folder; returns a summary.

    The folder loads with ``AutoModelForCausalLM.from_pretrained`` and
    converts with llama.cpp's ``convert_hf_to_gguf.py``. Llama / Mistral /
    Qwen2 / Qwen3 is chosen from the model (an imported model keeps its own).
    """
    model = _native_model(model)
    tokenizer_json, tokenizer_config = _tokenizer_files(tokenizer, chat_template, source,
                                                        model.max_seq_len)
    family = export_family(model, source)
    torch_dtype = _dtype(dtype)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = hf_config_for(model, family, tokenizer, source, torch_dtype)
    if family == "qwen3_5":           # 2.23: names, q_proj and the norms of Qwen3.5
        specs = [TensorSpec(name, shape, torch_dtype, lambda p=produce: p().to(torch_dtype))
                 for name, shape, produce in qwen35_export_tensors(model)]
    else:
        specs = [TensorSpec(native_to_hf(name), shape, torch_dtype,
                            lambda p=produce: p().to(torch_dtype))
                 for name, shape, produce in export_tensors(model)]
    shards: list[list[TensorSpec]] = [[]]
    size = 0
    for spec in specs:
        if shards[-1] and size + spec.nbytes > max_shard_bytes:
            shards.append([])
            size = 0
        shards[-1].append(spec)
        size += spec.nbytes
    total_bytes = sum(spec.nbytes for spec in specs)
    for stale in out.glob("model*.safetensors*"):
        stale.unlink()
    weight_map = {}
    for index, shard in enumerate(shards, start=1):
        name = ("model.safetensors" if len(shards) == 1
                else f"model-{index:05d}-of-{len(shards):05d}.safetensors")
        write_safetensors(out / name, shard, {"format": "pt"})
        weight_map.update({spec.name: name for spec in shard})
        if progress_callback is not None:
            progress_callback({"stage": "write", "done": index, "total": len(shards)})
    if len(shards) > 1:
        (out / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2),
            encoding="utf-8")
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    generation = dict((source or {}).get("generation_config") or {})
    generation.setdefault("bos_token_id", config.get("bos_token_id"))
    generation.setdefault("eos_token_id", config.get("eos_token_id"))
    generation.pop("transformers_version", None)
    (out / "generation_config.json").write_text(json.dumps(generation, indent=2),
                                                encoding="utf-8")
    (out / "tokenizer.json").write_text(json.dumps(tokenizer_json, ensure_ascii=False),
                                        encoding="utf-8")
    (out / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, ensure_ascii=False), encoding="utf-8")
    jinja = tokenizer_config.get("chat_template")
    if isinstance(jinja, str):
        (out / "chat_template.jinja").write_text(jinja, encoding="utf-8")
    return {"folder": str(out), "architecture": _FAMILY_ARCH[family], "family": family,
            "tensors": len(specs), "shards": len(shards), "bytes": total_bytes,
            "dtype": str(torch_dtype).replace("torch.", ""), "chat_template": chat_template}


def copy_tokenizer_files(folder: Path, out: Path) -> None:
    """Copy the tokenizer files of an imported model folder next to an export."""
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (folder / name).is_file():
            shutil.copy2(folder / name, out / name)


def describe_import(info: dict) -> str:
    """One line about an import, for the CLI and the GUI."""
    params = int(info.get("params", 0))
    size = f"{params / 1e9:.2f}B" if params >= 1e9 else f"{params / 1e6:.0f}M"
    quant = (f", {info['quantized_layers']} layers in 4 bits (NF4)"
             if info.get("quantized_layers") else "")
    return (f"{info.get('architecture')} {size} params, {info.get('dtype')}{quant}, "
            f"context {int(info.get('context', 0)):,}, chat template "
            f"{info.get('chat_template')}, {float(info.get('memory_bytes', 0)) / 1e9:.2f} GB "
            f"in memory, {float(info.get('seconds', 0)):.1f} s")


def estimate_import_bytes(hf: dict, quantize: str | None = None,
                          dtype: Any = "float32") -> int:
    """Memory the imported model will take (weights only)."""
    kwargs = native_config(hf)
    d, layers, ff = kwargs["d_model"], kwargs["n_layers"], kwargs["d_ff"]
    heads, kv = kwargs["n_heads"], kwargs["n_kv_heads"] or kwargs["n_heads"]
    head_dim = kwargs["head_dim"] or d // heads
    per_layer = d * head_dim * (2 * heads + 2 * kv) + 3 * d * ff
    rest = kwargs["vocab_size"] * d * (1 if kwargs["tie_word_embeddings"] else 2)
    element = torch.empty((), dtype=_dtype(dtype)).element_size()
    linear_bytes = per_layer * layers * (4.13 / 8 if quantize == "nf4" else element)
    return int(linear_bytes + rest * element)


def safetensors_header_dtypes(folder: Path) -> dict[str, int]:
    """How many tensors of each dtype a model folder holds (diagnostics)."""
    counts: dict[str, int] = {}
    for name, file in iter_folder_tensors(folder):
        kind = file.dtype(name)
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def read_hf_config(spec: str | os.PathLike) -> dict:
    """The ``config.json`` of a local model folder."""
    return _read_json(Path(spec) / "config.json")
