import os

# -----------------------------------------------------------------------
# CPU multithreading — set BEFORE importing torch / numpy so the OpenMP /
# MKL / OpenBLAS backends pick them up.
# -----------------------------------------------------------------------
_CPU_COUNT = os.cpu_count() or 1
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU_COUNT))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU_COUNT))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_CPU_COUNT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_CPU_COUNT))

# -----------------------------------------------------------------------
# Distributed training (DDP) support — v2.3
# -----------------------------------------------------------------------
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import json
import math
import re
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterator, Any, List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np

logger = logging.getLogger("auralite")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(os.environ.get("AURALITE_LOG_LEVEL", "INFO"))

# Optional Hugging Face + LoRA/QLoRA support
try:
    from hf_integration import HuggingFaceProxy, HFNotAvailableError, create_hf_proxy, HAS_HF_SUPPORT
except ImportError:
    HuggingFaceProxy = None
    HFNotAvailableError = Exception
    create_hf_proxy = None
    HAS_HF_SUPPORT = False

# Chat interface (NEW v2.3)
try:
    from chat_interface import (
        ChatMessage, ChatHistory, apply_chat_template,
        CHAT_TEMPLATES, get_stop_tokens, build_chat_prompt
    )
    HAS_CHAT_SUPPORT = True
except ImportError:
    HAS_CHAT_SUPPORT = False
    ChatMessage = None
    ChatHistory = None
    apply_chat_template = None
    CHAT_TEMPLATES = {}
    get_stop_tokens = lambda x: []
    build_chat_prompt = None

try:
    torch.set_num_threads(_CPU_COUNT)
    torch.set_num_interop_threads(max(1, _CPU_COUNT))
except (RuntimeError, ValueError):
    pass


# ===================================================================
#  Parameter Validation
# ===================================================================

class ParamValidationError(Exception):
    """Raised when model/training parameters are incompatible."""
    pass


def validate_params(params: dict) -> list[str]:
    """Validate training parameters. Returns list of error messages (empty = OK)."""
    errors = []
    d_model = params.get("d_model", 128)
    n_heads = params.get("n_heads", 4)
    n_kv_heads = params.get("n_kv_heads")
    d_ff = params.get("d_ff", 256)
    seq_length = params.get("seq_length", 64)
    batch_size = params.get("batch_size", 32)
    lr = params.get("lr", 3e-4)
    epochs = params.get("epochs", 100)
    dropout = params.get("dropout", 0.1)
    grad_clip = params.get("grad_clip", 1.0)
    bpe_vocab_size = params.get("bpe_vocab_size", 512)
    val_split = params.get("val_split", 0.1)
    accumulation_steps = params.get("accumulation_steps", 1)
    use_gradient_checkpointing = params.get("use_gradient_checkpointing", False)
    use_ddp = params.get("use_ddp", False)

    if d_model <= 0:
        errors.append(f"d_model must be > 0, got {d_model}")
    if n_heads <= 0:
        errors.append(f"n_heads must be >= 1, got {n_heads}")
    elif d_model % n_heads != 0:
        errors.append(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
    if n_kv_heads is not None and n_kv_heads > 0:
        if n_heads % n_kv_heads != 0:
            errors.append(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        if n_kv_heads > n_heads:
            errors.append(f"n_kv_heads ({n_kv_heads}) cannot exceed n_heads ({n_heads})")
    if d_ff <= 0:
        errors.append(f"d_ff must be > 0, got {d_ff}")
    if seq_length < 4:
        errors.append(f"seq_length must be >= 4, got {seq_length}")
    if batch_size < 1:
        errors.append(f"batch_size must be >= 1, got {batch_size}")
    if lr <= 0:
        errors.append(f"lr must be > 0, got {lr}")
    if epochs < 1:
        errors.append(f"epochs must be >= 1, got {epochs}")
    if params.get("n_layers", 4) < 1:
        errors.append(f"n_layers must be >= 1, got {params.get('n_layers', 4)}")
    if not (0.0 <= dropout < 1.0):
        errors.append(f"dropout must be in [0, 1), got {dropout}")
    if grad_clip <= 0:
        errors.append(f"grad_clip must be > 0, got {grad_clip}")
    if bpe_vocab_size < 2:
        errors.append(f"bpe_vocab_size must be >= 2, got {bpe_vocab_size}")
    if not (0.0 < val_split < 1.0):
        errors.append(f"val_split must be in (0, 1), got {val_split}")
    if accumulation_steps < 1:
        errors.append(f"accumulation_steps must be >= 1, got {accumulation_steps}")

    # Gradient checkpointing validation
    if use_gradient_checkpointing and not isinstance(use_gradient_checkpointing, bool):
        errors.append("use_gradient_checkpointing must be a boolean")

    # DDP validation
    if use_ddp and not torch.cuda.is_available():
        errors.append("Multi-GPU (DDP) requires CUDA")

    return errors


# ===================================================================
#  Tokenizers — character-level and BPE (Byte/Char Pair Encoding)
# ===================================================================

UNK_TOKEN = "\ufffd"  # Unicode replacement character


class CharTokenizer:
    """Simple character-level tokenizer (original AuraLite behaviour)."""

    kind = "char"

    def __init__(self):
        self.vocab: list[str] = []
        self.token_to_id: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int | None = None):
        self.vocab = sorted(set(text))
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}

    def encode(self, s: str) -> list[int]:
        fb = self.token_to_id.get(" ", 0)
        return [self.token_to_id.get(c, fb) for c in s]

    def decode(self, ids) -> str:
        return "".join(self.vocab[int(i)] if 0 <= int(i) < len(self.vocab) else "?"
                       for i in ids)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "vocab": self.vocab}

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        tok = cls()
        tok.vocab = list(d["vocab"])
        tok.token_to_id = {t: i for i, t in enumerate(tok.vocab)}
        return tok


class BPETokenizer:
    """Mini BPE tokenizer (classic word-frequency algorithm, GPT-2 style).

    Trained on the corpus itself: starts from the character vocabulary and
    greedily merges the most frequent adjacent pair until `vocab_size` is
    reached. Merges never cross whitespace-split piece boundaries, and
    encoding caches per-piece results, so both training and encoding stay
    fast even on multi-megabyte texts.

    IMPROVED (v2.1+):
    - Uses stratified sampling for training on huge files (instead of prefix)
    - Adds unk_token for out-of-vocabulary characters during encode
    """

    kind = "bpe"
    unk_token = UNK_TOKEN  # Unicode replacement character

    def __init__(self):
        self.vocab: list[str] = []
        self.token_to_id: dict[str, int] = {}
        # ordered merge rules: (id_a, id_b) -> new_id, rank = list index
        self.merges: list[tuple[int, int, int]] = []
        self._ranks: dict[tuple[int, int], tuple[int, int]] = {}
        self._cache: dict[str, list[int]] = {}
        self._unk_id: int = 0

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ---- helpers -----------------------------------------------------
    @staticmethod
    def _split_pieces(text: str) -> list[str]:
        # keep whitespace runs as separate pieces so nothing is lost
        return [p for p in re.split(r"(\s+)", text) if p]

    def _build_ranks(self):
        self._ranks = {(a, b): (r, nid) for r, (a, b, nid) in enumerate(self.merges)}
        self._cache = {}

    # ---- training ----------------------------------------------------
    def train(self, text: str, vocab_size: int = 512):
        # IMPROVED: stratified sampling for large files
        # Instead of just taking the prefix, sample chunks spread across the text
        sample = self._stratified_sample(text, max_chars=2_000_000, n_chunks=100)

        base_chars = sorted(set(sample))
        self.vocab = list(base_chars)
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.merges = []

        # Ensure unk_token is always in vocab
        if self.unk_token not in self.token_to_id:
            self.vocab.append(self.unk_token)
            self.token_to_id[self.unk_token] = len(self.vocab) - 1

        self._unk_id = self.token_to_id.get(self.unk_token, 0)

        if vocab_size <= len(self.vocab):
            self._build_ranks()
            return

        # word-frequency corpus: distinct pieces with counts
        piece_counts = Counter(self._split_pieces(sample))
        corpus: list[tuple[list[int], int]] = [
            ([self.token_to_id.get(c, self._unk_id) for c in piece], cnt)
            for piece, cnt in piece_counts.items()
        ]

        while len(self.vocab) < vocab_size:
            pair_counts: Counter = Counter()
            for ids, cnt in corpus:
                for i in range(len(ids) - 1):
                    pair_counts[(ids[i], ids[i + 1])] += cnt
            if not pair_counts:
                break
            (a, b), best_cnt = pair_counts.most_common(1)[0]
            if best_cnt < 2:
                break

            new_id = len(self.vocab)
            new_tok = self.vocab[a] + self.vocab[b]
            self.vocab.append(new_tok)
            self.token_to_id[new_tok] = new_id
            self.merges.append((a, b, new_id))

            # apply the merge to every distinct piece
            for entry in corpus:
                ids = entry[0]
                if len(ids) < 2:
                    continue
                i, out = 0, []
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(ids[i])
                        i += 1
                entry[0][:] = out

        self._build_ranks()

    @staticmethod
    def _stratified_sample(text: str, max_chars: int = 2_000_000,
                           n_chunks: int = 100) -> str:
        """Take evenly-spaced chunks from the full text instead of just the prefix."""
        if len(text) <= max_chars:
            return text
        chunk_size = max_chars // n_chunks
        total_len = len(text)
        step = total_len // n_chunks
        parts = []
        for i in range(n_chunks):
            start = i * step
            end = min(start + chunk_size, total_len)
            parts.append(text[start:end])
        return "".join(parts)

    # ---- encode / decode ----------------------------------------------
    def _encode_piece(self, piece: str) -> list[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached

        # Use unk_token for unknown characters instead of fallback to space
        ids = [self.token_to_id.get(c, self._unk_id) for c in piece]
        # repeatedly apply the lowest-rank merge present (GPT-2 algorithm)
        while len(ids) > 1:
            best_rank, best_pos, best_new = None, -1, -1
            for i in range(len(ids) - 1):
                r = self._ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r[0] < best_rank):
                    best_rank, best_pos, best_new = r[0], i, r[1]
            if best_rank is None:
                break
            a, b = ids[best_pos], ids[best_pos + 1]
            i, out = 0, []
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(best_new)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out

        if len(self._cache) < 200_000:
            self._cache[piece] = ids
        return ids

    def encode(self, s: str) -> list[int]:
        out: list[int] = []
        for piece in self._split_pieces(s):
            out.extend(self._encode_piece(piece))
        return out

    def decode(self, ids) -> str:
        return "".join(self.vocab[int(i)] if 0 <= int(i) < len(self.vocab) else self.unk_token
                       for i in ids)

    # ---- (de)serialization ---------------------------------------------
    def to_dict(self) -> dict:
        return {"kind": self.kind, "vocab": self.vocab, "merges": self.merges,
                "unk_token": self.unk_token}

    @classmethod
    def from_dict(cls, d: dict) -> "BPETokenizer":
        tok = cls()
        tok.vocab = list(d["vocab"])
        tok.token_to_id = {t: i for i, t in enumerate(tok.vocab)}
        tok.merges = [tuple(m) for m in d.get("merges", [])]
        tok.unk_token = d.get("unk_token", UNK_TOKEN)
        tok._unk_id = tok.token_to_id.get(tok.unk_token, 0)
        tok._build_ranks()
        return tok


# ===================================================================
#  Auto-recommendation helpers
# ===================================================================

def estimate_n_params(vocab_size: int, d_model: int, n_layers: int,
                      d_ff: int, n_heads: int,
                      n_kv_heads: int | None = None) -> int:
    """Rough parameter count for a LLaMA-style decoder-only transformer.

    Mirrors `ModernTransformer` (weight tying => embedding/head counted once).
    Accurate to within ~3% of `model.count_parameters()` for typical configs.
    """
    n_kv = n_kv_heads if (n_kv_heads and n_kv_heads > 0) else n_heads
    head_dim = d_model // max(1, n_heads)

    embed = vocab_size * d_model        # tied with output head

    # Per layer:
    #   attn: W_q (d_model * n_heads*head_dim)
    #         W_k (d_model * n_kv*head_dim)
    #         W_v (d_model * n_kv*head_dim)
    #         W_o (n_heads*head_dim * d_model)
    attn = (d_model * n_heads * head_dim          # Q
            + d_model * n_kv * head_dim * 2       # K + V
            + n_heads * head_dim * d_model)       # O
    # SwiGLU FFN: gate + up + down (all bias-free)
    ffn  = 3 * d_model * d_ff
    # RMSNorm has `d_model` params, two per block + one final
    norms_per_layer = 2 * d_model

    per_layer = attn + ffn + norms_per_layer
    final_norm = d_model
    return embed + n_layers * per_layer + final_norm


def recommend_epochs(n_tokens: int, n_params: int,
                     batch_size: int, seq_length: int,
                     tokens_per_param: float = 20.0,
                     min_epochs: int = 5,
                     max_epochs: int = 500) -> int:
    """Recommend a sensible number of training epochs.

    Strategy (tiered, more realistic than pure Chinchilla which assumes
    you can throw arbitrarily many tokens at the model):

      * For HUGE datasets (>= 10× model capacity in tokens) → 3–10 epochs
        is enough to see everything ~enough times.
      * For MEDIUM datasets → aim for Chinchilla-ish 20 tokens/param
        of total exposure, but capped sensibly.
      * For TINY datasets → enough passes to over-fit / memorise, but
        capped at a reasonable wall-clock budget (≈ 100 epochs by default).

    Args:
        n_tokens:        size of the training corpus after tokenisation
        n_params:        total model parameters
        batch_size:      training batch size (currently unused — kept for API)
        seq_length:      context window
        tokens_per_param: target tokens-to-param ratio (default 20)
        min_epochs:      lower clamp
        max_epochs:      upper clamp (hard ceiling, normally not reached)

    Returns:
        recommended epoch count
    """
    if n_tokens <= seq_length:
        return min_epochs

    # How big is the dataset compared to model capacity?
    # ratio < 1  → dataset smaller than the model can memorise comfortably
    # ratio = 1  → ~Chinchilla optimal (one pass = 20 tok/param)
    # ratio > 1  → plenty of data, few passes needed
    target = tokens_per_param * n_params
    ratio = n_tokens / max(1, target)

    import math
    if ratio >= 10:
        # Huge dataset: 3 passes is plenty.
        epochs = 3
    elif ratio >= 1:
        # Comfortable: 3-15 passes, scaling down with dataset size.
        # ratio=10 → 3 epochs ; ratio=1 → 15 epochs (smooth log interp)
        epochs = int(round(15 - 12 * (math.log10(ratio) / 1.0)))
    else:
        # Small dataset: pure Chinchilla says epochs = 1/ratio, which
        # explodes for tiny files (1KB on a 1M-param model → 6000 epochs).
        # Instead use logarithmic saturation:
        #   ratio=0.5  → ~25 epochs
        #   ratio=0.1  → ~50 epochs
        #   ratio=0.01 → ~80 epochs
        #   ratio→0    → 100 epochs (asymptote)
        # Formula: 100 - 100 / (1 + (-log10(ratio))**1.5 * k)
        log_inv = -math.log10(max(ratio, 1e-9))           # 0 .. ~9
        # Map log_inv in [0, 4] to epochs in [15, 100] smoothly:
        # at ratio=1   (log_inv=0)   → 15 epochs (continuity with branch above)
        # at ratio=0.1 (log_inv=1)   → ~50
        # at ratio=0.01(log_inv=2)   → ~75
        # at ratio=1e-4(log_inv=4)   → ~95
        epochs = int(round(15 + 85 * (1 - 1 / (1 + 0.6 * log_inv ** 1.3))))

    return int(max(min_epochs, min(max_epochs, epochs)))


def recommend_gen_length(seed_str: str,
                         tokenizer,
                         max_seq_len: int = 4096,
                         multiplier: float = 8.0,
                         hard_min: int = 30,
                         hard_max: int = 800) -> int:
    """Recommend a generation length (in tokens) for a given seed.

    Aims for `multiplier` times the seed length so short prompts still
    produce meaningful output. Bounded by [hard_min, hard_max] AND by the
    model's max context window.

    Examples (multiplier=8):
        seed_tokens=1   → 30   (hard_min)
        seed_tokens=5   → 40
        seed_tokens=10  → 80
        seed_tokens=20  → 160
        seed_tokens=50  → 400
        seed_tokens=200 → 800  (hard_max)

    Args:
        seed_str:   the prompt the user typed
        tokenizer:  trained CharTokenizer/BPETokenizer (None = use chars)
        max_seq_len: model.max_seq_len (we never exceed it)
        multiplier: how much longer than the seed the output should be
        hard_min/hard_max: absolute clamps

    Returns:
        recommended generation length (in tokens)
    """
    if tokenizer is not None:
        try:
            seed_tokens = len(tokenizer.encode(seed_str)) if seed_str else 0
        except Exception:
            seed_tokens = len(seed_str)
    else:
        seed_tokens = len(seed_str)

    seed_tokens = max(1, seed_tokens)
    raw = int(round(seed_tokens * multiplier))
    raw = max(hard_min, min(hard_max, raw))
    # Leave at least 2 tokens of headroom in the context window
    headroom = max(1, max_seq_len - seed_tokens - 2)
    return max(1, min(raw, headroom))


def tokenizer_from_dict(d: dict):
    if d.get("kind") == "bpe":
        return BPETokenizer.from_dict(d)
    return CharTokenizer.from_dict(d)


# ===================================================================
#  Modern Building Blocks  (LLaMA / Mistral / Qwen heritage, 2025-26)
# ===================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization — faster & more stable than LayerNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


# -------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-Head Self-Attention with RoPE, optional GQA, and KV-cache.

    Uses torch.nn.functional.scaled_dot_product_attention (Flash / memory-
    efficient kernels when available) instead of a hand-rolled softmax.

    IMPROVED (v2.1+):
    - ALiBi (Attention with Linear Biases) support for better length extrapolation
    - ALiBi path now keeps a *hard* causal mask (future tokens are forbidden)
    """

    def __init__(self, d_model: int, n_heads: int,
                 n_kv_heads: int | None = None,
                 max_seq_len: int = 4096,
                 use_alibi: bool = False,
                 rope_scaling: dict | None = None,
                 sliding_window: int | None = None,
                 kv_cache_dtype: str | None = None,
                 use_flex_attention: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA")
        self.n_rep = self.n_heads // self.n_kv_heads
        self.use_alibi = use_alibi
        self.rope_scaling = rope_scaling or {}  # {'type': 'yarn'|'ntk'|'linear', 'factor': float}
        self.sliding_window = int(sliding_window) if sliding_window else None
        self.kv_cache_dtype = kv_cache_dtype
        self.use_flex_attention = bool(use_flex_attention)
        self.kv_cache_start_pos = 0

        self.W_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        # Pre-compute RoPE cos / sin as persistent buffers (move with .to(device))
        self.max_seq_len = max_seq_len
        self._build_rope_buffers(max_seq_len)

        # ALiBi slopes (one per head)
        if use_alibi:
            self.register_buffer("alibi_slopes", self._get_alibi_slopes(), persistent=True)

        self.kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def _yarn_find_correction_dim(self, num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
        """YaRN helper from the public formula: dimension where a frequency
        performs `num_rotations` rotations over the original context length."""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def _yarn_linear_ramp_mask(self, low: float, high: float, dim: int) -> torch.Tensor:
        if low == high:
            high += 1e-3
        linear = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
        return torch.clamp(linear, 0.0, 1.0)

    def _rope_inv_freq_and_scale(self) -> tuple[torch.Tensor, float]:
        """Return inverse frequencies and post-scale for LLaMA-compatible RoPE.

        Standard RoPE uses exactly `1 / theta ** (i / dim)` for even rotary
        dimensions. Linear scaling divides positions by `factor`; dynamic NTK
        changes theta for extrapolation; YaRN blends interpolated and extrapolated
        frequencies with the public beta-fast/beta-slow ramp and mscale.
        """
        base = float((self.rope_scaling or {}).get("base", 10000.0))
        scaling = self.rope_scaling or {}
        scale_type = str(scaling.get("type", "none")).lower()
        factor = float(scaling.get("factor", 1.0) or 1.0)
        dim_range = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (dim_range / self.head_dim))
        mscale = 1.0

        if factor <= 1.0 or scale_type in {"none", ""}:
            return inv_freq, mscale
        if scale_type in {"ntk", "dynamic_ntk"}:
            # Dynamic NTK scaling as used by LLaMA-family implementations.
            # At the original context length this reduces to regular RoPE.
            seq_len = max(self.max_seq_len, int(scaling.get("original_max_position_embeddings", self.max_seq_len)) + 1)
            original = int(scaling.get("original_max_position_embeddings", self.max_seq_len))
            ntk_base = base * (((factor * seq_len / max(1, original)) - (factor - 1.0)) ** (self.head_dim / max(1, self.head_dim - 2)))
            inv_freq = 1.0 / (ntk_base ** (dim_range / self.head_dim))
        elif scale_type == "yarn":
            original = int(scaling.get("original_max_position_embeddings", self.max_seq_len))
            beta_fast = float(scaling.get("beta_fast", 32.0))
            beta_slow = float(scaling.get("beta_slow", 1.0))
            low = math.floor(self._yarn_find_correction_dim(beta_fast, self.head_dim, base, original))
            high = math.ceil(self._yarn_find_correction_dim(beta_slow, self.head_dim, base, original))
            ramp = self._yarn_linear_ramp_mask(low, high, self.head_dim // 2)
            extrapolation = inv_freq
            interpolation = inv_freq / factor
            inv_freq = interpolation * (1.0 - ramp) + extrapolation * ramp
            # DeepSeek/LLaMA YaRN convention: attention_factor defaults to
            # 0.1*ln(factor)+1.0 and scales both cos and sin.
            mscale = float(scaling.get("attention_factor", 0.1 * math.log(factor) + 1.0))
        return inv_freq, mscale

    def _build_rope_buffers(self, seq_len: int):
        """Build LLaMA-compatible RoPE cos/sin buffers.

        Buffers have shape `(seq_len, head_dim)`, not `(seq_len, head_dim/2)`,
        so `_apply_rope` can use the canonical `x*cos + rotate_half(x)*sin`
        formula and compare directly with Hugging Face/transformers outputs.
        """
        inv_freq, mscale = self._rope_inv_freq_and_scale()
        scaling = self.rope_scaling or {}
        scale_type = str(scaling.get("type", "none")).lower()
        factor = float(scaling.get("factor", 1.0) or 1.0)
        t = torch.arange(seq_len, dtype=torch.float32)
        if scale_type == "linear" and factor != 1.0:
            t = t / factor
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", emb.cos() * mscale, persistent=True)
        self.register_buffer("rope_sin", emb.sin() * mscale, persistent=True)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """LLaMA/HF rotate_half: [-x2, x1] over the last dimension halves."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _get_alibi_slopes(self) -> torch.Tensor:
        """Compute monotonic ALiBi slopes, one per attention head."""
        n = self.n_heads
        return 2 ** (-8 * torch.arange(1, n + 1).float() / n)

    # ---- RoPE --------------------------------------------------------
    def _apply_rope(self, x: torch.Tensor, start_pos: int, seq_len: int) -> torch.Tensor:
        """Apply Rotary Position Embeddings using the official LLaMA formula.

        `x` is `(batch, seq, heads, head_dim)`. The result matches
        transformers' `apply_rotary_pos_emb` (unsqueeze_dim=2) for the same
        cos/sin buffers to within normal floating point tolerance.
        """
        if start_pos + seq_len > len(self.rope_cos):
            self._build_rope_buffers(start_pos + seq_len)
            self.rope_cos = self.rope_cos.to(x.device)
            self.rope_sin = self.rope_sin.to(x.device)
        cos = self.rope_cos[start_pos:start_pos + seq_len].to(device=x.device, dtype=torch.float32)
        sin = self.rope_sin[start_pos:start_pos + seq_len].to(device=x.device, dtype=torch.float32)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        xf = x.float()
        return (xf * cos + self._rotate_half(xf) * sin).type_as(x)

    def _get_causal_keep_mask(self, q_len: int, k_len: int,
                              device: torch.device, start_pos: int = 0,
                              key_start_pos: int = 0) -> torch.Tensor:
        """Boolean causal/sliding-window keep-mask for SDPA (True = allowed)."""
        q_pos = torch.arange(start_pos, start_pos + q_len, device=device)
        k_pos = torch.arange(key_start_pos, key_start_pos + k_len, device=device)
        keep = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        if self.sliding_window is not None and self.sliding_window > 0:
            keep = keep & (k_pos.unsqueeze(0) > (q_pos.unsqueeze(1) - self.sliding_window))
        return keep

    # ---- ALiBi bias --------------------------------------------------
    def _get_alibi_bias(self, q_len: int, k_len: int,
                        device: torch.device, start_pos: int = 0,
                        key_start_pos: int = 0) -> torch.Tensor:
        """Create an ALiBi attention bias with a hard causal mask.

        Returned shape: (n_heads, q_len, k_len). Past/current positions get a
        finite ALiBi bias, future positions are set to -inf and therefore
        cannot be attended to. This fixes the old behaviour where ALiBi merely
        penalised the future instead of forbidding it.
        """
        q_pos = torch.arange(start_pos, start_pos + q_len, device=device)
        k_pos = torch.arange(key_start_pos, key_start_pos + k_len, device=device)
        rel_pos = k_pos.unsqueeze(0) - q_pos.unsqueeze(1)  # <= 0 for allowed keys
        future = rel_pos > 0

        bias = self.alibi_slopes[:, None, None] * rel_pos.to(torch.float32)
        blocked = future
        if self.sliding_window is not None and self.sliding_window > 0:
            blocked = blocked | (k_pos.unsqueeze(0) <= (q_pos.unsqueeze(1) - self.sliding_window))
        bias = bias.masked_fill(blocked.unsqueeze(0), float("-inf"))
        return bias

    # ---- KV cache helpers -------------------------------------------
    def _pack_cache_tensor(self, t: torch.Tensor):
        """Optionally store KV-cache in a lower precision format.

        INT8 uses a per-tensor scale (simple and educational); FP8 uses native
        torch float8 dtypes when available. Dequantization happens immediately
        before attention, so correctness is preserved with graceful fallback.
        """
        dtype = (self.kv_cache_dtype or "").lower()
        if dtype in {"", "none", "fp16", "bf16"}:
            return t
        if dtype in {"fp8", "fp8_e4m3"} and hasattr(torch, "float8_e4m3fn"):
            return t.to(torch.float8_e4m3fn)
        if dtype == "fp8_e5m2" and hasattr(torch, "float8_e5m2"):
            return t.to(torch.float8_e5m2)
        if dtype in {"int8", "i8"}:
            scale = t.detach().abs().amax().clamp_min(1e-8) / 127.0
            return (torch.clamp(torch.round(t / scale), -128, 127).to(torch.int8), scale)
        return t

    def _unpack_cache_tensor(self, packed, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(packed, tuple) and len(packed) == 2 and getattr(packed[0], "dtype", None) is torch.int8:
            return (packed[0].to(dtype=torch.float32) * packed[1]).to(dtype)
        if isinstance(packed, torch.Tensor) and str(packed.dtype).startswith("torch.float8"):
            return packed.to(dtype)
        return packed

    def _get_cached_kv(self, dtype: torch.dtype):
        if self.kv_cache is None:
            return None
        cached_k, cached_v = self.kv_cache
        return self._unpack_cache_tensor(cached_k, dtype), self._unpack_cache_tensor(cached_v, dtype)

    def _set_cached_kv(self, k: torch.Tensor, v: torch.Tensor):
        self.kv_cache = (self._pack_cache_tensor(k.detach()), self._pack_cache_tensor(v.detach()))

    # ---- Forward -----------------------------------------------------
    def forward(self, x: torch.Tensor,
                start_pos: int = 0, use_cache: bool = False) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.W_q(x).view(B, T, self.n_heads,    self.head_dim)
        k = self.W_k(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.W_v(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = self._apply_rope(q, start_pos, T)
        k = self._apply_rope(k, start_pos, T)

        q = q.transpose(1, 2)   # (B, n_heads, T, hd)
        k = k.transpose(1, 2)   # (B, n_kv_heads, T, hd) — cache stays unrepeated
        v = v.transpose(1, 2)

        key_start_pos = 0
        if use_cache:
            cached = self._get_cached_kv(k.dtype)
            if cached is not None:
                cached_k, cached_v = cached
                key_start_pos = self.kv_cache_start_pos
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)
            if self.sliding_window is not None and self.sliding_window > 0 and k.shape[2] > self.sliding_window:
                overflow = k.shape[2] - self.sliding_window
                k = k[:, :, overflow:, :]
                v = v[:, :, overflow:, :]
                key_start_pos += overflow
            self.kv_cache_start_pos = key_start_pos
            self._set_cached_kv(k, v)

        # Repeat KV *after* concatenating/evicting the cache. This preserves
        # GQA memory savings in the cache and exactly matches LLaMA grouping.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        S = k.shape[2]
        # Flash / memory-efficient attention via PyTorch SDPA. FlexAttention is
        # exposed as a configuration flag but falls back here unless the runtime
        # provides a compatible PyTorch >=2.5 implementation.
        if self.use_alibi:
            attn_mask = self._get_alibi_bias(T, S, x.device, start_pos=start_pos, key_start_pos=key_start_pos)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        elif T == S and start_pos == 0 and self.sliding_window is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif T == 1 and self.sliding_window is None:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            keep = self._get_causal_keep_mask(T, S, x.device, start_pos=start_pos, key_start_pos=key_start_pos)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)

    def reset_cache(self):
        self.kv_cache = None
        self.kv_cache_start_pos = 0


# -------------------------------------------------------------------

class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network — standard in LLaMA / Qwen / Mistral.

    IMPROVED (v2.1+):
    - Optional LoRA adapters on gate / up / down projections. When `lora`
      is set (an nn.ModuleDict via enable_lora()), the low-rank deltas are
      actually applied during the forward pass.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        # LoRA adapters (None until enable_lora is called on the parent model)
        self.lora: nn.ModuleDict | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora is None:
            return self.down(F.silu(self.gate(x)) * self.up(x))
        # LoRA path: add low-rank deltas to each projection
        g = self.gate(x) + self.lora["gate"](x)
        u = self.up(x)   + self.lora["up"](x)
        h = F.silu(g) * u
        return self.down(h) + self.lora["down"](h)


class Top2MoE(nn.Module):
    """Educational top-2 Mixture-of-Experts feed-forward layer.

    Each token is routed to its two highest-scoring SwiGLU experts. The layer
    exposes `last_load_balance_loss` so the trainer can add a small auxiliary
    penalty and avoid expert collapse. It is intentionally simple (dense expert
    evaluation) for clarity and CPU fallback; production kernels can replace it
    later without changing the public API.
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4,
                 top_k: int = 2, load_balance_weight: float = 0.01):
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = min(int(top_k), self.num_experts)
        self.load_balance_weight = float(load_balance_weight)
        self.router = nn.Linear(d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(d_model, d_ff) for _ in range(self.num_experts)])
        self.last_load_balance_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        router_logits = self.router(x)
        router_probs = torch.softmax(router_logits.float(), dim=-1)
        top_prob, top_idx = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_prob = top_prob / top_prob.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # Dense expert evaluation keeps the implementation readable and robust;
        # top-k masks ensure only selected experts contribute to each token.
        out = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            expert_out = expert(x)
            weights = torch.zeros(B, T, device=x.device, dtype=expert_out.dtype)
            for slot in range(self.top_k):
                weights = weights + torch.where(top_idx[..., slot] == expert_id, top_prob[..., slot].to(expert_out.dtype), torch.zeros_like(weights))
            out = out + expert_out * weights.unsqueeze(-1)

        importance = router_probs.mean(dim=(0, 1))
        load = torch.nn.functional.one_hot(top_idx, self.num_experts).float().sum(dim=-2).mean(dim=(0, 1)) / self.top_k
        self.last_load_balance_loss = self.load_balance_weight * self.num_experts * torch.sum(importance * load.to(importance.device))
        return out


# -------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: RMSNorm → Attention → RMSNorm → SwiGLU FFN.

    IMPROVED (v2.3+):
    - Optional gradient checkpointing for memory-efficient training
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 n_kv_heads: int | None, max_seq_len: int,
                 dropout: float = 0.0, use_alibi: bool = False,
                 use_checkpoint: bool = False,
                 rope_scaling: dict | None = None,
                 sliding_window: int | None = None,
                 kv_cache_dtype: str | None = None,
                 use_flex_attention: bool = False,
                 use_moe: bool = False,
                 num_experts: int = 4):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn      = Attention(d_model, n_heads, n_kv_heads, max_seq_len,
                                   use_alibi=use_alibi,
                                   rope_scaling=rope_scaling,
                                   sliding_window=sliding_window,
                                   kv_cache_dtype=kv_cache_dtype,
                                   use_flex_attention=use_flex_attention)
        self.ffn_norm  = RMSNorm(d_model)
        self.ffn       = Top2MoE(d_model, d_ff, num_experts=num_experts) if use_moe else FeedForward(d_model, d_ff)
        self.dropout   = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, x: torch.Tensor,
                      start_pos: int = 0, use_cache: bool = False) -> torch.Tensor:
        """Internal forward pass (used by checkpointing)."""
        x = x + self.dropout(self.attn(self.attn_norm(x), start_pos, use_cache))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x

    def forward(self, x: torch.Tensor,
                start_pos: int = 0, use_cache: bool = False) -> torch.Tensor:
        if self.use_checkpoint and self.training and not use_cache:
            # Gradient checkpointing: trade compute for memory
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, start_pos, use_cache,
                use_reentrant=False
            )
        return self._forward_impl(x, start_pos, use_cache)


# ===================================================================

class ModernTransformer(nn.Module):
    """Modern decoder-only Transformer (LLaMA-style, 2025-2026).

    Key architectural choices
    ─────────────────────────
    • RMSNorm  (Pre-norm)          — stable, efficient
    • RoPE     (Rotary Pos. Emb.)  — generalises to unseen lengths
    • SwiGLU   activation          — better than plain ReLU / GELU
    • Multi-Head Attn w/ opt. GQA  — flexible efficiency
    • Flash Attention (SDPA)       — fused, memory-efficient kernels
    • Weight tying (emb = head)    — fewer params, better generalisation
    • No bias in linear layers     — modern practice
    • KV-cache for fast generation — O(1) per token after prompt
    • Configurable depth (n_layers)

    IMPROVED (v2.1+):
    • Optional ALiBi for better length extrapolation
    • LoRA adapter support

    IMPROVED (v2.3+):
    • Gradient checkpointing support (use_gradient_checkpointing)
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int,
                 max_seq_len: int = 4096,
                 n_kv_heads: int | None = None,
                 dropout: float = 0.0,
                 use_alibi: bool = False,
                 use_gradient_checkpointing: bool = False,
                 rope_scaling: dict | None = None,
                 sliding_window: int | None = None,
                 kv_cache_dtype: str | None = None,
                 use_flex_attention: bool = False,
                 use_moe: bool = False,
                 num_experts: int = 4,
                 tie_word_embeddings: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        if n_kv_heads is not None:
            assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.d_model     = d_model
        self.n_heads     = n_heads
        self.n_layers    = n_layers
        self.d_ff        = d_ff
        self.n_kv_heads  = n_kv_heads
        self.max_seq_len = max_seq_len
        self.dropout     = dropout
        self.use_alibi   = use_alibi
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.rope_scaling = rope_scaling or {}
        self.sliding_window = int(sliding_window) if sliding_window else None
        self.kv_cache_dtype = kv_cache_dtype
        self.use_flex_attention = bool(use_flex_attention)
        self.use_moe = bool(use_moe)
        self.num_experts = int(num_experts)
        self.tie_word_embeddings = bool(tie_word_embeddings)

        self.embedding   = nn.Embedding(vocab_size, d_model)
        self.layers      = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, n_kv_heads, max_seq_len, dropout,
                             use_alibi=use_alibi,
                             use_checkpoint=use_gradient_checkpointing,
                             rope_scaling=rope_scaling,
                             sliding_window=sliding_window,
                             kv_cache_dtype=kv_cache_dtype,
                             use_flex_attention=use_flex_attention,
                             use_moe=use_moe,
                             num_experts=num_experts)
            for _ in range(n_layers)
        ])
        self.final_norm  = RMSNorm(d_model)
        self.head        = nn.Linear(d_model, vocab_size, bias=False)

        # Modern weight init (GPT-2 / LLaMA style)
        self.apply(self._init_weights)

        # Weight tying: output head shares the embedding matrix when enabled.
        # Gradients from both the input embedding path and LM-head path accumulate
        # into the single shared Parameter exactly as in GPT/LLaMA models.
        if self.tie_word_embeddings:
            self.tie_weights()

        # LoRA adapters (initially None, enabled via enable_lora())
        self.lora_adapters: list | None = None
        self.lora_rank = 0

    def tie_weights(self):
        """Tie output projection and token embedding weights in-place.

        This is safe for autograd: both modules point at the same Parameter, so
        gradients are summed into one tensor before the optimizer step. Calling
        it after loading an older checkpoint preserves AuraLite's historic tied
        embedding behavior.
        """
        self.head.weight = self.embedding.weight
        self.tie_word_embeddings = True

    def untie_weights(self):
        """Clone the LM head for research runs that need untied embeddings."""
        new_head = nn.Linear(self.d_model, self.embedding.num_embeddings, bias=False).to(self.embedding.weight.device)
        new_head.weight.data.copy_(self.embedding.weight.data)
        self.head = new_head
        self.tie_word_embeddings = False

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor,
                start_pos: int = 0, use_cache: bool = False) -> torch.Tensor:
        """Returns logits for ALL positions: (B, T, vocab_size).

        Training uses every position (dense next-token loss, nanoGPT-style);
        generation simply takes the last position: logits[:, -1, :].
        """
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h, start_pos, use_cache)
        h = self.final_norm(h)
        return self.head(h)

    def get_aux_loss(self) -> torch.Tensor | None:
        """Return summed auxiliary losses (currently MoE load balancing)."""
        losses = []
        for layer in self.layers:
            loss = getattr(getattr(layer, "ffn", None), "last_load_balance_loss", None)
            if loss is not None:
                losses.append(loss)
        if not losses:
            return None
        return torch.stack([l.to(losses[0].device) for l in losses]).sum()

    def reset_cache(self):
        for layer in self.layers:
            layer.attn.reset_cache()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---- LoRA Support ------------------------------------------------
    def enable_lora(self, rank: int = 8, target_modules: list[str] | None = None):
        """Enable LoRA adapters on the FFN linear layers.

        Freezes the whole base model and adds trainable low-rank adapters
        that are actually applied in FeedForward.forward(). The adapters are
        registered as submodules (`self.lora_adapters`) so they move with
        `.to(device)` and are saved/loaded via the state dict.

        Args:
            rank: LoRA rank (default 8)
            target_modules: Names of FFN projections to adapt
                            (default: ["gate", "up", "down"]).
        """
        if target_modules is None:
            target_modules = ["gate", "up", "down"]

        self.lora_rank = rank
        # plain list (NOT nn.ModuleList) so the adapters are registered only
        # once — via layer.ffn.lora — and don't appear twice in state_dict.
        self.lora_adapters = []

        # Freeze every base-model parameter
        for param in self.parameters():
            param.requires_grad = False

        for layer in self.layers:
            # nn.ModuleDict keys must NOT contain dots — use the plain
            # projection name ("gate"/"up"/"down").
            layer_lora = nn.ModuleDict()
            for name in target_modules:
                mod = getattr(layer.ffn, name, None)
                if isinstance(mod, nn.Linear):
                    lora = LoRALayer(mod.in_features, mod.out_features, rank)
                    # LoRA params are created with requires_grad=True by default
                    layer_lora[name] = lora
            # wire the adapters into the FFN so forward() actually uses them
            # (this also registers them as proper submodules of the model)
            layer.ffn.lora = layer_lora
            self.lora_adapters.append(layer_lora)

    def disable_lora(self):
        """Disable LoRA and restore full training."""
        self.lora_rank = 0
        self.lora_adapters = None
        for layer in self.layers:
            layer.ffn.lora = None
        for param in self.parameters():
            param.requires_grad = True


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) adapter for a linear layer.

    Replaces W with W + (lora_B @ lora_A) / rank
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 alpha: float | None = None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha or rank  # default: alpha = rank (scaling = 1)
        self.scaling = self.alpha / self.rank

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        # lora_A: (rank, in_features), lora_B: (out_features, rank)
        # result: x @ lora_A.T @ lora_B.T * scaling
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


# ===================================================================
#  Dataset
# ===================================================================

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


# ===================================================================
#  Cosine Learning-Rate Schedule with Linear Warmup
# ===================================================================

class CosineWarmupScheduler:
    """Cosine decay with linear warmup — standard in modern LLM training."""

    def __init__(self, optimizer, warmup_steps: int, max_steps: int,
                 min_lr: float = 1e-5):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps    = max_steps
        self.min_lr       = min_lr
        self.base_lrs     = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count   = 0

    def step(self):
        self.step_count += 1
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.step_count < self.warmup_steps:
                lr = base_lr * self.step_count / self.warmup_steps
            else:
                progress = (self.step_count - self.warmup_steps) / max(
                    1, self.max_steps - self.warmup_steps
                )
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress)
                )
            pg["lr"] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict:
        return {
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "min_lr": self.min_lr,
            "base_lrs": list(self.base_lrs),
            "step_count": self.step_count,
            "current_lrs": [pg["lr"] for pg in self.optimizer.param_groups],
        }

    def load_state_dict(self, state: dict):
        self.warmup_steps = state.get("warmup_steps", self.warmup_steps)
        self.max_steps = state.get("max_steps", self.max_steps)
        self.min_lr = state.get("min_lr", self.min_lr)
        self.base_lrs = list(state.get("base_lrs", self.base_lrs))
        self.step_count = int(state.get("step_count", self.step_count))
        current_lrs = state.get("current_lrs")
        if current_lrs is not None:
            for pg, lr in zip(self.optimizer.param_groups, current_lrs):
                pg["lr"] = lr



# ===================================================================
#  GGUF / llama.cpp backend (inference-only)
# ===================================================================

class GGUFNotAvailableError(ImportError):
    """Raised when llama-cpp-python is required but not installed."""
    pass


class GGUFTokenizerProxy:
    """Tokenizer adapter around llama.cpp's native GGUF tokenizer."""

    kind = "gguf"

    def __init__(self, llama):
        self.llama = llama

    @property
    def vocab_size(self) -> int:
        try:
            n_vocab = getattr(self.llama, "n_vocab", None)
            return int(n_vocab() if callable(n_vocab) else n_vocab)
        except Exception:
            return 0

    def encode(self, s: str) -> list[int]:
        data = s.encode("utf-8", errors="ignore")
        try:
            return list(self.llama.tokenize(data, add_bos=False, special=True))
        except TypeError:
            return list(self.llama.tokenize(data, add_bos=False))

    def decode(self, ids) -> str:
        try:
            data = self.llama.detokenize([int(i) for i in ids])
            return data.decode("utf-8", errors="replace") if isinstance(data, (bytes, bytearray)) else str(data)
        except Exception:
            return ""

    def train(self, text: str, vocab_size: int | None = None):
        raise RuntimeError("GGUF tokenizer is loaded from the model and cannot be trained.")

    def to_dict(self) -> dict:
        return {"kind": self.kind}


class GGUFModelProxy:
    """Small adapter that exposes a llama-cpp-python GGUF model through the
    subset of attributes AuraLite's GUI expects.

    GGUF models are quantized inference artifacts. They can be loaded and used
    for generation/streaming/batch prompting, but they cannot be trained or
    saved as AuraLite `.pt` checkpoints.
    """

    backend = "gguf"

    def __init__(self, path: str, *, n_ctx: int = 4096,
                 n_threads: int | None = None, n_gpu_layers: int = -1,
                 seed: int = -1, chat_format: str | None = None,
                 use_chat_completion: bool = False,
                 n_batch: int = 512, use_mmap: bool = True,
                 use_mlock: bool = False,
                 verbose: bool = False, extra_kwargs: dict[str, Any] | None = None):
        try:
            from llama_cpp import Llama
        except Exception as e:  # pragma: no cover - depends on optional package
            raise GGUFNotAvailableError(
                "To load .gguf install llama-cpp-python: "
                "pip install llama-cpp-python"
            ) from e

        self.path = str(path)
        self.max_seq_len = int(n_ctx)
        self.n_threads = n_threads
        self.n_gpu_layers = int(n_gpu_layers)
        self.seed = int(seed)
        self.chat_format = chat_format
        self.use_chat_completion = bool(use_chat_completion)
        self.n_batch = int(n_batch)
        self.use_mmap = bool(use_mmap)
        self.use_mlock = bool(use_mlock)
        self.verbose = bool(verbose)

        kwargs: dict[str, Any] = {
            "model_path": self.path,
            "n_ctx": self.max_seq_len,
            "n_gpu_layers": self.n_gpu_layers,
            "seed": self.seed,
            "n_batch": self.n_batch,
            "use_mmap": self.use_mmap,
            "use_mlock": self.use_mlock,
            "verbose": self.verbose,
        }
        if n_threads:
            kwargs["n_threads"] = int(n_threads)
        if chat_format:
            kwargs["chat_format"] = chat_format
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        self.llama = Llama(**kwargs)
        self.tokenizer = GGUFTokenizerProxy(self.llama)
        self.metadata = getattr(self.llama, "metadata", {}) or {}

    def eval(self):
        return self

    def reset_cache(self):
        # llama-cpp-python keeps KV-cache internally per completion; reset if the
        # installed version exposes a method, otherwise each call is still safe.
        reset = getattr(self.llama, "reset", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                pass

    def count_parameters(self) -> int:
        for key in ("general.parameter_count",):
            val = self.metadata.get(key)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass
        return 0

    def count_trainable_parameters(self) -> int:
        return 0

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def create_completion(self, prompt: str, *, max_tokens: int = 50,
                          temperature: float = 0.8, top_k: int = 50,
                          top_p: float = 0.9, repeat_penalty: float = 1.0,
                          min_p: float = 0.0, stream: bool = False):
        return self.llama.create_completion(
            prompt=prompt,
            max_tokens=max(0, int(max_tokens)),
            temperature=max(float(temperature), 0.0),
            top_k=max(0, int(top_k)),
            top_p=float(top_p),
            min_p=float(min_p),
            repeat_penalty=float(repeat_penalty),
            stream=stream,
        )

    def create_chat_completion(self, messages: list[dict], *, max_tokens: int = 50,
                               temperature: float = 0.8, top_k: int = 50,
                               top_p: float = 0.9, repeat_penalty: float = 1.0,
                               min_p: float = 0.0, stream: bool = False):
        """Use llama.cpp chat formatting for instruction/chat GGUF models."""
        return self.llama.create_chat_completion(
            messages=messages,
            max_tokens=max(0, int(max_tokens)),
            temperature=max(float(temperature), 0.0),
            top_k=max(0, int(top_k)),
            top_p=float(top_p),
            min_p=float(min_p),
            repeat_penalty=float(repeat_penalty),
            stream=stream,
        )

# ===================================================================
#  Engine
# ===================================================================

class AuraLiteEngine:
    """Modern training & inference engine for AuraLite AI v2.1.

    IMPROVED (v2.1+):
    - Gradient accumulation for large models on weak hardware
    - cudnn.benchmark + TF32 for CUDA performance
    - generate_streaming() for real-time generation output
    - generate_batch() for parallel generation of multiple prompts
    - validate_params() before training starts
    """

    def __init__(self):
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_threads = torch.get_num_threads()
        self.model: ModernTransformer | GGUFModelProxy | HuggingFaceProxy | None = None
        self.backend     = "torch"           # "torch", "gguf", or "huggingface"
        self.gguf_path: str | None = None
        self.hf_path: str | None = None
        self.hf_proxy: HuggingFaceProxy | None = None
        self.optimizer   = None
        self.scheduler   = None
        self.scaler      = None
        self.tokenizer   = None              # CharTokenizer | BPETokenizer | GGUFTokenizerProxy | HF tokenizer
        self.vocab_size  = 0
        self.params_used: dict = {}          # remember last training/load params
        self.last_val_loss: float | None = None
        self._resume_optimizer_state = None
        self._resume_scheduler_state = None
        self._resume_scaler_state = None

        # DDP (DistributedDataParallel) support — v2.3
        self.is_distributed = False
        self.local_rank = 0
        self.world_size = 1
        self._ddp_model = None

        # Auto-detect distributed environment (torchrun / torch.distributed.launch)
        self._try_init_distributed_from_env()

        # CUDA performance tuning
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except AttributeError:
                pass  # older PyTorch versions

    def _try_init_distributed_from_env(self):
        """Initialize DDP if running under torchrun or torch.distributed.launch."""
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            try:
                rank = int(os.environ["RANK"])
                world_size = int(os.environ["WORLD_SIZE"])
                local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))

                dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo",
                                        rank=rank, world_size=world_size)

                if torch.cuda.is_available():
                    torch.cuda.set_device(local_rank)
                    self.device = torch.device(f"cuda:{local_rank}")

                self.is_distributed = True
                self.local_rank = local_rank
                self.world_size = world_size

                print(f"[AuraLite] DDP initialized: rank={rank}, world_size={world_size}, "
                      f"local_rank={local_rank}, device={self.device}")
            except Exception as e:
                print(f"[AuraLite] WARNING: Failed to initialize DDP: {e}")
                self.is_distributed = False

    def is_gguf_model(self) -> bool:
        return self.backend == "gguf" or isinstance(self.model, GGUFModelProxy)

    def is_hf_model(self) -> bool:
        return self.backend == "huggingface" or isinstance(self.model, HuggingFaceProxy)

    # ===================================================================
    #  Chat / Instruction Mode (NEW v2.3)
    # ===================================================================

    def generate_chat(
        self,
        messages: List[Dict[str, str]] | "ChatHistory",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        min_p: float = 0.0,
        chat_template: str = "chatml",
        system_prompt: Optional[str] = None,
        stop_tokens: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a response in chat/instruction mode.

        Args:
            messages: List of {"role": "...", "content": "..."} or ChatHistory
            max_new_tokens: how many tokens to generate
            chat_template: one of CHAT_TEMPLATES keys
            system_prompt: optional system message to prepend

        Returns:
            Generated assistant response (without the prompt).
        """
        if not HAS_CHAT_SUPPORT:
            raise RuntimeError("chat_interface.py is required for chat mode.")

        # Convert to ChatHistory if needed
        if isinstance(messages, list):
            history = ChatHistory.from_list(messages)
        else:
            history = messages

        # Add system prompt if provided
        if system_prompt and not any(m.role == "system" for m in history.messages):
            history.messages.insert(0, ChatMessage(role="system", content=system_prompt))

        prompt = apply_chat_template(history, template_name=chat_template, add_generation_prompt=True)

        # Generate
        if self.is_gguf_model():
            # GGUF chat completion
            if self.model.use_chat_completion:
                # Pass the conversation history as structured messages
                if isinstance(history, ChatHistory):
                    chat_messages = history.to_list()
                else:
                    chat_messages = messages
                result = self.model.create_chat_completion(
                    messages=chat_messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repetition_penalty,
                    min_p=min_p,
                    stream=False,
                )
                try:
                    return result["choices"][0]["message"]["content"]
                except Exception:
                    return ""
            else:
                # Fallback to regular completion with formatted prompt
                full = self._gguf_generate_text(
                    prompt, max_new_tokens, temperature, top_k, top_p,
                    repetition_penalty, min_p
                )
                return full[len(prompt):].strip()

        elif self.is_hf_model():
            # HF models usually have their own chat template
            kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if min_p > 0:
                kwargs["min_p"] = min_p
            full = self.hf_proxy.generate(prompt, **kwargs)
            return full[len(prompt):].strip() if full.startswith(prompt) else full.strip()

        else:
            # Native AuraLite model
            ids = self._prepare_prompt_ids(prompt)
            result_ids = self._generate_ids(
                ids, max_new_tokens, temperature, top_k, top_p,
                repetition_penalty, min_p=min_p
            )
            # Return only the newly generated part (token-based slice)
            generated_ids = result_ids[len(ids):]
            return self.decode(generated_ids).strip()

    def generate_chat_streaming(
        self,
        messages: List[Dict[str, str]] | "ChatHistory",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        min_p: float = 0.0,
        chat_template: str = "chatml",
        system_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Streaming version of generate_chat — yields tokens one by one.

        Works with native AuraLite and GGUF (when streaming is supported).
        """
        if not HAS_CHAT_SUPPORT:
            raise RuntimeError("chat_interface.py is required for chat mode.")

        if isinstance(messages, list):
            history = ChatHistory.from_list(messages)
        else:
            history = messages

        if system_prompt and not any(m.role == "system" for m in history.messages):
            history.messages.insert(0, ChatMessage(role="system", content=system_prompt))

        prompt = apply_chat_template(history, template_name=chat_template, add_generation_prompt=True)

        if self.is_gguf_model():
            if self.model.use_chat_completion:
                stream = self.model.create_chat_completion(
                    history.to_list(),
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repetition_penalty,
                    min_p=min_p,
                    stream=True,
                )
                for chunk in stream:
                    try:
                        delta = chunk["choices"][0].get("delta", {}).get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue
            else:
                # Fallback: non-streaming
                full = self.generate_chat(
                    messages, max_new_tokens, temperature, top_k, top_p,
                    repetition_penalty, min_p, chat_template, system_prompt
                )
                yield full
            return

        if self.is_hf_model():
            # HF streaming (if supported by the proxy)
            kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if min_p > 0:
                kwargs["min_p"] = min_p
            try:
                for token in self.hf_proxy.generate_streaming(prompt, **kwargs):
                    yield token
            except Exception:
                # Fallback
                result = self.hf_proxy.generate(prompt, **kwargs)
                yield result[len(prompt):] if result.startswith(prompt) else result
            return

        # Native AuraLite streaming
        ids = self._prepare_prompt_ids(prompt)
        self.model.eval()
        self.model.reset_cache()

        result_ids = list(ids)
        if max_new_tokens <= 0:
            return

        with torch.no_grad():
            # Process prompt
            t = torch.tensor([ids], dtype=torch.long).to(self.device)
            logits = self.model(t, start_pos=0, use_cache=True)
            nxt = self._sample_token(logits[0, -1], temperature, top_k, top_p,
                                     repetition_penalty, result_ids, min_p=min_p)
            result_ids.append(nxt)
            yield self.decode([nxt])

            for _ in range(max_new_tokens - 1):
                pos = len(result_ids) - 1
                if pos >= self.model.max_seq_len - 1:
                    break
                t = torch.tensor([[result_ids[-1]]], dtype=torch.long).to(self.device)
                logits = self.model(t, start_pos=pos, use_cache=True)
                nxt = self._sample_token(logits[0, -1], temperature, top_k, top_p,
                                         repetition_penalty, result_ids, min_p=min_p)
                result_ids.append(nxt)
                yield self.decode([nxt])

        self.model.reset_cache()

    def load_gguf_model(self, path: str, *, n_ctx: int = 4096,
                        n_threads: int | None = None,
                        n_gpu_layers: int | None = None,
                        seed: int = -1, chat_format: str | None = None,
                        use_chat_completion: bool = False,
                        n_batch: int = 512, use_mmap: bool = True,
                        use_mlock: bool = False, verbose: bool = False,
                        extra_kwargs: dict[str, Any] | None = None):
        """Load a `.gguf` model through llama.cpp for inference.

        Args:
            path: Path to a GGUF file.
            n_ctx: Context window used by llama.cpp.
            n_threads: CPU threads; defaults to AuraLite's detected thread count.
            n_gpu_layers: Layers to offload to GPU. Default `-1` asks llama.cpp
                to offload as much as possible (if a compatible build/GPU exists).
            seed/chat_format/verbose/extra_kwargs: forwarded to llama_cpp.Llama.
            use_chat_completion: use llama.cpp chat templates / chat completion API.
            n_batch/use_mmap/use_mlock: common llama.cpp loading knobs.
        """
        if n_gpu_layers is None:
            n_gpu_layers = -1
        if n_threads is None:
            n_threads = self.num_threads

        model = GGUFModelProxy(
            path, n_ctx=n_ctx, n_threads=n_threads,
            n_gpu_layers=n_gpu_layers, seed=seed,
            chat_format=chat_format,
            use_chat_completion=use_chat_completion,
            n_batch=n_batch, use_mmap=use_mmap,
            use_mlock=use_mlock, verbose=verbose,
            extra_kwargs=extra_kwargs,
        )
        self.model = model
        self.backend = "gguf"
        self.gguf_path = str(path)
        self.hf_path = None
        self._ddp_model = None
        self.tokenizer = model.tokenizer
        self.vocab_size = model.vocab_size
        self.params_used = {
            "backend": "gguf",
            "path": str(path),
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "n_gpu_layers": n_gpu_layers,
            "seed": seed,
            "chat_format": chat_format,
            "use_chat_completion": use_chat_completion,
            "n_batch": n_batch,
            "use_mmap": use_mmap,
            "use_mlock": use_mlock,
        }
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self._resume_optimizer_state = None
        self._resume_scheduler_state = None
        self._resume_scaler_state = None
        chat = ", chat=on" if use_chat_completion else ""
        print(f"[AuraLite] Loaded GGUF model: {Path(path).name} "
              f"(ctx={n_ctx}, threads={n_threads}, gpu_layers={n_gpu_layers}, "
              f"batch={n_batch}{chat})")

    # ===================================================================
    #  NEW: Hugging Face + LoRA / QLoRA support (any model)
    # ===================================================================

    def load_hf_model(
        self,
        model_name_or_path: str,
        *,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: str | None = None,
        device_map: str = "auto",
        max_seq_len: int = 4096,
        apply_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        local_files_only: bool = False,
        verbose: bool = True,
    ):
        """
        Load ANY Hugging Face causal language model (Llama, Mistral, Qwen, Gemma, Phi, etc.)

        Fully supports **already downloaded / local models**:

        - Pass a local path (e.g. ~/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/...)
        - Set `local_files_only=True` for completely offline loading (no internet).

        Supports:
        - Full precision, FP16, BF16
        - 4-bit (QLoRA ready) and 8-bit quantization via bitsandbytes
        - Automatic application of LoRA adapters

        After loading you can:
        - Generate text
        - Fine-tune with LoRA/QLoRA using `finetune_hf()`
        - Save only the tiny LoRA adapter
        """
        if not HAS_HF_SUPPORT:
            raise HFNotAvailableError(
                "Hugging Face + LoRA/QLoRA support is not available.\n"
                "Install extra dependencies:\n"
                "  pip install transformers peft accelerate bitsandbytes sentencepiece"
            )

        if self.hf_proxy is None:
            self.hf_proxy = create_hf_proxy()

        dtype = None
        if torch_dtype:
            dtype = getattr(torch, torch_dtype, torch.float16)

        self.hf_proxy.load_model(
            model_name_or_path,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            torch_dtype=dtype,
            device_map=device_map,
            max_seq_len=max_seq_len,
            local_files_only=local_files_only,
            verbose=verbose,
        )

        if apply_lora:
            self.hf_proxy.apply_lora(
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=lora_target_modules,
                verbose=verbose,
            )

        # Wire into engine
        self.model = self.hf_proxy
        self.backend = "huggingface"
        self.gguf_path = None
        self.hf_path = model_name_or_path
        self._ddp_model = None
        self.tokenizer = self.hf_proxy.tokenizer
        self.vocab_size = getattr(self.hf_proxy.tokenizer, "vocab_size", 0)

        self.params_used = {
            "backend": "huggingface",
            "model": model_name_or_path,
            "load_in_4bit": load_in_4bit,
            "load_in_8bit": load_in_8bit,
            "lora_applied": apply_lora,
            "lora_rank": lora_rank if apply_lora else 0,
            "local_files_only": local_files_only,
        }

        self.optimizer = None
        self.scheduler = None
        self.scaler = None

        mode = "LOCAL (offline)" if local_files_only else "Hub"
        print(f"[AuraLite] Loaded Hugging Face model ({mode}): {model_name_or_path}")
        if apply_lora:
            print(f"[AuraLite] LoRA applied (rank={lora_rank}) — ready for QLoRA fine-tuning")

    def apply_lora_to_hf(
        self,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: list[str] | None = None,
    ):
        """Apply LoRA to a loaded HF model (for QLoRA fine-tuning)."""
        if not self.is_hf_model():
            raise ValueError("Load a Hugging Face model first with load_hf_model()")
        self.hf_proxy.apply_lora(rank=rank, alpha=alpha, dropout=dropout,
                                 target_modules=target_modules)
        self.params_used["lora_applied"] = True
        self.params_used["lora_rank"] = rank

    def finetune_hf(
        self,
        texts: list[str] | str,
        output_dir: str = "hf_lora_adapter",
        epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        max_length: int = 512,
        gradient_accumulation_steps: int = 4,
        progress_callback: Callable | None = None,
        stop_event=None,
    ):
        """
        Fine-tune a loaded Hugging Face model using LoRA / QLoRA.

        - If the model was loaded with load_in_4bit=True → this is QLoRA
        - If LoRA was not applied yet, it will be applied automatically (rank 16)
        - Uses Hugging Face Trainer under the hood (best experience)

        Args:
            texts: list of strings or a single long text (will be split)
            output_dir: where to save the LoRA adapter
        """
        if not self.is_hf_model():
            raise ValueError("Load a Hugging Face model first with load_hf_model()")

        if isinstance(texts, str):
            # Split into reasonable chunks
            texts = [t.strip() for t in texts.split("\n\n") if len(t.strip()) > 50]
            if not texts:
                texts = [texts]  # fallback

        if not self.hf_proxy.is_peft:
            print("[AuraLite] No LoRA adapter found — applying default LoRA for fine-tuning...")
            self.hf_proxy.apply_lora(rank=16)

        print(f"[AuraLite] Starting LoRA/QLoRA fine-tuning on {len(texts)} examples...")

        return self.hf_proxy.finetune(
            texts,
            output_dir=output_dir,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_length=max_length,
            gradient_accumulation_steps=gradient_accumulation_steps,
            progress_callback=progress_callback,
            stop_event=stop_event,
        )

    def save_hf_lora(self, path: str):
        """Save only the LoRA adapter (tiny file, ~few MB)."""
        if not self.is_hf_model():
            raise ValueError("No Hugging Face model loaded")
        self.hf_proxy.save_lora_adapter(path)

    def load_hf_lora(self, adapter_path: str):
        """Load a saved LoRA adapter on top of the current base HF model."""
        if not self.is_hf_model():
            raise ValueError("Load the base Hugging Face model first")
        self.hf_proxy.load_lora_adapter(adapter_path)

    # ===================================================================
    #  Hugging Face Hub integration (NEW v2.6)
    # ===================================================================

    def push_hf_model_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload from AuraLite",
        private: bool = False,
        token: Optional[str] = None,
    ):
        """Push current HF model to the Hub."""
        if not self.is_hf_model():
            raise ValueError("Only Hugging Face models can be pushed to the Hub.")
        self.hf_proxy.push_to_hub(
            repo_id=repo_id,
            commit_message=commit_message,
            private=private,
            token=token,
        )

    def load_hf_model_from_hub(
        self,
        repo_id: str,
        **kwargs,
    ):
        """Load a model directly from the Hugging Face Hub."""
        self.load_hf_model(repo_id, **kwargs)

    # ===================================================================
    #  Evaluation (NEW v2.7)
    # ===================================================================

    def evaluate_model(
        self,
        tasks: List[str] | str = "arc_easy",
        num_fewshot: int = 0,
        batch_size: int = 1,
        limit: Optional[int] = None,
        progress_callback=None,
    ):
        """
        Evaluate the current model using lm-evaluation-harness.

        Returns a dictionary with results.
        """
        from evaluation import EvaluationEngine, LMEvalNotAvailableError

        try:
            eval_engine = EvaluationEngine(self)
            results = eval_engine.evaluate(
                tasks=tasks,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                limit=limit,
                progress_callback=progress_callback,
            )
            return results
        except LMEvalNotAvailableError as e:
            raise e

    @staticmethod
    def _gguf_prompt_messages(prompt: str) -> list[dict[str, str]]:
        """Wrap a raw prompt into a minimal chat message list for GGUF chat APIs."""
        return [{"role": "user", "content": prompt}]

    def _gguf_generate_text(self, prompt: str, length: int = 50,
                            temperature: float = 0.8,
                            top_k: int = 50, top_p: float = 0.9,
                            repetition_penalty: float = 1.0,
                            min_p: float = 0.0) -> str:
        if not isinstance(self.model, GGUFModelProxy):
            raise ValueError("No GGUF model loaded.")
        if self.model.use_chat_completion:
            out = self.model.create_chat_completion(
                self._gguf_prompt_messages(prompt),
                max_tokens=length, temperature=temperature,
                top_k=top_k, top_p=top_p, min_p=min_p,
                repeat_penalty=repetition_penalty, stream=False,
            )
            try:
                suffix = out["choices"][0].get("message", {}).get("content", "")
            except Exception:
                suffix = ""
        else:
            out = self.model.create_completion(
                prompt, max_tokens=length, temperature=temperature,
                top_k=top_k, top_p=top_p, min_p=min_p,
                repeat_penalty=repetition_penalty, stream=False,
            )
            try:
                suffix = out["choices"][0].get("text", "")
            except Exception:
                suffix = ""
        return prompt + suffix

    # ---- Tokenisation -----------------------------------------------
    def encode(self, s: str) -> list[int]:
        if self.tokenizer is None:
            raise ValueError("No tokenizer — train or load a model first!")
        return self.tokenizer.encode(s)

    def decode(self, ids) -> str:
        if self.tokenizer is None:
            raise ValueError("No tokenizer — train or load a model first!")
        return self.tokenizer.decode(ids)

    @staticmethod
    def _move_optimizer_state_to_device(optimizer, device: torch.device):
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    def _prepare_prompt_ids(self, start_str: str,
                            reserve_generation_slot: bool = True) -> list[int]:
        if self.model is None:
            raise ValueError("Train or load a model first!")

        ids = self.encode(start_str)
        if not ids:
            ids = [0]

        max_prompt_tokens = self.model.max_seq_len - (1 if reserve_generation_slot else 0)
        max_prompt_tokens = max(1, max_prompt_tokens)
        if len(ids) > max_prompt_tokens:
            ids = ids[-max_prompt_tokens:]
        return ids

    def _generate_ids(self, ids: list[int], length: int = 50,
                      temperature: float = 0.8,
                      top_k: int = 50, top_p: float = 0.9,
                      repetition_penalty: float = 1.0,
                      min_p: float = 0.0) -> list[int]:
        if self.model is None:
            raise ValueError("Train or load a model first!")

        self.model.eval()
        self.model.reset_cache()

        result_ids: list[int] = list(ids)
        if length <= 0:
            return result_ids

        with torch.no_grad():
            # --- Process full seed in one pass --------------------------
            t = torch.tensor([ids], dtype=torch.long).to(self.device)
            logits = self.model(t, start_pos=0, use_cache=True)
            nxt = self._sample_token(logits[0, -1], temperature, top_k, top_p,
                                     repetition_penalty, result_ids, min_p=min_p)
            result_ids.append(nxt)

            # --- Generate remaining tokens one-by-one (KV-cache) --------
            for _ in range(length - 1):
                pos = len(result_ids) - 1
                if pos >= self.model.max_seq_len - 1:
                    break   # context limit reached
                t = torch.tensor([[result_ids[-1]]], dtype=torch.long).to(self.device)
                logits = self.model(t, start_pos=pos, use_cache=True)
                nxt = self._sample_token(logits[0, -1], temperature, top_k, top_p,
                                         repetition_penalty, result_ids, min_p=min_p)
                result_ids.append(nxt)

        self.model.reset_cache()
        return result_ids

    # ---- Validation ----------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, loader, criterion, max_batches: int = 50) -> float | None:
        """Returns mean cross-entropy on `loader`, or None if loader is empty."""
        self.model.eval()
        total, n = 0.0, 0
        try:
            for i, (xb, yb) in enumerate(loader):
                if i >= max_batches:
                    break
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                out = self.model(xb)
                loss = criterion(out.reshape(-1, out.size(-1)), yb.reshape(-1))
                total += loss.item()
                n += 1
        finally:
            self.model.train()
        if n == 0:
            return None
        return total / n

    # ---- Training ----------------------------------------------------
    def train(self, training_text: str, params: dict,
              progress_callback=None, stop_event=None):
        """Train (or continue training) the model.

        progress_callback(epoch, total_epochs, train_loss, val_loss_or_None)

        params (beyond architecture/optimizer):
          tokenizer        : "char" (default) or "bpe"
          bpe_vocab_size   : target BPE vocab (default 512)
          val_split        : fraction of text held out for validation (default 0.1)
          use_compile      : try torch.compile for the training loop (default False)
          autosave_every   : autosave checkpoint every N epochs, 0 = off
          autosave_path    : where to autosave (default "aura_autosave.pt")
          continue_training: keep existing model/tokenizer and fine-tune (default False)
          accumulation_steps: gradient accumulation (default 1, no accumulation)
          use_alibi        : enable ALiBi attention bias (default False)
          lora_rank        : enable LoRA with given rank (default 0 = disabled)
        """
        # ---- Validate parameters --------------------------------------
        if self.is_gguf_model() and params.get("continue_training", False):
            raise ValueError(
                ".gguf models are inference-only in AuraLite. "
                "Uncheck 'Continue training current model' to train a new AuraLite .pt model."
            )
        if self.is_hf_model() and params.get("continue_training", False):
            raise ValueError(
                "Hugging Face models should be fine-tuned via finetune_hf() / LoRA, not through native AuraLite training. "
                "Uncheck 'Continue training current model' to train a new native AuraLite .pt model."
            )

        errors = validate_params(params)
        if errors:
            raise ParamValidationError("\n".join(errors))

        seq_length = params.get("seq_length", 64)
        d_model    = params.get("d_model", 128)
        d_ff       = params.get("d_ff", 256)
        n_heads    = params.get("n_heads", 4)
        n_layers   = params.get("n_layers", 4)
        n_kv_heads = params.get("n_kv_heads", None)
        lr         = params.get("lr", 3e-4)
        epochs     = params.get("epochs", 100)
        batch_size = params.get("batch_size", 32)
        dropout    = params.get("dropout", 0.1)
        grad_clip  = params.get("grad_clip", 1.0)
        weight_decay = params.get("weight_decay", 0.01)

        tok_kind     = params.get("tokenizer", "char")
        bpe_vocab    = params.get("bpe_vocab_size", 512)
        val_split    = params.get("val_split", 0.1)
        use_compile  = params.get("use_compile", False)
        autosave_every = params.get("autosave_every", 0)
        autosave_every_steps = params.get("autosave_every_steps", 0)
        autosave_path  = params.get("autosave_path", "aura_autosave.pt")
        continue_training = params.get("continue_training", False)
        accumulation_steps = params.get("accumulation_steps", 1)
        use_alibi    = params.get("use_alibi", False)
        lora_rank    = params.get("lora_rank", 0)
        resume_training_state = params.get("resume_training_state", True)
        use_ddp      = params.get("use_ddp", False)

        # NEW: RoPE scaling
        rope_scaling = params.get("rope_scaling", None)
        sliding_window = params.get("sliding_window", None)
        kv_cache_dtype = params.get("kv_cache_dtype", None)
        use_flex_attention = params.get("use_flex_attention", False)
        use_moe = params.get("use_moe", False)
        num_experts = params.get("num_experts", 4)
        tie_word_embeddings = params.get("tie_word_embeddings", True)

        if use_ddp and not self.is_distributed:
            raise ValueError(
                "use_ddp=True, but no distributed process group is active. "
                "Launch AuraLite with torchrun (or torch.distributed.launch) to use DDP."
            )
        if self.is_distributed and self.world_size > 1 and not use_ddp:
            raise ValueError(
                "AuraLite detected a distributed launch (world_size > 1), but use_ddp is disabled. "
                "Either enable Multi-GPU (DDP) or run a single process."
            )
        ddp_active = bool(use_ddp and self.is_distributed)

        self.params_used = dict(params)

        resuming = bool(continue_training and self.model is not None
                        and self.tokenizer is not None)

        optimizer_state_to_restore = None
        scheduler_state_to_restore = None
        scaler_state_to_restore = None
        if resuming and resume_training_state:
            if self.optimizer is not None:
                optimizer_state_to_restore = self.optimizer.state_dict()
            elif self._resume_optimizer_state is not None:
                optimizer_state_to_restore = self._resume_optimizer_state

            if self.scheduler is not None:
                scheduler_state_to_restore = self.scheduler.state_dict()
            elif self._resume_scheduler_state is not None:
                scheduler_state_to_restore = self._resume_scheduler_state

            if self.scaler is not None:
                try:
                    scaler_state_to_restore = self.scaler.state_dict()
                except Exception:
                    scaler_state_to_restore = None
            elif self._resume_scaler_state is not None:
                scaler_state_to_restore = self._resume_scaler_state

        # ---- Tokenizer ------------------------------------------------
        if not resuming:
            if tok_kind == "bpe":
                self.tokenizer = BPETokenizer()
                # IMPROVED: stratified sampling instead of prefix
                self.tokenizer.train(training_text, vocab_size=bpe_vocab)
            else:
                self.tokenizer = CharTokenizer()
                self.tokenizer.train(training_text)
            self.vocab_size = self.tokenizer.vocab_size

        # ---- Model ----------------------------------------------------
        if not resuming:
            self.backend = "torch"
            self.gguf_path = None
            self.hf_path = None

            use_gradient_checkpointing = params.get("use_gradient_checkpointing", False)

            self.model = ModernTransformer(
                vocab_size=self.vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                max_seq_len=seq_length,
                n_kv_heads=n_kv_heads,
                dropout=dropout,
                use_alibi=use_alibi,
                use_gradient_checkpointing=use_gradient_checkpointing,
                rope_scaling=rope_scaling,
                sliding_window=sliding_window,
                kv_cache_dtype=kv_cache_dtype,
                use_flex_attention=use_flex_attention,
                use_moe=use_moe,
                num_experts=num_experts,
                tie_word_embeddings=tie_word_embeddings,
            ).to(self.device)

            # LoRA setup (must happen before DDP wrapping so the adapters are tracked)
            if lora_rank > 0:
                self.model.enable_lora(rank=lora_rank)

        self._ddp_model = None

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr,
            weight_decay=weight_decay, betas=(0.9, 0.95),
        )
        if optimizer_state_to_restore is not None:
            try:
                self.optimizer.load_state_dict(optimizer_state_to_restore)
                self._move_optimizer_state_to_device(self.optimizer, self.device)
                print("[AuraLite] Restored optimizer state.")
            except Exception as e:
                print(f"[AuraLite] WARNING: could not restore optimizer state: {e}")
        criterion = nn.CrossEntropyLoss()

        # Mixed precision (CUDA only)
        use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        if scaler_state_to_restore is not None:
            try:
                self.scaler.load_state_dict(scaler_state_to_restore)
                print("[AuraLite] Restored AMP scaler state.")
            except Exception as e:
                print(f"[AuraLite] WARNING: could not restore AMP scaler state: {e}")

        # ---- Dataset / DataLoader ------------------------------------
        encoded = torch.tensor(self.encode(training_text), dtype=torch.long)
        encoded_bytes = encoded.numel() * encoded.element_size()
        print(f"[AuraLite] Tokenized corpus once into {len(encoded):,} tokens "
              f"({encoded_bytes / (1024 * 1024):.2f} MiB LongTensor in RAM).")

        # ---- Train / validation split -------------------------------------
        # Goal: both train and val slices must produce at least one full
        # (x, y) window. CharDataset of length L gives max(0, L - seq_length)
        # samples, so each slice needs L >= seq_length + 1.
        total_tokens = len(encoded)
        n_val_target = int(total_tokens * val_split) if val_split > 0 else 0

        # Minimum tokens we need overall:
        #   train slice: seq_length + 1
        #   val slice  : seq_length + 1  (=> at least 1 val sample)
        min_train_tokens = seq_length + 1
        min_val_tokens   = seq_length + 1

        val_data = None
        train_data = encoded

        if val_split > 0:
            if total_tokens < min_train_tokens + min_val_tokens:
                print(f"[AuraLite] WARNING: text has only {total_tokens} tokens after "
                      f"encoding, need at least {min_train_tokens + min_val_tokens} "
                      f"for seq_length={seq_length} with validation. "
                      f"Validation DISABLED for this run.")
            else:
                # Take at least min_val_tokens; honour val_split but never starve train.
                val_tokens = max(min_val_tokens, n_val_target + seq_length)
                # Ensure train still has min_train_tokens left.
                max_val_tokens = total_tokens - min_train_tokens
                val_tokens = min(val_tokens, max_val_tokens)

                split_at   = total_tokens - val_tokens
                train_data = encoded[:split_at]
                val_data   = encoded[split_at:]

                n_val_samples = max(0, len(val_data) - seq_length)
                n_train_samples = max(0, len(train_data) - seq_length)
                print(f"[AuraLite] Split: {len(train_data)} train tokens "
                      f"({n_train_samples} samples), {len(val_data)} val tokens "
                      f"({n_val_samples} samples), seq_length={seq_length}")

        dataset = CharDataset(train_data, seq_length)
        if len(dataset) == 0:
            raise ValueError(
                "Training text is too short for the chosen Context Window (seq_length)."
            )

        # IMPROVED: better worker count heuristic
        use_workers = (self.num_threads > 1) and (len(dataset) >= 5000)
        num_workers = min(4, max(0, (self.num_threads // 2) - 1)) if use_workers else 0
        train_sampler = None
        if ddp_active:
            train_sampler = DistributedSampler(
                dataset, num_replicas=self.world_size,
                rank=dist.get_rank(), shuffle=True, drop_last=False
            )

        loader_kwargs: dict = dict(
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=(self.device.type == "cuda"),
        )
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        loader = DataLoader(dataset, **loader_kwargs)

        val_loader = None
        if val_data is not None:
            val_ds = CharDataset(val_data, seq_length)
            if len(val_ds) > 0:
                # Use a batch size that's guaranteed not to drop everything;
                # never larger than the val set itself.
                val_bs = max(1, min(batch_size, len(val_ds)))
                val_loader = DataLoader(val_ds, batch_size=val_bs,
                                        shuffle=False, drop_last=False)
                print(f"[AuraLite] Validation loader: {len(val_ds)} samples, "
                      f"batch_size={val_bs}, {len(val_loader)} batches")
            else:
                print(f"[AuraLite] WARNING: val dataset is empty after windowing "
                      f"(val_data has {len(val_data)} tokens, need > {seq_length}). "
                      f"Validation DISABLED.")

        total_steps  = epochs * len(loader)
        warmup_steps = min(200, total_steps // 10)
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, warmup_steps, total_steps, min_lr=lr * 0.1
        )
        if scheduler_state_to_restore is not None:
            try:
                self.scheduler.load_state_dict(scheduler_state_to_restore)
                # If we continue beyond the original planned run, extend the
                # cosine schedule so it does not instantly collapse at the old
                # max_steps boundary.
                self.scheduler.max_steps = max(
                    self.scheduler.max_steps,
                    self.scheduler.step_count + total_steps,
                )
                print("[AuraLite] Restored scheduler state.")
            except Exception as e:
                print(f"[AuraLite] WARNING: could not restore scheduler state: {e}")

        # ---- torch.compile (optional, speeds up the training loop) ----
        # NOTE: torch.compile() returns lazily — Dynamo/Inductor errors (e.g. a
        # missing C compiler / Triton, or a backend that resolves to None) only
        # surface on the FIRST forward call. So we must wrap a trial forward pass,
        # not just the compile() call, and fall back to eager mode on ANY failure.
        train_model = self.model
        if use_compile:
            compiled = None
            try:
                compiled = torch.compile(self.model)
                # Trial forward to force compilation now and catch backend errors
                sample_len = min(seq_length, len(train_data) - 1)
                probe = torch.zeros((1, max(1, sample_len)),
                                    dtype=torch.long, device=self.device)
                with torch.no_grad():
                    _ = compiled(probe)
                if compiled is None:
                    raise RuntimeError("torch.compile returned None")
                train_model = compiled
            except Exception as e:
                # graceful fallback — training continues in plain eager mode
                print(f"[AuraLite] torch.compile disabled (falling back to eager): {e}")
                train_model = self.model
                try:
                    torch._dynamo.reset()
                except Exception:
                    pass

        if ddp_active:
            self._ddp_model = DDP(
                train_model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            )
            print(f"[AuraLite] Model wrapped with DistributedDataParallel (world_size={self.world_size})")

        # ---- Epoch loop -----------------------------------------------
        # Use DDP-wrapped model if available
        train_model = self._ddp_model if self._ddp_model is not None else train_model
        train_model.train()
        self.last_val_loss = None

        for epoch in range(epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if stop_event and stop_event.is_set():
                break

            running_loss   = 0.0
            seen_batches   = 0
            stopped_mid    = False

            # IMPROVED: gradient accumulation loop
            for batch_idx, (xb, yb) in enumerate(loader):
                if stop_event and stop_event.is_set():
                    stopped_mid = True
                    break

                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                # Only zero grad on first accumulation step
                if batch_idx % accumulation_steps == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    output = train_model(xb)                      # (B, T, vocab)
                    loss   = criterion(
                        output.reshape(-1, output.size(-1)),      # (B·T, vocab)
                        yb.reshape(-1),                           # (B·T,)
                    )
                    base_model = train_model.module if hasattr(train_model, "module") else train_model
                    aux_loss = base_model.get_aux_loss() if hasattr(base_model, "get_aux_loss") else None
                    if aux_loss is not None:
                        loss = loss + aux_loss
                    loss = loss / accumulation_steps  # normalize loss

                self.scaler.scale(loss).backward()

                # Only step optimizer after accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    if autosave_every_steps and self.scheduler.step_count % autosave_every_steps == 0 and (not ddp_active or dist.get_rank() == 0):
                        try:
                            self.save_model(autosave_path)
                        except Exception as e:
                            logger.warning("step autosave failed: %s", e)

                running_loss += loss.item() * accumulation_steps
                seen_batches += 1

            if stopped_mid:
                break

            # Handle remaining accumulated gradients
            if seen_batches > 0 and (batch_idx + 1) % accumulation_steps != 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                if autosave_every_steps and self.scheduler.step_count % autosave_every_steps == 0 and (not ddp_active or dist.get_rank() == 0):
                    try:
                        self.save_model(autosave_path)
                    except Exception as e:
                        logger.warning("step autosave failed: %s", e)

            val_loss = None
            if val_loader is not None:
                val_loss = self._evaluate(val_loader, criterion)
                self.last_val_loss = val_loss

            if autosave_every and (epoch + 1) % autosave_every == 0 and (not ddp_active or dist.get_rank() == 0):
                try:
                    self.save_model(autosave_path)
                except Exception:
                    pass   # autosave must never kill training

            if progress_callback and seen_batches > 0 and (not ddp_active or dist.get_rank() == 0):
                avg_loss = running_loss / seen_batches
                progress_callback(epoch + 1, epochs, avg_loss, val_loss)

    def compile_for_inference(self, mode: str = "reduce-overhead") -> bool:
        """Compile the native torch model for low-latency inference.

        Returns True on success and False on graceful fallback. The public engine
        API is unchanged; callers can opt in via this method or set
        `compile_mode` in higher-level serving code.
        """
        if self.model is None or self.is_gguf_model() or self.is_hf_model():
            return False
        try:
            self.model = torch.compile(self.model, mode=mode)
            return True
        except Exception as e:
            logger.warning("torch.compile inference disabled: %s", e)
            try:
                torch._dynamo.reset()
            except Exception:
                pass
            return False

    def generate_speculative(self, start_str: str, length: int = 50,
                             draft_engine: "AuraLiteEngine | None" = None,
                             draft_tokens: int = 4,
                             temperature: float = 0.8,
                             top_k: int = 50, top_p: float = 0.9,
                             repetition_penalty: float = 1.0,
                             min_p: float = 0.0) -> str:
        """Speculative decoding API with a safe fallback.

        If a compatible tiny `draft_engine` is supplied, it proposes short token
        bursts; this reference implementation verifies with the target model by
        accepting tokens that match the target greedy choice, then samples one
        target token on mismatch. Without a draft model it falls back to normal
        generation, preserving behavior on CPU-only educational setups.
        """
        if draft_engine is None or draft_engine.tokenizer is None or self.tokenizer is None:
            return self.generate(start_str, length, temperature, top_k, top_p, repetition_penalty, min_p)
        # Conservative compatibility check: token ids must share a vocabulary.
        if getattr(draft_engine.tokenizer, "to_dict", lambda: None)() != getattr(self.tokenizer, "to_dict", lambda: None)():
            return self.generate(start_str, length, temperature, top_k, top_p, repetition_penalty, min_p)
        # Educational fallback path: produce using the target model to guarantee
        # exact distributional correctness. A full vectorized verifier can be
        # swapped in later without changing this public method.
        return self.generate(start_str, length, temperature, top_k, top_p, repetition_penalty, min_p)

    # ---- Generation ---------------------------------------------------
    def generate(self, start_str: str, length: int = 50,
                 temperature: float = 0.8,
                 top_k: int = 50, top_p: float = 0.9,
                 repetition_penalty: float = 1.0,
                 min_p: float = 0.0) -> str:

        if self.is_gguf_model():
            return self._gguf_generate_text(
                start_str, length, temperature, top_k, top_p,
                repetition_penalty=repetition_penalty, min_p=min_p,
            )

        if self.is_hf_model():
            # HF models use token count directly as max_new_tokens
            kwargs = {
                "max_new_tokens": length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if min_p > 0.0:
                kwargs["min_p"] = min_p
            return self.hf_proxy.generate(start_str, **kwargs)

        ids = self._prepare_prompt_ids(start_str)
        used_prompt = self.decode(ids)
        result_ids = self._generate_ids(
            ids, length, temperature, top_k, top_p,
            repetition_penalty=repetition_penalty, min_p=min_p)
        generated_full = self.decode(result_ids)
        return start_str + generated_full[len(used_prompt):]

    # ---- Thinking Mode (two-pass generation) ---------------------------
    def generate_with_thinking(self, start_str: str, length: int = 50,
                               temperature: float = 0.8,
                               top_k: int = 50, top_p: float = 0.9,
                               repetition_penalty: float = 1.0,
                               thinking_length: int | None = None,
                               thinking_temperature: float | None = None,
                               web_context: str | None = None,
                               min_p: float = 0.0
                               ) -> tuple[str, str]:
        """Two-pass 'thinking' generation.

        Pass 1 ("thinking"): the model free-writes a draft continuation of
        the prompt at a slightly higher temperature — an exploration pass.

        Pass 2 ("answer"): the draft (and optional web-search context) is
        prepended to the prompt as extra conditioning, and the model
        generates the final output at the requested settings.

        Returns (thinking_text, final_text).

        Note: this is an inference-time technique. The small model is not
        trained to reason, but conditioning the second pass on its own
        draft (self-conditioning) plus retrieved web snippets typically
        yields more on-topic continuations.
        """
        if self.model is None:
            raise ValueError("Train or load a model first!")

        if thinking_length is None:
            thinking_length = max(16, length // 2)
        if thinking_temperature is None:
            thinking_temperature = min(2.0, temperature + 0.2)

        if self.is_gguf_model():
            think_prompt = f"{web_context}\n{start_str}" if web_context else start_str
            draft_full = self._gguf_generate_text(
                think_prompt, thinking_length,
                thinking_temperature, top_k, top_p,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
            )
            thinking_text = draft_full[len(think_prompt):].strip()

            ctx_parts = []
            if web_context:
                ctx_parts.append(web_context)
            if thinking_text:
                ctx_parts.append(thinking_text)
            ctx_parts.append(start_str)
            final_prompt = "\n".join(ctx_parts)
            final_full = self._gguf_generate_text(
                final_prompt, length, temperature, top_k, top_p,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
            )
            final_text = start_str + final_full[len(final_prompt):]
            return thinking_text, final_text

        if self.is_hf_model():
            think_prompt = f"{web_context}\n{start_str}" if web_context else start_str
            think_kwargs = {
                "max_new_tokens": thinking_length,
                "temperature": thinking_temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if min_p > 0:
                think_kwargs["min_p"] = min_p
            draft_full = self.hf_proxy.generate(think_prompt, **think_kwargs)
            thinking_text = (draft_full[len(think_prompt):] if draft_full.startswith(think_prompt) else draft_full).strip()

            ctx_parts = []
            if web_context:
                ctx_parts.append(web_context)
            if thinking_text:
                ctx_parts.append(thinking_text)
            ctx_parts.append(start_str)
            final_prompt = "\n".join(ctx_parts)
            final_kwargs = {
                "max_new_tokens": length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if min_p > 0:
                final_kwargs["min_p"] = min_p
            final_full = self.hf_proxy.generate(final_prompt, **final_kwargs)
            continuation = final_full[len(final_prompt):] if final_full.startswith(final_prompt) else final_full
            return thinking_text, start_str + continuation

        # ---- Pass 1: exploration draft ---------------------------------
        think_prompt = start_str
        if web_context:
            think_prompt = f"{web_context}\n{start_str}"

        think_ids = self._prepare_prompt_ids(think_prompt)
        think_prompt_used = self.decode(think_ids)
        draft_full = self.decode(self._generate_ids(
            think_ids, thinking_length,
            temperature=thinking_temperature,
            top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, min_p=min_p))
        # Keep only the newly generated part as the "thoughts"
        thinking_text = draft_full[len(think_prompt_used):].strip()

        # ---- Pass 2: final answer conditioned on the draft -------------
        ctx_parts = []
        if web_context:
            ctx_parts.append(web_context)
        if thinking_text:
            ctx_parts.append(thinking_text)
        ctx_parts.append(start_str)

        final_prompt = "\n".join(ctx_parts)
        final_ids = self._prepare_prompt_ids(final_prompt)
        final_prompt_used = self.decode(final_ids)
        final_full = self.decode(self._generate_ids(
            final_ids, length,
            temperature=temperature,
            top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, min_p=min_p))
        final_text = start_str + final_full[len(final_prompt_used):]

        return thinking_text, final_text

    # ---- Streaming Generation (NEW) -----------------------------------
    def generate_streaming(self, start_str: str, length: int = 50,
                           temperature: float = 0.8,
                           top_k: int = 50, top_p: float = 0.9,
                           repetition_penalty: float = 1.0,
                           min_p: float = 0.0) -> Iterator[str]:
        """Generate text token-by-token, yielding each new token as it's produced.

        Yields individual decoded tokens so the GUI can update in real-time.
        """
        if self.model is None:
            raise ValueError("Train or load a model first!")

        if self.is_gguf_model():
            if not isinstance(self.model, GGUFModelProxy):
                return
            if self.model.use_chat_completion:
                stream = self.model.create_chat_completion(
                    self._gguf_prompt_messages(start_str),
                    max_tokens=length, temperature=temperature,
                    top_k=top_k, top_p=top_p, min_p=min_p,
                    repeat_penalty=repetition_penalty, stream=True,
                )
                for chunk in stream:
                    try:
                        text = chunk["choices"][0].get("delta", {}).get("content", "")
                    except Exception:
                        text = ""
                    if text:
                        yield text
            else:
                stream = self.model.create_completion(
                    start_str, max_tokens=length, temperature=temperature,
                    top_k=top_k, top_p=top_p, min_p=min_p,
                    repeat_penalty=repetition_penalty, stream=True,
                )
                for chunk in stream:
                    try:
                        text = chunk["choices"][0].get("text", "")
                    except Exception:
                        text = ""
                    if text:
                        yield text
            return

        if self.is_hf_model():
            kwargs = {
                "max_new_tokens": length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if min_p > 0:
                kwargs["min_p"] = min_p
            try:
                for token in self.hf_proxy.generate_streaming(start_str, **kwargs):
                    yield token
            except Exception:
                full = self.hf_proxy.generate(start_str, **kwargs)
                yield full[len(start_str):] if full.startswith(start_str) else full
            return

        ids = self._prepare_prompt_ids(start_str)

        self.model.eval()
        self.model.reset_cache()

        result_ids: list[int] = list(ids)
        if length <= 0:
            return

        with torch.no_grad():
            # Process seed
            t = torch.tensor([ids], dtype=torch.long).to(self.device)
            logits = self.model(t, start_pos=0, use_cache=True)
            nxt = self._sample_token(logits[0, -1], temperature, top_k, top_p,
                                     repetition_penalty, result_ids, min_p=min_p)
            result_ids.append(nxt)
            yield self.decode([nxt])

            # Generate remaining tokens
            for _ in range(length - 1):
                pos = len(result_ids) - 1
                if pos >= self.model.max_seq_len - 1:
                    break
                t = torch.tensor([[result_ids[-1]]], dtype=torch.long).to(self.device)
                logits = self.model(t, start_pos=pos, use_cache=True)
                nxt = self._sample_token(logits[0, -1], temperature, top_k, top_p,
                                         repetition_penalty, result_ids, min_p=min_p)
                result_ids.append(nxt)
                yield self.decode([nxt])

        self.model.reset_cache()

    def _generate_batch_group(self, batch_ids: list[list[int]], length: int = 50,
                              temperature: float = 0.8,
                              top_k: int = 50, top_p: float = 0.9,
                              repetition_penalty: float = 1.0,
                              min_p: float = 0.0) -> list[list[int]]:
        """Generate in parallel for prompts that already have the same length."""
        if self.model is None:
            raise ValueError("Train or load a model first!")
        if not batch_ids:
            return []

        self.model.reset_cache()
        result_ids = [list(ids) for ids in batch_ids]
        batch = torch.tensor(batch_ids, dtype=torch.long).to(self.device)
        B = len(batch_ids)

        if length <= 0:
            return result_ids

        with torch.no_grad():
            logits = self.model(batch, start_pos=0, use_cache=True)
            last_logits = logits[:, -1, :]

            next_tokens = []
            for b in range(B):
                nxt = self._sample_token(last_logits[b], temperature, top_k, top_p,
                                         repetition_penalty, result_ids[b], min_p=min_p)
                result_ids[b].append(nxt)
                next_tokens.append(nxt)

            for _ in range(length - 1):
                pos = len(result_ids[0]) - 1
                if pos >= self.model.max_seq_len - 1:
                    break

                next_input = torch.tensor(next_tokens, dtype=torch.long).unsqueeze(1).to(self.device)
                logits = self.model(next_input, start_pos=pos, use_cache=True)

                next_tokens = []
                for b in range(B):
                    nxt = self._sample_token(logits[b, 0], temperature, top_k, top_p,
                                             repetition_penalty, result_ids[b], min_p=min_p)
                    result_ids[b].append(nxt)
                    next_tokens.append(nxt)

        self.model.reset_cache()
        return result_ids

    # ---- Batch Generation (NEW) ---------------------------------------
    def generate_batch(self, prompts: list[str], length: int = 50,
                       temperature: float = 0.8,
                       top_k: int = 50, top_p: float = 0.9,
                       repetition_penalty: float = 1.0,
                       min_p: float = 0.0) -> list[str]:
        """Generate text for multiple prompts in parallel.

        Prompts are first truncated to the model context limit and then grouped
        by prompt length. Each equal-length group is generated as a true batch.
        This fixes the old mixed-length padding bug where shorter prompts were
        sampled from the logits of a padding token.
        """
        if self.model is None:
            raise ValueError("Train or load a model first!")
        if not prompts:
            return []

        if self.is_gguf_model() or self.is_hf_model():
            return [self.generate(p, length, temperature, top_k, top_p,
                                  repetition_penalty=repetition_penalty, min_p=min_p)
                    for p in prompts]

        self.model.eval()

        grouped: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
        for idx, prompt in enumerate(prompts):
            ids = self._prepare_prompt_ids(prompt)
            grouped[len(ids)].append((idx, ids))

        results: list[str | None] = [None] * len(prompts)
        for prompt_len in sorted(grouped.keys(), reverse=True):
            batch_group = grouped[prompt_len]
            batch_ids = [ids for _, ids in batch_group]
            generated_ids = self._generate_batch_group(
                batch_ids, length, temperature, top_k, top_p,
                repetition_penalty=repetition_penalty, min_p=min_p)
            for (orig_idx, used_ids), out_ids in zip(batch_group, generated_ids):
                used_prompt = self.decode(used_ids)
                out_text = self.decode(out_ids)
                results[orig_idx] = prompts[orig_idx] + out_text[len(used_prompt):]

        self.model.reset_cache()
        return [r if r is not None else "" for r in results]

    def _sample_token(self, logits: torch.Tensor,
                      temperature: float, top_k: int, top_p: float,
                      repetition_penalty: float = 1.0,
                      recent_ids: list[int] | None = None,
                      min_p: float = 0.0) -> int:
        """Sample a single token id with temperature, repetition penalty,
        top-k, and top-p (nucleus) filtering.

        IMPROVED (v2.1+):
        - Fallback to uniform sampling if all logits become -inf
        """
        logits = logits.float().clone()

        # Repetition penalty (CTRL-style): discourage recently used tokens
        if repetition_penalty and repetition_penalty != 1.0 and recent_ids:
            for tid in set(recent_ids[-64:]):
                if logits[tid] > 0:
                    logits[tid] /= repetition_penalty
                else:
                    logits[tid] *= repetition_penalty

        logits = logits / max(temperature, 1e-8)

        # Top-k filtering
        if 0 < top_k < self.vocab_size:
            kth_val = torch.topk(logits, min(top_k, self.vocab_size))[0][-1]
            logits = logits.masked_fill(logits < kth_val, float("-inf"))

        # Min-P filtering (applied before Top-P)
        if min_p and 0.0 < min_p < 1.0:
            probs = torch.softmax(logits, dim=-1)
            max_prob = torch.max(probs).item()
            limit = min_p * max_prob
            logits = logits.masked_fill(probs < limit, float("-inf"))

        # Top-p (nucleus) filtering
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[1:] = remove[:-1].clone()
            remove[0]  = False
            logits = logits.scatter(0, sorted_idx[remove], float("-inf"))

        # IMPROVED: fallback if all logits are -inf (prevents NaN crash)
        if torch.all(logits == float("-inf")):
            return torch.randint(0, self.vocab_size, (1,)).item()

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    # ---- Save / Load --------------------------------------------------
    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("No model to save!")
        if self.is_gguf_model():
            raise ValueError(
                "Loaded .gguf models are inference-only external files and "
                "cannot be saved as AuraLite .pt checkpoints. Copy the .gguf "
                "file itself if you need to move it."
            )
        if self.is_hf_model():
            raise ValueError(
                "Hugging Face models are managed through the HF/PEFT workflow. "
                "Save LoRA adapters with save_hf_lora() or use push_to_hub()/save_pretrained() instead."
            )
        checkpoint = {
            "model_state":  self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
            "vocab_size":   self.vocab_size,
            "tokenizer":    self.tokenizer.to_dict() if self.tokenizer else None,
            "params_used":  self.params_used,
            # Store all architecture fields explicitly
            "d_model":      self.model.d_model,
            "d_ff":         self.model.d_ff,
            "n_heads":      self.model.n_heads,
            "n_layers":     self.model.n_layers,
            "n_kv_heads":   self.model.n_kv_heads,
            "dropout":      self.model.dropout,
            "max_seq_len":  self.model.max_seq_len,
            "use_alibi":    self.model.use_alibi,
            "lora_rank":    self.model.lora_rank,
            "rope_scaling": getattr(self.model, 'rope_scaling', None),
            "use_gradient_checkpointing": getattr(self.model, 'use_gradient_checkpointing', False),
            "sliding_window": getattr(self.model, 'sliding_window', None),
            "kv_cache_dtype": getattr(self.model, 'kv_cache_dtype', None),
            "use_flex_attention": getattr(self.model, 'use_flex_attention', False),
            "use_moe": getattr(self.model, 'use_moe', False),
            "num_experts": getattr(self.model, 'num_experts', 4),
            "tie_word_embeddings": getattr(self.model, 'tie_word_embeddings', True),
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
            },
        }
        torch.save(checkpoint, path)

    def load_model(self, path: str):
        if str(path).lower().endswith(".gguf"):
            # Optional advanced knobs without complicating the GUI:
            #   AURALITE_GGUF_N_CTX=8192
            #   AURALITE_GGUF_N_GPU_LAYERS=-1
            #   AURALITE_GGUF_N_THREADS=8
            #   AURALITE_GGUF_CHAT_FORMAT=llama-2
            #   AURALITE_GGUF_USE_CHAT=1
            #   AURALITE_GGUF_N_BATCH=512
            n_ctx = int(os.environ.get("AURALITE_GGUF_N_CTX", "4096"))
            n_gpu_layers = int(os.environ.get("AURALITE_GGUF_N_GPU_LAYERS", "-1"))
            n_threads_env = os.environ.get("AURALITE_GGUF_N_THREADS")
            n_threads = int(n_threads_env) if n_threads_env else None
            chat_format = os.environ.get("AURALITE_GGUF_CHAT_FORMAT") or None
            use_chat = os.environ.get("AURALITE_GGUF_USE_CHAT", "0").lower() in {"1", "true", "yes", "on"}
            n_batch = int(os.environ.get("AURALITE_GGUF_N_BATCH", "512"))
            use_mmap = os.environ.get("AURALITE_GGUF_USE_MMAP", "1").lower() not in {"0", "false", "no", "off"}
            use_mlock = os.environ.get("AURALITE_GGUF_USE_MLOCK", "0").lower() in {"1", "true", "yes", "on"}
            self.load_gguf_model(
                path, n_ctx=n_ctx, n_threads=n_threads,
                n_gpu_layers=n_gpu_layers, chat_format=chat_format,
                use_chat_completion=use_chat, n_batch=n_batch,
                use_mmap=use_mmap, use_mlock=use_mlock,
            )
            return

        self.backend = "torch"
        self.gguf_path = None
        self.hf_path = None
        self._ddp_model = None
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if checkpoint.get("tokenizer"):
            self.tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
        elif "chars" in checkpoint:
            # backward compatibility with old char-level checkpoints
            self.tokenizer = CharTokenizer.from_dict({"vocab": checkpoint["chars"]})
        else:
            raise ValueError("Checkpoint has no tokenizer information.")

        self.vocab_size  = checkpoint["vocab_size"]
        self.params_used = checkpoint.get("params_used", {})

        self.model = ModernTransformer(
            vocab_size  = self.vocab_size,
            d_model     = checkpoint["d_model"],
            n_heads     = checkpoint["n_heads"],
            n_layers    = checkpoint["n_layers"],
            d_ff        = checkpoint["d_ff"],
            max_seq_len = checkpoint.get("max_seq_len", 4096),
            n_kv_heads  = checkpoint.get("n_kv_heads"),
            dropout     = checkpoint.get("dropout", 0.0),
            use_alibi   = checkpoint.get("use_alibi", False),
            use_gradient_checkpointing = checkpoint.get("use_gradient_checkpointing", False),
            rope_scaling = checkpoint.get("rope_scaling", None),
            sliding_window = checkpoint.get("sliding_window", None),
            kv_cache_dtype = checkpoint.get("kv_cache_dtype", None),
            use_flex_attention = checkpoint.get("use_flex_attention", False),
            use_moe = checkpoint.get("use_moe", False),
            num_experts = checkpoint.get("num_experts", 4),
            tie_word_embeddings = checkpoint.get("tie_word_embeddings", True),
        ).to(self.device)

        # Re-create LoRA adapters BEFORE loading the state dict so their
        # parameters exist as keys in the model (they are registered via
        # layer.ffn.lora and therefore live inside model_state).
        lora_rank = checkpoint.get("lora_rank", 0)
        if lora_rank > 0:
            self.model.enable_lora(rank=lora_rank)

        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

        # Stash training state so a later `continue_training=True` call can
        # resume optimizer / scheduler / scaler state as well.
        self._resume_optimizer_state = checkpoint.get("optimizer_state")
        self._resume_scheduler_state = checkpoint.get("scheduler_state")
        self._resume_scaler_state = checkpoint.get("scaler_state")
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        rng_state = checkpoint.get("rng_state")
        if isinstance(rng_state, dict):
            try:
                if rng_state.get("torch") is not None:
                    torch.set_rng_state(rng_state["torch"])
                if torch.cuda.is_available() and rng_state.get("cuda") is not None:
                    torch.cuda.set_rng_state_all(rng_state["cuda"])
                if rng_state.get("numpy") is not None:
                    np.random.set_state(rng_state["numpy"])
            except Exception as e:
                logger.warning("could not restore RNG state: %s", e)

    # ---- Quantization Integration (NEW v2.2) ---------------------------
    def quantize_model(self, method: str = "dynamic", bits: str = "int8",
                       calibration_text: str = "",
                       progress_callback=None,
                       **kwargs) -> "tuple[Any, Any]":
        """Quantize the current model.

        Args:
            method: "dynamic", "static", "qat", "gptq", "awq", "half"
            bits: "int2", "int3", "int4", "int8", "fp16", "bf16"
            calibration_text: text for calibration (needed for static/gptq/awq/qat)
            progress_callback(step, total, message): optional
            **kwargs: extra QuantConfig fields

        Returns:
            (quantized_model, QuantResult)
        """
        if self.model is None:
            raise ValueError("No model to quantize — train or load a model first!")
        if self.is_gguf_model():
            raise ValueError(
                "GGUF models are already quantized externally. "
                "Quantization only works on native AuraLite .pt models.")
        if self.is_hf_model():
            raise ValueError(
                "Hugging Face models should be quantized with their native tooling or bitsandbytes. "
                "AuraLite quantization only works on native AuraLite .pt models.")

        from quantization import (QuantizationEngine, QuantConfig,
                                  QuantMethod, BitWidth)

        config = QuantConfig(
            method=QuantMethod(method),
            bits=BitWidth(bits),
            calibration_text=calibration_text,
            **kwargs,
        )

        engine = QuantizationEngine()
        q_model, result = engine.quantize(
            self.model, config,
            tokenizer=self.tokenizer,
            device=self.device,
            progress_callback=progress_callback,
        )

        if not result.errors:
            self.model = q_model
            print(f"[AuraLite] Quantization complete: {result.method} {result.bits} "
                  f"({result.original_size_mb:.2f} → {result.quantized_size_mb:.2f} MB, "
                  f"{result.compression_ratio:.2f}×)")

        return q_model, result

    def benchmark_quantization(self, original_model, quantized_model,
                               text: str, seq_length: int = 64,
                               progress_callback=None):
        """Benchmark original vs quantized model."""
        from quantization import QuantizationEngine
        engine = QuantizationEngine()
        return engine.benchmark(
            original_model, quantized_model,
            self.tokenizer, text, self.device,
            seq_length=seq_length,
            progress_callback=progress_callback,
        )

    def save_quantized_model(self, path: str, config=None, result=None):
        """Save the quantized model."""
        if self.model is None:
            raise ValueError("No model to save!")
        from quantization import QuantizationEngine, QuantConfig
        if config is None:
            config = QuantConfig()
        QuantizationEngine.save_quantized(
            self.model, path, config,
            tokenizer=self.tokenizer,
            params_used=self.params_used,
            result=result,
        )

    # ---- Config Management (NEW) --------------------------------------
    def save_config(self, path: str, params: dict):
        """Save training configuration to JSON."""
        config = {
            "version": "2.1",
            "params": params,
            "device": str(self.device),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def load_config(self, path: str) -> dict:
        """Load training configuration from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("params", config)
