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

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

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

    if d_model <= 0:
        errors.append(f"d_model must be > 0, got {d_model}")
    if d_model % n_heads != 0:
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
    """

    def __init__(self, d_model: int, n_heads: int,
                 n_kv_heads: int | None = None,
                 max_seq_len: int = 4096,
                 use_alibi: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.use_alibi = use_alibi

        self.W_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        # Pre-compute RoPE cos / sin as persistent buffers (move with .to(device))
        freqs = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(max_seq_len).float()
        angles = torch.outer(t, freqs)                              # (max_seq_len, head_dim//2)
        self.register_buffer("rope_cos", angles.cos(), persistent=True)
        self.register_buffer("rope_sin", angles.sin(), persistent=True)

        # ALiBi slopes (one per head)
        if use_alibi:
            self.register_buffer("alibi_slopes", self._get_alibi_slopes(), persistent=True)

        self.kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def _get_alibi_slopes(self) -> torch.Tensor:
        """Compute ALiBi slopes: 2^(-8/n), 2^(-16/n), ... for n heads."""
        n = self.n_heads
        # NeMo convention: use 2^(-8/n) for the first head
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        # Simpler: standard ALiBi slopes
        slopes = 2 ** (-8 * torch.arange(1, n + 1).float() / n)
        return slopes

    # ---- RoPE --------------------------------------------------------
    def _apply_rope(self, x: torch.Tensor, start_pos: int, seq_len: int) -> torch.Tensor:
        """Apply Rotary Position Embeddings to a (*, seq_len, head_dim) tensor."""
        cos = self.rope_cos[start_pos:start_pos + seq_len]  # (T, hd//2)
        sin = self.rope_sin[start_pos:start_pos + seq_len]

        # x → (..., T, hd//2, 2)
        x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        x0, x1 = x_pairs[..., 0], x_pairs[..., 1]

        cos = cos[None, :, None, :]   # (1, T, 1, hd//2)
        sin = sin[None, :, None, :]

        out_x0 = x0 * cos - x1 * sin
        out_x1 = x0 * sin + x1 * cos
        return torch.stack([out_x0, out_x1], dim=-1).flatten(-2).type_as(x)

    # ---- ALiBi bias --------------------------------------------------
    def _get_alibi_bias(self, q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        """Create ALiBi causal attention bias of shape (n_heads, q_len, k_len)."""
        # Relative positions: q_pos - k_pos  (negative or zero for causal)
        q_pos = torch.arange(q_len, device=device)
        k_pos = torch.arange(k_len, device=device)
        rel_pos = q_pos.unsqueeze(1) - k_pos.unsqueeze(0)  # (q_len, k_len)
        # causal mask: only allow current and past positions
        rel_pos = rel_pos.clamp(max=0)  # zero out future positions
        # Apply slopes: (n_heads, 1, 1) * (1, q_len, k_len)
        alibi_bias = self.alibi_slopes.unsqueeze(-1).unsqueeze(-1) * rel_pos
        return alibi_bias  # (n_heads, q_len, k_len)

    # ---- Forward -----------------------------------------------------
    def forward(self, x: torch.Tensor,
                start_pos: int = 0, use_cache: bool = False) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.W_q(x).view(B, T, self.n_heads,    self.head_dim)
        k = self.W_k(x).view(B, T, self.n_kv_heads,  self.head_dim)
        v = self.W_v(x).view(B, T, self.n_kv_heads,  self.head_dim)

        q = self._apply_rope(q, start_pos, T)
        k = self._apply_rope(k, start_pos, T)

        # GQA: repeat KV heads to match query heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        q = q.transpose(1, 2)   # (B, nh, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # KV-cache
        if use_cache and self.kv_cache is not None:
            cached_k, cached_v = self.kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        if use_cache:
            self.kv_cache = (k, v)

        S = k.shape[2]
        # Flash / memory-efficient attention via PyTorch SDPA
        if T == S:
            # full sequence (training or seed pass) — standard causal mask
            if self.use_alibi:
                alibi = self._get_alibi_bias(T, S, x.device)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=alibi)
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif T == 1:
            # incremental decoding — the single query may attend to all keys
            if self.use_alibi:
                alibi = self._get_alibi_bias(1, S, x.device)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=alibi)
            else:
                out = F.scaled_dot_product_attention(q, k, v)
        else:
            # general case (chunked decoding with cache)
            if self.use_alibi:
                alibi = self._get_alibi_bias(T, S, x.device)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=alibi)
            else:
                q_pos = torch.arange(start_pos, start_pos + T, device=x.device)
                k_pos = torch.arange(S, device=x.device)
                keep = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)

    def reset_cache(self):
        self.kv_cache = None


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


# -------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: RMSNorm → Attention → RMSNorm → SwiGLU FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 n_kv_heads: int | None, max_seq_len: int,
                 dropout: float = 0.0, use_alibi: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn      = Attention(d_model, n_heads, n_kv_heads, max_seq_len,
                                   use_alibi=use_alibi)
        self.ffn_norm  = RMSNorm(d_model)
        self.ffn       = FeedForward(d_model, d_ff)
        self.dropout   = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor,
                start_pos: int = 0, use_cache: bool = False) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x), start_pos, use_cache))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


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
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int,
                 max_seq_len: int = 4096,
                 n_kv_heads: int | None = None,
                 dropout: float = 0.0,
                 use_alibi: bool = False):
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

        self.embedding   = nn.Embedding(vocab_size, d_model)
        self.layers      = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, n_kv_heads, max_seq_len, dropout,
                             use_alibi=use_alibi)
            for _ in range(n_layers)
        ])
        self.final_norm  = RMSNorm(d_model)
        self.head        = nn.Linear(d_model, vocab_size, bias=False)

        # Modern weight init (GPT-2 / LLaMA style)
        self.apply(self._init_weights)

        # Weight tying: output head shares the embedding matrix
        # (GPT-2 / LLaMA practice — fewer parameters, better generalisation)
        self.head.weight = self.embedding.weight

        # LoRA adapters (initially None, enabled via enable_lora())
        self.lora_adapters: list | None = None
        self.lora_rank = 0

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
    """Sliding-window token-level dataset.

    Stores the full encoded text as a single tensor and produces
    (input_seq, target_seq) samples on the fly, where target_seq is
    input_seq shifted by one — so the loss is computed over EVERY
    position of the window (dense next-token prediction), not just
    the last one. This makes training ~seq_length× more sample-efficient.
    """

    def __init__(self, encoded: torch.Tensor, seq_length: int):
        self.data = encoded
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1 : idx + self.seq_length + 1]
        return x, y


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
        self.model: ModernTransformer | None = None
        self.optimizer   = None
        self.scheduler   = None
        self.scaler      = None
        self.tokenizer   = None              # CharTokenizer | BPETokenizer
        self.vocab_size  = 0
        self.params_used: dict = {}          # remember last training params
        self.last_val_loss: float | None = None

        # CUDA performance tuning
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except AttributeError:
                pass  # older PyTorch versions

    # ---- Tokenisation -----------------------------------------------
    def encode(self, s: str) -> list[int]:
        if self.tokenizer is None:
            raise ValueError("No tokenizer — train or load a model first!")
        return self.tokenizer.encode(s)

    def decode(self, ids) -> str:
        if self.tokenizer is None:
            raise ValueError("No tokenizer — train or load a model first!")
        return self.tokenizer.decode(ids)

    # ---- Validation ----------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, loader, criterion, max_batches: int = 50) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for i, (xb, yb) in enumerate(loader):
            if i >= max_batches:
                break
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            out = self.model(xb)
            total += criterion(out.reshape(-1, out.size(-1)), yb.reshape(-1)).item()
            n += 1
        self.model.train()
        return total / max(1, n)

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
        autosave_path  = params.get("autosave_path", "aura_autosave.pt")
        continue_training = params.get("continue_training", False)
        accumulation_steps = params.get("accumulation_steps", 1)
        use_alibi    = params.get("use_alibi", False)
        lora_rank    = params.get("lora_rank", 0)

        self.params_used = dict(params)

        resuming = bool(continue_training and self.model is not None
                        and self.tokenizer is not None)

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
            self.model = ModernTransformer(
                vocab_size=self.vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                max_seq_len=4096,
                n_kv_heads=n_kv_heads,
                dropout=dropout,
                use_alibi=use_alibi,
            ).to(self.device)

            # LoRA setup
            if lora_rank > 0:
                self.model.enable_lora(rank=lora_rank)

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr,
            weight_decay=weight_decay, betas=(0.9, 0.95),
        )
        criterion = nn.CrossEntropyLoss()

        # Mixed precision (CUDA only)
        use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # ---- Dataset / DataLoader ------------------------------------
        encoded = torch.tensor(self.encode(training_text), dtype=torch.long)

        n_val = int(len(encoded) * val_split) if val_split > 0 else 0
        if n_val <= seq_length + 1:
            n_val = 0
        train_data = encoded[: len(encoded) - n_val] if n_val else encoded
        val_data   = encoded[len(encoded) - n_val - seq_length:] if n_val else None

        dataset = CharDataset(train_data, seq_length)
        if len(dataset) == 0:
            raise ValueError(
                "Training text is too short for the chosen Context Window (seq_length)."
            )

        # IMPROVED: better worker count heuristic
        use_workers = (self.num_threads > 1) and (len(dataset) >= 5000)
        num_workers = min(4, max(0, (self.num_threads // 2) - 1)) if use_workers else 0
        loader_kwargs: dict = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=(self.device.type == "cuda"),
        )
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        loader = DataLoader(dataset, **loader_kwargs)
        val_loader = (DataLoader(CharDataset(val_data, seq_length),
                                 batch_size=batch_size, shuffle=False)
                      if val_data is not None else None)

        total_steps  = epochs * len(loader)
        warmup_steps = min(200, total_steps // 10)
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, warmup_steps, total_steps, min_lr=lr * 0.1
        )

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

        # ---- Epoch loop -----------------------------------------------
        self.model.train()
        self.last_val_loss = None
        for epoch in range(epochs):
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
                    ) / accumulation_steps  # normalize loss

                self.scaler.scale(loss).backward()

                # Only step optimizer after accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()

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

            val_loss = None
            if val_loader is not None:
                val_loss = self._evaluate(val_loader, criterion)
                self.last_val_loss = val_loss

            if autosave_every and (epoch + 1) % autosave_every == 0:
                try:
                    self.save_model(autosave_path)
                except Exception:
                    pass   # autosave must never kill training

            if progress_callback and seen_batches > 0:
                avg_loss = running_loss / seen_batches
                progress_callback(epoch + 1, epochs, avg_loss, val_loss)

    # ---- Generation ---------------------------------------------------
    def generate(self, start_str: str, length: int = 50,
                 temperature: float = 0.8,
                 top_k: int = 50, top_p: float = 0.9,
                 repetition_penalty: float = 1.0) -> str:

        if self.model is None:
            raise ValueError("Train or load a model first!")

        self.model.eval()
        self.model.reset_cache()

        ids = self.encode(start_str)
        if not ids:
            ids = [0]
        result_ids: list[int] = list(ids)

        with torch.no_grad():
            # --- Process full seed in one pass --------------------------
            t = torch.tensor([ids], dtype=torch.long).to(self.device)
            logits = self.model(t, start_pos=0, use_cache=True)
            nxt = self._sample_token(logits[0, -1], temperature, top_k, top_p,
                                     repetition_penalty, result_ids)
            result_ids.append(nxt)

            # --- Generate remaining tokens one-by-one (KV-cache) --------
            for _ in range(length - 1):
                pos = len(result_ids) - 1
                if pos >= self.model.max_seq_len - 1:
                    break   # context limit reached
                t = torch.tensor([[result_ids[-1]]], dtype=torch.long).to(self.device)
                logits = self.model(t, start_pos=pos, use_cache=True)
                nxt = self._sample_token(logits[0, -1], temperature, top_k, top_p,
                                         repetition_penalty, result_ids)
                result_ids.append(nxt)

        self.model.reset_cache()
        return self.decode(result_ids)

    # ---- Streaming Generation (NEW) -----------------------------------
    def generate_streaming(self, start_str: str, length: int = 50,
                           temperature: float = 0.8,
                           top_k: int = 50, top_p: float = 0.9,
                           repetition_penalty: float = 1.0) -> Iterator[str]:
        """Generate text token-by-token, yielding each new token as it's produced.

        Yields individual decoded tokens so the GUI can update in real-time.
        """
        if self.model is None:
            raise ValueError("Train or load a model first!")

        self.model.eval()
        self.model.reset_cache()

        ids = self.encode(start_str)
        if not ids:
            ids = [0]
        result_ids: list[int] = list(ids)

        with torch.no_grad():
            # Process seed
            t = torch.tensor([ids], dtype=torch.long).to(self.device)
            logits = self.model(t, start_pos=0, use_cache=True)
            nxt = self._sample_token(logits[0, -1], temperature, top_k, top_p,
                                     repetition_penalty, result_ids)
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
                                         repetition_penalty, result_ids)
                result_ids.append(nxt)
                yield self.decode([nxt])

        self.model.reset_cache()

    # ---- Batch Generation (NEW) ---------------------------------------
    def generate_batch(self, prompts: list[str], length: int = 50,
                       temperature: float = 0.8,
                       top_k: int = 50, top_p: float = 0.9,
                       repetition_penalty: float = 1.0) -> list[str]:
        """Generate text for multiple prompts in parallel.

        Batches all prompts, pads to max length, runs single forward pass,
        then generates token-by-token for each sequence.
        """
        if self.model is None:
            raise ValueError("Train or load a model first!")

        self.model.eval()
        self.model.reset_cache()

        # Encode all prompts
        all_ids = [self.encode(p) for p in prompts]
        max_len = max(len(ids) for ids in all_ids)

        # Pad to same length
        padded_ids = []
        for ids in all_ids:
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [0] * pad_len)

        batch = torch.tensor(padded_ids, dtype=torch.long).to(self.device)
        B = len(prompts)

        result_ids = [list(ids) for ids in all_ids]

        with torch.no_grad():
            # Process full prompts in one pass
            logits = self.model(batch, start_pos=0, use_cache=True)
            last_logits = logits[:, -1, :]  # (B, vocab)

            next_tokens = []
            for b in range(B):
                nxt = self._sample_token(last_logits[b], temperature, top_k, top_p,
                                         repetition_penalty, result_ids[b])
                result_ids[b].append(nxt)
                next_tokens.append(nxt)

            # Generate remaining tokens one-by-one
            for _ in range(length - 1):
                pos = max(len(rids) for rids in result_ids) - 1
                if pos >= self.model.max_seq_len - 1:
                    break

                next_input = torch.tensor([[nt] for nt in next_tokens],
                                          dtype=torch.long).to(self.device)
                logits = self.model(next_input, start_pos=pos, use_cache=True)

                next_tokens = []
                for b in range(B):
                    nxt = self._sample_token(logits[b, 0], temperature, top_k, top_p,
                                             repetition_penalty, result_ids[b])
                    result_ids[b].append(nxt)
                    next_tokens.append(nxt)

        self.model.reset_cache()
        return [self.decode(rids) for rids in result_ids]

    def _sample_token(self, logits: torch.Tensor,
                      temperature: float, top_k: int, top_p: float,
                      repetition_penalty: float = 1.0,
                      recent_ids: list[int] | None = None) -> int:
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
        checkpoint = {
            "model_state":  self.model.state_dict(),
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
        }
        torch.save(checkpoint, path)

    def load_model(self, path: str):
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
