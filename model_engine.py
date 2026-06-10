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

import math
import re
from collections import Counter

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
#  Tokenizers — character-level and BPE (Byte/Char Pair Encoding)
# ===================================================================

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
    """

    kind = "bpe"

    def __init__(self):
        self.vocab: list[str] = []
        self.token_to_id: dict[str, int] = {}
        # ordered merge rules: (id_a, id_b) -> new_id, rank = list index
        self.merges: list[tuple[int, int, int]] = []
        self._ranks: dict[tuple[int, int], tuple[int, int]] = {}
        self._cache: dict[str, list[int]] = {}

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
        base_chars = sorted(set(text))
        self.vocab = list(base_chars)
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.merges = []

        if vocab_size <= len(self.vocab):
            self._build_ranks()
            return

        # word-frequency corpus: distinct pieces with counts
        piece_counts = Counter(self._split_pieces(text))
        corpus: list[tuple[list[int], int]] = [
            ([self.token_to_id[c] for c in piece], cnt)
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

    # ---- encode / decode ----------------------------------------------
    def _encode_piece(self, piece: str) -> list[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached

        fb = self.token_to_id.get(" ", 0)
        ids = [self.token_to_id.get(c, fb) for c in piece]
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
        return "".join(self.vocab[int(i)] if 0 <= int(i) < len(self.vocab) else "?"
                       for i in ids)

    # ---- (de)serialization ---------------------------------------------
    def to_dict(self) -> dict:
        return {"kind": self.kind, "vocab": self.vocab, "merges": self.merges}

    @classmethod
    def from_dict(cls, d: dict) -> "BPETokenizer":
        tok = cls()
        tok.vocab = list(d["vocab"])
        tok.token_to_id = {t: i for i, t in enumerate(tok.vocab)}
        tok.merges = [tuple(m) for m in d.get("merges", [])]
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
    """

    def __init__(self, d_model: int, n_heads: int,
                 n_kv_heads: int | None = None,
                 max_seq_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

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

        self.kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None

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
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif T == 1:
            # incremental decoding — the single query may attend to all keys
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            # general case (chunked decoding with cache)
            q_pos = torch.arange(start_pos, start_pos + T, device=x.device)
            k_pos = torch.arange(S, device=x.device)
            keep = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)         # True = attend
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)

    def reset_cache(self):
        self.kv_cache = None


# -------------------------------------------------------------------

class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network — standard in LLaMA / Qwen / Mistral."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# -------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: RMSNorm → Attention → RMSNorm → SwiGLU FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 n_kv_heads: int | None, max_seq_len: int,
                 dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn      = Attention(d_model, n_heads, n_kv_heads, max_seq_len)
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
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int,
                 max_seq_len: int = 4096,
                 n_kv_heads: int | None = None,
                 dropout: float = 0.0):
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

        self.embedding   = nn.Embedding(vocab_size, d_model)
        self.layers      = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, n_kv_heads, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm  = RMSNorm(d_model)
        self.head        = nn.Linear(d_model, vocab_size, bias=False)

        # Modern weight init (GPT-2 / LLaMA style)
        self.apply(self._init_weights)

        # Weight tying: output head shares the embedding matrix
        # (GPT-2 / LLaMA practice — fewer parameters, better generalisation)
        self.head.weight = self.embedding.weight

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
    """Modern training & inference engine for AuraLite AI v2."""

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
        """
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

        self.params_used = dict(params)

        resuming = bool(continue_training and self.model is not None
                        and self.tokenizer is not None)

        # ---- Tokenizer ------------------------------------------------
        if not resuming:
            if tok_kind == "bpe":
                self.tokenizer = BPETokenizer()
                # train merges on a capped sample for speed on huge files
                sample = training_text[:2_000_000]
                self.tokenizer.train(sample, vocab_size=bpe_vocab)
                # make sure every char of the full text is representable
                missing = sorted(set(training_text) - set(self.tokenizer.vocab))
                for ch in missing:
                    self.tokenizer.token_to_id[ch] = len(self.tokenizer.vocab)
                    self.tokenizer.vocab.append(ch)
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
            ).to(self.device)

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

        use_workers = (self.num_threads > 1) and (len(dataset) >= 5000)
        num_workers = min(self.num_threads, 4) if use_workers else 0
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
        train_model = self.model
        if use_compile:
            try:
                train_model = torch.compile(self.model)
            except Exception:
                train_model = self.model   # graceful fallback

        # ---- Epoch loop -----------------------------------------------
        self.model.train()
        self.last_val_loss = None
        for epoch in range(epochs):
            if stop_event and stop_event.is_set():
                break

            running_loss   = 0.0
            seen_batches   = 0
            stopped_mid    = False

            for xb, yb in loader:
                if stop_event and stop_event.is_set():
                    stopped_mid = True
                    break

                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    output = train_model(xb)                      # (B, T, vocab)
                    loss   = criterion(
                        output.reshape(-1, output.size(-1)),      # (B·T, vocab)
                        yb.reshape(-1),                           # (B·T,)
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                running_loss += loss.item()
                seen_batches += 1

            if stopped_mid:
                break

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

    def _sample_token(self, logits: torch.Tensor,
                      temperature: float, top_k: int, top_p: float,
                      repetition_penalty: float = 1.0,
                      recent_ids: list[int] | None = None) -> int:
        """Sample a single token id with temperature, repetition penalty,
        top-k, and top-p (nucleus) filtering."""
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
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
