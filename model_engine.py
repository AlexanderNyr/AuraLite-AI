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
    """Multi-Head Self-Attention with RoPE, optional GQA, and KV-cache."""

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
        end_pos = start_pos + T

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
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask — query at absolute position *p* may attend to keys at ≤ *p*
        q_pos = torch.arange(start_pos, end_pos,   device=x.device).unsqueeze(1)   # (T, 1)
        k_pos = torch.arange(S,                     device=x.device).unsqueeze(0)   # (1, S)
        mask  = k_pos > q_pos                                           # (T, S) True = masked
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = torch.softmax(scores.float(), dim=-1).type_as(q)
        out  = torch.matmul(attn, v)                                    # (B, nh, T, hd)
        out  = out.transpose(1, 2).contiguous().view(B, T, -1)
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
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h, start_pos, use_cache)
        h = self.final_norm(h)
        return self.head(h[:, -1, :])          # next-token prediction from last position

    def reset_cache(self):
        for layer in self.layers:
            layer.attn.reset_cache()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ===================================================================
#  Dataset
# ===================================================================

class CharDataset(Dataset):
    """Sliding-window character-level dataset.

    Stores the full encoded text as a single tensor and produces
    (input_seq, next_char) samples on the fly.
    """

    def __init__(self, encoded: torch.Tensor, seq_length: int):
        self.data = encoded
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + self.seq_length]
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
        self.chars: list[str]         = []
        self.char_to_idx: dict         = {}
        self.idx_to_char: dict         = {}
        self.vocab_size  = 0
        self.params_used: dict         = {}     # remember last training params

    # ---- Tokenisation -----------------------------------------------
    def encode(self, s: str) -> list[int]:
        fallback = self.char_to_idx.get(" ", 0)
        return [self.char_to_idx.get(c, fallback) for c in s]

    def decode(self, ids) -> str:
        return "".join(self.idx_to_char.get(int(i), "?") for i in ids)

    # ---- Training ----------------------------------------------------
    def train(self, training_text: str, params: dict,
              progress_callback=None, stop_event=None):

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

        self.params_used = dict(params)

        # ---- Vocabulary -----------------------------------------------
        self.chars = sorted(list(set(training_text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # ---- Model ----------------------------------------------------
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
        dataset = CharDataset(encoded, seq_length)
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
        total_steps  = epochs * len(loader)
        warmup_steps = min(200, total_steps // 10)
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, warmup_steps, total_steps, min_lr=lr * 0.1
        )

        # ---- Epoch loop -----------------------------------------------
        self.model.train()
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

                with torch.amp.autocast("cuda", enabled=use_amp):
                    output = self.model(xb)
                    loss   = criterion(output, yb)

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

            if progress_callback and seen_batches > 0:
                avg_loss = running_loss / seen_batches
                progress_callback(epoch + 1, epochs, avg_loss)

    # ---- Generation ---------------------------------------------------
    def generate(self, start_str: str, length: int = 50,
                 temperature: float = 0.8,
                 top_k: int = 50, top_p: float = 0.9) -> str:

        if self.model is None:
            raise ValueError("Train or load a model first!")

        self.model.eval()
        self.model.reset_cache()

        seed_ids = self.encode(start_str)
        result: list[str] = list(start_str)

        with torch.no_grad():
            # --- Process full seed in one pass (no padding needed) ------
            ids_tensor = torch.tensor([seed_ids], dtype=torch.long).to(self.device)
            logits = self.model(ids_tensor, start_pos=0, use_cache=True)
            result.append(self._sample_token(logits[0], temperature, top_k, top_p))

            # --- Generate remaining tokens one-by-one (KV-cache) --------
            for i in range(length - 1):
                last_id = self.encode(result[-1])
                ids_tensor = torch.tensor([[last_id[0]]], dtype=torch.long).to(self.device)
                logits = self.model(
                    ids_tensor,
                    start_pos=len(seed_ids) + i,
                    use_cache=True,
                )
                result.append(self._sample_token(logits[0], temperature, top_k, top_p))

        self.model.reset_cache()
        return "".join(result)

    def _sample_token(self, logits: torch.Tensor,
                      temperature: float, top_k: int, top_p: float) -> str:
        """Sample a single token with temperature, top-k, and top-p (nucleus)."""
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

        probs   = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        return self.idx_to_char[next_id]

    # ---- Save / Load --------------------------------------------------
    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("No model to save!")
        checkpoint = {
            "model_state":  self.model.state_dict(),
            "vocab_size":   self.vocab_size,
            "chars":        self.chars,
            "char_to_idx":  self.char_to_idx,
            "idx_to_char":  {int(k): v for k, v in self.idx_to_char.items()},
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

        self.chars        = checkpoint["chars"]
        self.vocab_size   = checkpoint["vocab_size"]
        self.char_to_idx  = checkpoint["char_to_idx"]
        self.idx_to_char  = {int(k): v for k, v in checkpoint["idx_to_char"].items()}
        self.params_used  = checkpoint.get("params_used", {})

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
