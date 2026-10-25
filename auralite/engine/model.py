"""model.py — Model definitions: `ModernTransformer` (LLaMA-style decoder) and LoRA adapters.

Extracted verbatim from `auralite/engine/_legacy.py` (v2.6.5); `_legacy` re-exports
every name below, so `auralite.engine.<name>` keeps working.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import logging
from typing import cast

from .layers import Attention, RMSNorm, TransformerBlock

logger = logging.getLogger("auralite")


def estimate_n_params(vocab_size: int, d_model: int, n_layers: int,
                      d_ff: int, n_heads: int,
                      n_kv_heads: int | None = None) -> int:
    """Rough parameter count for a LLaMA-style decoder-only transformer.

    Mirrors `ModernTransformer`, which by default keeps the input embedding and
    the output LM head as two independent matrices (untied), so both are counted.
    Accurate to within ~3% of `model.count_parameters()` for typical configs.
    """
    n_kv = n_kv_heads if (n_kv_heads and n_kv_heads > 0) else n_heads
    head_dim = d_model // max(1, n_heads)

    # Input embedding + separate output head (untied by default).
    embed = vocab_size * d_model        # token embedding
    head = vocab_size * d_model         # output projection (LM head)

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
    return embed + head + n_layers * per_layer + final_norm

class ModernTransformer(nn.Module):
    """Modern decoder-only Transformer (LLaMA-style, 2025-2026).

    Key architectural choices
    ─────────────────────────
    • RMSNorm  (Pre-norm)          — stable, efficient
    • RoPE     (Rotary Pos. Emb.)  — generalises to unseen lengths
    • SwiGLU   activation          — better than plain ReLU / GELU
    • Multi-Head Attn w/ opt. GQA  — flexible efficiency
    • Flash Attention (SDPA)       — fused, memory-efficient kernels
    • Untied input/output (default) — the token embedding (read/context) and the
      LM head (write/classification) are two mathematically distinct layers.
      Sharing one matrix forces it to both cluster synonyms (for the input) and
      separate them (for the output); the opposing gradients partially cancel,
      which slows convergence. They are kept as independent parameters by
      default. Legacy weight tying is still available via
      ``tie_word_embeddings=True`` / ``tie_weights()``.
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
                 tie_word_embeddings: bool = False,
                 use_qk_norm: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        # n_kv_heads == 0 means "no GQA" (use MHA, i.e. n_kv_heads = n_heads).
        # Normalise here so the %-modulo below never divides by zero.
        if n_kv_heads in (0, "0"):
            n_kv_heads = None
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
        self.use_qk_norm = bool(use_qk_norm)

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
                             num_experts=num_experts,
                             use_qk_norm=use_qk_norm)
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
        self.lora_config: dict | None = None      # 2.16: rank, targets, alpha, rslora

        # 2.11: generation attends to the trained length at most (see
        # `default_context_window`); `set_context_window(None)` lifts it.
        self.context_window: int | None = None
        self.set_context_window(self.default_context_window())

    # ---- Context window (2.11) -------------------------------------------
    def default_context_window(self) -> int | None:
        """The inference window a model of this configuration gets by default.

        RoPE attention scores depend on the distance between tokens, and a
        model trained on ``max_seq_len``-token windows has never seen a longer
        distance. Past that length 2.10 generated noise (docs/engine.md,
        "Context length"); attending to the newest ``max_seq_len`` positions
        keeps every distance inside the trained range. A model trained with its
        own (shorter) ``sliding_window`` already has a window: ``None``.
        """
        if self.sliding_window and self.sliding_window <= int(self.max_seq_len or 0):
            return None
        return int(self.max_seq_len) if self.max_seq_len else None

    def set_context_window(self, window: int | None) -> None:
        """Attend to at most the `window` newest positions outside training.

        ``None`` (or 0) removes the limit, which is the 2.10 behaviour. The
        window never applies in ``train()`` mode and changes nothing for
        sequences that fit into it. Drops every layer's KV cache.
        """
        value = int(window) if window else None
        if value is not None and value < 1:
            raise ValueError(f"context window must be a positive number of tokens, got {window!r}")
        self.context_window = value
        for attn in self._attention_layers():
            attn.context_window = value
            attn.reset_cache()

    def extend_context(self, max_seq_len: int) -> bool:
        """Raise the trained length to `max_seq_len` before training on it (2.11).

        Continued training on longer windows ("train short, extend long")
        calls this: the RoPE tables grow, and a context window that followed
        the old length follows the new one. RoPE scaling settings that derive
        their original length from ``max_seq_len`` are pinned to the old
        length, so the rotary frequencies the model was trained with do not
        change underneath it. Returns False when nothing had to change.
        """
        new_len = int(max_seq_len)
        old_len = int(self.max_seq_len or 0)
        if new_len <= old_len:
            return False
        follow = self.context_window is not None and self.context_window == self.default_context_window()
        layers = self._attention_layers()
        scaling = dict(self.rope_scaling or {})
        kind = str(scaling.get("type", "none")).lower()
        if kind in ("ntk", "dynamic_ntk") and "ntk_base" not in scaling and layers:
            # NTK derives its base from max_seq_len: keep the trained one.
            scaling["ntk_base"] = layers[0]._ntk_base()
        if kind not in ("none", "") and "original_max_position_embeddings" not in scaling:
            scaling["original_max_position_embeddings"] = old_len
        self.rope_scaling = scaling
        self.max_seq_len = new_len
        for attn in layers:
            attn.rope_scaling = dict(self.rope_scaling or {})
            attn.max_seq_len = new_len
            device = attn.rope_cos.device
            attn._build_rope_buffers(new_len)
            attn.rope_cos = attn.rope_cos.to(device)
            attn.rope_sin = attn.rope_sin.to(device)
        if follow:
            self.set_context_window(self.default_context_window())
        return True

    def tie_weights(self):
        """Tie output projection and token embedding weights in-place.

        This is safe for autograd: both modules point at the same Parameter, so
        gradients are summed into one tensor before the optimizer step. Calling
        it after loading an older checkpoint preserves AuraLite's historic tied
        embedding behavior.
        """
        self.head.weight = self.embedding.weight
        self.tie_word_embeddings = True

    def resize_token_embeddings(self, vocab_size: int) -> int:
        """Grow the vocabulary to `vocab_size` tokens; returns how many were added (2.15).

        New tokens (the special tokens of a chat template, say) get the mean
        of the existing rows, in the token embedding and in the output head:
        the model then gives them an average logit instead of a random one,
        which keeps its predictions unchanged until training teaches them
        (Hewitt, 2021). Existing rows, the tying of the head to the embedding
        and the ``requires_grad`` flags stay as they were. A vocabulary never
        shrinks.
        """
        old = int(self.embedding.num_embeddings)
        new = int(vocab_size)
        if new <= old:
            return 0
        weight = self.embedding.weight
        embedding = nn.Embedding(new, self.d_model).to(device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            embedding.weight[:old] = weight
            embedding.weight[old:] = weight.mean(dim=0)
        embedding.weight.requires_grad_(weight.requires_grad)
        self.embedding = embedding
        if self.tie_word_embeddings:
            self.head.weight = self.embedding.weight
        else:
            head_weight = self.head.weight
            head = nn.Linear(self.d_model, new, bias=False).to(device=head_weight.device,
                                                               dtype=head_weight.dtype)
            with torch.no_grad():
                head.weight[:old] = head_weight
                head.weight[old:] = head_weight.mean(dim=0)
            head.weight.requires_grad_(head_weight.requires_grad)
            self.head = head
        return new - old

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
                start_pos: int = 0, use_cache: bool = False,
                capture: dict | None = None,
                logits_to_keep: int | None = None) -> torch.Tensor:
        """Returns logits (B, T, vocab_size) — for all positions by default.

        Training uses every position (dense next-token loss, nanoGPT-style).
        Generation only needs the newest position: ``logits_to_keep=1`` (2.8)
        runs the final norm and the LM head on the last position only, which
        skips a (T x d_model x vocab) matmul per prompt. Speculative decoding
        keeps as many positions as it verifies.

        ``capture`` is a diagnostic-only side channel: pass a dict and every
        layer stores its (detached) RoPE-rotated q/k under its index, which is
        what `AuraLiteEngine.attention_maps()` turns into an attention matrix
        for the GUI's heatmap. Default None keeps the hot path untouched.
        """
        h = self.embedding(x)
        for index, layer in enumerate(self.layers):
            layer_capture = capture.setdefault(index, {}) if capture is not None else None
            h = layer(h, start_pos, use_cache, layer_capture)
        if logits_to_keep is not None and 0 < int(logits_to_keep) < h.shape[1]:
            h = h[:, -int(logits_to_keep):]
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

    def reset_cache(self, capacity: int | None = None):
        """Drop every layer's KV cache.

        `capacity` (2.8, optional) preallocates storage for that many
        positions on the next write, so a generation of known length never
        reallocates (the cache grows geometrically otherwise).
        """
        for attn in self._attention_layers():
            attn.reset_cache(capacity)

    def _attention_layers(self) -> list[Attention]:
        return [cast(TransformerBlock, layer).attn for layer in self.layers]

    def truncate_cache(self, end: int) -> None:
        """Roll every layer's cache back to positions ``< end`` (2.8).

        Speculative decoding calls this after a rejected draft token instead
        of re-running the model over the whole prefix.
        """
        for attn in self._attention_layers():
            attn.truncate_cache(end)

    @property
    def cache_length(self) -> int:
        """Positions currently cached (the next token's position)."""
        layers = self._attention_layers()
        return layers[0].cache_length if layers else 0

    def kv_cache_bytes(self) -> int:
        """Memory held by the KV caches of all layers, in bytes."""
        return sum(attn.kv_cache_bytes() for attn in self._attention_layers())

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---- LoRA Support ------------------------------------------------
    def enable_lora(self, rank: int = 8, target_modules: list[str] | None = None, *,
                    targets: str = "ffn", alpha: float | None = None,
                    rslora: bool = False):
        """Enable LoRA adapters on the linear layers of every block.

        Freezes the whole base model and adds trainable low-rank adapters
        that the forward pass of the FFN (`layer.ffn.lora`) and, 2.16, of the
        attention (`layer.attn.lora`) actually applies. The adapters are
        submodules, so they move with `.to(device)` and live in the state dict.

        Args:
            rank: LoRA rank (default 8)
            target_modules: names of the projections to adapt, from
                ``LORA_TARGETS["all"]``; overrides `targets` (2.15 passed
                FFN names only).
            targets: 2.16, ``"ffn"`` (gate/up/down, the default and all 2.15
                did), ``"attn"`` (W_q/W_k/W_v/W_o) or ``"all"`` of them.
            alpha: scale numerator (default `rank`).
            rslora: scale ``alpha / sqrt(rank)`` instead of ``alpha / rank``.
        """
        if target_modules is None:
            target_modules = list(lora_target_names(targets))
        names = [str(n) for n in target_modules]
        unknown = sorted(set(names) - set(LORA_TARGETS["all"]))
        if unknown:
            raise ValueError(f"unknown LoRA target modules {unknown}: use names from "
                             f"{', '.join(LORA_TARGETS['all'])}")
        if self.lora_adapters:
            self.disable_lora()

        self.lora_rank = rank
        self.lora_config = {"rank": int(rank), "targets": names,
                            "alpha": float(alpha) if alpha else float(rank),
                            "rslora": bool(rslora)}
        # plain list (NOT nn.ModuleList) so the adapters are registered only
        # once — via layer.ffn.lora / layer.attn.lora — and don't appear twice
        # in state_dict.
        self.lora_adapters = []

        # Freeze every base-model parameter
        for param in self.parameters():
            param.requires_grad = False

        for layer in self.layers:
            # nn.ModuleDict keys must NOT contain dots — use the plain
            # projection name ("gate", "W_q", ...).
            for part, part_names in (("ffn", LORA_TARGETS["ffn"]), ("attn", LORA_TARGETS["attn"])):
                wanted = [n for n in names if n in part_names]
                block = getattr(layer, part, None)
                if not wanted or block is None:
                    continue
                layer_lora = nn.ModuleDict()
                for name in wanted:
                    mod = getattr(block, name, None)
                    if isinstance(mod, nn.Linear):
                        # LoRA params are created with requires_grad=True by default
                        layer_lora[name] = LoRALayer(mod.in_features, mod.out_features, rank,
                                                     alpha=alpha, rslora=rslora).to(
                            device=mod.weight.device)
                # wire the adapters into the block so forward() uses them
                # (this also registers them as proper submodules of the model)
                block.lora = layer_lora
                self.lora_adapters.append(layer_lora)

    def disable_lora(self):
        """Disable LoRA and restore full training."""
        self.lora_rank = 0
        self.lora_adapters = None
        self.lora_config = None
        for layer in self.layers:
            for part in ("ffn", "attn"):
                block = getattr(layer, part, None)
                if block is not None and hasattr(block, "lora"):
                    block.lora = None
        for param in self.parameters():
            param.requires_grad = True

    def lora_pairs(self) -> list[tuple[str, nn.Linear, LoRALayer]]:
        """``(name, base linear, adapter)`` of every LoRA adapter (2.16)."""
        pairs = []
        for i, layer in enumerate(self.layers):
            for part in ("ffn", "attn"):
                block = getattr(layer, part, None)
                adapters = getattr(block, "lora", None) if block is not None else None
                for name, adapter in (adapters.items() if adapters is not None else ()):
                    pairs.append((f"layers.{i}.{part}.{name}", getattr(block, name), adapter))
        return pairs

    @torch.no_grad()
    def merge_lora(self) -> int:
        """Fold the adapters into the weights they adapt and remove them (2.16).

        ``W += scale * B @ A`` for every adapter; afterwards the model is a
        plain one (no LoRA, every parameter trainable) that computes the same
        function, without the adapters' extra matrix products. Returns how
        many weight matrices changed.
        """
        pairs = self.lora_pairs()
        for _name, linear, adapter in pairs:
            delta = adapter.delta_weight().to(device=linear.weight.device,
                                              dtype=linear.weight.dtype)
            linear.weight.add_(delta)
        if self.lora_adapters is not None:
            self.disable_lora()
        return len(pairs)

class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) adapter for a linear layer.

    Adds ``scale * (x @ A.T @ B.T)`` to the layer's output, ``scale = alpha /
    rank`` (``alpha / sqrt(rank)`` with `rslora`, 2.16). B starts at zero, so
    a new adapter does not change the model.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 alpha: float | None = None, rslora: bool = False):
        super().__init__()
        self.rank = rank
        self.alpha = alpha or rank  # default: alpha = rank (scaling = 1)
        self.rslora = bool(rslora)
        self.scaling = self.alpha / (math.sqrt(self.rank) if self.rslora else self.rank)

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        # lora_A: (rank, in_features), lora_B: (out_features, rank)
        # result: x @ lora_A.T @ lora_B.T * scaling
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        """The change this adapter makes to the weight it adapts (out, in)."""
        return (self.lora_B @ self.lora_A) * self.scaling


#: Projections a LoRA adapter can sit on (2.16): the FFN's and the attention's.
LORA_TARGETS = {"ffn": ("gate", "up", "down"), "attn": ("W_q", "W_k", "W_v", "W_o"),
                "all": ("gate", "up", "down", "W_q", "W_k", "W_v", "W_o")}


def lora_target_names(targets: str | list | tuple) -> tuple[str, ...]:
    """The projection names of `targets` ("ffn", "attn", "all" or a list of names)."""
    if isinstance(targets, (list, tuple)):
        return tuple(str(t) for t in targets)
    key = str(targets or "ffn").strip().lower()
    if key not in LORA_TARGETS:
        raise ValueError(f"lora_targets must be one of {', '.join(LORA_TARGETS)}, got {targets!r}")
    return LORA_TARGETS[key]


__all__ = ["LORA_TARGETS", "LoRALayer", "ModernTransformer", "estimate_n_params",
           "lora_target_names"]
