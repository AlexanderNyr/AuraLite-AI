"""LoRA adapters of the model in memory (2.16): merge, save alone, load again.

A LoRA run (``lora_rank`` in the training params) trains small adapter
matrices next to frozen weights. :meth:`LoRAMixin.merge_lora` folds them into
the weights (a plain model again, as fast as before the adapters);
:meth:`LoRAMixin.save_adapter` writes only what the run trained — the
adapters, the rows of the token embedding and output head for tokens the run
added (a chat template's markers), the tokenizer and the chat template — to a
file a few percent the size of the model; :meth:`LoRAMixin.load_adapter` puts
such a file on its base model again. The file records a fingerprint of the
base weights, so an adapter is not put on a model it was not trained on.

The file is a ``torch.save`` dict of tensors and plain values (it loads with
``weights_only=True``)::

    format "auralite-lora", version 1, lora_config (rank, targets, alpha,
    rslora, base_vocab, embedding_rows), adapters (state-dict entries),
    vocab_size, rows_from, embedding_rows, head_rows (None when tied),
    tokenizer, chat_template, base_sha256, auralite_version
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from typing import Any

import torch

from auralite.engine.safe_io import safe_torch_load
from auralite.engine.tokenizers import tokenizer_from_dict

logger = logging.getLogger("auralite")

__all__ = ["ADAPTER_FORMAT", "ADAPTER_VERSION", "LoRAMixin", "base_fingerprint"]

#: The ``format`` field of an adapter file, and its layout version.
ADAPTER_FORMAT = "auralite-lora"
ADAPTER_VERSION = 1


def base_fingerprint(model: Any, rows: int) -> str:
    """sha256 of the weights a LoRA adapter sits on: every parameter that is
    not an adapter, of the token embedding and output head the first `rows`
    rows (the base model's tokens)."""
    digest = hashlib.sha256()
    for name, param in sorted(model.named_parameters(), key=lambda item: item[0]):
        if ".lora." in name:
            continue
        tensor = param.detach()
        if name.split(".")[0] in ("embedding", "head"):
            tensor = tensor[:rows]
        tensor = tensor.contiguous().cpu()
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype};".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tied(model: Any) -> bool:
    return model.head.weight is model.embedding.weight


class LoRAMixin:
    """`AuraLiteEngine` methods for LoRA adapters (2.16)."""

    model: Any
    tokenizer: Any
    vocab_size: int
    default_chat_template: str

    def _lora_model(self) -> Any:
        model = getattr(self.model, "module", self.model)
        if model is None or not hasattr(model, "lora_pairs"):
            raise ValueError("no AuraLite model is loaded (LoRA works on native .pt models)")
        return model

    def has_lora(self) -> bool:
        """True when the model in memory has LoRA adapters."""
        model = getattr(self.model, "module", self.model)
        return bool(getattr(model, "lora_adapters", None))

    def merge_lora(self) -> int:
        """Fold the LoRA adapters into the weights; returns how many matrices changed.

        The model computes the same function afterwards (up to rounding), as a
        plain model: no adapters, every parameter trainable, the speed of a
        model without LoRA. Save it with ``save_model``.
        """
        model = self._lora_model()
        if not model.lora_adapters:
            raise ValueError("the model has no LoRA adapters to merge")
        merged = int(model.merge_lora())
        logger.info(f"[AuraLite] LoRA merged into {merged} weight matrices; the model has no "
                    f"adapters now")
        return merged

    def save_adapter(self, path: str) -> dict:
        """Write the LoRA adapters and the rows of new tokens to `path` (2.16).

        Returns ``{"path", "bytes", "adapters", "params", "new_rows"}``.
        """
        model = self._lora_model()
        if not model.lora_adapters:
            raise ValueError("the model has no LoRA adapters (train with lora_rank > 0)")
        config = dict(model.lora_config or {})
        vocab = int(model.embedding.num_embeddings)
        # The rows of the base model's tokens stay as they were unless the run
        # trained every row (or is a 2.15 one, which did): then the file holds all.
        rows_from = 0
        if "base_vocab" in config and config.get("embedding_rows", "new") == "new":
            rows_from = min(int(config["base_vocab"] or 0), vocab)
        adapters = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    if ".lora." in k}
        embedding_rows = model.embedding.weight.detach()[rows_from:].cpu().clone()
        head_rows = None if _tied(model) else model.head.weight.detach()[rows_from:].cpu().clone()
        from auralite import __version__
        data = {
            "format": ADAPTER_FORMAT, "version": ADAPTER_VERSION,
            "lora_config": config, "adapters": adapters,
            "vocab_size": vocab, "rows_from": rows_from,
            "embedding_rows": embedding_rows, "head_rows": head_rows,
            "tokenizer": self.tokenizer.to_dict() if self.tokenizer is not None else None,
            "chat_template": self.default_chat_template,
            "base_sha256": base_fingerprint(model, rows_from),
            "auralite_version": __version__,
        }
        folder = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp_path = tempfile.mkstemp(prefix=".adapter-", suffix=".tmp", dir=folder)
        os.close(fd)
        try:
            torch.save(data, tmp_path)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        params = sum(v.numel() for v in adapters.values()) + embedding_rows.numel() + (
            head_rows.numel() if head_rows is not None else 0)
        info = {"path": str(path), "bytes": os.path.getsize(path),
                "adapters": len(model.lora_pairs()), "params": int(params),
                "new_rows": vocab - rows_from}
        logger.info(f"[AuraLite] LoRA adapter saved: {path} ({info['bytes'] / 1e6:.2f} MB, "
                    f"{info['adapters']} adapters, {info['new_rows']} new token rows)")
        return info

    def load_adapter(self, path: str, *, check_base: bool = True, merge: bool = False) -> dict:
        """Put the LoRA adapter file `path` on the model in memory, its base (2.16).

        The model must be the one the adapter was trained on (the file's
        fingerprint of the base weights must match; ``check_base=False`` skips
        that). Adds the adapter's token rows, tokenizer and chat template;
        ``merge=True`` then folds the adapters into the weights. Returns
        ``{"adapters", "new_rows", "merged"}``.
        """
        data = safe_torch_load(path, map_location="cpu")
        if not isinstance(data, dict) or data.get("format") != ADAPTER_FORMAT:
            raise ValueError(f"{path} is not an AuraLite LoRA adapter file (save_adapter or "
                             f"`auralite lora save` writes one)")
        if int(data.get("version", 0)) > ADAPTER_VERSION:
            raise ValueError(f"{path} was written by a newer AuraLite "
                             f"({data.get('auralite_version')}); update AuraLite to load it")
        model = self._lora_model()
        if model.lora_adapters:
            raise ValueError("the model already has LoRA adapters: merge them first "
                             "(merge_lora) or load the base model")
        rows_from, vocab = int(data["rows_from"]), int(data["vocab_size"])
        have = int(model.embedding.num_embeddings)
        if not rows_from <= have <= vocab:
            raise ValueError(f"the adapter was trained on a model with {rows_from:,} token rows "
                             f"(grown to {vocab:,}); this one has {have:,}")
        if check_base and base_fingerprint(model, rows_from) != data["base_sha256"]:
            raise ValueError("this adapter was trained on another base model (its weights "
                             "differ from the ones loaded); load that model, or pass "
                             "check_base=False to put the adapter on this one anyway")
        weight = model.embedding.weight
        if vocab > have:
            model.resize_token_embeddings(vocab)
        with torch.no_grad():
            model.embedding.weight[rows_from:] = data["embedding_rows"].to(
                device=weight.device, dtype=weight.dtype)
            if data.get("head_rows") is not None and not _tied(model):
                model.head.weight[rows_from:] = data["head_rows"].to(
                    device=model.head.weight.device, dtype=model.head.weight.dtype)
        config = dict(data["lora_config"])
        model.enable_lora(rank=int(config["rank"]), target_modules=config.get("targets"),
                          alpha=config.get("alpha"), rslora=bool(config.get("rslora", False)))
        model.lora_config.update({key: config[key] for key in ("base_vocab", "embedding_rows")
                                  if key in config})
        expected = {k for k in model.state_dict() if ".lora." in k}
        if set(data["adapters"]) != expected:
            model.disable_lora()
            raise ValueError("the adapter's matrices do not fit this model's layers")
        model.load_state_dict(data["adapters"], strict=False)
        model.to(weight.device)
        model.eval()
        if data.get("tokenizer") is not None:
            self.tokenizer = tokenizer_from_dict(data["tokenizer"])
        self.vocab_size = vocab
        if data.get("chat_template"):
            self.default_chat_template = str(data["chat_template"])
        # The optimizer state of the base model does not fit the new parameters.
        for name in ("_resume_optimizer_state", "_resume_scheduler_state",
                     "_resume_scaler_state", "optimizer", "scheduler", "scaler"):
            if hasattr(self, name):
                setattr(self, name, None)
        info = {"adapters": len(model.lora_pairs()), "new_rows": vocab - have, "merged": False}
        logger.info(f"[AuraLite] LoRA adapter loaded: {path} ({info['adapters']} adapters, "
                    f"{info['new_rows']} new token rows)")
        if merge:
            self.merge_lora()
            info["merged"] = True
        return info
