"""Training: :class:`TrainerMixin` (moved out of ``_legacy.py`` in 2.14).

``AuraLiteEngine.train`` and the training loop behind it, ``_train_impl``,
with their helpers (validation passes, progress callbacks, the corpus cache
and AMP scaler steps). The code moved unchanged; the methods are reached
through ``AuraLiteEngine`` as before.

2.15: ``train`` also takes a :class:`~auralite.engine.corpus.TextCorpus`
(several documents, validated on whole held-out documents) and a
:class:`~auralite.engine.dialogs.DialogSet` (chat fine-tuning: the loss
learns the assistant's replies only).
"""
from __future__ import annotations

import math
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, cast
from uuid import uuid4

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from auralite.engine import cluster as _cluster
from auralite.engine import compute_thread as _compute
from auralite.engine import dialogs as _dialogs
from auralite.engine.corpus import TextCorpus
from auralite.engine.dataset import (
    IGNORE_INDEX,
    CharDataset,
    WindowBatches,
    spread_indices,
    validation_split_point,
)
from auralite.engine.dialogs import DialogSet
from auralite.engine.model import ModernTransformer, estimate_n_params
from auralite.engine.optimizers import (
    Muon,
    _ChainedOptimizers,
    adamw_param_groups,
    split_parameters_for_muon,
)
from auralite.engine.params import (
    ParamValidationError,
    describe_unknown_params,
    unknown_train_params,
    validate_params,
)
from auralite.engine.layers import ResidualDropout
from auralite.engine.recipe import (
    AUTO_DROPOUT,
    AUTO_DROPOUT_MAX_PASSES,
    DEFAULT_RECIPE,
    auto_dropout,
    is_auto,
)
from auralite.engine.scheduler import CosineWarmupScheduler, WSDScheduler
from auralite.engine.tokenizers import BPETokenizer, CharTokenizer, tokenizer_from_dict
from auralite.engine import tokenizer_io as _tok_io
from auralite.engine.train_control import TrainControl

import logging

logger = logging.getLogger("auralite")

__all__ = ["TrainerMixin"]


class _SetupStopped(Exception):
    """Stop or finish pressed before the first training step (2.14.1)."""

    def __init__(self, stage: str):
        super().__init__(stage)
        self.stage = stage


#: Engine attributes a run changes before its first step; a run stopped then
#: puts them back, so the model in memory stays usable (2.14.1).
_SETUP_STATE = ("tokenizer", "vocab_size", "model", "optimizer", "optimizer_name", "scheduler",
                "scaler", "params_used", "backend", "gguf_path", "hf_path",
                "default_chat_template", "last_val_loss", "last_val_bpb", "_ddp_model")


class _ReplyLoss(nn.Module):
    """Cross-entropy over the learned targets only (2.15, dialogs).

    The mean over the targets that are not ``IGNORE_INDEX``, like
    ``nn.CrossEntropyLoss``, but 0 (with a gradient of zeros) for a batch
    without any, where ``nn.CrossEntropyLoss`` returns NaN: a window can lie
    entirely inside a long user turn.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        learned = (targets != IGNORE_INDEX).sum()
        total = F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX, reduction="sum")
        return total / learned.clamp(min=1)


def _undo_resize(model: Any, embedding: nn.Module, head: nn.Module) -> None:
    """Put back the embedding and head a `resize_token_embeddings` replaced (2.15)."""
    model.embedding = embedding
    model.head = head
    if getattr(model, "tie_word_embeddings", False):
        model.head.weight = embedding.weight


def _remove_file_quietly(path: str) -> None:
    """atexit fallback for `AuraLiteEngine._release_corpus_tempfile`."""
    try:
        os.remove(path)
    except OSError:
        pass


class TrainerMixin:
    """``train()`` and the training loop of :class:`~auralite.engine.AuraLiteEngine`."""

    def _scaler_unscale_optimizer(self):
        """GradScaler.unscale_ that also supports chained (Muon + AdamW) optimizers."""
        if isinstance(self.optimizer, _ChainedOptimizers):
            for opt in self.optimizer.optimizers:
                self.scaler.unscale_(opt)
        else:
            self.scaler.unscale_(self.optimizer)

    def _scaler_step_optimizer(self):
        """GradScaler.step that also supports chained (Muon + AdamW) optimizers."""
        if isinstance(self.optimizer, _ChainedOptimizers):
            for opt in self.optimizer.optimizers:
                self.scaler.step(opt)
        else:
            self.scaler.step(self.optimizer)

    @staticmethod
    def _move_optimizer_state_to_device(optimizer, device: torch.device):
        optimizers = optimizer.optimizers if isinstance(optimizer, _ChainedOptimizers) else [optimizer]
        for opt in optimizers:
            for state in opt.state.values():
                for key, value in list(state.items()):
                    if torch.is_tensor(value):
                        state[key] = value.to(device)

    # ---- Validation ----------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, loader, criterion, max_batches: int | None = None,
                  on_batch: Callable[[int, int], None] | None = None) -> float | None:
        """Mean per-token cross-entropy on `loader`, or None if it is empty.

        Batches are weighted by their token count (2.8 averaged batch means, so a
        short last batch counted as much as a full one). ``on_batch(done,
        total)`` follows the batches (2.14.1, the progress bar). 2.15: only
        targets that are not ``IGNORE_INDEX`` count (dialogs: the replies).
        """
        self.model.eval()
        total, count = 0.0, 0
        try:
            n_batches = len(loader) if hasattr(loader, "__len__") else 0
            if max_batches is not None and n_batches:
                n_batches = min(n_batches, max_batches)
            for i, (xb, yb) in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                out = self.model(xb)
                loss = criterion(out.reshape(-1, out.size(-1)), yb.reshape(-1))
                # 2.15: dialogs learn only some targets; plain text learns all
                learned = int((yb != IGNORE_INDEX).sum())
                total += loss.item() * learned
                count += learned
                if on_batch is not None:
                    on_batch(i + 1, max(n_batches, i + 1))
        finally:
            self.model.train()
        cluster = getattr(self, "_cluster", None)
        if cluster is not None:            # 2.14: each computer validated its share
            total, count_f = cluster.allreduce_sum([total, float(count)])
            count = int(count_f)
        if count == 0:
            return None
        return total / count

    # ---- Training ----------------------------------------------------
    @staticmethod
    def _invoke_progress(progress_callback, epoch, total_epochs, train_loss,
                         val_loss=None, info=None):
        """Call a progress callback with optional live-training `info` dict.

        Compatible with both:
          cb(epoch, total, loss, val_loss)          — legacy: only on epoch end
          cb(epoch, total, loss, val_loss, info)    — live batch/setup updates

        `info` keys (all optional): phase, message, batch, batches, global_step,
        total_steps, percent, epoch_percent, lr, samples_per_sec, tokens_per_sec,
        elapsed, eta_seconds, is_epoch_end, epoch_loss.
        """
        if progress_callback is None:
            return
        if info is None:
            info = {}
        mode = getattr(progress_callback, "_auralite_progress_mode", None)
        if mode is None:
            # Probe once: prefer the 5-arg live API, fall back to legacy 4-arg.
            try:
                progress_callback(epoch, total_epochs, train_loss, val_loss, info)
                try:
                    progress_callback._auralite_progress_mode = "info"
                except Exception:
                    pass
                return
            except TypeError:
                try:
                    progress_callback._auralite_progress_mode = "legacy"
                except Exception:
                    pass
                mode = "legacy"
        if mode == "info":
            progress_callback(epoch, total_epochs, train_loss, val_loss, info)
            return
        # Legacy 4-arg: keep the historic contract — one call per finished epoch.
        if info.get("is_epoch_end") and int(epoch) > 0:
            progress_callback(epoch, total_epochs, train_loss, val_loss)

    def _iter_token_chunks(self, text: str, chunk_chars: int = 2_000_000,
                           overlap_chars: int = 256,
                           on_progress: Callable[[int], None] | None = None):
        """Yield token ids in chunks — the mmap corpus path never holds them all.

        Each chunk is encoded with a small overlap before its start; the tokens
        produced by that overlap are dropped again, so chunk boundaries do not
        duplicate text. For the char tokenizer this is *exact* (a char is a
        token); for BPE a few boundary merges can differ from encoding the whole
        text at once, which is the documented trade-off of streaming corpora.
        ``on_progress(characters done)`` follows the chunks (2.14.1).
        """
        if not text:
            return
        step = max(1, int(chunk_chars))
        overlap = max(0, int(overlap_chars))
        for start in range(0, len(text), step):
            left = max(0, start - overlap)
            chunk = text[left:start + step]
            ids = self.encode(chunk)
            if start > 0 and left != start:
                prefix_ids = self.encode(text[left:start])
                ids = ids[len(prefix_ids):]
            if on_progress is not None:
                on_progress(min(len(text), start + step))
            yield ids

    def _apply_auto_dropout(self, passes: float, *, bound: bool) -> float:
        """Set the residual dropout ``"auto"`` picks for `passes` (2.14).

        `bound`: `passes` is only an upper bound so far (a time budget may
        end the run sooner), so dropout is only switched off, never on.
        """
        chosen = auto_dropout(passes)
        model = getattr(self.model, "module", self.model)
        if chosen > 0.0 and bound:
            return AUTO_DROPOUT
        layers = [m for m in model.modules() if isinstance(m, ResidualDropout)]
        if not layers:
            chosen = 0.0          # a continued model built without dropout keeps none
        for module in layers:
            module.p = chosen
        model.dropout = chosen
        logger.info(f"[AuraLite] Dropout auto: {chosen} (the plan reads the training text "
                    f"{passes:.1f} times; dropout above {AUTO_DROPOUT_MAX_PASSES})")
        return chosen

    def train(self, training_text: str | TextCorpus | DialogSet, params: dict,
              progress_callback=None, stop_event=None, finish_event=None, tokenizer=None):
        """Train (or continue training) the model.

        `training_text` (2.15): a text, a :class:`TextCorpus` of several
        documents or a :class:`DialogSet` of chat dialogs, as
        ``auralite.engine.load_training_data(paths)`` reads them from files,
        folders and globs. A corpus is validated on whole documents held out
        at random (``val_split`` of the text) and trained on the others,
        joined with blank lines. Dialogs fine-tune a chat model: each is
        written with the ``chat_template`` (default "chatml", or the
        continued model's own), whose markers become special tokens (a
        continued model gets rows for them), and the loss learns only the
        assistant's replies and the marker that ends them, so the model
        answers in the template and stops by itself; ``val_split`` of the
        dialogs are held out.

        `stop_event` (a ``threading.Event``) stops at the next batch;
        `finish_event` (2.12) ends the run well instead: the learning rate
        decays for ``cooldown_ratio`` x the steps taken so far, then the run
        ends with ``stop_reason="finished"``. The GUI's first Stop and the
        CLI's first Ctrl+C finish, the second one stops. Either one before the
        first training step (2.14.1: while the tokenizer trains or the corpus
        is tokenized) ends the run at once and puts the engine back as it
        was: the summary then has ``stopped_during`` (the stage) and
        ``epochs_completed`` 0.

        `tokenizer` (2.14.1): train a new model with this tokenizer instead
        of building one (see also the ``tokenizer_path`` parameter).

        progress_callback(epoch, total_epochs, train_loss, val_loss_or_None, info=None)

        The optional 5th argument `info` is a dict with live batch/ETA fields so
        UIs can update the progress bar *between* epochs. Old 4-arg callbacks
        still work and are only invoked once per finished epoch.

        info keys: phase (setup|train|val|epoch_end|stopped), message, batch,
        batches, global_step, total_steps, percent, epoch_percent, lr,
        tokens_per_sec, batches_per_sec, elapsed, eta_seconds, is_epoch_end.
        2.14.1, the stage of the work, in setup, train and val events:
        stage (tokenizer|tokenize|model|data|compile|train|val), stage_label,
        stage_step, stage_done, stage_total, stage_unit, stage_percent,
        stage_rate (units per second), stage_eta_seconds, stage_elapsed.

        Throttle knobs in params:
          progress_every_batches  : emit at least every N batches (default ~1%)
          progress_every_seconds  : also emit at least every N seconds (default 0.5)

        Defaults not listed come from ``auralite.engine.recipe.DEFAULT_RECIPE``
        (2.12: the GUI and the CLI read the same recipe): Muon lr 0.02 with
        QK-norm (AdamW lr 1e-3 for the embeddings, norms and head), WSD,
        BPE 512, batch 32, dropout 0.1, early stopping after 3 epochs. A
        continued model keeps the optimizer it was trained with unless
        ``optimizer`` is given.

        params (beyond architecture/optimizer):
          tokenizer        : "bpe" (default since 2.12) or "char"
          bpe_vocab_size   : target BPE vocab (default 512)
          tokenizer_path   : train with the tokenizer in this file instead
                             (2.14.1): an AuraLite tokenizer file
                             (`auralite tokenizer train`), a Hugging Face
                             tokenizer.json (BPE), GPT-2 vocab.json +
                             merges.txt, or a checkpoint; `tokenizer` and
                             `bpe_vocab_size` are then not used
          val_split        : fraction of text held out for validation (default 0.1)
          use_compile      : try torch.compile for the training loop (default False)
          autosave_every   : autosave checkpoint every N epochs, 0 = off
          autosave_path    : where to autosave (default "aura_autosave.pt")
          continue_training: keep existing model/tokenizer and fine-tune (default False)
          accumulation_steps: gradient accumulation (default 1, no accumulation)
          max_steps        : stop after N optimizer steps (default 0 = no limit);
                             the LR schedule is sized to the budget (2.8)
          sample_stride    : distance between the starts of training windows
                             (default seq_length: an epoch is one pass over the
                             training tokens, with a fresh random window offset
                             per epoch; 1 = every window, the 2.8 behaviour)
          val_max_tokens   : validation budget in tokens (default
                             50 * batch_size * seq_length), spread evenly over
                             the validation split in non-overlapping windows
          early_stopping_patience: stop after N epochs without a better
                             validation loss (default 3 since 2.12; 0 = off;
                             needs a validation split)
          restore_best_weights: end with the weights of the best validation
                             epoch (default: on when early stopping is on)
          early_stopping_cooldown: decay the learning rate for at most two
                             epochs before an early stop (default True, 2.13)
          dropout          : residual dropout, or "auto" (default, 2.14): 0
                             when the plan reads the training text at most 8
                             times (epochs, max_steps or the time budget),
                             0.1 otherwise
          muon_ns_steps    : Newton-Schulz iterations per Muon step (default 3,
                             2.14; 2.13 ran 5)
          cluster_workers  : train together with this many other computers
                             (2.14, default 0): wait for them on
                             cluster_port (default 29500), cluster_host
                             (default all interfaces), at most
                             cluster_timeout_s (default 600); they run
                             `auralite cluster-worker HOST`. batch_size
                             stays the batch of a step (split between the
                             computers); cluster_token (optional) must match
          time_budget_s    : end the run after this many seconds, counted from
                             the start of train(), with the LR decay sized to
                             end at the deadline (default 0 = no budget)
          cooldown_ratio   : length of the decay that ends a finished run, as
                             a fraction of the steps taken (default 0.2)

        Returns a summary dict (also kept as ``engine.last_train_summary``):
        epochs_completed, optimizer_steps, tokens, train_loss, val_loss,
        best_val_loss, val_bpb, best_val_bpb (bits per UTF-8 byte, 2.11),
        best_epoch, restored_best, stop_reason (None | "max_steps" |
        "early_stopping" | "finished" | "time_budget" | "stopped"),
        cooldown_steps, tokenizer_fit_chars (2.13: characters the new BPE
        was fitted on), train_tokens, val_tokens, dropout and
        dropout_planned_passes (2.14: the dropout used, and for "auto" how
        many times the plan reads the training text), cluster_size (2.14: the
        computers that trained), seconds, tokenizer_source (2.14.1:
        "trained", "file", "given", "main computer" or "continued") and
        tokenizer (a description), stopped_during (2.14.1, see above).
          use_alibi        : enable ALiBi attention bias (default False)
          lora_rank        : enable LoRA with given rank (default 0 = disabled);
                             2.15: also for a continued model (fine-tuning
                             with adapters). On dialogs the token embedding
                             and the output head train too (the template's
                             markers are new tokens)
          val_start_char   : 2.15, validation starts at this character of the
                             text (a TextCorpus sets it to its first held-out
                             document); default: the last val_split of it
          dialog_loss      : 2.15, dialogs only: "assistant" (default, the
                             replies) or "all" (every token, as plain text)

        2.15: dialogs add ``dialogs`` to the summary (train, val, template,
        loss, learned_tokens, added_tokens, skipped records), a TextCorpus
        ``documents`` and ``val_documents``. A dialog run starts a fresh
        optimizer and learning-rate schedule and runs on one computer.
        """
        # 2.13: training stays on this thread; an idle OpenMP team of the
        # compute thread would only slow it down (compute_thread.py).
        _compute.release_compute_pool()
        session = None
        distributed_before = (self.is_distributed, self.world_size, self.local_rank)
        state = {name: getattr(self, name, None) for name in _SETUP_STATE}
        started = time.time()
        # 2.15: changes to the model in memory that a stop before the first
        # step takes back (new token rows, LoRA adapters), newest last
        self._setup_undo: list[Callable[[], Any]] = []
        try:
            training_text, params, corpus_info = self._resolve_training_data(training_text,
                                                                             params)
            with _compute.run_inline():
                cluster = _cluster.cluster_params(params)
                if cluster["workers"] > 0 and getattr(self, "_cluster", None) is None:
                    # 2.14: wait for the workers, then hand them the run.
                    session = self._host_cluster(cluster, params, progress_callback,
                                                 stop_event)
                    job = dict(params)
                    if tokenizer is not None and not job.get("tokenizer_path"):
                        # The workers get the tokenizer itself later; the key
                        # only makes them cut the validation text alike.
                        job["tokenizer_path"] = "(the main computer's tokenizer)"
                    session.send_job(str(training_text), job)
                    self._cluster = session
                return self._train_impl(training_text, params, progress_callback,
                                        stop_event, finish_event, tokenizer=tokenizer,
                                        corpus_info=corpus_info)
        except _SetupStopped as stopped:
            for undo in reversed(self._setup_undo):
                undo()
            for name, value in state.items():
                setattr(self, name, value)
            logger.info(f"[AuraLite] Stopped before the first training step "
                        f"({stopped.stage}); the model in memory is unchanged.")
            self.last_train_summary = {
                "epochs_completed": 0, "optimizer_steps": 0, "tokens": 0,
                "train_loss": None, "val_loss": None, "best_val_loss": None,
                "val_bpb": None, "best_val_bpb": None, "best_epoch": None,
                "restored_best": False, "stop_reason": "stopped",
                "stopped_during": stopped.stage, "cooldown_steps": 0,
                "cluster_size": 1, "seconds": time.time() - started,
            }
            return dict(self.last_train_summary)
        finally:
            if session is not None:
                self._cluster = None
                self._ddp_model = None
                # The model stays on the CPU it was trained on.
                self.is_distributed, self.world_size, self.local_rank = distributed_before
                session.close()
            # Success, Ctrl+C/stop_event and exceptions alike: never leave the
            # anonymous corpus memmap behind (see `_release_corpus_tempfile`).
            self._release_corpus_tempfile()
            self._setup_undo = []
            _compute.release_caller_pool()

    def _resolve_training_data(self, data: Any, params: dict
                               ) -> tuple[str | DialogSet, dict, dict]:
        """``(text or dialogs, params, corpus info)`` for `train` (2.15).

        A :class:`TextCorpus` becomes its training text; when it holds whole
        documents out for validation, ``val_start_char`` in the returned
        params says where they start (a cluster's workers get it too).
        """
        if isinstance(data, DialogSet):
            if int(params.get("cluster_workers") or 0) > 0 \
                    or getattr(self, "_cluster", None) is not None:
                raise ValueError("chat fine-tuning on dialogs runs on one computer: "
                                 "train without cluster_workers")
            return data, params, {}
        if isinstance(data, TextCorpus):
            val_split = float(params.get("val_split", DEFAULT_RECIPE["val_split"]) or 0.0)
            held = data.validation_documents(val_split)
            text, start = data.layout(val_split)
            info = {"documents": len(data), "val_documents": len(held) if start is not None else 0}
            if start is not None:
                params = {**params, "val_start_char": start}
                logger.info(f"[AuraLite] Corpus: {len(data):,} documents; {len(held):,} of them "
                            f"({(len(text) - start) / max(1, len(text)):.1%} of the text) are "
                            f"held out whole for validation")
            elif len(data) > 1:
                logger.info(f"[AuraLite] Corpus: {len(data):,} documents; validation is the "
                            f"end of the text (no document split fits val_split={val_split:g})")
            return text, params, info
        if not isinstance(data, str):
            raise TypeError(f"train() takes a text, a TextCorpus or a DialogSet, not "
                            f"{type(data).__name__}")
        return data, params, {}

    # ---- Only the tokenizer (2.14.1) -----------------------------------
    def train_tokenizer(self, text: str | TextCorpus, vocab_size: int | None = None, *,
                        kind: str = "bpe",
                        pretokenizer: str = "gpt", val_split: float | None = None,
                        progress_callback: Callable[[dict], None] | None = None,
                        stop_event: Any = None, path: str | None = None,
                        progress_every_seconds: float = 0.5) -> Any:
        """Train only a tokenizer for `text` and return it (2.14.1).

        The engine, its model and its tokenizer stay as they are: train a
        model with the result through ``train(..., tokenizer=tok)`` or, once
        saved (`path`, or :meth:`save_tokenizer`), ``params["tokenizer_path"]``.
        `vocab_size` defaults to the recipe's 512 and `val_split` to its 0.1,
        so the validation text of a later ``train()`` on the same text stays
        unseen. ``progress_callback(info)`` gets the stage events of
        ``train()`` (``info["message"]``, ``info["stage_percent"]``, ...), at
        most every `progress_every_seconds`; `stop_event` ends the training
        with the merges learned so far. How it went:
        ``engine.last_tokenizer_info``. 2.15: `text` can be a
        :class:`TextCorpus`; the tokenizer is then fitted on the documents
        ``train()`` trains on, not on those it holds out.
        """
        split = DEFAULT_RECIPE["val_split"] if val_split is None else float(val_split)
        fit_chars = None
        if isinstance(text, TextCorpus):
            text, fit_chars = text.layout(split)
        tok, info = _tok_io.train_tokenizer(
            text, kind=kind, vocab_size=int(vocab_size or DEFAULT_RECIPE["bpe_vocab_size"]),
            pretokenizer=pretokenizer, val_split=split,
            progress=progress_callback,
            should_stop=stop_event.is_set if stop_event is not None else None,
            every_s=progress_every_seconds, fit_chars=fit_chars)
        self.last_tokenizer_info = info
        if path:
            _tok_io.save_tokenizer(tok, path, training=info)
        return tok

    def save_tokenizer(self, path: str, tokenizer: Any = None,
                       training: dict | None = None) -> str:
        """Save `tokenizer` (default: the model's) as an AuraLite tokenizer file (2.14.1)."""
        tok = tokenizer if tokenizer is not None else self.tokenizer
        if tok is None:
            raise ValueError("No tokenizer — train or load a model first!")
        return _tok_io.save_tokenizer(tok, path, training=training)

    @staticmethod
    def load_tokenizer(path: str) -> Any:
        """The tokenizer in a file, folder or checkpoint (2.14.1; the engine is not changed)."""
        return _tok_io.load_tokenizer(path)

    def _host_cluster(self, cluster: dict, params: dict, progress_callback: Any,
                      stop_event: Any) -> Any:
        """Main computer of a cluster run: open the port and wait for the workers."""
        epochs = int(params.get("epochs", 100) or 100)

        def status(message: str) -> None:
            logger.info(f"[AuraLite] {message}")
            self._invoke_progress(progress_callback, 0, epochs, 0.0, None, info={
                "phase": "cluster", "message": message, "percent": 0.0, "elapsed": 0.0,
                "is_epoch_end": False, "batch": 0, "batches": 0})

        return _cluster.host_cluster(cluster["workers"], port=cluster["port"],
                                     host=cluster["host"], timeout_s=cluster["timeout_s"],
                                     token=cluster["token"], on_status=status,
                                     stop_event=stop_event)

    def _release_corpus_tempfile(self) -> None:
        """Delete the temp-dir corpus memmap that `train()` created for this run.

        Only the anonymous ``corpus_cache="mmap"`` file is removed: its name is
        unique per run, so no later run could ever reuse it, and on a tmpfs
        ``/tmp`` it would pin RAM until reboot (a 1 GB char-level corpus becomes
        a uint32 file of up to 4 GB). An explicit ``corpus_cache_path`` is the
        user's file and is kept.
        """
        path = getattr(self, "_corpus_tmp_file", None)
        self._corpus_tmp_file = None
        if not path:
            return
        for attempt in (1, 2):
            try:
                os.remove(path)
                break
            except FileNotFoundError:
                break
            except OSError as exc:
                if attempt == 1:
                    # Windows refuses to delete a file that is still memory-mapped:
                    # drop memmap views that only a reference cycle keeps alive.
                    import gc
                    gc.collect()
                    continue
                # Still mapped (e.g. by a DataLoader worker that has not exited).
                import atexit
                logger.debug(f"[AuraLite] corpus memmap {path} still in use ({exc}); "
                             f"removing it at interpreter exit")
                atexit.register(_remove_file_quietly, path)
                return
        if getattr(self, "_corpus_cache_path", None) == path:
            self._corpus_cache_path = None
        logger.debug(f"[AuraLite] removed temporary corpus memmap {path}")

    def _train_impl(self, training_text: str | DialogSet, params: dict,
                    progress_callback=None, stop_event=None, finish_event=None,
                    tokenizer=None, corpus_info: dict | None = None):
        """Body of :meth:`train`, split out so its temp files are released on every exit."""
        # 2.15: dialogs (chat fine-tuning) instead of a text
        dialogs: DialogSet | None = None
        if isinstance(training_text, DialogSet):
            dialogs, training_text = training_text, ""
        if not isinstance(getattr(self, "_setup_undo", None), list):
            self._setup_undo = []          # called without train()
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
        unknown = unknown_train_params(params)
        if unknown:
            logger.warning(f"[AuraLite] WARNING: unknown training parameter(s) ignored: "
                           f"{describe_unknown_params(unknown)}")

        seq_length = params.get("seq_length", 64)
        d_model    = params.get("d_model", 128)
        d_ff       = params.get("d_ff", 256)
        n_heads    = params.get("n_heads", 4)
        n_layers   = params.get("n_layers", 4)
        n_kv_heads = params.get("n_kv_heads", None)
        # Treat n_kv_heads == 0 as "no GQA" (MHA). Zero is a common GUI default and
        # previously reached ModernTransformer where the %-modulo raised ZeroDivisionError.
        if n_kv_heads in (0, "0"):
            n_kv_heads = None
        # Training defaults come from one recipe (2.12, auralite/engine/recipe.py);
        # the measurements behind each value are in docs/training.md § 8.
        recipe = DEFAULT_RECIPE
        lr         = params.get("lr", recipe["lr"])
        epochs     = params.get("epochs", 100)
        batch_size = params.get("batch_size", recipe["batch_size"])
        dropout    = params.get("dropout", recipe["dropout"])
        # 2.14: "auto" builds the model with dropout and switches it off once
        # the plan turns out to read the text at most eight times (recipe.py).
        dropout_auto = is_auto(dropout)
        if dropout_auto:
            dropout = AUTO_DROPOUT
        dropout_passes: float | None = None
        grad_clip  = params.get("grad_clip", recipe["grad_clip"])
        weight_decay = params.get("weight_decay", recipe["weight_decay"])

        tok_kind     = params.get("tokenizer", recipe["tokenizer"])
        bpe_vocab    = params.get("bpe_vocab_size", recipe["bpe_vocab_size"])
        val_split    = params.get("val_split", recipe["val_split"])
        use_compile  = params.get("use_compile", False)
        autosave_every = params.get("autosave_every", 0)
        autosave_every_steps = params.get("autosave_every_steps", 0)
        autosave_path  = params.get("autosave_path", "aura_autosave.pt")
        continue_training = params.get("continue_training", False)
        accumulation_steps = max(1, int(params.get("accumulation_steps", 1) or 1))
        # 2.8: a budget in optimizer steps (0 = no limit); the LR schedule is
        # sized to whichever ends first, the epochs or this budget.
        max_steps = max(0, int(params.get("max_steps", 0) or 0))
        # 2.9: an epoch is one pass over the training tokens. Windows start every
        # `sample_stride` tokens (default seq_length = no overlap) from a random
        # offset drawn per epoch; sample_stride=1 restores the 2.8 epochs, which
        # visited every window and were therefore seq_length passes long.
        sample_stride = params.get("sample_stride")
        sample_stride = max(1, int(sample_stride)) if sample_stride else int(seq_length)
        val_max_tokens = params.get("val_max_tokens")
        early_stopping_patience = max(0, int(params.get(
            "early_stopping_patience", recipe["early_stopping_patience"]) or 0))
        restore_best_weights = bool(params.get("restore_best_weights",
                                               early_stopping_patience > 0))
        early_stopping_cooldown = bool(params.get("early_stopping_cooldown",
                                                  recipe["early_stopping_cooldown"]))
        time_budget_s = max(0.0, float(params.get("time_budget_s", 0) or 0))
        cooldown_ratio = float(params.get("cooldown_ratio", recipe["cooldown_ratio"]))
        use_alibi    = params.get("use_alibi", False)
        lora_rank    = params.get("lora_rank", 0)
        resume_training_state = params.get("resume_training_state", True)
        use_ddp      = params.get("use_ddp", False)
        # 2.14: a run on several computers (cluster.py): DDP over gloo on the
        # CPU, the main computer (rank 0) decides when to stop.
        cluster = getattr(self, "_cluster", None)
        user_finish_event, user_stop_event = finish_event, stop_event
        if cluster is not None:
            if params.get("continue_training", False):
                raise ValueError("a cluster run trains a new model; continue a model "
                                 "on one computer (continue_training)")
            use_ddp = True
            self.is_distributed = True
            self.world_size = cluster.world_size
            self.local_rank = 0
            self.device = torch.device("cpu")
            # Every computer follows these copies, set from the main computer's
            # events before each step.
            finish_event, stop_event = threading.Event(), threading.Event()

        # NEW: RoPE scaling
        rope_scaling = params.get("rope_scaling", None)
        sliding_window = params.get("sliding_window", None)
        kv_cache_dtype = params.get("kv_cache_dtype", None)
        use_flex_attention = params.get("use_flex_attention", False)
        use_moe = params.get("use_moe", False)
        num_experts = params.get("num_experts", 4)
        # Untie the input embedding and the output (LM head) by default: they are
        # two distinct layers and sharing one matrix cancels gradients and slows
        # convergence. Opt back into legacy weight tying with
        # {"tie_word_embeddings": True}.
        tie_word_embeddings = params.get("tie_word_embeddings", False)

        # NEW (v2.6): modern training stack
        #   QK-norm          — per-head RMS normalization of q/k before RoPE
        #   Muon optimizer   — Newton–Schulz orthogonalized momentum for the
        #                      hidden 2-D matrices (embeddings/head/norms stay
        #                      on AdamW)
        #   WSD schedule     — Warmup-Stable-Decay replaces plain cosine decay
        use_qk_norm     = bool(params.get("use_qk_norm", recipe["use_qk_norm"]))
        # v2.6.2: reproducibility + AMP dtype + persisted chat template
        seed             = params.get("seed", None)
        amp_dtype        = str(params.get("amp_dtype", "fp16")).strip().lower()
        chat_template    = params.get("chat_template", None)
        if chat_template:
            self.default_chat_template = str(chat_template)
        optimizer_name  = str(params.get("optimizer", recipe["optimizer"])).strip().lower()
        muon_lr         = float(params.get("muon_lr", recipe["muon_lr"]))
        muon_momentum   = float(params.get("muon_momentum", 0.95))
        muon_adjust_lr  = str(params.get("muon_adjust_lr", "match_rms_adamw")).strip()
        muon_ns_steps   = int(params.get("muon_ns_steps", recipe["muon_ns_steps"]))
        lr_schedule     = str(params.get("lr_schedule", recipe["lr_schedule"])).strip().lower()
        wsd_stable_ratio  = float(params.get("wsd_stable_ratio", 0.8))
        wsd_min_lr_ratio  = float(params.get("wsd_min_lr_ratio", 0.2))
        wsd_decay       = str(params.get("wsd_decay", "cosine")).strip().lower()

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

        previous_params = dict(self.params_used or {})
        self.params_used = dict(params)

        # ---- Reproducibility (v2.6.2) --------------------------------
        # Seed torch/numpy BEFORE the tokenizer trains, the model initializes
        # and the DataLoader starts shuffling so a full run can be replayed.
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
            logger.info(f"[AuraLite] Seed set: {seed} (torch & numpy RNG)")

        # A compiled module must never be trained: torch.compile captures the
        # inference graph, and the weights belong to the wrapper.
        if self._compiled_inference_model is not None:
            self.disable_inference_compile()

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
        # 2.15: dialogs — the chat template (the continued model's own unless
        # one is given), what the loss learns, and a fresh optimizer: the run
        # is a new task, and new template tokens change the vocabulary.
        template = ""
        dialog_loss = str(params.get("dialog_loss", "assistant")).strip().lower()
        if dialogs is not None:
            own = self.default_chat_template
            template = _dialogs.check_template(
                chat_template or (own if resuming and own in _dialogs.SFT_TEMPLATES else "chatml"))
            self.default_chat_template = template
            if dialog_loss not in _dialogs.DIALOG_LOSSES:
                raise ValueError(f"dialog_loss must be one of "
                                 f"{', '.join(_dialogs.DIALOG_LOSSES)}, got {dialog_loss!r}")
            if optimizer_state_to_restore is not None or scheduler_state_to_restore is not None:
                logger.info("[AuraLite] Dialogs: a fresh optimizer and learning-rate schedule "
                            "(the continued model's are not restored)")
            optimizer_state_to_restore = scheduler_state_to_restore = None
            scaler_state_to_restore = None
        if "optimizer" not in params and optimizer_state_to_restore is not None:
            # 2.12: the default optimizer changed from AdamW to Muon. A model
            # continued with default params keeps the optimizer whose state is
            # restored (a chained Muon state is a list, an AdamW state a dict).
            optimizer_name = "muon" if isinstance(optimizer_state_to_restore, list) else "adamw"

        # ---- Live progress helpers ------------------------------------
        # progress_every_batches: 0 = auto (≈1% of epoch, min 1, max 25)
        # progress_every_seconds: also emit at least this often (default 0.5s)
        progress_every_batches = int(params.get("progress_every_batches", 0) or 0)
        progress_every_seconds = float(params.get("progress_every_seconds", 0.5) or 0.5)
        train_t0 = time.time()

        def _setup_progress(message: str, frac: float = 0.0, stage: dict | None = None):
            """Setup-phase updates (tokenizer / encode) before epoch 1."""
            info = {
                "phase": "setup",
                "message": message,
                "percent": max(0.0, min(2.0, float(frac) * 2.0)),  # keep setup <2%
                "elapsed": time.time() - train_t0,
                "is_epoch_end": False,
                "batch": 0,
                "batches": 0,
            }
            if stage:
                info.update(stage)
                info["message"] = message
            self._invoke_progress(progress_callback, 0, epochs, 0.0, None, info=info)

        def setup_stage(stage: str, label: str, low: float, high: float) -> Any:
            """2.14.1: a setup stage with done/total, rate and ETA (StageReporter)."""
            def emit(info: dict) -> None:
                share = float(info.get("stage_percent") or 0.0) / 100.0
                _setup_progress(info["message"], low + (high - low) * share, stage=info)
            return _tok_io.StageReporter(emit if progress_callback is not None else None,
                                         stage, label, every_s=progress_every_seconds)

        def stop_requested() -> bool:
            """Stop or finish pressed (the caller's events, not a cluster's copies)."""
            return bool((user_stop_event is not None and user_stop_event.is_set())
                        or (user_finish_event is not None and user_finish_event.is_set()))

        def check_stop(stage: str) -> None:
            """2.14.1: before the first step a stop ends the run at once."""
            if stop_requested():
                raise _SetupStopped(stage)

        _setup_progress("Preparing: validating parameters…", 0.02)

        # ---- Tokenizer ------------------------------------------------
        # 2.13: a new BPE never sees the validation text. Fitted on all of it,
        # it learned merges for words found only in the held-out tail, and the
        # validation loss came out up to 9 % too low (docs/training.md § 10).
        # 2.14.1: a tokenizer from a file (or given) is not fitted here, but
        # the text is cut at the same place, so its validation is comparable.
        tokenizer_path = params.get("tokenizer_path") or None
        tokenizer_source = "continued" if resuming else "trained"
        if resuming and (tokenizer is not None or tokenizer_path):
            logger.warning("[AuraLite] WARNING: continued training keeps the model's tokenizer; "
                           "the given tokenizer is not used")
        # 2.15: dialogs are held out whole; a new tokenizer is fitted on the
        # text of the training dialogs (without the template's markers, which
        # become special tokens below)
        train_dialogs: list[list[dict]] = []
        val_dialogs: list[list[dict]] = []
        dialog_fit_chars = 0
        if dialogs is not None:
            train_index, val_index = dialogs.split(val_split)
            train_dialogs = [dialogs.dialogs[i] for i in train_index]
            val_dialogs = [dialogs.dialogs[i] for i in val_index]
            training_text = "\n".join(_dialogs.dialog_plain_text(d, template)
                                      for d in train_dialogs)
            dialog_fit_chars = len(training_text)
            if val_dialogs:
                training_text += "\n" + "\n".join(_dialogs.dialog_plain_text(d, template)
                                                  for d in val_dialogs)
            logger.info(f"[AuraLite] Dialogs: {len(train_dialogs):,} to train on, "
                        f"{len(val_dialogs):,} held out for validation; {template} template, "
                        f"the loss learns "
                        f"{'the replies' if dialog_loss == 'assistant' else 'every token'}")
            if dialogs.skipped:
                logger.warning(f"[AuraLite] WARNING: {dialogs.skipped:,} records were not "
                               f"dialogs with a reply and are skipped (the first: "
                               f"{dialogs.first_skipped})")
        val_start = params.get("val_start_char")
        fit_chars = len(training_text)
        if dialogs is not None:
            fit_chars = dialog_fit_chars
        elif val_start is not None and val_split > 0 and 0 < int(val_start) < len(training_text):
            # 2.15: validation starts at the first held-out document (TextCorpus)
            fit_chars = int(val_start)
        elif not resuming and (tok_kind == "bpe" or tokenizer_path or tokenizer is not None) \
                and val_split > 0:
            fit_chars = validation_split_point(training_text, val_split)
        if not resuming and cluster is not None and not cluster.is_main:
            _setup_progress("Receiving the tokenizer from the main computer…", 0.08,
                            stage={"stage": "tokenizer", "stage_label": "Receiving the tokenizer"})
            self.tokenizer = tokenizer_from_dict(cluster.broadcast_json(None))
            self.vocab_size = self.tokenizer.vocab_size
            tokenizer_source = "main computer"
        elif not resuming:
            new_tokenizer: Any
            if tokenizer is not None or tokenizer_path:
                tokenizer_source = "given" if tokenizer is not None else "file"
                shown = "the given tokenizer" if tokenizer is not None else \
                    os.path.basename(str(tokenizer_path))
                _setup_progress(f"Loading the tokenizer ({shown})…", 0.08,
                                stage={"stage": "tokenizer", "stage_label": "Loading the tokenizer"})
                file_info: dict = {}
                if tokenizer is not None:
                    new_tokenizer = tokenizer
                else:
                    new_tokenizer, file_info = _tok_io.read_tokenizer_file(str(tokenizer_path))
                if not callable(getattr(new_tokenizer, "to_dict", None)):
                    raise ValueError("the tokenizer must be an AuraLite or plugged-in BPE "
                                     "tokenizer (to_dict/encode/decode)")
                logger.info(f"[AuraLite] Tokenizer from {shown}: "
                            f"{_tok_io.describe_tokenizer(new_tokenizer)}")
                seen = (file_info.get("training") or {})
                if (seen.get("text_sha256") == _tok_io.text_sha256(training_text)
                        and int(seen.get("fit_chars") or 0) > fit_chars):
                    logger.warning("[AuraLite] WARNING: this tokenizer was fitted on the "
                                   "validation text too: the validation loss can come out "
                                   "too low (train it with the same val_split, "
                                   "docs/training.md § 13)")
            elif tok_kind == "bpe":
                # 2.8: GPT-style pre-tokenization by default; "ws" = 2.7 behaviour
                new_tokenizer = BPETokenizer(
                    pretokenizer=str(params.get("bpe_pretokenizer", "gpt")))
                reporter = setup_stage("tokenizer", f"Training the BPE tokenizer "
                                                    f"(vocab {bpe_vocab:,})", 0.08, 0.25)

                def on_step(step: str, done: int, total: int) -> None:
                    words, unit = _tok_io.BPE_STEPS.get(step, (step, ""))
                    reporter.update(done, total, step=step, words=words, unit=unit)

                reporter.update(0, 0, step="sample", words="sampling the text")
                # IMPROVED: stratified sampling instead of prefix
                new_tokenizer.train(training_text[:fit_chars], vocab_size=bpe_vocab,
                                    extra_chars=training_text[fit_chars:],
                                    progress=on_step, should_stop=stop_requested)
            else:
                _setup_progress("Training char tokenizer…", 0.08,
                                stage={"stage": "tokenizer", "stage_label": "Character tokenizer"})
                new_tokenizer = CharTokenizer()
                new_tokenizer.train(training_text)
            check_stop("tokenizer")
            self.tokenizer = new_tokenizer
            if cluster is not None:          # the workers use exactly this one
                cluster.broadcast_json(self.tokenizer.to_dict())
            self.vocab_size = self.tokenizer.vocab_size
            if tokenizer_source != "trained" or tok_kind == "bpe":
                missing, occurrences = _tok_io.coverage_report(self.tokenizer, training_text)
                if missing:
                    shown_chars = ", ".join(repr(c) for c in missing)
                    logger.warning(f"[AuraLite] WARNING: the tokenizer does not know "
                                   f"{occurrences:,} characters of the text (most often "
                                   f"{shown_chars}); they are encoded as its unknown token "
                                   f"and cannot be generated")
            _setup_progress(
                f"Tokenizer ready: {_tok_io.describe_tokenizer(self.tokenizer)}", 0.25,
                stage={"stage": "tokenizer", "stage_label": "Tokenizer ready",
                       "stage_percent": 100.0},
            )
        else:
            _setup_progress("Continued training — tokenizer already in memory", 0.15)
            if dialogs is None:
                # 2.15: continued training warns about unknown characters too
                missing, occurrences = _tok_io.coverage_report(self.tokenizer, training_text)
                if missing:
                    logger.warning(f"[AuraLite] WARNING: the model's tokenizer does not know "
                                   f"{occurrences:,} characters of the text (most often "
                                   f"{', '.join(repr(c) for c in missing)}); they are encoded "
                                   f"as its unknown token")
        added_tokens = 0
        if dialogs is not None:
            # 2.15: the template's markers are one token each, and characters
            # of the dialogs the tokenizer lacks (a continued model's, from
            # another corpus) get tokens too. On a copy of the tokenizer, so
            # a stop before the first step leaves it as it was.
            markers = _dialogs.template_special_tokens(template)
            self.tokenizer, added_tokens = _tok_io.with_special_tokens(self.tokenizer, markers)
            if added_tokens:
                logger.info(f"[AuraLite] Special tokens of the {template} template: "
                            f"{', '.join(markers)} ({added_tokens} new)")
            self.tokenizer, new_chars = _tok_io.with_characters(self.tokenizer, training_text)
            if new_chars:
                added_tokens += len(new_chars)
                shown = ", ".join(repr(c) for c in new_chars[:12])
                logger.info(f"[AuraLite] Characters of the dialogs the tokenizer did not know, "
                            f"added as tokens: {len(new_chars)} ({shown}"
                            f"{', ...' if len(new_chars) > 12 else ''})")
            rows = int(self.vocab_size or 0) if resuming else 0
            self.vocab_size = max(rows, int(self.tokenizer.vocab_size))

        # ---- Model ----------------------------------------------------
        check_stop("model")
        if not resuming:
            _setup_progress("Building model…", 0.30,
                            stage={"stage": "model", "stage_label": "Building the model"})
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
                use_qk_norm=use_qk_norm,
            ).to(self.device)

            # LoRA setup (must happen before DDP wrapping so the adapters are tracked)
            if lora_rank > 0:
                self.model.enable_lora(rank=lora_rank)

            # v2.6.5: report the *actual* size. `estimate_n_params` works from
            # the requested vocab_size (BPE may shrink it a lot), so GUI/CLI
            # estimates can differ noticeably from what was really built.
            actual_params = self.model.count_parameters()
            estimated = None
            if not (use_moe or use_qk_norm or tie_word_embeddings):
                estimated = estimate_n_params(
                    self.vocab_size, d_model, n_layers, d_ff,
                    n_heads=n_heads, n_kv_heads=n_kv_heads)
            note = (f"; pre-tokenizer estimate was {estimated:,}" if estimated is not None
                    and estimated != actual_params else "")
            logger.info(f"[AuraLite] Model built: {actual_params:,} params "
                        f"(vocab={self.vocab_size:,}{note})")
        else:
            # 2.11 "train short, extend long": continuing on longer windows
            # raises the trained length, and with it the generation window.
            native = getattr(self.model, "module", self.model)
            extend = getattr(native, "extend_context", None)
            old_len = int(getattr(native, "max_seq_len", 0) or 0)
            if callable(extend) and extend(int(seq_length)):
                logger.info(f"[AuraLite] Context extended: {old_len} → {seq_length} tokens "
                            f"(continued training on longer windows; generation now "
                            f"attends to the newest {seq_length} tokens)")
                old_batch = previous_params.get("batch_size")
                old_tokens = int(old_batch or 0) * old_len
                new_tokens = int(batch_size) * int(seq_length)
                if old_tokens and old_tokens < new_tokens:
                    keep = max(1, old_tokens // int(seq_length))
                    logger.info(f"[AuraLite] Tokens per optimizer step: {old_tokens:,} → "
                                f"{new_tokens:,}. batch_size={keep} keeps them (and the "
                                f"steps per epoch) as before; docs/training.md, section 7.")
            # 2.15: rows for the new special tokens of a chat template, and
            # LoRA adapters on the continued model (fine-tuning with adapters;
            # 2.14 applied lora_rank to new models only)
            grown: Any = native
            rows = int(getattr(getattr(grown, "embedding", None), "num_embeddings", 0) or 0)
            if rows and int(self.vocab_size) > rows and hasattr(grown, "resize_token_embeddings"):
                old_modules = (grown.embedding, grown.head)
                grown.resize_token_embeddings(int(self.vocab_size))
                self._setup_undo.append(
                    lambda m=grown, e=old_modules[0], h=old_modules[1]: _undo_resize(m, e, h))
                optimizer_state_to_restore = scheduler_state_to_restore = None
                logger.info(f"[AuraLite] Vocabulary {rows:,} → {self.vocab_size:,} tokens: the "
                            f"new ones start as the mean of the others")
            if int(lora_rank or 0) > 0 and not getattr(grown, "lora_adapters", None) \
                    and hasattr(grown, "enable_lora"):
                grown.enable_lora(rank=int(lora_rank))
                self._setup_undo.append(grown.disable_lora)
                optimizer_state_to_restore = scheduler_state_to_restore = None
                logger.info(f"[AuraLite] LoRA rank {int(lora_rank)} on the continued model "
                            f"(a fresh optimizer)")
        native_model: Any = getattr(self.model, "module", self.model)
        if dialogs is not None and getattr(native_model, "lora_adapters", None):
            # The template's markers may be new tokens: with LoRA the token
            # embedding and the output head must still learn them.
            flags = [(p, p.requires_grad) for p in (native_model.embedding.weight,
                                                    native_model.head.weight)]
            for p, _ in flags:
                p.requires_grad_(True)
            self._setup_undo.append(lambda flags=flags: [p.requires_grad_(f) for p, f in flags])
            logger.info("[AuraLite] LoRA on dialogs: the token embedding and the output head "
                        "train too")

        self._ddp_model = None

        # ---- Optimizer ------------------------------------------------
        # "adamw" (classic) or "muon" (v2.6 — Newton–Schulz orthogonalized
        # momentum). Muon handles only the hidden 2-D matrices of the
        # transformer blocks; embeddings / norm scales / an untied LM head
        # are trained by a companion AdamW, following Moonlight et al.
        has_lora = bool(getattr(self.model, "lora_adapters", None))
        if optimizer_name == "muon" and has_lora:
            logger.warning("[AuraLite] WARNING: LoRA adapters are trained with AdamW "
                           "(Muon is not recommended for low-rank adapters). Falling back to AdamW.")
            optimizer_name = "adamw"

        if optimizer_name == "muon":
            muon_params, adam_decay_params, adam_no_decay_params = split_parameters_for_muon(self.model)
            if not muon_params:
                logger.warning("[AuraLite] WARNING: no eligible 2-D matrices for Muon (LoRA-only?). Using AdamW.")
                optimizer_name = "adamw"
            else:
                sub_optimizers = [
                    Muon(muon_params, lr=muon_lr, momentum=muon_momentum,
                         weight_decay=weight_decay, adjust_lr_fn=muon_adjust_lr,
                         ns_steps=muon_ns_steps),
                ]
                adam_groups = []
                if adam_decay_params:
                    adam_groups.append({"params": adam_decay_params, "weight_decay": weight_decay})
                if adam_no_decay_params:
                    adam_groups.append({"params": adam_no_decay_params, "weight_decay": 0.0})
                if adam_groups:
                    sub_optimizers.append(optim.AdamW(adam_groups, lr=lr, betas=(0.9, 0.95)))
                self.optimizer = _ChainedOptimizers(*sub_optimizers)
                logger.info(f"[AuraLite] Optimizer: Muon ({len(muon_params)} matrices, "
                            f"lr={muon_lr}, momentum={muon_momentum}, adjust_lr={muon_adjust_lr}, "
                            f"ns_steps={muon_ns_steps}) + "
                            f"AdamW ({len(adam_decay_params) + len(adam_no_decay_params)} params, lr={lr})")

        if optimizer_name != "muon":
            single_group_state = (
                isinstance(optimizer_state_to_restore, dict)
                and len(optimizer_state_to_restore.get("param_groups", [])) == 1)
            if single_group_state:
                # Resuming a 2.7 optimizer state (one group): keep its layout so
                # the Adam moments carry over exactly.
                self.optimizer = optim.AdamW(
                    self.model.parameters(), lr=lr,
                    weight_decay=weight_decay, betas=(0.9, 0.95),
                )
            else:
                # 2.8: weight decay on the weight matrices only (no norm gains,
                # embeddings or biases), as the Muon path already did.
                self.optimizer = optim.AdamW(
                    adamw_param_groups(self.model, weight_decay), lr=lr,
                    betas=(0.9, 0.95),
                )
        self.optimizer_name = optimizer_name
        if optimizer_state_to_restore is not None:
            try:
                self.optimizer.load_state_dict(optimizer_state_to_restore)
                self._move_optimizer_state_to_device(self.optimizer, self.device)
                logger.info("[AuraLite] Restored optimizer state.")
            except Exception as e:
                logger.warning(f"[AuraLite] WARNING: could not restore optimizer state: {e}")
        # 2.15: dialogs learn the replies only (0 for a window without any)
        criterion: nn.Module = _ReplyLoss() if dialogs is not None else nn.CrossEntropyLoss()

        # Mixed precision. History: fp16 autocast + GradScaler on CUDA only.
        # v2.6.2 adds `amp_dtype`:
        #   "fp16" — old behaviour (CUDA autocast fp16 with a GradScaler)
        #   "bf16" — autocast bf16 on CUDA *and* CPU (bf16 needs no scaler);
        #            bf16 is far more forgiving than fp16 for training
        #   "none" — force full fp32 everywhere (debugging / determinism)
        if amp_dtype == "none":
            use_amp = False
            amp_t = None
        elif amp_dtype == "bf16":
            use_amp = True
            amp_t = torch.bfloat16
        else:  # fp16
            use_amp = self.device.type == "cuda"
            amp_t = torch.float16
        amp_device = self.device.type if self.device.type in ("cuda", "cpu") else "cpu"
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=bool(use_amp and amp_t is torch.float16))
        if use_amp:
            logger.info(f"[AuraLite] AMP: {amp_dtype} autocast on {amp_device}")
        if scaler_state_to_restore is not None:
            try:
                self.scaler.load_state_dict(scaler_state_to_restore)
                logger.info("[AuraLite] Restored AMP scaler state.")
            except Exception as e:
                logger.warning(f"[AuraLite] WARNING: could not restore AMP scaler state: {e}")

        # ---- Dataset / DataLoader ------------------------------------
        # v2.6.5: `corpus_cache="mmap"` (or AURALITE_CORPUS_CACHE=mmap) streams
        # the corpus into a uint32 memmap instead of holding an int64 tensor in
        # RAM. Measured cost of the RAM path: ~16 MiB per 1 MiB of text, i.e.
        # a 500 MiB corpus needs ~8 GiB peak; memmap+uint32 cuts that by ~4x
        # and never materialises the token list.
        _setup_progress("Tokenizing corpus (encode)…", 0.45,
                        stage={"stage": "tokenize", "stage_label": "Tokenizing the corpus",
                               "stage_done": 0, "stage_total": len(training_text),
                               "stage_unit": "chars", "stage_percent": 0.0})
        tokenizing = setup_stage("tokenize", "Tokenizing the corpus", 0.45, 0.55)

        def encoded_upto(offset: int) -> Callable[[int], None]:
            def report(done: int) -> None:
                tokenizing.update(offset + done, len(training_text), step="encode",
                                  unit="chars")
                check_stop("tokenize")
            return report

        cache_mode = str(params.get("corpus_cache")
                         or os.environ.get("AURALITE_CORPUS_CACHE", "")).strip().lower()
        use_mmap = cache_mode in ("1", "true", "yes", "on", "mmap", "memmap")
        corpus_file = None
        encoded = None
        dialog_targets: torch.Tensor | None = None
        learned_tokens = 0
        if dialogs is not None:
            # 2.15: each dialog with its targets (the replies), the training
            # dialogs first; held in RAM (dialog sets are small)
            if use_mmap:
                logger.info("[AuraLite] Dialogs are tokenized in memory (corpus_cache "
                            "applies to texts)")
                use_mmap = False
            ids, targets = [], []
            fit_tokens = 0
            every = max(1, len(dialogs) // 200)
            for number, dialog in enumerate(train_dialogs + val_dialogs, 1):
                dialog_ids, dialog_targets_list = _dialogs.encode_dialog(
                    self.tokenizer, dialog, template, dialog_loss)
                ids += dialog_ids
                targets += dialog_targets_list
                if number == len(train_dialogs):
                    fit_tokens = len(ids)
                    learned_tokens = sum(1 for t in targets if t != IGNORE_INDEX)
                if number % every == 0 or number == len(dialogs):
                    tokenizing.update(number, len(dialogs), step="encode", unit="dialogs")
                    check_stop("tokenize")
            if learned_tokens == 0:
                raise ValueError("the training dialogs have no reply tokens to learn")
            encoded = torch.tensor(ids, dtype=torch.long)
            dialog_targets = torch.tensor(targets, dtype=torch.long)
            logger.info(f"[AuraLite] Tokenized {len(dialogs):,} dialogs into {len(ids):,} tokens; "
                        f"the loss learns {learned_tokens:,} of the "
                        f"{fit_tokens:,} training tokens")
        elif use_mmap:
            from auralite.engine.dataset import write_tokens_memmap
            cache_path = params.get("corpus_cache_path")
            if cache_path:
                corpus_file = Path(cache_path)
            else:
                corpus_file = Path(tempfile.gettempdir()) / (
                    f"auralite_corpus_{uuid4().hex}.uint32.bin")
                # Unique name, so never reusable: train() deletes it when it returns.
                self._corpus_tmp_file = str(corpus_file)
            fit_count = [0]

            def _chunks():
                for ids in self._iter_token_chunks(training_text[:fit_chars],
                                                   on_progress=encoded_upto(0)):
                    fit_count[0] += len(ids)
                    yield ids
                yield from self._iter_token_chunks(training_text[fit_chars:],
                                                   on_progress=encoded_upto(fit_chars))
            total_tokens = write_tokens_memmap(corpus_file, _chunks(), dtype="uint32")
            fit_tokens = fit_count[0]
            size_mib = corpus_file.stat().st_size / (1024 * 1024)
            logger.info(f"[AuraLite] Tokenized corpus once into {total_tokens:,} tokens "
                        f"({size_mib:.2f} MiB uint32 memmap: {corpus_file}).")
            self._corpus_cache_path = str(corpus_file)
        else:
            # 2.14.1: in chunks, for the progress bar (the same ids as encode)
            ids = _tok_io.encode_with_progress(self.tokenizer, training_text[:fit_chars],
                                               report=encoded_upto(0))
            fit_tokens = len(ids)
            if fit_chars < len(training_text):
                ids += _tok_io.encode_with_progress(self.tokenizer, training_text[fit_chars:],
                                                    report=encoded_upto(fit_chars))
            encoded = torch.tensor(ids, dtype=torch.long)
            encoded_bytes = encoded.numel() * encoded.element_size()
            logger.info(f"[AuraLite] Tokenized corpus once into {len(encoded):,} tokens "
                        f"({encoded_bytes / (1024 * 1024):.2f} MiB LongTensor in RAM).")
        token_count = total_tokens if use_mmap else len(cast(torch.Tensor, encoded))
        per_token = len(training_text) / max(1, token_count)
        _setup_progress(
            f"Dialogs tokenized: {token_count:,} tokens, {learned_tokens:,} of them to learn"
            if dialogs is not None else
            f"Corpus tokenized: {token_count:,} tokens ({per_token:.2f} characters per token)",
            0.55, stage={"stage": "tokenize", "stage_label": "Corpus tokenized",
                         "stage_done": len(training_text), "stage_total": len(training_text),
                         "stage_unit": "chars", "stage_percent": 100.0},
        )
        check_stop("tokenize")

        # ---- Train / validation split -------------------------------------
        # Goal: both train and val slices must produce at least one full
        # (x, y) window. CharDataset of length L gives max(0, L - seq_length)
        # samples, so each slice needs L >= seq_length + 1.
        total_tokens = token_count
        n_val_target = int(total_tokens * val_split) if val_split > 0 else 0

        # Minimum tokens we need overall:
        #   train slice: seq_length + 1
        #   val slice  : seq_length + 1  (=> at least 1 val sample)
        min_train_tokens = seq_length + 1
        min_val_tokens   = seq_length + 1

        val_data = None
        train_data = encoded
        split_at = total_tokens
        n_train_tokens = total_tokens

        # 2.15: dialogs validate on held-out dialogs only (none: no validation)
        if val_split > 0 and (dialogs is None or val_dialogs):
            if total_tokens < min_train_tokens + min_val_tokens:
                logger.warning(f"[AuraLite] WARNING: text has only {total_tokens} tokens after "
                               f"encoding, need at least {min_train_tokens + min_val_tokens} "
                               f"for seq_length={seq_length} with validation. "
                               f"Validation DISABLED for this run.")
            else:
                # Take at least min_val_tokens; honour val_split but never starve train.
                val_tokens = max(min_val_tokens, n_val_target + seq_length)
                # Ensure train still has min_train_tokens left.
                max_val_tokens = total_tokens - min_train_tokens
                val_tokens = min(val_tokens, max_val_tokens)
                # 2.13: validate on exactly the text the tokenizer did not see.
                if min_train_tokens <= fit_tokens <= total_tokens - min_val_tokens:
                    val_tokens = total_tokens - fit_tokens

                split_at   = total_tokens - val_tokens
                n_train_tokens = split_at
                if encoded is not None:
                    train_data = encoded[:split_at]
                    val_data   = encoded[split_at:]

                logger.info(f"[AuraLite] Split: {n_train_tokens} train tokens, "
                            f"{val_tokens} val tokens, seq_length={seq_length}")

        train_targets = val_targets = None
        if dialog_targets is not None:        # 2.15: the targets follow the same cut
            train_targets = dialog_targets[:split_at]
            val_targets = dialog_targets[split_at:] if val_data is not None else None
            if val_data is not None and split_at != fit_tokens:
                logger.warning(f"[AuraLite] WARNING: the held-out dialogs have fewer than "
                               f"{min_val_tokens} tokens (seq_length + 1): validation also "
                               f"reads the end of the training dialogs; hold out more "
                               f"dialogs (val_split)")
        net = cast(nn.Module, self.model)      # the module being trained (typed)
        if use_mmap:
            from auralite.engine.dataset import PagedDataset
            assert corpus_file is not None
            dataset = PagedDataset(corpus_file, seq_length, offset=0, limit=split_at,
                                   stride=sample_stride)
        else:
            assert train_data is not None
            dataset = CharDataset(train_data, seq_length, stride=sample_stride,
                                  targets=train_targets)
        if len(dataset) == 0:
            raise ValueError(
                "Training text is too short for the chosen Context Window (seq_length)."
            )

        # 2.9: batches are gathered in-process (WindowBatches) instead of by
        # DataLoader workers: a batch is one indexing op (~0.02 ms), while worker
        # processes cost start-up time, IPC per batch and — with spawn on
        # Windows — a pickled copy of the whole corpus per worker.
        if seed is not None:
            loader_seed = int(seed)
        elif ddp_active:
            loader_seed = 0                    # every rank must draw the same order
        else:
            loader_seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        rank_batch = batch_size
        if cluster is not None:
            rank_batch = max(1, round(batch_size / cluster.world_size))
            logger.info(f"[AuraLite] Cluster: batch {rank_batch * cluster.world_size} = "
                        f"{rank_batch} windows on each of {cluster.world_size} computers")
        loader = WindowBatches(
            dataset, rank_batch, shuffle=True, seed=loader_seed,
            rank=dist.get_rank() if ddp_active else 0,
            world_size=self.world_size if ddp_active else 1,
            pin_memory=(self.device.type == "cuda"),
            drop_last=bool(use_compile),
        )
        logger.info(f"[AuraLite] Epoch: {len(dataset):,} windows of {seq_length} tokens "
                    f"(stride {sample_stride}) = {len(loader):,} batches")

        val_loader = None
        if use_mmap:
            assert corpus_file is not None
            val_ds = (PagedDataset(corpus_file, seq_length, offset=split_at,
                                   limit=total_tokens - split_at, stride=seq_length)
                      if split_at < total_tokens else None)
        else:
            val_ds = (CharDataset(val_data, seq_length, stride=seq_length, targets=val_targets)
                      if val_data is not None else None)
        if val_ds is not None:
            if len(val_ds) > 0:
                # 2.9: non-overlapping windows spread over the whole split. 2.8
                # read the first 50 batches of stride-1 windows, i.e. only the
                # first ~50*batch_size tokens of the split, ~seq_length times over.
                from auralite.engine.dataset import _Subset
                budget = int(val_max_tokens) if val_max_tokens else 50 * batch_size * seq_length
                picked = spread_indices(len(val_ds), max(1, budget // seq_length))
                val_bs = max(1, min(batch_size, len(picked)))
                shard = picked if cluster is None else picked[cluster.rank::cluster.world_size]
                val_loader = WindowBatches(_Subset(val_ds, shard), val_bs,
                                           shuffle=False, random_offsets=False)
                spread = ("" if len(picked) == len(val_ds)
                          else f", spread over {len(val_ds):,}")
                logger.info(f"[AuraLite] Validation: {len(picked):,} windows "
                            f"({len(picked) * seq_length:,} tokens{spread}), "
                            f"batch_size={val_bs}")
            else:
                logger.warning(f"[AuraLite] WARNING: val dataset is empty after windowing "
                               f"(val slice has {total_tokens - split_at} tokens, need > "
                               f"{seq_length}). Validation DISABLED.")
        if val_loader is not None:
            _setup_progress("Preparing the validation windows…", 0.70,
                            stage={"stage": "data", "stage_label": "Preparing the data"})
        val_bytes_per_token = self._bytes_per_token(val_loader) if val_loader is not None else None

        # Scheduler steps = optimizer steps. 2.7 counted *batches*, so with
        # accumulation_steps=N the schedule only reached 1/N of its length:
        # WSD never decayed and cosine stopped at a high LR.
        opt_steps_per_epoch = math.ceil(len(loader) / accumulation_steps)
        total_steps  = epochs * opt_steps_per_epoch
        if max_steps:
            total_steps = min(total_steps, max_steps)
        step_batch = rank_batch * (cluster.world_size if cluster is not None else 1)
        tokens_per_opt_step = step_batch * seq_length * accumulation_steps
        if dropout_auto:
            # The epochs and max_steps bound the plan; a time budget can only
            # shorten it, and is looked at again once the step time is known.
            plan_passes = total_steps * tokens_per_opt_step / max(1, n_train_tokens)
            dropout_passes = plan_passes
            dropout = self._apply_auto_dropout(plan_passes, bound=bool(time_budget_s))
        warmup_steps = min(200, total_steps // 10)
        # v2.6: WSD (Warmup-Stable-Decay) is the default schedule — it matches
        # or beats cosine convergence and supports branching decay runs from
        # any stable-phase checkpoint. "cosine" remains available.
        if lr_schedule == "wsd":
            self.scheduler = WSDScheduler(
                self.optimizer, warmup_steps, total_steps,
                stable_ratio=wsd_stable_ratio,
                min_lr_ratio=wsd_min_lr_ratio,
                decay_type=wsd_decay,
            )
            logger.info(f"[AuraLite] LR schedule: WSD (stable={wsd_stable_ratio:.0%}, "
                        f"min_lr_ratio={wsd_min_lr_ratio}, decay={wsd_decay})")
        else:
            self.scheduler = CosineWarmupScheduler(
                self.optimizer, warmup_steps, total_steps, min_lr=lr * 0.1
            )
            logger.info("[AuraLite] LR schedule: cosine + warmup")
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
                logger.info("[AuraLite] Restored scheduler state.")
            except Exception as e:
                logger.warning(f"[AuraLite] WARNING: could not restore scheduler state: {e}")

        # ---- torch.compile (optional, speeds up the training loop) ----
        # NOTE: torch.compile() returns lazily — Dynamo/Inductor errors (e.g. a
        # missing C compiler / Triton, or a backend that resolves to None) only
        # surface on the FIRST forward call. So we must wrap a trial forward pass,
        # not just the compile() call, and fall back to eager mode on ANY failure.
        train_model = self.model
        if use_compile:
            compiled = None
            _setup_progress("Compiling the model with torch.compile (the first step takes a "
                            "while)…", 0.80, stage={"stage": "compile",
                                                    "stage_label": "Compiling the model"})
            try:
                compile_t0 = time.time()
                compiled = torch.compile(self.model)
                if compiled is None:
                    raise RuntimeError("torch.compile returned None")
                # 2.9: the trial step is a real training step — the batch shape
                # of the loader, grad mode, forward *and* backward. 2.8 probed
                # one sequence under no_grad, which compiled an inference graph
                # that training never used and could not catch a backward
                # failure; the training graph was then compiled inside the loop.
                probe = torch.zeros((min(batch_size, len(dataset)), seq_length),
                                    dtype=torch.long, device=self.device)
                with torch.amp.autocast(amp_device, enabled=use_amp, dtype=amp_t):
                    probe_out = compiled(probe)
                probe_out.float().mean().backward()
                net.zero_grad(set_to_none=True)
                train_model = compiled
                logger.info(f"[AuraLite] torch.compile: training graph ready in "
                            f"{time.time() - compile_t0:.1f}s")
            except Exception as e:
                # graceful fallback — training continues in plain eager mode
                logger.warning(f"[AuraLite] torch.compile disabled (falling back to eager): {e}")
                train_model = self.model
                net.zero_grad(set_to_none=True)
                loader.drop_last = False
                try:
                    torch._dynamo.reset()
                except Exception:
                    pass

        if ddp_active:
            self._ddp_model = DDP(
                train_model,
                device_ids=([self.local_rank] if torch.cuda.is_available()
                            and self.device.type == "cuda" else None),
                find_unused_parameters=bool(use_moe),
            )
            logger.info(f"[AuraLite] Model wrapped with DistributedDataParallel (world_size={self.world_size})")

        # ---- Epoch loop -----------------------------------------------
        # Use DDP-wrapped model if available
        train_model = self._ddp_model if self._ddp_model is not None else train_model
        train_model.train()
        self.last_val_loss = None
        self.last_val_bpb = None

        def _bpb(loss: float | None) -> float | None:
            if loss is None or not val_bytes_per_token:
                return None
            return float(loss) / math.log(2.0) / val_bytes_per_token

        n_loader_batches = max(1, len(loader))
        if progress_every_batches <= 0:
            # ~1% of the epoch, but keep UI responsive without flooding Tk
            progress_every_batches = max(1, min(25, n_loader_batches // 100 or 1))
        total_opt_steps = max(1, epochs * n_loader_batches)
        tokens_per_batch = batch_size * seq_length
        report_rank_ok = (not ddp_active or dist.get_rank() == 0)

        check_stop("setup")
        _setup_progress(
            f"Training start: {epochs} epochs × {n_loader_batches} batches "
            f"(~{epochs * n_loader_batches} steps)",
            0.95, stage={"stage": "train", "stage_label": f"Epoch 1/{epochs}",
                         "stage_done": 0, "stage_total": n_loader_batches,
                         "stage_unit": "batches", "stage_percent": 0.0},
        )
        if report_rank_ok:
            self._invoke_progress(
                progress_callback, 0, epochs, 0.0, None,
                info={
                    "phase": "train",
                    "message": f"Epoch 1/{epochs} — start",
                    "batch": 0,
                    "batches": n_loader_batches,
                    "global_step": 0,
                    "total_steps": total_opt_steps,
                    "percent": 0.0,
                    "epoch_percent": 0.0,
                    "elapsed": time.time() - train_t0,
                    "eta_seconds": None,
                    "is_epoch_end": False,
                    "lr": self.scheduler.get_lr() if self.scheduler else lr,
                },
            )

        last_report_t = 0.0
        completed_batches_total = 0
        run_opt_steps = 0           # optimizer steps taken in this run (max_steps)
        budget_done = False
        # 2.12: a finish (finish_event) or a time budget ends the run with a
        # learning-rate decay instead of at the peak (train_control.py).
        # 2.14: on a cluster every computer runs TrainControl on the main
        # computer's clock and events, so all of them decide alike.
        shared_elapsed = [0.0]

        def sync_with_main() -> None:
            assert cluster is not None
            elapsed, finish, stop = cluster.broadcast_floats([
                time.time() - train_t0,
                float(bool(user_finish_event is not None and user_finish_event.is_set())),
                float(bool(user_stop_event is not None and user_stop_event.is_set()))])
            shared_elapsed[0] = elapsed
            if finish and finish_event is not None:
                finish_event.set()
            if stop and stop_event is not None:
                stop_event.set()

        clock = time.time
        if cluster is not None:
            sync_with_main()
            clock = lambda: train_t0 + shared_elapsed[0]  # noqa: E731
        ctl = TrainControl(self.scheduler, start_time=train_t0, finish_event=finish_event,
                           time_budget_s=time_budget_s, cooldown_ratio=cooldown_ratio,
                           max_run_steps=total_steps, steps_per_epoch=opt_steps_per_epoch,
                           clock=clock)

        best_val = None
        best_epoch = None
        best_state = None
        epochs_without_gain = 0
        epochs_completed = 0
        tokens_seen = 0
        last_train_loss = None
        stop_reason = None

        for epoch in range(epochs):
            if stop_event and stop_event.is_set():
                stop_reason = "stopped"
                break
            loader.set_epoch(epoch)

            running_loss   = 0.0
            seen_batches   = 0
            stopped_mid    = False
            epoch_t0       = time.time()
            last_report_t  = 0.0

            # IMPROVED: gradient accumulation loop
            for batch_idx, (xb, yb) in enumerate(loader):
                if cluster is not None:
                    sync_with_main()
                if stop_event and stop_event.is_set():
                    stopped_mid = True
                    break

                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                # Only zero grad on first accumulation step
                if batch_idx % accumulation_steps == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(amp_device, enabled=use_amp, dtype=amp_t):
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
                    self._scaler_unscale_optimizer()
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self._scaler_step_optimizer()
                    self.scaler.update()
                    self.scheduler.step()
                    run_opt_steps += 1
                    if autosave_every_steps and self.scheduler.step_count % autosave_every_steps == 0 and (not ddp_active or dist.get_rank() == 0):
                        try:
                            self.save_model(autosave_path)
                        except Exception as e:
                            logger.warning("step autosave failed: %s", e)

                running_loss += loss.item() * accumulation_steps
                seen_batches += 1
                completed_batches_total += 1
                tokens_seen += yb.numel()
                if (max_steps and run_opt_steps >= max_steps) or ctl.after_step(run_opt_steps):
                    budget_done = True
                if (dropout_auto and time_budget_s and dropout > 0.0
                        and ctl.planned_run_steps() is not None):
                    # The time budget's first plan: how far the run will get.
                    planned = ctl.planned_run_steps() or 0
                    budget_passes = planned * tokens_per_opt_step / max(1, n_train_tokens)
                    if dropout_passes is not None:
                        budget_passes = min(budget_passes, dropout_passes)
                    dropout_passes = budget_passes
                    dropout = self._apply_auto_dropout(budget_passes, bound=False)
                    dropout_auto = False               # decided once

                # ---- Live batch progress (between epochs) ----------------
                if report_rank_ok and progress_callback is not None:
                    now = time.time()
                    is_last_batch = (batch_idx + 1) >= n_loader_batches
                    due_by_count = (seen_batches == 1
                                    or seen_batches % progress_every_batches == 0
                                    or is_last_batch)
                    due_by_time = (progress_every_seconds > 0
                                   and (now - last_report_t) >= progress_every_seconds)
                    if due_by_count or due_by_time:
                        last_report_t = now
                        avg_loss = running_loss / max(1, seen_batches)
                        epoch_frac = seen_batches / float(n_loader_batches)
                        # overall percent across all epochs
                        overall_frac = ctl.fraction((epoch + epoch_frac) / float(max(1, epochs)))
                        elapsed = now - train_t0
                        epoch_elapsed = now - epoch_t0
                        # speed from this epoch so far
                        sps = (seen_batches / epoch_elapsed) if epoch_elapsed > 0 else 0.0
                        tps = sps * tokens_per_batch
                        # ETA: remaining batches in this epoch + future epochs
                        remain_batches = (n_loader_batches - seen_batches) +                             max(0, epochs - epoch - 1) * n_loader_batches
                        eta = ctl.eta_seconds((remain_batches / sps) if sps > 0 else None)
                        cur_lr = self.scheduler.get_lr() if self.scheduler else lr
                        self._invoke_progress(
                            progress_callback,
                            epoch + 1, epochs, avg_loss, None,
                            info={
                                "phase": "train",
                                "message": (
                                    f"Epoch {epoch + 1}/{epochs} · "
                                    f"batch {seen_batches}/{n_loader_batches}"
                                ),
                                "batch": seen_batches,
                                "batches": n_loader_batches,
                                "global_step": completed_batches_total,
                                "total_steps": total_opt_steps,
                                "percent": overall_frac * 100.0,
                                "epoch_percent": epoch_frac * 100.0,
                                "lr": cur_lr,
                                "samples_per_sec": sps * batch_size,
                                "batches_per_sec": sps,
                                "tokens_per_sec": tps,
                                "elapsed": elapsed,
                                "eta_seconds": eta,
                                "is_epoch_end": False,
                                "epoch_loss": avg_loss,
                                # 2.14.1: the stage line follows the epoch
                                "stage": "train",
                                "stage_label": f"Epoch {epoch + 1}/{epochs}",
                                "stage_step": "batches",
                                "stage_done": seen_batches,
                                "stage_total": n_loader_batches,
                                "stage_unit": "batches",
                                "stage_percent": epoch_frac * 100.0,
                                "stage_rate": sps,
                                "stage_eta_seconds": ((n_loader_batches - seen_batches) / sps
                                                      if sps > 0 else None),
                                "stage_elapsed": epoch_elapsed,
                            },
                        )

                if budget_done:
                    break

            if seen_batches:
                last_train_loss = running_loss / seen_batches
            if stopped_mid:
                stop_reason = "stopped"
                # Final status for interrupted epoch
                if report_rank_ok and seen_batches > 0:
                    avg_loss = running_loss / seen_batches
                    self._invoke_progress(
                        progress_callback, epoch + 1, epochs, avg_loss, None,
                        info={
                            "phase": "stopped",
                            "message": f"Stopped at epoch {epoch + 1}, batch {seen_batches}/{n_loader_batches}",
                            "batch": seen_batches,
                            "batches": n_loader_batches,
                            "global_step": completed_batches_total,
                            "total_steps": total_opt_steps,
                            "percent": ((epoch + seen_batches / float(n_loader_batches))
                                        / float(max(1, epochs))) * 100.0,
                            "elapsed": time.time() - train_t0,
                            "is_epoch_end": False,
                        },
                    )
                break

            # Handle remaining accumulated gradients
            if seen_batches > 0 and not budget_done and (batch_idx + 1) % accumulation_steps != 0:
                self._scaler_unscale_optimizer()
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self._scaler_step_optimizer()
                self.scaler.update()
                self.scheduler.step()
                run_opt_steps += 1
                if autosave_every_steps and self.scheduler.step_count % autosave_every_steps == 0 and (not ddp_active or dist.get_rank() == 0):
                    try:
                        self.save_model(autosave_path)
                    except Exception as e:
                        logger.warning("step autosave failed: %s", e)

            val_loss = None
            if val_loader is not None:
                if report_rank_ok:
                    self._invoke_progress(
                        progress_callback, epoch + 1, epochs,
                        running_loss / max(1, seen_batches), None,
                        info={
                            "phase": "val",
                            "message": f"Epoch {epoch + 1}/{epochs} — validation…",
                            "batch": n_loader_batches,
                            "batches": n_loader_batches,
                            "global_step": completed_batches_total,
                            "total_steps": total_opt_steps,
                            "percent": ((epoch + 1) / float(max(1, epochs))) * 100.0 - 0.01,
                            "epoch_percent": 100.0,
                            "elapsed": time.time() - train_t0,
                            "is_epoch_end": False,
                            "lr": self.scheduler.get_lr() if self.scheduler else lr,
                        },
                    )
                val_t0 = time.time()
                val_hook: Callable[[int, int], None] | None = None
                if report_rank_ok and progress_callback is not None:
                    # 2.14.1: validation batches as a stage
                    val_info = {"batch": n_loader_batches, "batches": n_loader_batches,
                                "global_step": completed_batches_total,
                                "total_steps": total_opt_steps,
                                "percent": ((epoch + 1) / float(max(1, epochs))) * 100.0 - 0.01,
                                "epoch_percent": 100.0, "is_epoch_end": False}
                    val_epoch, val_loss_so_far = epoch + 1, running_loss / max(1, seen_batches)

                    def emit_val(stage_info: dict, val_epoch: int = val_epoch,
                                 val_loss_so_far: float = val_loss_so_far,
                                 val_info: dict = val_info) -> None:
                        self._invoke_progress(
                            progress_callback, val_epoch, epochs, val_loss_so_far, None,
                            info={**val_info, **stage_info, "phase": "val",
                                  "elapsed": time.time() - train_t0,
                                  "lr": self.scheduler.get_lr() if self.scheduler else lr})

                    validating = _tok_io.StageReporter(
                        emit_val, "val", f"Epoch {val_epoch}/{epochs} — validation",
                        every_s=progress_every_seconds)

                    def on_val_batch(done: int, total: int,
                                     validating: Any = validating) -> None:
                        validating.update(done, total, step="batches", unit="batches")
                    val_hook = on_val_batch
                val_loss = self._evaluate(val_loader, criterion, on_batch=val_hook)
                val_seconds = time.time() - val_t0
                if cluster is not None:
                    val_seconds = cluster.broadcast_floats([val_seconds])[0]
                ctl.note_validation(val_seconds)
                self.last_val_loss = val_loss
                self.last_val_bpb = _bpb(val_loss)
                if val_loss is not None:
                    if best_val is None or val_loss < best_val:
                        best_val, best_epoch, epochs_without_gain = val_loss, epoch + 1, 0
                        if restore_best_weights:
                            best_state = {k: v.detach().clone()
                                          for k, v in net.state_dict().items()}
                    else:
                        epochs_without_gain += 1
            epochs_completed = epoch + 1

            if autosave_every and (epoch + 1) % autosave_every == 0 and (not ddp_active or dist.get_rank() == 0):
                try:
                    self.save_model(autosave_path)
                except Exception:
                    pass   # autosave must never kill training

            if progress_callback and seen_batches > 0 and report_rank_ok:
                avg_loss = running_loss / seen_batches
                elapsed = time.time() - train_t0
                # ETA for remaining full epochs after this one
                remain_epochs = max(0, epochs - (epoch + 1))
                epoch_dur = time.time() - epoch_t0
                eta = ctl.eta_seconds((epoch_dur * remain_epochs) if epoch_dur > 0 else None)
                self._invoke_progress(
                    progress_callback, epoch + 1, epochs, avg_loss, val_loss,
                    info={
                        "phase": "epoch_end",
                        "message": f"Epoch {epoch + 1}/{epochs} finished",
                        "batch": n_loader_batches,
                        "batches": n_loader_batches,
                        "global_step": completed_batches_total,
                        "total_steps": total_opt_steps,
                        "percent": ((epoch + 1) / float(max(1, epochs))) * 100.0,
                        "epoch_percent": 100.0,
                        "lr": self.scheduler.get_lr() if self.scheduler else lr,
                        "batches_per_sec": (n_loader_batches / epoch_dur) if epoch_dur > 0 else 0.0,
                        "tokens_per_sec": ((n_loader_batches * tokens_per_batch) / epoch_dur) if epoch_dur > 0 else 0.0,
                        "elapsed": elapsed,
                        "eta_seconds": eta,
                        "is_epoch_end": True,
                        "epoch_loss": avg_loss,
                        "epoch_seconds": epoch_dur,
                        "val_bpb": _bpb(val_loss),
                    },
                )

            if budget_done:
                stop_reason = ctl.stop_reason or "max_steps"
                what = {"max_steps": f"max_steps={max_steps} reached",
                        "finished": "finished after a learning-rate cooldown",
                        "time_budget": f"time budget of {time_budget_s:g}s spent",
                        "early_stopping": "early stopping after a learning-rate cooldown",
                        }.get(stop_reason, stop_reason)
                logger.info(f"[AuraLite] {what} (epoch {epoch + 1}/{epochs}); training stops.")
                break
            if (early_stopping_patience and epochs_without_gain >= early_stopping_patience
                    and not ctl.finishing):
                if early_stopping_cooldown and ctl.finish("early_stopping"):
                    logger.info(f"[AuraLite] No better validation loss for {epochs_without_gain} "
                                f"epochs: decaying the learning rate before stopping.")
                    continue
                stop_reason = "early_stopping"
                logger.info(f"[AuraLite] Early stopping: no better validation loss for "
                            f"{epochs_without_gain} epochs (best {best_val:.4f} at epoch "
                            f"{best_epoch}).")
                break

        restored = False
        if (restore_best_weights and best_state is not None
                and (best_epoch != epochs_completed or stop_reason == "stopped")):
            net.load_state_dict(best_state)
            self.last_val_loss = best_val
            self.last_val_bpb = _bpb(best_val)
            restored = True
            logger.info(f"[AuraLite] Restored the weights of epoch {best_epoch} "
                        f"(best validation loss {best_val:.4f}).")
        elapsed = time.time() - train_t0
        if cluster is not None:           # tokens of every computer
            tokens_seen = int(cluster.allreduce_sum([float(tokens_seen)])[0])
        val_note = ""
        if self.last_val_loss is not None:
            val_note = f", val loss {self.last_val_loss:.4f}"
            if self.last_val_bpb is not None:
                val_note += f" ({self.last_val_bpb:.4f} bits/byte)"
            if best_epoch is not None and not restored and best_epoch != epochs_completed:
                val_note += f" (best {best_val:.4f} at epoch {best_epoch})"
        logger.info(f"[AuraLite] Training done: {epochs_completed} epochs, "
                    f"{run_opt_steps:,} optimizer steps, {tokens_seen:,} tokens in "
                    f"{elapsed:.1f}s ({tokens_seen / max(elapsed, 1e-9):,.0f} tokens/s)"
                    f"{val_note}" + (f"; stopped: {stop_reason}" if stop_reason else ""))
        self.last_train_summary = {
            "epochs_completed": epochs_completed,
            "optimizer_steps": run_opt_steps,
            "tokens": tokens_seen,
            "train_loss": last_train_loss,
            "val_loss": self.last_val_loss,
            "best_val_loss": best_val,
            "val_bpb": self.last_val_bpb,
            "best_val_bpb": _bpb(best_val),
            "best_epoch": best_epoch,
            "restored_best": restored,
            "stop_reason": stop_reason,
            "cooldown_steps": (ctl.run_steps - ctl.finish_started_at
                               if ctl.finish_started_at is not None else 0),
            "tokenizer_fit_chars": fit_chars,
            "tokenizer_source": tokenizer_source,
            "tokenizer": _tok_io.describe_tokenizer(self.tokenizer),
            "train_tokens": n_train_tokens,
            "val_tokens": total_tokens - split_at,
            "dropout": float(dropout),
            "dropout_planned_passes": dropout_passes,
            "cluster_size": cluster.world_size if cluster is not None else 1,
            "seconds": elapsed,
        }
        if corpus_info:                    # 2.15: a TextCorpus
            self.last_train_summary.update(corpus_info)
        if dialogs is not None:            # 2.15: chat fine-tuning
            self.last_train_summary["dialogs"] = {
                "train": len(train_dialogs), "val": len(val_dialogs), "template": template,
                "loss": dialog_loss, "learned_tokens": learned_tokens,
                "added_tokens": added_tokens, "skipped": dialogs.skipped,
            }
        return dict(self.last_train_summary)
