"""AuraLite command line interface.

    auralite train     --text corpus.txt --params params.json --model out.pt
    auralite train     --text books/ notes/*.md --model out.pt      (2.15: folders, globs)
    auralite train     --text chats.jsonl --resume base.pt --model chat.pt   (2.15: dialogs)
    auralite generate  --model out.pt --prompt "Hello" --length 40
    auralite chat      --model out.pt [--template chatml] [--system "..."]
    auralite serve     --model out.pt [--host 0.0.0.0] [--port 8000]
    auralite info      --model out.pt
    auralite export    --model out.pt [--format pt2|onnx|torchscript] [--out PATH]
    auralite image-train    --images DIR [--type gan|flow] [--preset small] [--params p.json] --model img.pt
    auralite image-generate --model img.pt [-n 16] [--steps 20] [--out grid.png]
    auralite image-eval     --model img.pt --images DIR [-n 256]
    auralite tokenizer train   --text corpus.txt --vocab-size 4096 --out tok.json
    auralite tokenizer info    tok.json [--text sample.txt]
    auralite tokenizer encode  tok.json "some text"
    auralite tokenizer convert tokenizer.json --out tok.json
    auralite chat-eval --model chat.pt --data held_out.jsonl [--limit 300] [--json]
    auralite lora merge --model lora.pt --out merged.pt
    auralite lora save  --model lora.pt --out adapter.pt
    auralite lora apply --base base.pt --adapter adapter.pt --out chat.pt [--merge]
    auralite distill --teacher teacher.gguf --prompts prompts.txt --out chats.jsonl
    auralite train   --data prefs.jsonl --resume chat.pt --method dpo --model dpo.pt
    auralite doctor  [--data dialogs.jsonl] [--goal finetune|pretrain] [--write-params plan.json]
                     [--studio]   (2.19.1: also check that AuraLite Studio's window starts)
    auralite studio                     (2.19: the desktop app on Qt 6; pip install "auralite-ai[studio]")

2.14.1: `auralite train --tokenizer FILE` trains with your own tokenizer: a file
of `auralite tokenizer train`, a Hugging Face tokenizer.json (BPE), GPT-2
vocab.json + merges.txt, or a checkpoint (its tokenizer). Progress lines name
every stage (tokenizer, tokenizing, epochs, validation) with done/total, the
speed and the time left.

`--mmap` (or `--corpus-cache PATH`) streams the corpus into a uint32 memory map
instead of an in-RAM int64 tensor — the same tokens, ~4x less memory and no
Python list of the whole corpus.

`--params` JSON accepts every key of `AuraLiteEngine.train()` (d_model,
n_heads, n_layers, d_ff, seq_length, epochs, batch_size, lr, tokenizer,
optimizer, muon_lr, lr_schedule, use_qk_norm, sliding_window, seed,
amp_dtype, chat_template, ...).

`auralite train` (2.12): Ctrl+C finishes the run with a short learning-rate
decay and saves the model; a second Ctrl+C stops at once (and still saves).
`--time-budget 10m` sizes the run to a time limit.

2.14.2: `auralite image-train --type flow|gan` trains a flow-matching or a GAN
image generator instead of the diffusion model; `auralite image-eval` measures
an image generator (patch SWD against a folder of real images).

2.15: `auralite train --text` takes several files, folders and globs (the
documents are joined with blank lines and validated on whole held-out
documents); a `.jsonl` file of dialogs fine-tunes a chat model instead
(`--template`, `--lora`): it learns to answer in the chat template and to stop
at the end of its reply.

2.16: dialogs train for every epoch asked for (early stopping only when
asked); `--prompt-weight` learns the prompt too, with that weight;
`--pack-dialogs` packs whole dialogs into the windows. LoRA puts adapters on
every linear layer (`--lora-targets`), learns at 10x the learning rate
(`--lora-lr`), and `--save-adapter` writes what the run trained to a file of
its own; `auralite lora` merges, saves and applies adapters. `auralite
chat-eval` measures a chat model on held-out dialogs (stop rate, exact
replies, reply loss). `auralite distill` has a teacher model (GGUF, a llama.cpp
server, Hugging Face or AuraLite) answer a file of prompts and writes its
replies as dialogs (`--judge`, `--pairs`: preference pairs; an interrupted run
resumes). A `.jsonl` of preference pairs (prompt, chosen, rejected) trains the
chat model with DPO or SimPO (`--method`, `--beta`); `chat-eval` on pairs
reports how often the chosen reply is the likelier one.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from typing import Any

#: Model size and epochs of `auralite train`. Everything else (optimizer,
#: tokenizer, batch size, early stopping, ...) is the engine's recipe,
#: `auralite.engine.recipe.DEFAULT_RECIPE` (2.12), which `train()` applies.
DEFAULT_TRAIN_PARAMS: dict = {
    "d_model": 128, "n_heads": 4, "n_layers": 4, "d_ff": 256,
    "seq_length": 64, "epochs": 10,
}


def _parse_duration(value: str) -> float:
    """Seconds from ``"90"``, ``"90s"``, ``"10m"`` or ``"1.5h"`` (``--time-budget``)."""
    text = str(value).strip().lower()
    scale = {"s": 1.0, "m": 60.0, "h": 3600.0}.get(text[-1:], None)
    number = text[:-1] if scale else text
    try:
        seconds = float(number) * (scale or 1.0)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a duration such as 90, 90s, 10m or 1.5h, got {value!r}") from None
    if seconds < 0:
        raise argparse.ArgumentTypeError(f"a duration cannot be negative, got {value!r}")
    return seconds


def _graceful_sigint(finish_event: threading.Event, stop_event: threading.Event):
    """Ctrl+C: finish with a cooldown, again: stop now, a third time: interrupt.

    Returns the previous handler (None when signals cannot be installed, e.g.
    outside the main thread).
    """
    def handler(signum, frame):
        if not finish_event.is_set():
            finish_event.set()
            print("\n^C: finishing — decaying the learning rate for a few steps "
                  "(Ctrl+C again to stop now)", file=sys.stderr, flush=True)
        elif not stop_event.is_set():
            stop_event.set()
            print("\n^C: stopping now", file=sys.stderr, flush=True)
        else:
            raise KeyboardInterrupt

    try:
        return signal.signal(signal.SIGINT, handler)
    except (ValueError, OSError):        # not the main thread / no signals
        return None


def _stop_on_sigint(stop_event: threading.Event, message: str):
    """Ctrl+C sets `stop_event` (2.14.2, image-train); a second Ctrl+C interrupts.

    Returns the previous handler (None when signals cannot be installed).
    """
    def handler(signum, frame):
        if not stop_event.is_set():
            stop_event.set()
            print(f"\n^C: {message} (Ctrl+C again to abort)", file=sys.stderr, flush=True)
        else:
            raise KeyboardInterrupt

    try:
        return signal.signal(signal.SIGINT, handler)
    except (ValueError, OSError):        # not the main thread / no signals
        return None


def _add_common_flags(parser: argparse.ArgumentParser, *,
                      suppress_defaults: bool = False) -> None:
    """`--quiet` / `--log-level` shared by the root parser and every subcommand."""
    quiet_default = argparse.SUPPRESS if suppress_defaults else False
    level_default = (argparse.SUPPRESS if suppress_defaults
                     else os.environ.get("AURALITE_LOG_LEVEL", "INFO"))
    parser.add_argument("--quiet", action="store_true", default=quiet_default,
                        help="suppress progress output (results are still printed)")
    parser.add_argument("--log-level", default=level_default,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="library log level (default: INFO or $AURALITE_LOG_LEVEL)")


def _add_context_flags(parser: argparse.ArgumentParser) -> None:
    """`--context-window` / `--full-context` (2.11) for the generating commands."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--context-window", default=None, metavar="N",
                       help="tokens a generated token attends to: 'auto' (default: the "
                            "length the model was trained on), 'full' or a number")
    group.add_argument("--full-context", action="store_true",
                       help="attend to the whole history (the 2.10 behaviour); past the "
                            "trained length the output degrades into noise")


def _context_window_arg(args: argparse.Namespace) -> str:
    if getattr(args, "full_context", False):
        return "full"
    value = getattr(args, "context_window", None)
    return str(value) if value not in (None, "") else "auto"


def _configure_logging(args: argparse.Namespace) -> None:
    """Apply --quiet/--log-level (and $AURALITE_LOG_LEVEL) to library loggers.

    Since v2.6.5 the engine reports its progress through the ``auralite``
    logger instead of `print()`. The CLI preserves the historical UX: messages
    go to **stdout** (so `auralite train ... > log.txt` still captures them),
    the default level is INFO, and ``--quiet`` silences the library entirely.
    The handler re-resolves ``sys.stdout`` on every record so a later
    redirection (tests, wrappers, the GUI) is honoured.

    The level is also exported through ``AURALITE_LOG_LEVEL`` so that modules
    imported *later* (``auralite.engine`` reads the variable at import time) keep
    the level requested on the command line instead of resetting it to INFO.
    """
    level = "CRITICAL" if getattr(args, "quiet", False) else str(
        getattr(args, "log_level", "INFO")).upper()
    os.environ["AURALITE_LOG_LEVEL"] = level

    # Imported here, after the variable is set, and not at module level:
    # `auralite --version` / `--help` exit inside argparse before this runs, so
    # they never import the engine (and torch) — 2.7.0.
    from auralite.engine.logging_utils import reset_engine_logging

    logger = reset_engine_logging(level=level, stream_name="stdout")
    # The library installs its own stderr handler at import time; replacing it
    # keeps a single copy of every line, and `propagate = False` stops records
    # from reaching the root logger (and being printed a second time).
    logger.propagate = False
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(message)s")


class _ProgressPrinter:
    """Progress lines of `auralite train` and `auralite tokenizer train`.

    2.14.1: the stages of the setup (tokenizer training, tokenizing the
    corpus, ...) and the validation print their done/total, speed and time
    left: on a terminal as one line updated in place, otherwise (a log file)
    at most every few seconds. Each stage ends with a full line.
    """

    def __init__(self, stream=None, interval: float | None = None):
        self.stream = stream
        self.interactive = self._isatty()
        self.interval = interval if interval is not None else (0.25 if self.interactive else 5.0)
        self.key: tuple | None = None
        self.last = 0.0
        self.width = 0
        self.last_setup = ""

    def _out(self):
        return self.stream if self.stream is not None else sys.stdout

    def _isatty(self) -> bool:
        try:
            return bool(self._out().isatty())
        except (AttributeError, ValueError):
            return False

    def _line(self, text: str, final: bool) -> None:
        out = self._out()
        if self.interactive and not final:
            out.write("\r" + text.ljust(self.width))
            self.width = max(self.width, len(text))
        else:
            if self.width:
                out.write("\r" + text.ljust(self.width) + "\n")
            else:
                out.write(text + "\n")
            self.width = 0
        out.flush()

    def _end_inline(self) -> None:
        if self.width:
            self._out().write("\n")
            self._out().flush()
            self.width = 0

    def stage(self, info: dict, prefix: str | None = None) -> None:
        """A stage event (`StageReporter` info): throttled, in place on a terminal."""
        message = str(info.get("message") or info.get("stage_label") or "")
        name = prefix or str(info.get("stage") or "setup")
        key = (name, info.get("stage_label"), info.get("stage_step"))
        try:
            done, total = float(info.get("stage_done") or 0), float(info.get("stage_total") or 0)
        except (TypeError, ValueError):
            done, total = 0.0, 0.0
        finished = total > 0 and done >= total
        now = time.monotonic()
        if key != self.key:
            self._end_inline()
            self.key, self.last = key, now
            self._line(f"  {name} | {message}", final=not self.interactive or finished)
            return
        if not finished and now - self.last < self.interval:
            return
        self.last = now
        self._line(f"  {name} | {message}", final=finished or not self.interactive)

    def __call__(self, epoch: int, total: int, train_loss: float, val_loss: float | None,
                 info: dict | None = None) -> None:
        info = info or {}
        phase = info.get("phase", "train")
        if phase in ("setup", "cluster"):
            if info.get("stage_total") or info.get("stage_step"):
                self.stage(info)
                return
            msg = info.get("message") or "setup"
            if msg != self.last_setup:
                self._end_inline()
                self.key = None
                self.last_setup = msg
                self._line(f"  setup | {msg}", final=True)
            return
        if phase == "val":
            if info.get("stage_total"):
                self.stage(info, prefix=f"epoch {epoch}/{total}")
            else:
                self._end_inline()
                self._line(f"  epoch {epoch}/{total} | validating…", final=True)
            return
        self._end_inline()
        self.key = None
        # Live batch lines (same epoch)
        if phase == "train" and not info.get("is_epoch_end"):
            batch = info.get("batch")
            batches = info.get("batches")
            pct = info.get("percent")
            eta = info.get("eta_seconds")
            tps = info.get("tokens_per_sec")
            parts = [f"  epoch {epoch}/{total}"]
            if batch is not None and batches is not None:
                parts.append(f"batch {batch}/{batches}")
            if info.get("d_loss") is not None:          # 2.14.2: image GAN
                parts.append(f"G={float(train_loss):.4f} D={float(info['d_loss']):.4f}")
                if info.get("aug_p") is not None:
                    parts.append(f"aug p={float(info['aug_p']):.2f}")
            elif train_loss:
                parts.append(f"loss={train_loss:.4f}")
            if tps:
                parts.append(f"{float(tps):.0f} tok/s")
            elif info.get("images_per_second"):
                parts.append(f"{float(info['images_per_second']):.0f} img/s")
            if pct is not None:
                parts.append(f"{float(pct):.1f}%")
            if eta is not None:
                m, sec = divmod(int(max(0, float(eta))), 60)
                h, m = divmod(m, 60)
                parts.append(f"ETA {h}h{m:02d}m{sec:02d}s" if h else f"ETA {m}m{sec:02d}s")
            self._line(" | ".join(parts), final=True)
            return
        if phase == "stopped":
            self._line(f"  {info.get('message') or 'stopped'}", final=True)
            return
        metric = "val SWD" if info.get("val_metric") == "swd" else "val"
        val = f" | {metric}={val_loss:.4f}" if val_loss is not None else ""
        preference = info.get("preference") or {}
        if preference.get("accuracy") is not None:          # 2.16: DPO / SimPO
            val += (f" | accuracy {float(preference['accuracy']):.1%}, margin "
                    f"{float(preference['margin']):.3f}")
            if preference.get("kl") is not None:
                val += f", KL {float(preference['kl']):.4f}"
        if val and info.get("val_bpb") is not None:
            val += f" ({float(info['val_bpb']):.4f} bits/byte)"
        if info.get("d_loss") is not None:              # 2.14.2: image GAN
            losses = f"G={float(train_loss):.4f} D={float(info['d_loss']):.4f}"
            if info.get("aug_p") is not None:
                losses += f" | aug p={float(info['aug_p']):.2f}"
        else:
            losses = f"loss={train_loss:.4f}"
        self._line(f"  epoch {epoch}/{total} done | {losses}{val}", final=True)


def _progress(epoch: int, total: int, train_loss: float, val_loss: float | None,
              info: dict | None = None) -> None:
    """One-off progress line (kept for callers of 2.14.0; the CLI uses _ProgressPrinter)."""
    _ProgressPrinter(interval=0.0)(epoch, total, train_loss, val_loss, info)


def _describe_data(data) -> str:
    """One line about what `auralite train` read (2.15)."""
    from auralite.engine import DialogSet, PreferenceSet, TextSources, TokenShards
    if isinstance(data, (TokenShards, TextSources)):        # 2.18: big data
        return data.describe()
    if isinstance(data, PreferenceSet):                     # 2.16
        line = (f"{len(data):,} preference pairs from "
                f"{', '.join(os.path.basename(p) for p in data.sources)}")
        if data.skipped:
            line += f" ({data.skipped:,} records skipped, the first: {data.first_skipped})"
        return line
    if isinstance(data, DialogSet):
        line = f"{len(data):,} dialogs from {', '.join(os.path.basename(p) for p in data.sources)}"
        if data.skipped:
            line += f" ({data.skipped:,} records skipped, the first: {data.first_skipped})"
        return line
    chars = sum(len(doc) for doc in data.documents)
    line = f"{chars:,} characters" if len(data) == 1 else \
        f"{len(data):,} documents, {chars:,} characters"
    if data.skipped:
        line += f" ({len(data.skipped)} files skipped: not UTF-8)"
    return line


def cmd_train(args: argparse.Namespace) -> int:
    from auralite.engine import AuraLiteEngine, DataFormatError, DialogSet, load_training_data
    # 2.15: files, folders and globs; text documents or dialogs
    from auralite.engine import TextSources, TokenShards
    stream = {"always": True, "never": False}.get(str(getattr(args, "stream", "") or ""))
    try:
        data = load_training_data(args.text, stream=stream)
    except (FileNotFoundError, DataFormatError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if isinstance(data, TextSources):           # 2.18: Hugging Face datasets, record fields
        data.text_field = getattr(args, "text_field", None) or data.text_field
        data.hf_config = getattr(args, "hf_config", None)
        data.hf_split = getattr(args, "hf_split", None)
    if not getattr(args, "quiet", False):
        print(f"data: {_describe_data(data)}")
    big = isinstance(data, (TokenShards, TextSources))
    params = dict(DEFAULT_TRAIN_PARAMS)
    if big:
        params.pop("epochs")                    # 2.18: one pass over big data by default
    if getattr(args, "preset", None):
        # 2.18: a model size from the presets (auralite plan --list); --params wins
        from auralite.engine import preset_params
        try:
            params.update(preset_params(args.preset))
        except ValueError as err:
            print(f"error: {err}", file=sys.stderr)
            return 2
    given: dict = {}
    if args.params:
        with open(args.params, "r", encoding="utf-8") as f:
            given = json.load(f)
        params.update(given)
    for flag, key in (("max_tokens", "max_tokens"), ("eval_every", "eval_every_steps"),
                      ("memory_plan", "memory_plan"), ("shards_dir", "shards_dir"),
                      ("autosave", "autosave_path"), ("autosave_steps", "autosave_every_steps"),
                      ("autosave_minutes", "autosave_every_minutes")):
        if getattr(args, flag, None) is not None:
            params[key] = getattr(args, flag)
    if getattr(args, "shards_dir", None):
        params["shards_dir"] = os.path.abspath(args.shards_dir)
    if getattr(args, "autosave", None) and not (params.get("autosave_every_steps")
                                                or params.get("autosave_every_minutes")
                                                or params.get("autosave_every")):
        params["autosave_every_minutes"] = 30.0
    # 2.16: preference pairs train with DPO or SimPO, on the model of --resume
    from auralite.engine import PreferenceSet
    method = getattr(args, "method", None) or str(params.get("method") or "auto")
    pairs = isinstance(data, PreferenceSet) and method != "sft"
    if getattr(args, "method", None):
        params["method"] = args.method
    if getattr(args, "beta", None) is not None:
        params["preference_beta"] = float(args.beta)
    if pairs:
        # the epochs of a new model are not those of a preference run (default 3)
        params = {k: v for k, v in params.items()
                  if k in given or k not in DEFAULT_TRAIN_PARAMS}
        if not getattr(args, "resume", None):
            print("error: preference pairs train a chat model: give it with --resume "
                  "(fine-tune a model on dialogs first)", file=sys.stderr)
            return 2
    elif method in ("dpo", "simpo"):
        print(f"error: --method {method} needs preference pairs (records with a prompt, a "
              f"chosen and a rejected reply)", file=sys.stderr)
        return 2
    if getattr(args, "mmap", False) or getattr(args, "corpus_cache", None):
        params["corpus_cache"] = "mmap"
    if getattr(args, "corpus_cache", None):
        params["corpus_cache_path"] = os.path.abspath(args.corpus_cache)
    if getattr(args, "template", None):
        params["chat_template"] = args.template
    if getattr(args, "lora", None):
        params["lora_rank"] = int(args.lora)
    # 2.16: the prompt's weight, packing, LoRA's places, scale and learning rate
    for flag, key in (("prompt_weight", "prompt_loss_weight"), ("lora_targets", "lora_targets"),
                      ("lora_lr", "lora_lr"), ("lora_alpha", "lora_alpha")):
        if getattr(args, flag, None) is not None:
            params[key] = getattr(args, flag)
    if getattr(args, "rslora", False):
        params["lora_rslora"] = True
    if getattr(args, "optimizer", None):
        params["optimizer"] = args.optimizer
    if getattr(args, "quantize", None):
        if not params.get("lora_rank"):
            print("error: --quantize nf4 (QLoRA) trains LoRA adapters: add --lora RANK",
                  file=sys.stderr)
            return 2
        if not getattr(args, "resume", None):
            print("error: --quantize nf4 needs a model to fine-tune: --resume CHECKPOINT or "
                  "hf:ORG/NAME", file=sys.stderr)
            return 2
        params["quantize_base"] = args.quantize
    if getattr(args, "pack_dialogs", False):
        params["dialog_packing"] = True
    if getattr(args, "save_adapter", None) and not params.get("lora_rank"):
        print("error: --save-adapter needs a LoRA run (--lora RANK)", file=sys.stderr)
        return 2
    if isinstance(data, DialogSet) and getattr(args, "cluster_workers", None):
        print("error: chat fine-tuning on dialogs runs on one computer "
              "(drop --cluster-workers)", file=sys.stderr)
        return 2
    if isinstance(data, DialogSet) and params.get("chat_template"):
        from auralite.engine.dialogs import check_template
        try:
            check_template(params["chat_template"])
        except ValueError as err:
            print(f"error: {err}", file=sys.stderr)
            return 2
    if getattr(args, "tokenizer", None):
        # 2.14.1: train with your own tokenizer (auralite tokenizer train, a
        # Hugging Face tokenizer.json, GPT-2 vocab.json + merges.txt, a checkpoint);
        # 2.18: or hf:MODEL, the tokenizer.json of a Hugging Face model
        if str(args.tokenizer).lower().startswith("hf:"):
            params["tokenizer_path"] = args.tokenizer
        elif not os.path.exists(args.tokenizer):
            print(f"error: tokenizer file not found: {args.tokenizer}", file=sys.stderr)
            return 2
        else:
            params["tokenizer_path"] = os.path.abspath(args.tokenizer)
    # v2.6.5: validate the merged params against the typed config so typos in
    # --params (e.g. "epoch" instead of "epochs") fail loudly instead of being
    # silently ignored by train().
    try:
        from auralite.engine.config import AuraLiteConfig
        AuraLiteConfig(**{k: v for k, v in params.items()})
    except Exception as err:
        print(f"error: invalid training parameters: {err}", file=sys.stderr)
        return 2
    # 2.9: the typed config accepts extra keys, so a misspelt key used to train
    # silently with the default. A near-miss of a known key is now an error.
    from auralite.engine import describe_unknown_params, unknown_train_params
    typos = [(key, hint) for key, hint in unknown_train_params(params) if hint]
    if typos:
        print(f"error: unknown training parameter(s): {describe_unknown_params(typos)}",
              file=sys.stderr)
        return 2
    engine = AuraLiteEngine()
    if getattr(args, "resume", None):
        # 2.11: continue a checkpoint — e.g. on longer windows ("train short,
        # extend long", docs/training.md); a longer seq_length extends the
        # model's context.
        from auralite.engine.hf_models import HFModelError, describe_import, is_hf_model_spec
        if is_hf_model_spec(args.resume):
            # 2.18.1: a Hugging Face model, imported (and quantized) in memory
            try:
                info = engine.import_hf_model(args.resume, quantize=getattr(args, "quantize", None),
                                              progress_callback=_TransferPrinter(
                                                  getattr(args, "quiet", False)))
            except (HFModelError, FileNotFoundError, InterruptedError) as err:
                print(f"error: {err}", file=sys.stderr)
                return 2
            if not getattr(args, "quiet", False):
                print(f"imported: {describe_import(info)}")
        elif not os.path.exists(args.resume):
            print(f"error: checkpoint not found: {args.resume}", file=sys.stderr)
            return 2
        else:
            engine.load_model(args.resume)
        params["continue_training"] = True
        run = getattr(engine, "_resume_run_state", None)
        if getattr(args, "continue_run", False):
            # 2.18: the interrupted run's own parameters (only these may change)
            keep = ("autosave_path", "autosave_every_steps", "autosave_every_minutes",
                    "autosave_every", "time_budget_s", "eval_every_steps", "memory_plan")
            params = {key: value for key, value in params.items() if key in keep}
            params.update(continue_training=True, continue_run=True)
        elif isinstance(run, dict) and not run.get("finished") and \
                not getattr(args, "quiet", False):
            print(f"note: {args.resume} holds an unfinished run (optimizer step "
                  f"{int(run.get('run_opt_steps', 0)):,}); add --continue-run to continue it "
                  f"exactly where it stopped")
    elif getattr(args, "continue_run", False):
        print("error: --continue-run needs --resume CHECKPOINT (an autosave of the run)",
              file=sys.stderr)
        return 2
    if getattr(args, "time_budget", None):
        params["time_budget_s"] = float(args.time_budget)
    if getattr(args, "cluster_workers", None):
        # 2.14: train together with other computers (auralite cluster-worker).
        params["cluster_workers"] = int(args.cluster_workers)
        params["cluster_port"] = int(args.cluster_port)
        if args.cluster_host:
            params["cluster_host"] = args.cluster_host
        if args.cluster_token:
            params["cluster_token"] = args.cluster_token
        if args.cluster_timeout:
            params["cluster_timeout_s"] = float(args.cluster_timeout)
    # 2.20: the run goes into the run library (auralite runs) unless --no-record
    recorder = _record_train_run(args, data, params, engine, pairs)
    finish_event, stop_event = threading.Event(), threading.Event()
    previous = _graceful_sigint(finish_event, stop_event)
    from auralite.engine.hf_bpe import TokenizerFormatError
    printer = None if getattr(args, "quiet", False) else _ProgressPrinter()
    try:
        summary = engine.train(
            data, params,
            progress_callback=recorder.callback(printer) if recorder is not None else printer,
            stop_event=stop_event, finish_event=finish_event)
    except (TokenizerFormatError, FileNotFoundError, DataFormatError) as err:
        if recorder is not None:
            recorder.fail(err)
        print(f"error: {err}", file=sys.stderr)
        return 2
    except BaseException as err:
        if recorder is not None:
            recorder.fail(err)
        raise
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    if isinstance(summary, dict) and summary.get("stopped_during"):
        # 2.14.1: stopped before the first step — there is no new model
        if recorder is not None:
            recorder.finish(summary)
        print(f"stopped before training began ({summary['stopped_during']}); nothing was saved")
        return 130
    try:
        engine.save_model(args.model)
    except BaseException as err:
        if recorder is not None:
            recorder.fail(err)
        raise
    line = f"saved: {args.model}"
    if getattr(args, "save_adapter", None):
        info = engine.save_adapter(args.save_adapter)
        line += f" | adapter: {args.save_adapter} ({info['bytes'] / 1e6:.2f} MB)"
    summary = engine.last_train_summary or {}
    if recorder is not None:
        recorder.finish(summary, result=args.model,
                        checkpoint=_cli_continue_point(args.model, params, summary))
    if summary.get("val_loss") is not None:
        line += f" | val loss {summary['val_loss']:.4f}"
        if summary.get("val_bpb") is not None:
            line += f" ({summary['val_bpb']:.4f} bits/byte)"
    if summary.get("dialogs"):
        line += f" | chat template {summary['dialogs']['template']}"
    report = summary.get("preference") or {}
    if (report.get("final") or {}).get("accuracy") is not None:     # 2.16
        line += (f" | {str(summary.get('method', '')).upper()}: held-out accuracy "
                 f"{report['initial']['accuracy']:.1%} -> {report['final']['accuracy']:.1%}")
    reason = str(summary.get("stop_reason") or "")
    if reason:
        line += " | " + {"early_stopping": "stopped early (best epoch kept)",
                         "finished": "finished after a learning-rate cooldown",
                         "time_budget": "time budget spent",
                         "stopped": "stopped",
                         "max_steps": "max_steps reached"}.get(reason, reason)
    print(line)
    if reason == "stopped" and not summary.get("restored_best") and \
            not getattr(args, "quiet", False):
        print(f"continue this run later: auralite train --data ... --resume {args.model} "
              f"--continue-run")
        if recorder is not None:
            print(f"  or: auralite runs continue {recorder.id}")
    if recorder is not None and not getattr(args, "quiet", False):
        print(f"run: {recorder.id} (auralite runs show {recorder.id})")
    return 0


def _data_kind(data: Any) -> str:
    """The kind of `data` as the Studio's data check names it (2.20)."""
    from auralite.engine import DialogSet, PreferenceSet, TextSources, TokenShards
    if isinstance(data, DialogSet):
        return "dialogs"
    if isinstance(data, PreferenceSet):
        return "preferences"
    if isinstance(data, TokenShards):
        return "shards"
    if isinstance(data, TextSources):
        return "big"
    return "text"


def _record_train_run(args: argparse.Namespace, data: Any, params: dict, engine: Any,
                      pairs: bool) -> Any:
    """2.20: a :class:`~auralite.runs.RunRecorder` for `auralite train`, or None
    (--no-record, AURALITE_NO_RUNS=1). --continue-run of a recorded run adds a
    session to it; a run whose files the library cannot write trains anyway."""
    from auralite import runs as runlib
    if getattr(args, "no_record", False) or not runlib.recording_enabled():
        return None
    resume = getattr(args, "resume", None)
    try:
        if getattr(args, "continue_run", False) and resume:
            wanted = os.path.abspath(resume)
            for run in runlib.list_runs():
                if wanted in (run.record.get("checkpoint"), run.record.get("result"),
                              run.record.get("output")) and run.status != "running":
                    return runlib.RunRecorder.resume(run)
        if pairs:
            kind, method = "preference", str(params.get("method") or "dpo")
            method = "dpo" if method == "auto" else method
        elif resume:
            kind = "finetune"
            method = ("qlora" if params.get("quantize_base") == "nf4" else
                      "lora" if params.get("lora_rank") else "full")
        else:
            kind, method = "train", "lora" if params.get("lora_rank") else "new"
        run_state = getattr(engine, "_resume_run_state", None)
        recorded = dict(params)
        if getattr(args, "continue_run", False) and isinstance(run_state, dict):
            recorded = dict(run_state.get("params") or {}, **params)
        options = {"stream": {"always": True, "never": False}.get(str(args.stream or ""))
                   if getattr(args, "stream", None) else None,
                   "text_field": getattr(args, "text_field", None),
                   "hf_config": getattr(args, "hf_config", None),
                   "hf_split": getattr(args, "hf_split", None)}
        try:
            records = len(data)
        except TypeError:
            records = None
        recorder = runlib.RunRecorder(
            kind=kind, source="cli", name=getattr(args, "run_name", None) or "",
            base={"spec": resume or "",
                  "label": runlib.base_label(resume or "", kind, getattr(args, "preset", "") or "")},
            method=method,
            data={"paths": list(args.text), "kind": _data_kind(data), "records": records,
                  "options": {k: v for k, v in options.items() if v is not None}},
            params=recorded, hardware=runlib.machine_summary(engine), files="paths")
        recorder.record["output"] = os.path.abspath(args.model)
        if params.get("autosave_path"):
            recorder.record["checkpoint"] = os.path.abspath(str(params["autosave_path"]))
        recorder._write()
        return recorder
    except OSError as err:
        print(f"note: the run is not recorded ({err})", file=sys.stderr)
        return None


def _cli_continue_point(model: str, params: dict, summary: dict) -> str | None:
    """Where a run of `auralite train` continues from (2.20): the saved model
    after a stop, else its autosave."""
    if summary.get("stop_reason") == "stopped" and not summary.get("restored_best"):
        return model
    autosave = params.get("autosave_path")
    if autosave and os.path.isfile(str(autosave)) and summary.get("stop_reason") == "stopped":
        return str(autosave)
    return None


def cmd_data_prepare(args: argparse.Namespace) -> int:
    """2.18: tokenize big data into token shards (``auralite data prepare``)."""
    from auralite.engine import DataFormatError, prepare_dataset
    from auralite.engine.hf_bpe import TokenizerFormatError
    if args.tokenizer and not str(args.tokenizer).lower().startswith("hf:") \
            and not os.path.exists(args.tokenizer):
        print(f"error: tokenizer file not found: {args.tokenizer}", file=sys.stderr)
        return 2
    finish_event, stop_event = threading.Event(), threading.Event()
    previous = _graceful_sigint(finish_event, stop_event)
    printer = None if args.quiet else _StagePrinter()
    try:
        shards = prepare_dataset(
            args.sources, args.out, tokenizer=args.tokenizer, vocab_size=args.vocab_size,
            tokenizer_kind=args.kind, pretokenizer=args.pretokenizer,
            val_fraction=args.val_fraction, val_max_tokens=args.val_max_tokens,
            shard_tokens=args.shard_tokens, workers=args.workers, text_field=args.text_field,
            hf_config=args.hf_config, hf_split=args.hf_split, max_tokens=args.max_tokens,
            max_documents=args.max_documents, separator=args.separator,
            fast_tokenizer=not args.no_fast, overwrite=args.overwrite,
            progress_callback=printer, stop_event=_EitherEvent(stop_event, finish_event))
    except (FileNotFoundError, DataFormatError, TokenizerFormatError, ValueError,
            RuntimeError) as err:
        if printer is not None:
            printer.end()
        print(f"error: {err}", file=sys.stderr)
        return 2
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    if printer is not None:
        printer.end()
    print(f"{'stopped' if shards.stopped else 'ready'}: {shards.describe()} in {shards.folder}")
    if shards.stopped:
        print("run the same command again to continue where it stopped")
        return 130
    print(f"train on it: auralite train --data {shards.folder} --preset 125m")
    return 0


def cmd_data_info(args: argparse.Namespace) -> int:
    """2.18: what a prepared dataset holds."""
    from auralite.engine import DataFormatError, TokenShards
    try:
        shards = TokenShards.open(args.dataset)
    except (FileNotFoundError, DataFormatError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    manifest = shards.manifest
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False))
        return 0
    stats = manifest.get("stats") or {}
    tok = manifest.get("tokenizer") or {}
    print(shards.describe())
    print(f"  folder:     {shards.folder}")
    print(f"  tokenizer:  {tok.get('description', '?')}")
    print(f"  token type: {shards.dtype}; documents end with "
          f"{json.dumps((manifest.get('separator') or {}).get('token') or (manifest.get('separator') or {}).get('text', ''))}")
    if stats.get("chars_per_token"):
        print(f"  {stats.get('chars', 0):,} characters, {stats['chars_per_token']:.2f} per token")
    for source in manifest.get("sources") or []:
        print(f"  source:     {source.get('spec')}")
    skipped = stats.get("skipped") or {}
    if skipped:
        print("  skipped:    " + ", ".join(f"{n:,} {why}" for why, n in skipped.items()))
    if stats.get("not_utf8"):
        print(f"  not UTF-8:  {len(stats['not_utf8'])} file(s)")
    if not shards.complete:
        print("  the preparation did not finish: run it again to continue")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """2.18: the memory a training run needs, and a batch plan that fits."""
    from auralite.engine import MODEL_PRESETS, available_memory, plan_training, preset_params
    from auralite.engine.sizing import count_parameters, format_bytes, parse_size
    if args.list:
        print(f"{'preset':>6}  {'d_model':>7}  {'layers':>6}  {'heads':>5}  {'d_ff':>5}  "
              f"{'params (vocab ' + format(args.vocab, ',') + ')':>22}")
        for name, preset in MODEL_PRESETS.items():
            counts = count_parameters(vocab_size=args.vocab, **{k: v for k, v in preset.items()
                                                                 if k != "seq_length"})
            print(f"{name:>6}  {preset['d_model']:>7}  {preset['n_layers']:>6}  "
                  f"{preset['n_heads']:>5}  {preset['d_ff']:>5}  {counts['total'] / 1e6:>20.1f}M")
        return 0
    model: dict = {"d_model": 128, "n_layers": 4, "n_heads": 4, "d_ff": 256, "seq_length": 64}
    if args.preset:
        try:
            model.update(preset_params(args.preset))
        except ValueError as err:
            print(f"error: {err}", file=sys.stderr)
            return 2
    if args.params:
        with open(args.params, "r", encoding="utf-8") as f:
            given = json.load(f)
        model.update({k: v for k, v in given.items() if k in (
            "d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff", "seq_length", "batch_size",
            "accumulation_steps", "tie_word_embeddings", "optimizer", "amp_dtype", "lora_rank",
            "dropout")})
    seq = int(args.seq_length or model.get("seq_length", 64))
    batch = int(args.batch_size or model.get("batch_size", 32))
    try:
        memory = parse_size(args.memory) if args.memory else available_memory(args.device)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    dims = {"d_model": int(model["d_model"]), "n_layers": int(model["n_layers"]),
            "n_heads": int(model["n_heads"]), "d_ff": int(model["d_ff"]),
            "n_kv_heads": model.get("n_kv_heads") or None, "vocab_size": int(args.vocab),
            "tie_word_embeddings": bool(model.get("tie_word_embeddings", False)),
            "optimizer": str(args.optimizer or model.get("optimizer") or "muon"),
            "amp": str(args.amp or model.get("amp_dtype") or "none"),
            "lora_rank": int(model.get("lora_rank") or 0),
            "dropout": float(model.get("dropout", 0.0) if not isinstance(model.get("dropout"),
                                                                           str) else 0.0),
            "device": args.device, "seq_length": seq}
    from auralite.engine import estimate_training_memory
    estimate = estimate_training_memory(batch_size=batch,
                                        gradient_checkpointing=bool(args.checkpointing), **dims)
    print(f"model: {estimate['parameters'] / 1e6:,.1f}M parameters "
          f"(d_model {dims['d_model']}, {dims['n_layers']} layers, vocab {dims['vocab_size']:,})"
          f"; windows of {seq} tokens, batch {batch}")
    for part in ("weights", "gradients", "optimizer", "activations", "logits", "overhead"):
        print(f"  {part:<12} {format_bytes(estimate[part]):>10}")
    print(f"  {'total':<12} {format_bytes(estimate['total']):>10}"
          + (" (loss in chunks)" if estimate["chunked_loss"] else ""))
    if not memory:
        print("free memory unknown: give --memory 16GB")
        return 0
    print(f"memory: {format_bytes(memory)} {'given' if args.memory else 'free now'} "
          f"({args.device})")
    plan = plan_training(int(memory * 0.9), batch_size=batch, gradient_checkpointing=False,
                         **dims)
    if not plan["fits"]:
        print(f"does not fit even 1 window per step with gradient checkpointing "
              f"({format_bytes(plan['estimate']['total'])}): a smaller preset, LoRA / QLoRA "
              f"(--quantize nf4), or a GPU")
        return 1
    advice = [f"batch_size {plan['micro_batch']}", f"accumulation_steps "
              f"{plan['accumulation_steps']}"]
    if plan["gradient_checkpointing"]:
        advice.append("use_gradient_checkpointing true")
    print(f"fits as: {', '.join(advice)} (about {format_bytes(plan['estimate']['total'])}); "
          f"auralite train --memory-plan auto does this by itself")
    return 0


class _EitherEvent:
    """``is_set()`` of two events (the first Ctrl+C of a preparation stops it too)."""

    def __init__(self, first: threading.Event, second: threading.Event):
        self.first, self.second = first, second

    def is_set(self) -> bool:
        return self.first.is_set() or self.second.is_set()


class _StagePrinter:
    """One updating line for stage events (``auralite data prepare``)."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.last = 0.0
        self.width = 0

    def __call__(self, info: dict) -> None:
        now = time.time()
        if now - self.last < self.interval and info.get("stage_percent") != 100.0:
            return
        self.last = now
        text = str(info.get("message", ""))[:150]
        pad = max(0, self.width - len(text))
        self.width = len(text)
        if sys.stdout.isatty():
            print("\r" + text + " " * pad, end="", flush=True)
        else:
            print(text, flush=True)

    def end(self) -> None:
        if self.width and sys.stdout.isatty():
            print()
        self.width = 0


def _load_engine(path: str) -> Any:
    """An engine with the checkpoint `path` loaded, or None (the error printed)."""
    from auralite.engine import AuraLiteEngine
    if not os.path.exists(path):
        print(f"error: checkpoint not found: {path}", file=sys.stderr)
        return None
    engine = AuraLiteEngine()
    engine.load_model(path)
    return engine


def cmd_chat_eval(args: argparse.Namespace) -> int:
    """2.16: stop rate, exact replies and reply loss on held-out dialogs."""
    from auralite.engine import DataFormatError
    engine = _load_engine(args.model)
    if engine is None:
        return 2
    quiet = args.json or getattr(args, "quiet", False)
    from auralite.engine import (DialogSet, PreferenceSet, evaluate_preferences,
                                 load_training_data)
    try:
        data = load_training_data(args.data)
    except (FileNotFoundError, DataFormatError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if isinstance(data, PreferenceSet):
        # 2.16: how often the chosen reply of held-out pairs is the likelier one
        try:
            ranked = evaluate_preferences(engine, data, limit=args.limit)
        except ValueError as err:
            print(f"error: {err}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(ranked))
            return 0
        print(f"pairs: {ranked['pairs']} | template {engine.default_chat_template}")
        print(f"chosen reply likelier:          {ranked['accuracy']:.1%}")
        print(f"chosen likelier per token:      {ranked['mean_accuracy']:.1%}")
        print(f"log-probability margin:         {ranked['margin']:.3f} "
              f"({ranked['mean_margin']:.4f} per token)")
        return 0

    def progress(done: int, total: int) -> None:
        if not quiet and (done == total or done % max(1, total // 20) == 0):
            print(f"\rreplies: {done}/{total}", end="" if done < total else "\n", flush=True)

    try:
        result = engine.evaluate_chat(data if isinstance(data, DialogSet) else args.data,
                                      limit=args.limit, max_new_tokens=args.max_new_tokens,
                                      temperature=args.temperature, on_progress=progress)
    except (FileNotFoundError, DataFormatError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
        return 0
    loss = result["reply_loss"]
    bpb = result["reply_bpb"]
    temperature = result["temperature"]
    sampling = "greedy" if temperature <= 0 else f"temperature {temperature}"
    loss_text = f"{loss:.4f} nats/token" if loss is not None else "n/a"
    bpb_text = f"{bpb:.4f}" if bpb is not None else "n/a"
    print(f"dialogs: {result['dialogs']} | template {result['template']} | {sampling}")
    print(f"stopped by itself: {result['stop_rate']:.1%}")
    print(f"exact replies:     {result['exact']:.1%}")
    print(f"no stray markers:  {result['clean_rate']:.1%}")
    print(f"reply loss:        {loss_text}")
    print(f"reply bits/byte:   {bpb_text}")
    for sample in result["samples"]:
        mark = "=" if sample["exact"] else ("~" if sample["stopped"] else "!")
        print(f"  {mark} {sample['user']!r} -> {sample['reply']!r}"
              + ("" if sample["exact"] else f" (expected {sample['expected']!r})"))
    return 0


def cmd_lora(args: argparse.Namespace) -> int:
    """2.16: merge LoRA adapters into a model, save them alone, or apply them to a base."""
    if args.lora_command == "apply":
        engine = _load_engine(args.base)
        if engine is None:
            return 2
        if not os.path.exists(args.adapter):
            print(f"error: adapter not found: {args.adapter}", file=sys.stderr)
            return 2
        try:
            info = engine.load_adapter(args.adapter, check_base=not args.no_check,
                                       merge=args.merge)
        except ValueError as err:
            print(f"error: {err}", file=sys.stderr)
            return 2
        engine.save_model(args.out)
        print(f"saved: {args.out} | {info['adapters']} adapters"
              + (", merged" if info["merged"] else "") + f", {info['new_rows']} new tokens")
        return 0
    engine = _load_engine(args.model)
    if engine is None:
        return 2
    if not engine.has_lora():
        print(f"error: {args.model} has no LoRA adapters", file=sys.stderr)
        return 2
    if args.lora_command == "merge":
        merged = engine.merge_lora()
        engine.save_model(args.out)
        print(f"saved: {args.out} | {merged} weight matrices merged (no adapters left)")
        return 0
    info = engine.save_adapter(args.out)
    print(f"saved: {args.out} | {info['bytes'] / 1e6:.2f} MB, {info['adapters']} adapters, "
          f"{info['new_rows']} new token rows")
    return 0


def cmd_distill(args: argparse.Namespace) -> int:
    """2.16: a teacher model answers a file of prompts; its replies become dialogs."""
    from auralite.engine.distill import distill_dialogs, load_teacher, meta_path, read_prompts
    if not os.path.exists(args.prompts):
        print(f"error: prompts file not found: {args.prompts}", file=sys.stderr)
        return 2
    try:
        prompts, skipped = read_prompts(args.prompts)
    except (OSError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.limit:
        prompts = prompts[:int(args.limit)]
    if not prompts:
        print(f"error: no prompts in {args.prompts} (one per line in a .txt; records with "
              f"\"prompt\" or dialogs in a .jsonl)", file=sys.stderr)
        return 2
    try:
        teacher = load_teacher(args.teacher, license=args.license, n_ctx=args.n_ctx,
                               n_threads=args.threads)
    except (FileNotFoundError, ValueError, ImportError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    quiet = getattr(args, "quiet", False)
    if not quiet:
        print(f"teacher: {teacher.name} ({teacher.kind}) | license: "
              f"{teacher.license or 'unknown (check the model card)'}")
        print(f"prompts: {len(prompts):,} from {os.path.basename(args.prompts)}"
              + (f" ({skipped:,} records without a question skipped)" if skipped else ""))
    stop = threading.Event()
    previous = _stop_on_sigint(stop, "stopping after this prompt; run the same command "
                                     "again to continue")
    shown = [0.0]

    def progress(info: dict) -> None:
        now = time.monotonic()
        last = info["done"] == info["total"]
        if quiet or (now - shown[0] < 1.0 and not last):
            return
        shown[0] = now
        rate = float(info.get("prompts_per_second") or 0.0)
        eta = info.get("eta_seconds")
        text = (f"distill: {info['done']:,}/{info['total']:,} prompts | {info['kept']:,} "
                f"dialogs | {info['pairs']:,} pairs | {60.0 * rate:.1f}/min")
        if eta:
            text += f" | ETA {_fmt_eta(float(eta))}"
        end = "\n" if last or not sys.stdout.isatty() else ""
        print(("\r" if sys.stdout.isatty() else "") + text, end=end, flush=True)

    try:
        meta = distill_dialogs(
            teacher, prompts, args.out, samples=args.samples, system=args.system,
            max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p,
            judge=args.judge, pairs_path=args.pairs, keep_truncated=args.keep_truncated,
            overwrite=args.overwrite, seed=args.seed,
            prompts_file=os.path.abspath(args.prompts), on_progress=progress, stop_event=stop)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    counts, speed = meta["counts"], meta["speed"]
    line = f"saved: {args.out} ({meta['files']['dialogs']['records']:,} dialogs)"
    if args.pairs:
        line += f" | {args.pairs} ({meta['files']['pairs']['records']:,} pairs)"
    if counts.get("truncated"):
        line += f" | {counts['truncated']:,} replies cut at --max-tokens dropped"
    if counts.get("rejected"):
        line += f" | {counts['rejected']:,} rejected by the judge"
    if counts.get("no_reference"):
        line += f" | {counts['no_reference']:,} prompts without a reference answer"
    if speed.get("prompts_per_hour"):
        line += (f" | {speed['prompts_per_hour']:,.0f} prompts/hour, "
                 f"{speed.get('tokens_per_second') or 0:.1f} tokens/s")
    print(line)
    print(f"metadata: {meta_path(args.out)}")
    if meta.get("stopped"):
        print(f"stopped at {meta['progress']['done']:,} of {meta['progress']['total']:,}: run "
              f"the same command again to continue")
        return 130
    return 0


def _fmt_eta(seconds: float) -> str:
    m, sec = divmod(int(max(0.0, seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


def cmd_cluster_worker(args: argparse.Namespace) -> int:
    """Join the training runs of another computer (2.14, auralite/engine/cluster.py)."""
    from auralite.engine.cluster import ClusterError, run_worker

    def status(message: str) -> None:
        print(f"[cluster] {message}", flush=True)

    stop = threading.Event()
    try:
        served = run_worker(args.main, once=args.once, timeout_s=float(args.timeout),
                            token=args.token, on_status=status, stop_event=stop)
    except KeyboardInterrupt:
        stop.set()
        print("[cluster] stopped", flush=True)
        return 130
    except (ClusterError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(f"[cluster] served {served} run(s)", flush=True)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from auralite.engine import AuraLiteEngine
    engine = AuraLiteEngine()
    engine.load_model(args.model)
    engine.set_context_window(_context_window_arg(args))
    if getattr(args, "compile", False):
        seconds = engine.enable_inference_compile()
        if seconds:
            print(f"torch.compile: ready in {seconds:.2f}s "
                  f"(kernels cached in $AURALITE_HOME/torch_inductor)")
        else:
            print("torch.compile unavailable — running eagerly", file=sys.stderr)
    out = engine.generate(args.prompt, args.length,
                          temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    print(out)
    return 0


def _chat_grammar(args: argparse.Namespace):
    """The JSON grammar of `auralite chat --json / --json-schema FILE` (2.17), or None."""
    from auralite.engine import structured

    if getattr(args, "json_schema", None):
        with open(args.json_schema, "r", encoding="utf-8") as f:
            return structured.compile_schema(json.load(f))
    if getattr(args, "json", False):
        return structured.json_object_grammar()
    return None


def cmd_chat(args: argparse.Namespace) -> int:
    from auralite.engine import AuraLiteEngine
    from auralite.engine.structured import SchemaError
    try:
        grammar = _chat_grammar(args)
    except (OSError, ValueError, SchemaError) as err:
        print(f"error: --json-schema: {err}", file=sys.stderr)
        return 2
    engine = AuraLiteEngine()
    engine.load_model(args.model)
    engine.set_context_window(_context_window_arg(args))
    if getattr(args, "compile", False):
        engine.enable_inference_compile()
    if grammar is not None:
        try:
            engine.structured_check(grammar, args.length)
        except ValueError as err:
            print(f"error: {err}", file=sys.stderr)
            return 2
    history: list[dict] = []
    if args.system:
        history.append({"role": "system", "content": args.system})
    if not args.template and engine.continues_text():
        # 2.21.3: a model of plain text has never seen a chat template
        from auralite import bundle
        print("note: this model was trained on plain text, not on dialogs: it continues "
              "text and knows no chat template, so its replies may begin mid-word and run "
              "on. To write on from a prompt: "
              + bundle.console_command("generate", "--model", args.model, "--prompt", "...")
              + "; to chat, train it on dialogs", file=sys.stderr)

    def _respond(user_text: str) -> None:
        history.append({"role": "user", "content": user_text})
        reply = engine.generate_chat(history, args.length,
                                     temperature=args.temperature,
                                     top_k=args.top_k, top_p=args.top_p,
                                     chat_template=args.template, grammar=grammar)
        history.append({"role": "assistant", "content": reply})
        print(f"assistant: {reply}")

    if args.prompt:
        _respond(args.prompt)
        return 0
    print("AuraLite chat — Ctrl-D / 'exit' to quit")
    while True:
        try:
            user = input("you: ").strip()
        except EOFError:
            break
        if user.lower() in {"exit", "quit", ""}:
            break
        _respond(user)
    return 0


#: `auralite export --format` → default file suffix.
EXPORT_SUFFIXES = {"pt2": ".pt2", "onnx": ".onnx", "torchscript": ".torchscript.pt",
                   # 2.18.1: a Hugging Face folder, a GGUF file (llama.cpp, Ollama)
                   "hf": "-hf", "gguf": ".gguf"}


class _TransferPrinter:
    """Progress of `auralite import` / `export`: downloads, conversion, writing (2.18.1)."""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.last = 0.0
        self.shown = ""

    def __call__(self, info: dict) -> None:
        if self.quiet:
            return
        now = time.monotonic()
        stage = info.get("stage")
        if stage == "download":
            total = int(info.get("all_total") or info.get("total") or 0)
            done = int(info.get("all_bytes") or info.get("bytes") or 0)
            text = f"downloading {info.get('file')}: {done / 1e6:,.0f}" + (
                f" / {total / 1e6:,.0f} MB ({100 * done / total:.0f} %)" if total else " MB")
        else:
            text = f"{stage}: {info.get('done')}/{info.get('total')}"
        final = stage != "download" and info.get("done") == info.get("total")
        if not final and now - self.last < 2.0:
            return
        self.last = now
        if text != self.shown:
            print(text, flush=True)
            self.shown = text


def cmd_import(args: argparse.Namespace) -> int:
    """Import a Hugging Face Llama / Mistral / Qwen2 / Qwen3 as a native checkpoint (2.18.1)."""
    from auralite.engine import AuraLiteEngine
    from auralite.engine.hf_models import HFModelError, describe_import
    source = args.source
    if not str(source).lower().startswith("hf:") and not os.path.isdir(source):
        print(f"error: {source} is neither hf:ORG/NAME nor a model folder", file=sys.stderr)
        return 2
    if getattr(args, "revision", None) and str(source).lower().startswith("hf:") \
            and "@" not in source:
        source = f"{source}@{args.revision}"
    engine = AuraLiteEngine()
    try:
        info = engine.import_hf_model(source, quantize=args.quantize, dtype=args.dtype,
                                      progress_callback=_TransferPrinter(args.quiet))
    except (HFModelError, FileNotFoundError, InterruptedError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    engine.save_model(args.out, include_optimizer=False)
    print(f"imported: {describe_import(info)}")
    print(f"saved: {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB) | fine-tune: auralite "
          f"train --data DIALOGS.jsonl --resume {args.out} --lora 16"
          + ("" if args.quantize else " [--quantize nf4]"))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export a native checkpoint as .pt2 (torch.export), ONNX or TorchScript."""
    import warnings

    from auralite.engine import AuraLiteEngine
    from auralite.export import ModelExporter

    if not os.path.exists(args.model):
        print(f"error: checkpoint not found: {args.model}", file=sys.stderr)
        return 2
    engine = AuraLiteEngine()
    engine.load_model(args.model)
    if engine.is_gguf_model() or engine.is_hf_model():
        print("error: only native AuraLite checkpoints can be exported "
              "(.gguf and Hugging Face models have their own tooling)", file=sys.stderr)
        return 2
    out = args.out or os.path.splitext(args.model)[0] + EXPORT_SUFFIXES[args.format]
    if args.format in ("hf", "gguf"):
        # 2.18.1: for transformers / vLLM and for llama.cpp / Ollama / LM Studio
        from auralite.engine.hf_models import HFModelError
        try:
            if args.format == "hf":
                info = engine.export_hf_model(out, dtype=args.dtype or "bfloat16",
                                              progress_callback=_TransferPrinter(args.quiet))
                print(f"exported: {out}/ ({info['architecture']}, {info['dtype']}, "
                      f"{info['bytes'] / 1e9:.2f} GB in {info['shards']} file(s), chat "
                      f"template {info['chat_template']}; LoRA adapters merged)")
            else:
                info = engine.export_gguf_model(out, quant=args.quant,
                                                progress_callback=_TransferPrinter(args.quiet))
                for warning in info["warnings"]:
                    print(f"warning: {warning}", file=sys.stderr)
                print(f"exported: {out} (GGUF {info['type']}, llama.cpp architecture "
                      f"{info['architecture']}, pre-tokenizer {info['pre_tokenizer']}, "
                      f"{info['bytes'] / 1e9:.2f} GB) | smaller: llama-quantize {out} "
                      f"OUT.gguf Q4_K_M")
        except HFModelError as err:
            print(f"error: {err}", file=sys.stderr)
            return 2
        return 0
    exporter = ModelExporter(engine.model, engine.tokenizer, engine.device)
    try:
        if args.format == "pt2":
            exporter.export_pt2(out, dynamic=not args.static)
            note = "load with torch.export.load(path).module() or auralite.export.load_pt2"
        elif args.format == "onnx":
            exporter.export_onnx(out, opset_version=args.opset, dynamic_axes=not args.static,
                                 exporter=args.onnx_exporter)
            note = f"{exporter.last_onnx_exporter} exporter"
        else:
            print("warning: TorchScript is deprecated by PyTorch — prefer --format pt2",
                  file=sys.stderr)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                exporter.export_torchscript(out)
            note = "deprecated format"
    except ImportError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(f"exported: {out} ({note})")
    return 0


#: Import string of the ASGI app `auralite serve` runs. It must resolve in an
#: installed AuraLite, where only the `auralite` package exists (the 2.6 flat name
#: `server.openai_server:app` would only work inside a checkout).
SERVER_APP = "auralite.server.openai_server:app"


def _image_train_params(args: argparse.Namespace) -> tuple[dict, str | None]:
    """Merge --preset, --params and the flag overrides; validate strictly.

    The image CLI is new in 2.10, so unlike `train` it has no legacy keys to
    tolerate: every key must be known, and near-misses get a suggestion.
    """
    import difflib

    from auralite.image_engine import (_TRAIN_EXTRAS, IMAGE_PRESETS, ImageGenEngine,
                                       preset_slug, validate_image_params)
    params: dict = {}
    model_type = getattr(args, "model_type", None)
    if args.preset:
        wanted = args.preset.strip().lower()
        slugs = {preset_slug(n): n for n in IMAGE_PRESETS}
        typed = f"{model_type}-{wanted}"
        if model_type and model_type != "diffusion" and typed in slugs:
            names = [slugs[typed]]       # 2.14.2: `--type gan --preset tiny` → "GAN Tiny"
        else:
            names = [slugs[wanted]] if wanted in slugs else []
        if not names:
            names = [n for n in IMAGE_PRESETS if n.lower().startswith(wanted)]
        if len(names) != 1:
            choices = ", ".join(slugs)
            return {}, f"unknown --preset {args.preset!r} (choose: {choices})"
        params.update(IMAGE_PRESETS[names[0]])
    if args.params:
        with open(args.params, "r", encoding="utf-8") as f:
            params.update(json.load(f))
    if model_type:
        params["model_type"] = model_type
    for key in ("epochs", "batch_size", "seed", "val_split", "img_size"):
        value = getattr(args, key, None)
        if value is not None:
            params[key] = value
    if getattr(args, "grayscale", False):
        params["channels"] = 1
    if getattr(args, "classes", False):
        params["class_conditional"] = True
    known = set(ImageGenEngine.DEFAULTS) | set(_TRAIN_EXTRAS)
    unknown = sorted(k for k in params if k not in known)
    if unknown:
        hints = []
        for key in unknown:
            close = difflib.get_close_matches(key, sorted(known), n=1)
            hints.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        return {}, "unknown image training parameter(s): " + ", ".join(hints)
    errors = validate_image_params({**ImageGenEngine.DEFAULTS, **params})
    if errors:
        return {}, "invalid image training parameters: " + "; ".join(errors)
    return params, None


def cmd_image_train(args: argparse.Namespace) -> int:
    if not os.path.isdir(args.images):
        print(f"error: image folder not found: {args.images}", file=sys.stderr)
        return 2
    params, error = _image_train_params(args)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    from auralite.image_engine import ImageGenEngine, save_image_grid, scan_image_folder
    if not scan_image_folder(args.images):
        print(f"error: no supported images under {args.images}", file=sys.stderr)
        return 2
    engine = ImageGenEngine()
    if args.resume:
        engine.load_model(args.resume)
        params["continue_training"] = True
    stop_event = threading.Event()
    previous = _stop_on_sigint(stop_event, "stopping after this batch; the weights trained "
                                           "so far are saved")
    try:
        engine.train(args.images, params,
                     progress_callback=None if getattr(args, "quiet", False)
                     else _ProgressPrinter(), stop_event=stop_event)
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    summary = engine.last_train_summary
    if summary.get("stopped_during"):
        print("stopped while decoding the images; nothing saved")
        return 130
    engine.save_model(args.model)
    line = (f"saved: {args.model} | {engine.model_type} | {summary.get('steps')} steps in "
            f"{summary.get('seconds')}s | params {engine.parameter_summary()}")
    if engine.class_names:
        line += f" | classes: {', '.join(engine.class_names)}"
    if summary.get("best_val_loss") is not None:
        metric = "SWD " if summary.get("val_metric") == "swd" else ""
        line += (f" | best val {metric}{summary['best_val_loss']:.4f} "
                 f"(epoch {summary.get('best_epoch')})")
    if summary.get("augment") == "ada":
        line += f" | augmentation p {summary['augment_p']:.2f}"
    elif summary.get("augment") == "diffaug":
        line += " | DiffAugment"
    print(line)
    if args.samples:
        images = engine.generate(n=16, steps=20, sampler="dpm++2m", seed=0,
                                 class_label="all" if engine.class_names else None,
                                 guidance_scale=2.0)
        save_image_grid(images, args.samples)
        print(f"samples: {args.samples}")
    return 0


def cmd_image_generate(args: argparse.Namespace) -> int:
    from auralite.image_engine import ImageGenEngine, save_image_grid
    if not os.path.exists(args.model):
        print(f"error: checkpoint not found: {args.model}", file=sys.stderr)
        return 2
    engine = ImageGenEngine()
    engine.load_model(args.model)
    if args.class_label is not None and not engine.class_names:
        print("error: --class needs a class-conditional model (image-train --classes)",
              file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    try:
        images = engine.generate(**_image_generate_kwargs(args, args.n))
    except ValueError as err:          # unknown class name / index, sampler of another type
        print(f"error: {err}", file=sys.stderr)
        return 2
    seconds = time.perf_counter() - t0
    if args.out_dir:
        from PIL import Image
        os.makedirs(args.out_dir, exist_ok=True)
        for i, im in enumerate(images):
            Image.fromarray(im).save(os.path.join(args.out_dir, f"sample_{i:03d}.png"))
        print(f"saved {len(images)} images to {args.out_dir}")
    if args.out:
        save_image_grid(images, args.out)
        print(f"saved: {args.out}")
    if engine.model_type == "gan":
        print(f"{len(images)} images, GAN (one network call each), {seconds:.2f}s")
    else:
        print(f"{len(images)} images, {args.sampler} x {args.steps} steps, {seconds:.2f}s")
    return 0


def _image_generate_kwargs(args: argparse.Namespace, n: int) -> dict:
    """`ImageGenEngine.generate` keyword arguments from the image-generate/-eval flags."""
    kwargs: dict[str, Any] = dict(n=n, steps=args.steps, eta=args.eta, seed=args.seed,
                                  use_ema=not args.no_ema, sampler=args.sampler,
                                  class_label=args.class_label, guidance_scale=args.guidance)
    if getattr(args, "truncation", None):
        kwargs["truncation"] = float(args.truncation)
    return kwargs


def cmd_image_eval(args: argparse.Namespace) -> int:
    """2.14.2: patch SWD of generated images against a folder of real images."""
    from auralite.image_engine import ImageGenEngine, scan_image_folder
    if not os.path.exists(args.model):
        print(f"error: checkpoint not found: {args.model}", file=sys.stderr)
        return 2
    if not os.path.isdir(args.images) or not scan_image_folder(args.images):
        print(f"error: no supported images under {args.images}", file=sys.stderr)
        return 2
    engine = ImageGenEngine()
    engine.load_model(args.model)
    if args.class_label is not None and not engine.class_names:
        print("error: --class needs a class-conditional model (image-train --classes)",
              file=sys.stderr)
        return 2
    kwargs = _image_generate_kwargs(args, args.n)
    kwargs.pop("n")
    seed = kwargs.pop("seed")
    try:
        result = engine.evaluate(args.images, n=args.n,
                                 seed=0 if seed is None else int(seed), **kwargs)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"model": args.model, "model_type": engine.model_type, **{
            k: ({str(r): v for r, v in val.items()} if k == "levels" else val)
            for k, val in result.items()}}))
        return 0
    levels = ", ".join(f"{r}px {v:.2f}" for r, v in result["levels"].items())
    print(f"model: {args.model} ({engine.model_type}, {engine.parameter_summary()} params)")
    print(f"SWD: {result['swd']:.2f}  (lower is better; per level: {levels})")
    if result.get("real_floor") is not None:
        print(f"real-vs-real floor: {result['real_floor']:.2f}  "
              f"({result['floor_images']} vs {result['floor_images']} real images)")
    print(f"{result['images']} generated vs {result['images']} real images, "
          f"{result['seconds']:.2f}s")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        from auralite import bundle
        if bundle.packaged():                 # 2.21: the app leaves the server out
            print("error: the server needs FastAPI and uvicorn", file=sys.stderr)
            for line in bundle.missing_part("uvicorn", "serve"):
                print(f"  {line}", file=sys.stderr)
            return 2
        print("error: install serving dependencies: pip install fastapi uvicorn pydantic",
              file=sys.stderr)
        return 2
    os.environ["AURALITE_MODEL"] = os.path.abspath(args.model)
    window = _context_window_arg(args)
    if window != "auto" or "AURALITE_CONTEXT_WINDOW" not in os.environ:
        os.environ["AURALITE_CONTEXT_WINDOW"] = window
    # Single worker on purpose: engine/KV-cache and the rate limiter live in
    # process memory (see the auralite/server/openai_server.py docstring).
    uvicorn.run(SERVER_APP, host=args.host, port=args.port, workers=1)
    return 0


def _bench(model_path: str, n_tokens: int, modes: bool = False,
           compile_bench: bool = False) -> int:
    """Measure generation speed, weight memory and (optionally) KV-cache modes.

    `--bench-modes` rebuilds the model with fp16/int8 KV caches and reports
    tokens/s next to the cache size per token; `--bench-compile` compares prompt
    processing (prefill) eager vs `torch.compile`. Both are *measurements*, not
    expectations — on CPU the compiled decode path is slower, which is exactly
    why the engine only routes prefill through it.
    """
    from auralite.engine import AuraLiteEngine
    engine = AuraLiteEngine()
    engine.load_model(model_path)
    engine.generate("hello", 1, temperature=0.0, top_k=1)          # warm-up
    start = time.perf_counter()
    engine.generate("hello", n_tokens, temperature=0.0, top_k=1)
    elapsed = max(1e-9, time.perf_counter() - start)
    print(f"bench: {n_tokens} tokens in {elapsed:.2f}s "
          f"→ {n_tokens / elapsed:.1f} tokens/s")
    try:
        weights = sum(p.numel() * p.element_size() for p in engine.model.parameters())
        print(f"weights_mb: {weights / 2**20:.2f}")
    except (AttributeError, TypeError):
        print("weights_mb: n/a (external backend)")
    model = engine.model
    for attr in ("kv_cache_dtype", "sliding_window", "use_flex_attention",
                 "use_alibi", "lora_rank"):
        print(f"{attr}: {getattr(model, attr, None)}")

    if modes:
        try:
            rows = engine.bench_generation_modes(tokens=max(8, n_tokens), repeats=2)
        except RuntimeError as err:
            print(f"bench-modes: n/a ({err})", file=sys.stderr)
            return 1
        print("kv-cache modes (same weights, rebuilt model):")
        print(f"  {'mode':<10} {'tokens/s':>10} {'kv B/token':>12}")
        for row in rows:
            print(f"  {row['mode']:<10} {row['tokens_per_sec']:>10.1f} "
                  f"{row['kv_bytes_per_token']:>12}")
        print("  note: lower precision saves KV memory; on CPU it does not "
              "speed up decoding.")
    if compile_bench:
        try:
            rows = engine.bench_prefill(repeats=2)
        except RuntimeError as err:
            print(f"bench-compile: n/a ({err})", file=sys.stderr)
            return 1
        compile_seconds = rows[0]["compile_seconds"] if rows else 0.0
        print(f"torch.compile prefill (first compile {compile_seconds:.1f}s, "
              f"kernels cached in $AURALITE_HOME/torch_inductor):")
        print(f"  {'shape':<10} {'eager tok/s':>12} {'compiled tok/s':>15} {'speedup':>8}")
        for row in rows:
            print(f"  {row['shape']:<10} {row['eager_tokens_per_sec']:>12.0f} "
                  f"{row['compiled_tokens_per_sec']:>15.0f} {row['speedup']:>7.2f}×")
        print("  note: single-token decode stays eager on purpose (compilation "
              "re-checks guards per KV length and loses).")
    return 0


def _gguf_perplexity(args) -> int:
    """Score a `.gguf` model on a text file, with caching (roadmap P2.12)."""
    from auralite.engine.gguf_proxy import (GGUFNotAvailableError, bench_gguf_perplexity,
                                         clear_perplexity_cache, perplexity_cache_path)
    if not os.path.exists(args.text):
        print(f"error: text file not found: {args.text}", file=sys.stderr)
        return 2
    try:
        text = open(args.text, "r", encoding="utf-8", errors="replace").read()
    except OSError as err:
        print(f"error: cannot read {args.text}: {err}", file=sys.stderr)
        return 2
    if getattr(args, "clear_cache", False):
        removed = clear_perplexity_cache()
        print(f"perplexity cache cleared: {removed} ({perplexity_cache_path()})")
    if not text.strip():
        print("error: the text file is empty", file=sys.stderr)
        return 2

    try:
        result = bench_gguf_perplexity(
            args.model, text, n_ctx=int(getattr(args, "n_ctx", 2048) or 2048),
            use_cache=not getattr(args, "no_cache", False),
            n_threads=getattr(args, "threads", None),
        )
    except GGUFNotAvailableError as err:
        print(f"error: {err}", file=sys.stderr)
        return 3
    source = "cache" if result["cached"] else f"{result['seconds']:.2f}s"
    print(f"perplexity: {result['perplexity']:.4f}  ({source})")
    print(f"bits_per_token: {result['bits_per_token']:.4f}")
    print(f"scored_tokens: {result['n_tokens']}  text_chars: {result['text_chars']}  "
          f"n_ctx: {result['n_ctx']}")
    print(f"cache_key: {result['key']}")
    if not result["cached"]:
        print(f"cache_file: {perplexity_cache_path()}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    # GGUF scoring never touches the checkpoint reader (a .gguf is not a .pt).
    if getattr(args, "perplexity", False):
        return _gguf_perplexity(args)
    # v2.6.5: checkpoints are read with the safe unpickler (see safe_io.py).
    from auralite.engine.safe_io import safe_torch_load
    if not os.path.exists(args.model):
        print(f"error: checkpoint not found: {args.model}", file=sys.stderr)
        return 2
    try:
        ckpt = safe_torch_load(args.model, map_location="cpu")
    except Exception as err:  # refusal to unpickle / corrupt file / not a .pt
        print(f"error: cannot read {args.model}: {err}", file=sys.stderr)
        return 2
    if not isinstance(ckpt, dict):
        print(f"error: {args.model} is not an AuraLite checkpoint", file=sys.stderr)
        return 2
    keys = ["checkpoint_version", "chat_template", "vocab_size", "d_model", "n_heads",
            "n_kv_heads", "n_layers", "d_ff", "max_seq_len", "use_alibi", "use_qk_norm",
            "sliding_window", "rope_scaling", "dropout", "lora_rank",
            # 2.18.1: the fields of a model imported from Hugging Face
            "head_dim", "qkv_bias", "norm_eps", "qk_norm_shared"]
    for k in keys:
        if k in ckpt:
            print(f"{k}: {ckpt[k]}")
    source = ckpt.get("hf_source")
    if isinstance(source, dict):
        print(f"imported_from: {source.get('repo')} ({source.get('architecture')})")
    quantization = ckpt.get("quantization")
    if isinstance(quantization, dict):
        print(f"quantization: {quantization.get('method')} on "
              f"{len(quantization.get('modules') or [])} layers (block "
              f"{quantization.get('block_size')}, double quantization "
              f"{bool(quantization.get('double_quant'))})")
    if isinstance(ckpt.get("lora_config"), dict):        # 2.16: where the adapters sit
        config = ckpt["lora_config"]
        print(f"lora_targets: {', '.join(config.get('targets') or [])} "
              f"(alpha {config.get('alpha')}, rslora {bool(config.get('rslora'))})")
    if ckpt.get("max_seq_len"):
        trained = ckpt.get("sliding_window")
        window = min(int(trained), int(ckpt["max_seq_len"])) if trained else int(ckpt["max_seq_len"])
        print(f"generation_window: {window} tokens (the trained length; "
              f"generate/chat/serve --full-context lifts it)")
    tok = ckpt.get("tokenizer") or {}
    if tok:
        try:                                # 2.14.1: plugged-in tokenizers have no "vocab" list
            from auralite.engine.tokenizer_io import describe_tokenizer
            from auralite.engine.tokenizers import tokenizer_from_dict
            built = tokenizer_from_dict(tok)
            print(f"tokenizer: {tok.get('kind', '?')} (vocab={built.vocab_size})")
            print(f"tokenizer_detail: {describe_tokenizer(built)}")
            if getattr(built, "special_tokens", None):      # 2.15: a chat model's markers
                print(f"special_tokens: {', '.join(built.special_tokens)}")
        except Exception:                   # noqa: BLE001 - a damaged dict still prints
            print(f"tokenizer: {tok.get('kind', '?')} (vocab={len(tok.get('vocab', []))})")
    params = ckpt.get("params_used") or {}
    if params:
        print("params_used:")
        for k in sorted(params):
            print(f"  {k}: {params[k]}")
    if getattr(args, "backends", False):
        from auralite.quantization_backends import describe_backends
        print("quantization backends:")
        for key, value in describe_backends().items():
            print(f"  {key}: {value}")
    if getattr(args, "bench", 0):
        return _bench(args.model, int(args.bench),
                      modes=bool(getattr(args, "bench_modes", False)),
                      compile_bench=bool(getattr(args, "bench_compile", False)))
    return 0


# ----------------------------------------------------------------------
#  auralite tokenizer ... (2.14.1)
# ----------------------------------------------------------------------
def _read_text(path: str) -> str | None:
    if not os.path.exists(path):
        print(f"error: text file not found: {path}", file=sys.stderr)
        return None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if not text:
        print(f"error: the text file is empty: {path}", file=sys.stderr)
        return None
    return text


def cmd_tokenizer_train(args: argparse.Namespace) -> int:
    """Train only a tokenizer and save it (`auralite train --tokenizer` uses it)."""
    from auralite.engine import DataFormatError, DialogSet, load_training_data
    from auralite.engine.tokenizer_io import describe_tokenizer, save_tokenizer, train_tokenizer
    # 2.15: files, folders and globs, cut like `auralite train` cuts them
    try:
        data = load_training_data(args.text)
    except (FileNotFoundError, DataFormatError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if isinstance(data, DialogSet):
        print("error: train the tokenizer on text; `auralite train` on dialogs adds the chat "
              "template's tokens to the model's tokenizer", file=sys.stderr)
        return 2
    text, fit_chars = data.layout(float(args.val_split or 0.0))
    if not text:
        print("error: the training text is empty", file=sys.stderr)
        return 2
    printer = None if getattr(args, "quiet", False) else _ProgressPrinter()
    stop = threading.Event()

    def handler(signum, frame):
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set()
        print("\n^C: stopping — the merges learned so far are kept (Ctrl+C again to abort)",
              file=sys.stderr, flush=True)

    try:
        previous = signal.signal(signal.SIGINT, handler)
    except (ValueError, OSError):            # not the main thread
        previous = None
    try:
        tokenizer, info = train_tokenizer(
            text, kind=args.kind, vocab_size=args.vocab_size, pretokenizer=args.pretokenizer,
            val_split=args.val_split, progress=printer.stage if printer else None,
            should_stop=stop.is_set, source=", ".join(os.path.basename(p) for p in args.text),
            fit_chars=fit_chars)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    save_tokenizer(tokenizer, args.out, training=info)
    line = f"saved: {args.out} | {describe_tokenizer(tokenizer)}"
    if info.get("chars_per_token"):
        line += f" | {info['chars_per_token']:.2f} characters per token"
    print(line)
    if args.kind == "bpe" and fit_chars is not None:
        print(f"fitted on the training documents ({info['fit_chars']:,} of "
              f"{info['text_chars']:,} characters): the held-out documents are the "
              f"validation text of `auralite train` with the same val_split")
    elif args.kind == "bpe" and info["fit_chars"] < info["text_chars"]:
        print(f"fitted on the first {info['fit_chars']:,} of {info['text_chars']:,} characters: "
              f"the last {info['val_split']:g} is the validation text of `auralite train` "
              f"with the same val_split")
    if info.get("stopped"):
        print(f"stopped early: {info['merges']:,} of {info['requested_vocab_size']:,} tokens' "
              f"merges learned")
    elif info.get("exhausted"):
        print(f"no pair occurs twice any more: {info['vocab_size']:,} tokens instead of "
              f"{info['requested_vocab_size']:,}")
    return 0


def _tokenizer_or_error(path: str):
    from auralite.engine.hf_bpe import TokenizerFormatError
    from auralite.engine.tokenizer_io import read_tokenizer_file
    try:
        return read_tokenizer_file(path)
    except (TokenizerFormatError, FileNotFoundError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return None


def _token_piece(tokenizer, token_id: int) -> str:
    text_of = getattr(tokenizer, "token_text", None)
    piece = text_of(token_id) if callable(text_of) else tokenizer.decode([token_id])
    return repr(piece)[1:-1]


def cmd_tokenizer_info(args: argparse.Namespace) -> int:
    from auralite.engine.tokenizer_io import describe_tokenizer
    loaded = _tokenizer_or_error(args.file)
    if loaded is None:
        return 2
    tokenizer, info = loaded
    print(f"file: {info.get('path', args.file)}")
    print(f"format: {info.get('format')}")
    print(f"tokenizer: {describe_tokenizer(tokenizer)}")
    print(f"kind: {tokenizer.kind}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    special = getattr(tokenizer, "special_tokens", None)
    if special:
        print(f"special_tokens: {', '.join(special[:12])}" + (" ..." if len(special) > 12 else ""))
    for key, value in sorted((info.get("training") or {}).items()):
        if key != "text_sha256":
            print(f"training.{key}: {value}")
    if getattr(args, "text", None):
        text = _read_text(args.text)
        if text is None:
            return 2
        ids = tokenizer.encode(text)
        print(f"sample: {len(text):,} characters -> {len(ids):,} tokens "
              f"({len(text) / max(1, len(ids)):.2f} characters per token)")
        print("first tokens: " + " | ".join(_token_piece(tokenizer, i) for i in ids[:24]))
    return 0


def cmd_tokenizer_encode(args: argparse.Namespace) -> int:
    loaded = _tokenizer_or_error(args.file)
    if loaded is None:
        return 2
    tokenizer = loaded[0]
    ids = tokenizer.encode(args.text)
    print(" ".join(str(i) for i in ids))
    print(" | ".join(_token_piece(tokenizer, i) for i in ids))
    back = tokenizer.decode(ids)
    print(f"{len(ids)} tokens, {len(args.text) / max(1, len(ids)):.2f} characters per token, "
          + ("decodes back exactly" if back == args.text else f"decodes back as {back!r}"))
    return 0


def cmd_tokenizer_convert(args: argparse.Namespace) -> int:
    from auralite.engine.tokenizer_io import describe_tokenizer, save_tokenizer
    loaded = _tokenizer_or_error(args.source)
    if loaded is None:
        return 2
    tokenizer, info = loaded
    save_tokenizer(tokenizer, args.out, training=info.get("training") or None)
    print(f"saved: {args.out} | {describe_tokenizer(tokenizer)} (read as {info.get('format')})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from auralite import __version__     # not auralite.engine: --version needs no torch

    parser = argparse.ArgumentParser(prog="auralite", description="AuraLite AI CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}",
                        help="print the installed AuraLite version and exit")
    _add_common_flags(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train a model on text files, or fine-tune one on dialogs")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("--text", "--data", dest="text", nargs="+", required=True, metavar="PATH",
                   help="training data (UTF-8): text files, folders (their .txt/.md/.jsonl "
                        "files), globs; .jsonl/.json files of {\"text\": ...} records, or of "
                        "dialogs ({\"messages\": [...]}, ShareGPT, Alpaca, prompt/completion) "
                        "to fine-tune a chat model; 2.18: a folder of `auralite data prepare`, "
                        "compressed (.gz .bz2 .xz .zst) and .parquet files, hf:NAME datasets "
                        "(text over 256 MB is tokenized into shards first)")
    # 2.18: big models and big data
    p.add_argument("--preset", metavar="SIZE",
                   help="a model size: 25m, 45m, 125m, 350m, 700m, 1.2b, 2.4b (auralite plan "
                        "--list); --params overrides its fields")
    p.add_argument("--max-tokens", type=int, metavar="N",
                   help="train on this many tokens (a budget; the epochs follow from it)")
    p.add_argument("--eval-every", type=int, metavar="STEPS",
                   help="also validate every STEPS optimizer steps")
    p.add_argument("--memory-plan", choices=["off", "warn", "auto"],
                   help="warn (default): log the estimated memory and warn when it is more "
                        "than is free; auto: also split the batch (gradient accumulation) and "
                        "turn on gradient checkpointing to fit (when nothing fits: the "
                        "smallest step, one window with checkpointing)")
    p.add_argument("--autosave", metavar="PATH",
                   help="save the run here as it trains (every 30 minutes unless "
                        "--autosave-steps / --autosave-minutes); --resume PATH "
                        "--continue-run continues it after a stop or a crash")
    p.add_argument("--autosave-steps", type=int, metavar="N", help="autosave every N steps")
    p.add_argument("--autosave-minutes", type=float, metavar="M", help="autosave every M minutes")
    p.add_argument("--continue-run", action="store_true",
                   help="with --resume: continue the run saved in that checkpoint exactly "
                        "where it stopped (same data order, schedule and counters)")
    p.add_argument("--stream", choices=["auto", "always", "never"], default="auto",
                   help="tokenize text into shards on disk first: auto (over 256 MB, "
                        "compressed, Parquet, hf:), always, never")
    p.add_argument("--shards-dir", metavar="DIR",
                   help="where raw data is tokenized to (default: under ~/.auralite/datasets, "
                        "reused by later runs)")
    p.add_argument("--text-field", metavar="NAME",
                   help="the text field of .jsonl/.parquet/hf: records (default text)")
    p.add_argument("--hf-config", metavar="NAME", help="hf: datasets: the configuration")
    p.add_argument("--hf-split", metavar="NAME", help="hf: datasets: the split (default train)")
    p.add_argument("--template", metavar="NAME",
                   help="dialogs: the chat template (chatml, gemma, phi or llama3; default: the "
                        "continued model's, else chatml)")
    p.add_argument("--lora", type=int, metavar="RANK",
                   help="train LoRA adapters of this rank instead of every weight (with "
                        "--resume: fine-tune a model with adapters)")
    p.add_argument("--lora-targets", choices=["all", "ffn", "attn"],
                   help="LoRA: the layers that get adapters (default all: FFN and attention)")
    p.add_argument("--lora-lr", type=float, metavar="LR",
                   help="LoRA: learning rate (default 10x the learning rate)")
    p.add_argument("--lora-alpha", type=float, metavar="ALPHA",
                   help="LoRA: adapter scale numerator (default the rank)")
    p.add_argument("--rslora", action="store_true",
                   help="LoRA: scale adapters by alpha/sqrt(rank) (rank-stabilised LoRA)")
    p.add_argument("--save-adapter", metavar="PATH",
                   help="LoRA: also write the adapters and new token rows alone to PATH "
                        "(apply them to the base with `auralite lora apply`)")
    p.add_argument("--quantize", choices=["nf4"],
                   help="QLoRA: keep the --resume model's attention and feed-forward weights "
                        "in 4 bits while LoRA adapters train (needs --lora)")
    p.add_argument("--optimizer", choices=["muon", "adamw", "adamw8bit"],
                   help="default muon (AdamW for LoRA); adamw8bit: AdamW with 8-bit states, "
                        "a quarter of AdamW's optimizer memory")
    p.add_argument("--prompt-weight", type=float, metavar="W",
                   help="dialogs: weight of the prompt tokens in the loss, 0 (the replies "
                        "only) to 1 (every token); default 0.1, or 0 with --lora")
    p.add_argument("--pack-dialogs", action="store_true",
                   help="dialogs: pack whole dialogs into the windows (a long one split where "
                        "a user turn starts) instead of cutting them from the stream")
    p.add_argument("--method", choices=["auto", "sft", "dpo", "simpo"],
                   help="preference pairs (prompt, chosen, rejected): dpo (default for pairs), "
                        "simpo, or sft (fine-tune on the chosen replies); needs --resume")
    p.add_argument("--beta", type=float, metavar="BETA",
                   help="DPO / SimPO: reward scale beta (default 0.1 for DPO, 2.0 for SimPO; "
                        "--params for lr, epochs, dpo_positive, simpo_gamma)")
    p.add_argument("--params", help="JSON file with training params")
    p.add_argument("--model", default="model.pt", help="output checkpoint")
    p.add_argument("--mmap", action="store_true",
                   help="stream the corpus into a uint32 memmap (low RAM use; "
                        "required for corpora that do not fit in memory). The "
                        "temporary file is deleted when training ends")
    p.add_argument("--corpus-cache", metavar="PATH",
                   help="where to keep the tokenised corpus (implies --mmap)")
    p.add_argument("--tokenizer", metavar="FILE",
                   help="train with this tokenizer instead of a new one: a file of "
                        "`auralite tokenizer train`, a Hugging Face tokenizer.json (BPE), "
                        "GPT-2 vocab.json (+ merges.txt) or a checkpoint")
    p.add_argument("--resume", metavar="CHECKPOINT",
                   help="continue training this checkpoint (a longer seq_length in "
                        "--params extends its context); 2.18.1: or hf:ORG/NAME / a model "
                        "folder, imported first (Llama, Mistral, Qwen2, Qwen3)")
    p.add_argument("--time-budget", type=_parse_duration, metavar="DURATION",
                   help="end training after this long (90, 90s, 10m, 1.5h), counted from "
                        "the start; the learning-rate decay is sized to end on time")
    p.add_argument("--cluster-workers", type=int, metavar="N",
                   help="train together with N other computers on the local network: "
                        "wait for them, then split every step between all of them "
                        "(they run: auralite cluster-worker THIS_COMPUTER)")
    p.add_argument("--cluster-port", type=int, default=29500, metavar="PORT",
                   help="port the workers connect to (default 29500)")
    p.add_argument("--cluster-host", metavar="ADDRESS",
                   help="address to listen on (default: every interface)")
    p.add_argument("--cluster-token", metavar="TEXT",
                   help="workers must give the same token (keeps runs on one network apart)")
    p.add_argument("--cluster-timeout", type=_parse_duration, metavar="DURATION",
                   help="how long to wait for the workers (default 10m)")
    # 2.20: the run library
    p.add_argument("--no-record", action="store_true",
                   help="keep this run out of the run library (auralite runs; also "
                        "AURALITE_NO_RUNS=1)")
    p.add_argument("--run-name", metavar="NAME",
                   help="the run's name in the run library (default: what it trains on what)")
    p.set_defaults(func=cmd_train)

    # 2.18: big datasets as token shards, and memory plans
    p = sub.add_parser("data", help="prepare big datasets: tokenize them once into shards")
    data_sub = p.add_subparsers(dest="data_command", required=True)
    q = data_sub.add_parser("prepare", help="tokenize files, folders or a Hugging Face dataset "
                                            "into token shards (resumable)")
    _add_common_flags(q, suppress_defaults=True)
    q.add_argument("sources", nargs="+", metavar="SOURCE",
                   help="files, folders, globs (text, .jsonl, .json, .parquet; .gz .bz2 .xz "
                        ".zst) and hf:NAME[:CONFIG][@SPLIT] datasets")
    q.add_argument("--out", required=True, metavar="DIR", help="the dataset folder to write")
    q.add_argument("--tokenizer", metavar="FILE",
                   help="tokenizer file, checkpoint, or hf:MODEL (its tokenizer.json); "
                        "default: fit a new BPE on a sample of the data")
    q.add_argument("--vocab-size", type=int, default=8192, metavar="N",
                   help="a new tokenizer's vocabulary (default 8192)")
    q.add_argument("--kind", choices=["bpe", "char"], default="bpe",
                   help="a new tokenizer: bpe (default) or char")
    q.add_argument("--pretokenizer", choices=["gpt", "ws"], default="gpt",
                   help="a new BPE's pre-tokenizer (default gpt)")
    q.add_argument("--val-fraction", type=float, default=0.01, metavar="F",
                   help="share of the documents held out for validation (default 0.01)")
    q.add_argument("--val-max-tokens", type=int, default=10_000_000, metavar="N",
                   help="at most this many validation tokens (default 10M)")
    q.add_argument("--max-tokens", type=int, metavar="N", help="stop after this many tokens")
    q.add_argument("--max-documents", type=int, metavar="N",
                   help="stop after this many documents")
    q.add_argument("--workers", type=int, metavar="N",
                   help="tokenizer processes (default: cores - 1, one for small data)")
    q.add_argument("--text-field", default="text", metavar="NAME",
                   help="the text field of records (default text)")
    q.add_argument("--hf-config", metavar="NAME", help="hf: datasets: the configuration")
    q.add_argument("--hf-split", metavar="NAME", help="hf: datasets: the split (default train)")
    q.add_argument("--separator", choices=["auto", "eos", "blank-line", "none"],
                   default="auto", help="what ends a document (default: the tokenizer's "
                                        "end-of-text token, else a blank line)")
    q.add_argument("--shard-tokens", type=int, default=1 << 27, metavar="N",
                   help="tokens per shard file (default 134M)")
    q.add_argument("--no-fast", action="store_true",
                   help="never use the tokenizers package (Rust) for Hugging Face tokenizers")
    q.add_argument("--overwrite", action="store_true",
                   help="replace a dataset already in --out")
    q.set_defaults(func=cmd_data_prepare)
    q = data_sub.add_parser("info", help="describe a prepared dataset")
    q.add_argument("dataset", metavar="DIR")
    q.add_argument("--json", action="store_true", help="print the whole manifest")
    q.set_defaults(func=cmd_data_info)

    p = sub.add_parser("plan", help="the memory a training run needs, and a batch that fits")
    p.add_argument("--preset", metavar="SIZE", help="a model size (see --list)")
    p.add_argument("--params", metavar="FILE", help="JSON training params (model fields)")
    p.add_argument("--list", action="store_true", help="list the model presets")
    p.add_argument("--vocab", type=int, default=32000, metavar="N",
                   help="vocabulary size (default 32000)")
    p.add_argument("--seq-length", type=int, metavar="N", help="tokens per window")
    p.add_argument("--batch-size", type=int, metavar="N", help="windows per optimizer step")
    p.add_argument("--memory", metavar="SIZE",
                   help="memory to plan for (16GB); default: free now on --device")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--optimizer", choices=["muon", "adamw"], help="default muon")
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], help="mixed precision")
    p.add_argument("--checkpointing", action="store_true",
                   help="estimate with gradient checkpointing")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("cluster-worker",
                       help="help another computer train: join its runs over the network")
    p.add_argument("main", help="the main computer: HOST or HOST:PORT (default port 29500)")
    p.add_argument("--once", action="store_true", help="exit after one run")
    p.add_argument("--token", help="the main computer's --cluster-token")
    p.add_argument("--timeout", type=_parse_duration, default=600.0, metavar="DURATION",
                   help="network timeout of a run (default 10m)")
    p.set_defaults(func=cmd_cluster_worker)

    # 2.14.1: train / inspect / convert tokenizers on their own
    p = sub.add_parser("tokenizer", help="train only a tokenizer, inspect or convert one")
    tok_sub = p.add_subparsers(dest="tokenizer_command", required=True)
    q = tok_sub.add_parser("train", help="train only a BPE (or character) tokenizer and save it")
    _add_common_flags(q, suppress_defaults=True)
    q.add_argument("--text", nargs="+", required=True, metavar="PATH",
                   help="training text (UTF-8): files, folders, globs, as for `auralite train`")
    q.add_argument("--out", default="tokenizer.json", help="output file (default tokenizer.json)")
    q.add_argument("--vocab-size", type=int, default=512, metavar="N",
                   help="tokens to learn (default 512, as `auralite train`)")
    q.add_argument("--kind", choices=["bpe", "char"], default="bpe",
                   help="bpe (default) or char (one token per character)")
    q.add_argument("--pretokenizer", choices=["gpt", "ws"], default="gpt",
                   help="where merges may happen: gpt (default) or ws (2.7, whitespace only)")
    q.add_argument("--val-split", type=float, default=0.1, metavar="F",
                   help="leave the last F of the text out, as training does for validation "
                        "(default 0.1; 0 = fit on all of it)")
    q.set_defaults(func=cmd_tokenizer_train)
    q = tok_sub.add_parser("info", help="describe a tokenizer file")
    _add_common_flags(q, suppress_defaults=True)
    q.add_argument("file", help="tokenizer file, tokenizer.json, vocab.json or checkpoint")
    q.add_argument("--text", metavar="FILE", help="also encode this text file as a sample")
    q.set_defaults(func=cmd_tokenizer_info)
    q = tok_sub.add_parser("encode", help="show the tokens of a text")
    _add_common_flags(q, suppress_defaults=True)
    q.add_argument("file", help="tokenizer file, tokenizer.json, vocab.json or checkpoint")
    q.add_argument("text", help="the text to encode")
    q.set_defaults(func=cmd_tokenizer_encode)
    q = tok_sub.add_parser("convert", help="save any supported tokenizer as an AuraLite "
                                           "tokenizer file")
    _add_common_flags(q, suppress_defaults=True)
    q.add_argument("source", help="tokenizer.json, vocab.json, merges.txt, a folder or a checkpoint")
    q.add_argument("--out", required=True, help="output file")
    q.set_defaults(func=cmd_tokenizer_convert)

    p = sub.add_parser("generate", help="generate text from a checkpoint")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model for prompt processing "
                        "(kernel cache in $AURALITE_HOME/torch_inductor)")
    _add_context_flags(p)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("chat", help="chat with a checkpoint (interactive or one-shot)")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", help="one-shot message (omit for interactive mode)")
    p.add_argument("--template", default=None, help="chat template override")
    p.add_argument("--system", help="system prompt")
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model for prompt processing")
    p.add_argument("--json", action="store_true",
                   help="answer with a JSON object (JSON mode, constrained decoding)")
    p.add_argument("--json-schema", metavar="FILE",
                   help="answer with JSON that follows this JSON schema (a .json file)")
    _add_context_flags(p)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("chat-eval", help="measure a chat model on held-out dialogs: stop "
                                         "rate, exact replies, reply loss (or on preference "
                                         "pairs: how often the chosen reply is likelier)")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, nargs="+", metavar="PATH",
                   help="dialogs (.jsonl/.json: OpenAI messages, ShareGPT, Alpaca, ...) or "
                        "preference pairs")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="the first N dialogs only")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 (default) replies greedily")
    p.add_argument("--json", action="store_true", help="print the result as one JSON line")
    p.set_defaults(func=cmd_chat_eval)

    p = sub.add_parser("distill", help="have a teacher model answer prompts; save its "
                                       "replies as dialogs (and preference pairs)")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("--teacher", required=True, metavar="MODEL",
                   help="a .gguf file (llama-cpp-python), an http:// llama.cpp server, a "
                        "Hugging Face model (folder or hub id) or an AuraLite checkpoint")
    p.add_argument("--prompts", required=True, metavar="FILE",
                   help="one prompt per line (.txt), or .jsonl records: {\"prompt\": ..., "
                        "\"answer\": ...} or dialogs (a final reply is the reference answer)")
    p.add_argument("--out", required=True, metavar="FILE",
                   help="dialogs to write (.jsonl, OpenAI messages: auralite train --data)")
    p.add_argument("--samples", type=int, default=1, metavar="N",
                   help="replies per prompt (default 1)")
    p.add_argument("--system", metavar="TEXT",
                   help="system message for prompts that have none")
    p.add_argument("--max-tokens", type=int, default=256, metavar="N",
                   help="reply length limit (default 256); replies cut there are dropped")
    p.add_argument("--keep-truncated", action="store_true",
                   help="keep replies cut at --max-tokens")
    p.add_argument("--temperature", type=float, default=None,
                   help="sampling temperature (default 0 = greedy for one sample, 0.8 for more)")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge", metavar="NAME",
                   help="keep only replies the judge accepts: exact, contains or number (vs "
                        "the reference answer), regex:PATTERN, or teacher (it picks the better "
                        "of two replies; --samples 2)")
    p.add_argument("--pairs", metavar="FILE",
                   help="also write preference pairs (prompt, chosen, rejected) for "
                        "--method dpo|simpo; needs --judge")
    p.add_argument("--license", metavar="NAME",
                   help="the teacher's license, recorded in the metadata (when the model "
                        "file names none)")
    p.add_argument("--limit", type=int, metavar="N", help="the first N prompts only")
    p.add_argument("--overwrite", action="store_true",
                   help="start again instead of resuming an earlier run on --out")
    p.add_argument("--n-ctx", type=int, default=4096, metavar="N",
                   help="GGUF: context length (default 4096)")
    p.add_argument("--threads", type=int, metavar="N", help="GGUF: CPU threads")
    p.set_defaults(func=cmd_distill)

    p = sub.add_parser("lora", help="LoRA adapters: merge them, save them alone, apply them")
    _add_common_flags(p, suppress_defaults=True)
    lora_sub = p.add_subparsers(dest="lora_command", required=True)
    q = lora_sub.add_parser("merge", help="fold the adapters into the weights (a plain model)")
    _add_common_flags(q, suppress_defaults=True)
    q.add_argument("--model", required=True, help="checkpoint with LoRA adapters")
    q.add_argument("--out", required=True, help="merged checkpoint")
    q.set_defaults(func=cmd_lora)
    q = lora_sub.add_parser("save", help="write the adapters and new token rows alone")
    _add_common_flags(q, suppress_defaults=True)
    q.add_argument("--model", required=True, help="checkpoint with LoRA adapters")
    q.add_argument("--out", required=True, help="adapter file")
    q.set_defaults(func=cmd_lora)
    q = lora_sub.add_parser("apply", help="put an adapter on the base model it was trained on")
    _add_common_flags(q, suppress_defaults=True)
    q.add_argument("--base", required=True, help="the base checkpoint")
    q.add_argument("--adapter", required=True, help="adapter file (auralite lora save)")
    q.add_argument("--out", required=True, help="checkpoint to write")
    q.add_argument("--merge", action="store_true", help="fold the adapters into the weights")
    q.add_argument("--no-check", action="store_true",
                   help="skip the check that the base is the adapter's own")
    q.set_defaults(func=cmd_lora)

    p = sub.add_parser("serve", help="start the OpenAI-compatible API server")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    _add_context_flags(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("info", help="inspect a checkpoint")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("--model", required=True)
    p.add_argument("--backends", action="store_true",
                   help="also print the quantization backend diagnostics")
    p.add_argument("--bench", type=int, default=0, metavar="N",
                   help="generate N tokens and report tokens/s + memory")
    p.add_argument("--bench-modes", action="store_true",
                   help="also compare fp32/fp16/int8 KV-cache modes (tokens/s + memory)")
    p.add_argument("--bench-compile", action="store_true",
                   help="also compare eager vs torch.compile prompt processing (prefill)")
    p.add_argument("--perplexity", action="store_true",
                   help="score a .gguf model on --text and report perplexity (cached)")
    p.add_argument("--text", metavar="FILE", help="text file used by --perplexity")
    p.add_argument("--n-ctx", type=int, default=2048,
                   help="context window used by --perplexity (default 2048)")
    p.add_argument("--no-cache", action="store_true",
                   help="ignore and do not write the perplexity cache")
    p.add_argument("--clear-cache", action="store_true",
                   help="drop the perplexity cache before scoring")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("import", help="import a Hugging Face Llama / Mistral / Qwen2 / Qwen3 "
                                      "as a native checkpoint (to fine-tune, chat, export)")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("source", metavar="SOURCE",
                   help="hf:ORG/NAME[@REVISION] (downloaded to ~/.auralite/hf, resumable; "
                        "HF_TOKEN for gated models) or a local model folder")
    p.add_argument("--out", required=True, help="checkpoint to write (.pt or .safetensors)")
    p.add_argument("--quantize", choices=["nf4"],
                   help="nf4: attention and feed-forward weights in 4 bits (QLoRA base; "
                        "about 1/7 of the float32 memory)")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32",
                   help="of the weights that are not quantized (default float32, for CPU "
                        "training; bfloat16 for GPU inference)")
    p.add_argument("--revision", help="branch, tag or commit (default main)")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("export", help="export a checkpoint as a Hugging Face folder, GGUF, "
                                      ".pt2 (torch.export), ONNX or TorchScript")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("--model", required=True, help="native AuraLite checkpoint (.pt/.safetensors)")
    p.add_argument("--format", choices=sorted(EXPORT_SUFFIXES), default="pt2",
                   help="pt2 (default, torch.export), onnx, torchscript (deprecated); 2.18.1: "
                        "hf (a transformers folder) and gguf (llama.cpp, Ollama, LM Studio)")
    p.add_argument("--quant", choices=["f16", "q8_0", "bf16", "f32"], default="f16",
                   help="gguf: weight type (default f16; llama-quantize makes Q4_K_M etc.)")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                   help="hf: weight dtype (default bfloat16)")
    p.add_argument("--out", help="output file (default: the checkpoint name with the "
                                 "format's suffix)")
    p.add_argument("--static", action="store_true",
                   help="fix batch and sequence length to the example shape "
                        "(default: dynamic)")
    p.add_argument("--opset", type=int, default=None,
                   help="ONNX opset (default: the exporter's; < 18 uses the TorchScript "
                        "exporter)")
    p.add_argument("--onnx-exporter", choices=["auto", "dynamo", "torchscript"], default="auto",
                   help="ONNX exporter (default auto: torch.export-based when onnxscript "
                        "is installed)")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("image-train", help="train a from-scratch image generator (diffusion, "
                                           "flow matching or GAN) on a folder of images")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("--images", required=True, help="folder of training images (recursive)")
    p.add_argument("--type", dest="model_type", choices=["diffusion", "flow", "gan"],
                   help="model type (default: the preset's, else diffusion): DDPM diffusion, "
                        "flow matching (rectified flow) or a GAN (R3GAN-style)")
    p.add_argument("--preset", help="tiny | small | medium | large | flow-tiny | flow-small | "
                                    "gan-tiny | gan-small (with --type: tiny = <type>-tiny)")
    p.add_argument("--params", help="JSON file with image training params "
                                    "(keys of ImageGenEngine.DEFAULTS)")
    p.add_argument("--model", default="image_model.pt",
                   help="output checkpoint (.pt or .safetensors)")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--img-size", dest="img_size", type=int)
    p.add_argument("--grayscale", action="store_true", help="train a 1-channel model")
    p.add_argument("--seed", type=int)
    p.add_argument("--val-split", dest="val_split", type=float, default=0.05,
                   help="held-out fraction for the validation loss (default 0.05; 0 = off)")
    p.add_argument("--resume", metavar="CKPT",
                   help="continue training this checkpoint (its architecture is kept)")
    p.add_argument("--samples", metavar="PNG",
                   help="also save a 4x4 grid of samples here after training")
    p.add_argument("--classes", action="store_true",
                   help="class-conditional: every first-level sub-folder is a class")
    p.set_defaults(func=cmd_image_train)

    p = sub.add_parser("image-generate", help="sample images from an image-generator "
                                              "checkpoint")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("--model", required=True)
    p.add_argument("-n", type=int, default=16, help="number of images (default 16)")
    _add_image_sampling_flags(p)
    p.add_argument("--out", default="samples.png", help="grid image (default samples.png)")
    p.add_argument("--out-dir", help="also save every image separately here")
    p.set_defaults(func=cmd_image_generate)

    p = sub.add_parser("image-eval", help="measure an image generator: patch sliced "
                                          "Wasserstein distance (SWD) to real images")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("--model", required=True)
    p.add_argument("--images", required=True, help="folder of real images to compare with")
    p.add_argument("-n", type=int, default=256,
                   help="generated images, and real ones (default 256; fewer if the folder "
                        "is smaller)")
    _add_image_sampling_flags(p)
    p.add_argument("--json", action="store_true", help="print the result as one JSON line")
    p.set_defaults(func=cmd_image_eval)

    # 2.19: what this computer can train, and the desktop app
    p = sub.add_parser("doctor", help="check this computer and recommend a model and settings "
                                      "for training on it")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("--data", nargs="+", metavar="PATH",
                   help="the training data: sizes the model and the time estimate to it")
    p.add_argument("--goal", choices=["finetune", "pretrain"], default="finetune",
                   help="fine-tune a pretrained model (default) or train one from scratch")
    p.add_argument("--hours", type=float, default=8.0, metavar="H",
                   help="how long a run may take (default 8: overnight)")
    p.add_argument("--no-speed", action="store_true",
                   help="skip the 1-second speed test (time estimates are then rough)")
    p.add_argument("--write-params", metavar="FILE",
                   help="write the recommended settings as a --params JSON file for "
                        "auralite train")
    p.add_argument("--json", action="store_true", help="print everything as JSON")
    p.add_argument("--studio", action="store_true",
                   help="also check that AuraLite Studio's window starts here (a fresh "
                        "process builds it without showing it; exit code 1 if it fails)")
    p.add_argument("--out", metavar="FILE",
                   help="also write the report to FILE, in UTF-8 - for an issue (PowerShell's "
                        "`>` would garble letters outside ASCII)")
    p.set_defaults(func=cmd_doctor)

    # 2.22: the benchmark
    p = sub.add_parser("bench", help="measure this computer for AuraLite: how fast three sizes "
                                     "of a new model train and write - offline, a few minutes")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("--quick", action="store_true",
                   help="two sizes, 15 s each: a check that it all runs (about a minute)")
    p.add_argument("--sizes", metavar="LIST",
                   help="the sizes, comma-separated: tiny (0.9M parameters), small (7M), "
                        "medium (19M); default: all three")
    p.add_argument("--seconds", type=float, metavar="S",
                   help="seconds of training per size (default 60)")
    p.add_argument("--data", metavar="FILE",
                   help="train on this text file instead of the built-in text (the report "
                        "then says how long one pass over it takes)")
    p.add_argument("--finetune", nargs="?", const="hf:Qwen/Qwen3.5-0.8B",
                   metavar="MODEL",
                   help="also fine-tune a ready model with LoRA: hf:ORG/NAME (downloaded "
                        "once), the folder of a model downloaded before, or an AuraLite "
                        "checkpoint (.pt); default hf:Qwen/Qwen3.5-0.8B (1.7 GB)")
    p.add_argument("--method", choices=["lora", "qlora"],
                   help="how --finetune trains (default: QLoRA below 8 GB of memory, else LoRA)")
    p.add_argument("--threads", type=int, metavar="N", help="training threads (default: AuraLite's)")
    p.add_argument("--out", metavar="FOLDER",
                   help="where to write bench-<processor>-<date>.md and .json (default "
                        "~/.auralite/bench)")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.add_argument("--child", metavar="SPEC", help=argparse.SUPPRESS)   # one stage, alone
    p.set_defaults(func=cmd_bench)

    # 2.20: the run library
    p = sub.add_parser("runs", help="the run library: every training run, its curve and "
                                    "files; compare runs, continue a stopped one")
    p.set_defaults(func=cmd_runs, runs_command="list")
    runs_sub = p.add_subparsers(dest="runs_command")
    q = runs_sub.add_parser("list", help="the runs, newest first")
    q.add_argument("--all", action="store_true", help="every run (default: the newest 30)")
    q.add_argument("--json", action="store_true", help="as JSON")
    q.set_defaults(func=cmd_runs)
    q = runs_sub.add_parser("show", help="one run: settings, losses, files, how to continue")
    q.add_argument("run", help="its id, the start of it, or last")
    q.add_argument("--json", action="store_true", help="the whole record as JSON")
    q.set_defaults(func=cmd_runs)
    q = runs_sub.add_parser("compare", help="what differs between runs")
    q.add_argument("runs", nargs="+", metavar="RUN", help="two or more runs")
    q.add_argument("--json", action="store_true", help="as JSON (with the curves)")
    q.set_defaults(func=cmd_runs)
    q = runs_sub.add_parser("continue", help="continue a stopped, failed or interrupted run "
                                             "exactly where it stood")
    q.add_argument("run", help="its id, the start of it, or last")
    _add_common_flags(q, suppress_defaults=True)
    q.set_defaults(func=cmd_runs)
    q = runs_sub.add_parser("delete", help="delete a run (or only its model files: --files)")
    q.add_argument("run", help="its id, the start of it, or last")
    q.add_argument("--files", action="store_true",
                   help="delete only the model files in its folder; keep its record and curve")
    q.add_argument("--yes", action="store_true", help="do not ask")
    q.set_defaults(func=cmd_runs)

    p = sub.add_parser("studio", help="open AuraLite Studio, the desktop app (needs PySide6: "
                                      "pip install \"auralite-ai[studio]\"; "
                                      "auralite doctor --studio checks it)")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("--self-test", action="store_true",
                   help="build the window without showing it and print what was found, "
                        "like python -m auralite.studio --self-test (2.21)")
    p.set_defaults(func=cmd_studio)
    return parser


def _add_image_sampling_flags(p: argparse.ArgumentParser) -> None:
    """The sampling flags of image-generate and image-eval."""
    p.add_argument("--steps", type=int, default=20, help="sampler steps (default 20)")
    p.add_argument("--sampler", choices=["dpm++2m", "ddim", "ddpm", "euler", "heun"],
                   default="dpm++2m",
                   help="diffusion: dpm++2m | ddim | ddpm; flow matching: dpm++2m | euler | "
                        "heun (ddim = euler); a GAN needs none")
    p.add_argument("--eta", type=float, default=0.0,
                   help="DDIM stochasticity (0 = deterministic; ddpm uses 1)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-ema", action="store_true", help="sample with the raw weights")
    p.add_argument("--class", dest="class_label", metavar="NAME",
                   help="class name or index of a class-conditional model, or 'all'")
    p.add_argument("--guidance", type=float, default=2.0,
                   help="classifier-free guidance scale with --class (default 2.0; 1 = off)")
    p.add_argument("--truncation", type=float, default=0.0,
                   help="GAN: resample latent values beyond this magnitude (e.g. 1.5: cleaner, "
                        "less varied images; default 0 = off)")


def _optional_packages() -> dict[str, str]:
    """Installed optional packages and their versions ("" = not installed)."""
    from importlib import metadata
    found = {}
    for module, dist in (("transformers", "transformers"), ("peft", "peft"),
                         ("llama_cpp", "llama-cpp-python"), ("PySide6", "PySide6-Essentials"),
                         ("onnx", "onnx"), ("fastapi", "fastapi"), ("datasets", "datasets")):
        version = ""
        for name in (dist, "PySide6") if module == "PySide6" else (dist,):
            try:
                version = metadata.version(name)
                break
            except metadata.PackageNotFoundError:
                continue
        found[module] = version
    return found


def _studio_check_command() -> list[str]:
    """``python -m auralite.studio --self-test``; 2.21: in the app that needs no Python, its
    console program (``auralite.exe studio --self-test``) - the Studio's own program has no
    console, and a pipe reads nothing from it."""
    from auralite import bundle
    if bundle.packaged():
        return bundle.cli_command("studio", "--self-test")
    return [sys.executable, "-m", "auralite.studio", "--self-test"]


def _studio_check(timeout: float = 300.0) -> tuple[int | None, str]:
    """2.19.1: ``python -m auralite.studio --self-test`` in a fresh process: its exit code
    (None: it did not finish) and what it printed. A crash inside Qt or PyTorch ends that
    process, not this one."""
    import subprocess
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.run(_studio_check_command(),
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=timeout, env=env, check=False)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, bytes) else b""
        return None, out.decode("utf-8", "replace") + f"\n(no answer after {timeout:.0f} s)"
    except OSError as exc:
        return None, str(exc)
    return proc.returncode, proc.stdout.decode("utf-8", "replace")


def _studio_verdict(code: int | None) -> str:
    from auralite.studio.app import _ending
    if code == 0:
        return "the window starts"
    if code == 2:
        return "it cannot start here (see above)"
    if code == 1:
        return "it failed (see above)"
    if code is None:
        return "the check did not finish"
    return f"it crashed ({_ending(code)})"


def _doctor_command(rec, data: list[str], params_file: str) -> list[str]:
    """`auralite` commands that run the recommendation. 2.21.3: in the app they begin with
    its console program as a console runs it - ``.\\auralite.exe`` (bundle.console_command;
    2.21.2 said so in every hint but these) - and a path with spaces is in quotes."""
    from auralite import bundle
    cli = bundle.console_command()
    data_text = bundle.quoted(list(data)) if data else "DIALOGS.jsonl"
    params = f" --params {bundle.quoted([params_file])}" if params_file else ""
    if rec.goal == "pretrain":
        preset = rec.settings.get("preset", "25m")
        minutes = int(rec.settings.get("time_limit_min", 480))
        return [f"{cli} train --data {data_text} --preset {preset} --time-budget {minutes}m "
                f"--memory-plan auto{params} --model my-model.pt"]
    choice = rec.choice
    if choice is None:
        return []
    quantize = " --quantize nf4" if choice.method == "qlora" else ""
    optimizer = " --optimizer adamw8bit" if rec.settings.get("optimizer") == "adamw8bit" else ""
    lora = "" if choice.method == "full" else f" --lora {rec.settings.get('lora_rank', 16)}"
    pack = " --pack-dialogs" if rec.settings.get("pack_dialogs") else ""
    return [f"{cli} train --data {data_text} --resume {choice.model.spec}{quantize}{lora}"
            f"{optimizer}{pack} --memory-plan auto{params} --model my-chat-model.pt",
            f"{cli} chat --model my-chat-model.pt",
            f"{cli} export --model my-chat-model.pt --format gguf --out my-chat-model.gguf"]


class _Tee:
    """A text stream that writes to several (2.21.2: ``doctor --out``)."""

    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def cmd_bench(args: argparse.Namespace) -> int:
    """2.22: ``auralite bench`` - this computer's numbers for training and writing."""
    import threading

    from auralite import bench
    if args.child:                      # a stage in a process of its own (bench._run_child)
        logging.getLogger("auralite").setLevel(logging.WARNING)
        return bench.child_main(args.child)
    if args.sizes:
        sizes = [s.strip() for s in str(args.sizes).split(",") if s.strip()]
    else:
        sizes = list(bench.QUICK["sizes"] if args.quick else bench.SIZES)
    seconds = float(args.seconds or (bench.QUICK["seconds"] if args.quick
                                     else bench.DEFAULT_SECONDS))
    tokens = int(bench.QUICK["generate_tokens"] if args.quick else bench.GENERATE_TOKENS)
    quiet = bool(getattr(args, "quiet", False))
    minutes = bench.estimated_minutes(sizes, seconds, finetune=bool(args.finetune))
    if not quiet:
        print(f"AuraLite bench: {', '.join(sizes)} - {seconds:g} s of training each"
              + (f", then fine-tuning {args.finetune}" if args.finetune else "")
              + f"; about {max(1.0, minutes):,.0f} min. Other programs slow it down: close "
              "what you can.", file=sys.stderr, flush=True)

    def progress(line: str) -> None:
        if not quiet:
            print(line, file=sys.stderr, flush=True)

    # the stages train in child processes (bench --child), whose engine lines stay there
    stop = threading.Event()
    try:
        report = bench.run_bench(sizes=sizes, seconds=seconds, generate_tokens=tokens,
                                 data=args.data, finetune=args.finetune,
                                 method=args.method or "", threads=args.threads,
                                 progress=progress, stop=stop)
    except bench.BenchError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, InterruptedError):
        stop.set()
        print("stopped: no report", file=sys.stderr)
        return 130
    md, js = bench.save_report(report, args.out)
    if args.json:
        print(json.dumps(report, indent=1, ensure_ascii=False))
        print(f"saved: {md} and {js.name}", file=sys.stderr)
        return 0
    print(bench.to_markdown(report))
    print(f"Saved: {md}")
    print(f"       {js}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """2.19: the computer, the data, the model and settings that fit, the commands to run.
    2.21.2: ``--out FILE`` writes the same report to FILE in UTF-8 - PowerShell's ``>``
    decodes a program's output in the console's code page and garbles what is not ASCII."""
    out = getattr(args, "out", None)
    if not out:
        return _doctor_report(args)
    import contextlib
    import io
    from pathlib import Path
    copy = io.StringIO()
    with contextlib.redirect_stdout(_Tee(sys.stdout, copy)):  # type: ignore[type-var]
        status = _doctor_report(args)
    target = Path(out)
    target.write_text(copy.getvalue(), encoding="utf-8")
    print(f"\nThe report is in {target.resolve()}")
    return status


def _doctor_report(args: argparse.Namespace) -> int:
    import platform

    import torch

    from auralite import __version__
    from auralite.studio import (detect_hardware, inspect_data, measure_gflops,
                                 recommend_finetune, recommend_pretrain)
    hw = detect_hardware()
    if not args.no_speed:
        hw.gflops = measure_gflops(hw.accelerator if hw.accelerator != "cpu" else "cpu")
    report = inspect_data(args.data) if args.data else None
    if args.goal == "pretrain":
        rec = recommend_pretrain(hw, report, max_hours=args.hours)
    else:
        rec = recommend_finetune(hw, report, max_hours=args.hours)
    params: dict = {}
    if rec.goal == "finetune" and rec.choice is not None:
        s = rec.settings
        params = {"epochs": s["epochs"], "seq_length": s["seq_length"],
                  "batch_size": s["batch_size"], "val_split": s["val_split"]}
        if rec.choice.method != "full":
            params["lora_lr"] = s["lora_lr"]
        else:
            params["lr"] = s["lr"]
    elif rec.goal == "pretrain":
        params = {"epochs": rec.settings["epochs"], "bpe_vocab_size":
                  rec.settings["bpe_vocab_size"]}
    if args.write_params and params:
        with open(args.write_params, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=1)
    commands = _doctor_command(rec, list(args.data or []), args.write_params or "")
    packages = _optional_packages()
    from auralite.studio.app import installation
    installed, stale = installation()
    studio: dict = {"pyside6": packages.get("PySide6", ""), "installed": installed,
                    "install_problem": stale}
    if args.studio:
        code, output = _studio_check()
        studio["check"] = {"code": code, "result": _studio_verdict(code), "output": output}
    status = 1 if args.studio and studio["check"]["code"] != 0 else 0
    if args.json:
        out = {"version": __version__, "python": platform.python_version(),
               "torch": torch.__version__, "hardware": hw.to_dict(), "packages": packages,
               "studio": studio,
               "goal": rec.goal, "budget": rec.budget, "settings": rec.settings,
               "params": params, "commands": commands, "warnings": rec.warnings,
               "data": None if report is None else {
                   "kind": report.kind, "records": report.records, "tokens": report.tokens,
                   "script": report.script, "problems": report.problems,
                   "warnings": report.warnings}}
        if rec.goal == "finetune":
            out["choice"] = None if rec.choice is None else rec.choice.model.repo
            out["options"] = [{"repo": f.model.repo, "method": f.method, "memory": f.memory,
                               "fits": f.fits, "tight": f.tight,
                               "tokens_per_second": f.tokens_per_second, "hours": f.hours}
                              for f in rec.options]
        else:
            out["choice"] = rec.preset.preset if rec.preset is not None else None
        print(json.dumps(out, indent=1))
        return status
    print(f"AuraLite {__version__} - Python {platform.python_version()}, "
          f"PyTorch {torch.__version__}")
    from auralite import bundle
    if bundle.packaged():                         # 2.21: which app, and where
        print(f"App: {bundle.describe()}")
        other = bundle.other_cli_on_path()        # 2.21.2: what `auralite` alone runs
        if other is not None:
            print(f"  ! In PowerShell `auralite` alone runs {other}, another AuraLite (from "
                  f"pip?), and so does cmd outside the app's folder. For this app's type "
                  f"{bundle.console_command()} in its folder.")
    for line in hw.summary_lines():
        print(line)
    print("Optional: " + ", ".join(f"{name} {version or '-'}"
                                   for name, version in packages.items()))
    if "check" in studio:
        print("\nAuraLite Studio, checked in a fresh process "
              f"({'auralite studio' if bundle.packaged() else 'python -m auralite.studio'}"
              " --self-test):")
        for line in studio["check"]["output"].rstrip().splitlines():
            print("    " + line)
        print(f"  => {studio['check']['result']}")
    elif studio["pyside6"]:
        print(f"Studio: PySide6 {studio['pyside6']} - `auralite doctor --studio` checks that "
              f"its window starts")
    else:
        from auralite.studio.app import install_command
        print(f"Studio: needs PySide6 - {install_command()}")
    if stale and "check" not in studio:           # the self-test's output names it already
        print(f"  ! {stale}")
    print()
    if report is not None:
        for line in report.summary_lines():
            print(line)
        print()
    for line in rec.lines():
        print(line)
    if rec.goal == "finetune":
        usable = [f for f in rec.options if f.usable]
        if usable:
            print("\nModels that fit:")
            for fit in usable:
                print("  " + fit.line())
    if params and args.write_params:
        print(f"\nSettings written to {args.write_params}")
    if commands:
        print("\nTo run it:")
        for command in commands:
            print("  " + command)
    studio_program = bundle.studio_program()
    print("\nOr in the desktop app: "
          + (f"{studio_program.name} in {studio_program.parent}" if studio_program
             else "auralite studio"))
    return status


# ----------------------------------------------------------------------
#  2.20: the run library (auralite runs)
# ----------------------------------------------------------------------
def _bytes_text(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} GB"


def _run_dict(run: Any) -> dict:
    ok, why = run.continue_state()
    return {"id": run.id, "name": run.name, "status": run.status, "kind": run.kind,
            "source": run.source, "method": run.method, "created": run.created,
            "seconds": run.seconds, "best_val_loss": run.best_val, "val_bpb": run.val_bpb,
            "folder": str(run.folder),
            "result": str(run.result_path) if run.result_path else None,
            "checkpoint": str(run.checkpoint_path) if run.checkpoint_path else None,
            "can_continue": ok, "why_not": why}


def _runs_list(args: argparse.Namespace) -> int:
    from auralite import runs as runlib
    runs = runlib.list_runs()
    if getattr(args, "json", False):
        print(json.dumps([_run_dict(r) for r in runs], indent=1))
        return 0
    root = runlib.runs_root()
    if not runs:
        print(f"no runs yet: auralite train and AuraLite Studio record them in {root}")
        return 0
    shown = runs if getattr(args, "all", False) else runs[:30]
    print(f"{'ID':<28} {'STATUS':<12} {'KIND':<11} {'BEST VAL':>9} {'TIME':>9}  NAME")
    for run in shown:
        best = f"{run.best_val:.4f}" if run.best_val is not None else ""
        print(f"{run.id:<28} {run.status:<12} {run.kind_label:<11} {best:>9} "
              f"{runlib._duration(run.seconds):>9}  {run.name}")
    if len(shown) < len(runs):
        print(f"... and {len(runs) - len(shown)} older (--all)")
    total = sum(run.disk_bytes() for run in runs)
    print(f"{len(runs)} runs, {_bytes_text(total)} in {root}")
    return 0


def _runs_show(args: argparse.Namespace) -> int:
    from auralite import runs as runlib
    run = runlib.find_run(args.run)
    if getattr(args, "json", False):
        print(json.dumps({**run.record, **_run_dict(run)}, indent=1, default=str))
        return 0
    print(f"{run.name}  [{run.id}]")
    for label, text in runlib.describe_run(run):
        print(f"  {label + ':':<12} {text}")
    for label, path, _size in run.files():
        print(f"  {label.capitalize() + ':':<12} {path}")
    if run.notes:
        print(f"  {'Notes:':<12} {run.notes}")
    print(f"  {'Folder:':<12} {run.folder}")
    if run.continue_state()[0]:
        print(f"\nauralite runs continue {run.id}")
    return 0


def _runs_compare(args: argparse.Namespace) -> int:
    from auralite import runs as runlib
    runs = [runlib.find_run(key) for key in args.runs]
    table = runlib.compare_runs(runs)
    if getattr(args, "json", False):
        print(json.dumps(table, indent=1))
        return 0
    width = 26
    label_width = max(len(label) for label, _values, _differs in table["rows"]) + 2
    print(" " * (label_width + 2) + "".join(f"{run.id[:width - 2]:<{width}}" for run in runs))
    for label, values, differs in table["rows"]:
        mark = "*" if differs else " "
        cells = "".join(f"{(v if len(v) < width - 1 else v[:width - 4] + '...'):<{width}}"
                        for v in values)
        print(f"{mark} {label:<{label_width}}{cells}")
    print("\n* differs")
    return 0


def _runs_continue(args: argparse.Namespace) -> int:
    from auralite import runs as runlib
    from auralite.engine import AuraLiteEngine, DataFormatError
    run = runlib.find_run(args.run)
    try:
        path = runlib.continue_point(run)
        data = runlib.load_run_data(run)
    except (ValueError, FileNotFoundError, DataFormatError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    quiet = getattr(args, "quiet", False)
    if not quiet:
        print(f"continuing {run.name} from {path}")
    engine = AuraLiteEngine()
    engine.load_model(str(path))
    recorder = runlib.RunRecorder.resume(run)
    params: dict = {"continue_training": True, "continue_run": True}
    folder = run.record.get("files") == "folder"
    if folder:
        params.update(autosave_every_minutes=runlib.CHECKPOINT_MINUTES,
                      autosave_path=str(recorder.checkpoint_path))
    finish_event, stop_event = threading.Event(), threading.Event()
    previous = _graceful_sigint(finish_event, stop_event)
    try:
        summary = engine.train(data, params,
                               progress_callback=recorder.callback(
                                   None if quiet else _ProgressPrinter()),
                               stop_event=stop_event, finish_event=finish_event)
    except ValueError as err:
        recorder.fail(err)
        print(f"error: {err}", file=sys.stderr)
        return 2
    except BaseException as err:
        recorder.fail(err)
        raise
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    if not folder and not (run.record.get("output") or run.record.get("result")):
        recorder.record["output"] = str(path)          # nowhere else to keep it
    done = recorder.save_and_finish(summary, engine)
    out = done.result_path
    reason = str(summary.get("stop_reason") or "")
    line = f"run {done.id}: {done.status}"
    if out is not None:
        line += f" | saved: {out}"
    if summary.get("val_loss") is not None:
        line += f" | val loss {summary['val_loss']:.4f}"
    print(line)
    if reason == "stopped" and done.continue_state()[0] and not quiet:
        print(f"continue it later: auralite runs continue {done.id}")
    return 0


def _runs_delete(args: argparse.Namespace) -> int:
    from auralite import runs as runlib
    run = runlib.find_run(args.run)
    if run.status == "running":
        print(f"error: {run.id} is training now", file=sys.stderr)
        return 2
    what = (f"the model files of {run.id}" if args.files else
            f"run {run.id} ({run.name}) and its folder")
    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print(f"error: add --yes to delete {what}", file=sys.stderr)
            return 2
        answer = input(f"Delete {what}? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("nothing deleted")
            return 1
    freed = runlib.delete_run(run, files_only=args.files)
    print(f"deleted {what}: {_bytes_text(freed)} freed")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    """2.20: the run library: every run of auralite train and AuraLite Studio."""
    action = getattr(args, "runs_command", None) or "list"
    handler = {"list": _runs_list, "show": _runs_show, "compare": _runs_compare,
               "continue": _runs_continue, "delete": _runs_delete}[action]
    try:
        return handler(args)
    except (KeyError, ValueError) as err:
        message = err.args[0] if isinstance(err, KeyError) and err.args else err
        print(f"error: {message}", file=sys.stderr)
        return 2


def cmd_studio(args: argparse.Namespace) -> int:
    """2.19: AuraLite Studio, the desktop app on Qt 6."""
    from auralite.studio.app import SELF_TEST
    from auralite.studio.app import main as studio_main
    return studio_main([SELF_TEST] if getattr(args, "self_test", False) else [])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args)
    try:
        return args.func(args)
    except ModuleNotFoundError as err:
        from auralite import bundle
        if not bundle.packaged() or not err.name:
            raise
        # 2.21: the app that needs no Python leaves some optional parts out - say which,
        # and where they come with, instead of a traceback
        top = err.name.split(".")[0]
        print(f"error: {err}", file=sys.stderr)
        for line in bundle.missing_part(top, bundle.EXTRAS.get(top, "")):
            print(f"  {line}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
