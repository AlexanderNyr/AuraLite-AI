"""AuraLite command line interface.

    auralite train     --text corpus.txt --params params.json --model out.pt
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


def cmd_train(args: argparse.Namespace) -> int:
    from auralite.engine import AuraLiteEngine
    text_path = args.text
    if not os.path.exists(text_path):
        print(f"error: text file not found: {text_path}", file=sys.stderr)
        return 2
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    params = dict(DEFAULT_TRAIN_PARAMS)
    if args.params:
        with open(args.params, "r", encoding="utf-8") as f:
            params.update(json.load(f))
    if getattr(args, "mmap", False) or getattr(args, "corpus_cache", None):
        params["corpus_cache"] = "mmap"
    if getattr(args, "corpus_cache", None):
        params["corpus_cache_path"] = os.path.abspath(args.corpus_cache)
    if getattr(args, "tokenizer", None):
        # 2.14.1: train with your own tokenizer (auralite tokenizer train, a
        # Hugging Face tokenizer.json, GPT-2 vocab.json + merges.txt, a checkpoint)
        if not os.path.exists(args.tokenizer):
            print(f"error: tokenizer file not found: {args.tokenizer}", file=sys.stderr)
            return 2
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
        if not os.path.exists(args.resume):
            print(f"error: checkpoint not found: {args.resume}", file=sys.stderr)
            return 2
        engine.load_model(args.resume)
        params["continue_training"] = True
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
    finish_event, stop_event = threading.Event(), threading.Event()
    previous = _graceful_sigint(finish_event, stop_event)
    from auralite.engine.hf_bpe import TokenizerFormatError
    try:
        summary = engine.train(
            text, params,
            progress_callback=None if getattr(args, "quiet", False) else _ProgressPrinter(),
            stop_event=stop_event, finish_event=finish_event)
    except (TokenizerFormatError, FileNotFoundError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    if isinstance(summary, dict) and summary.get("stopped_during"):
        # 2.14.1: stopped before the first step — there is no new model
        print(f"stopped before training began ({summary['stopped_during']}); nothing was saved")
        return 130
    engine.save_model(args.model)
    line = f"saved: {args.model}"
    summary = engine.last_train_summary or {}
    if summary.get("val_loss") is not None:
        line += f" | val loss {summary['val_loss']:.4f}"
        if summary.get("val_bpb") is not None:
            line += f" ({summary['val_bpb']:.4f} bits/byte)"
    reason = str(summary.get("stop_reason") or "")
    if reason:
        line += " | " + {"early_stopping": "stopped early (best epoch kept)",
                         "finished": "finished after a learning-rate cooldown",
                         "time_budget": "time budget spent",
                         "stopped": "stopped",
                         "max_steps": "max_steps reached"}.get(reason, reason)
    print(line)
    return 0


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


def cmd_chat(args: argparse.Namespace) -> int:
    from auralite.engine import AuraLiteEngine
    engine = AuraLiteEngine()
    engine.load_model(args.model)
    engine.set_context_window(_context_window_arg(args))
    if getattr(args, "compile", False):
        engine.enable_inference_compile()
    history: list[dict] = []
    if args.system:
        history.append({"role": "system", "content": args.system})

    def _respond(user_text: str) -> None:
        history.append({"role": "user", "content": user_text})
        reply = engine.generate_chat(history, args.length,
                                     temperature=args.temperature,
                                     top_k=args.top_k, top_p=args.top_p,
                                     chat_template=args.template)
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
EXPORT_SUFFIXES = {"pt2": ".pt2", "onnx": ".onnx", "torchscript": ".torchscript.pt"}


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
            "sliding_window", "rope_scaling", "dropout", "lora_rank"]
    for k in keys:
        if k in ckpt:
            print(f"{k}: {ckpt[k]}")
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
    from auralite.engine.tokenizer_io import describe_tokenizer, save_tokenizer, train_tokenizer
    text = _read_text(args.text)
    if text is None:
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
            should_stop=stop.is_set, source=os.path.basename(args.text))
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
    if args.kind == "bpe" and info["fit_chars"] < info["text_chars"]:
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

    p = sub.add_parser("train", help="train a model from a text file")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("--text", required=True, help="training text file (UTF-8)")
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
                        "--params extends its context)")
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
    p.set_defaults(func=cmd_train)

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
    q.add_argument("--text", required=True, help="training text file (UTF-8)")
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
    _add_context_flags(p)
    p.set_defaults(func=cmd_chat)

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

    p = sub.add_parser("export", help="export a checkpoint as .pt2 (torch.export), ONNX "
                                      "or TorchScript")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("--model", required=True, help="native AuraLite checkpoint (.pt/.safetensors)")
    p.add_argument("--format", choices=sorted(EXPORT_SUFFIXES), default="pt2",
                   help="pt2 (default, torch.export), onnx, torchscript (deprecated)")
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
