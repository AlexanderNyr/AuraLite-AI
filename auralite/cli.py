"""AuraLite command line interface.

    auralite train     --text corpus.txt --params params.json --model out.pt
    auralite generate  --model out.pt --prompt "Hello" --length 40
    auralite chat      --model out.pt [--template chatml] [--system "..."]
    auralite serve     --model out.pt [--host 0.0.0.0] [--port 8000]
    auralite info      --model out.pt
    auralite export    --model out.pt [--format pt2|onnx|torchscript] [--out PATH]

`--mmap` (or `--corpus-cache PATH`) streams the corpus into a uint32 memory map
instead of an in-RAM int64 tensor — the same tokens, ~4x less memory and no
Python list of the whole corpus.

`--params` JSON accepts every key of `AuraLiteEngine.train()` (d_model,
n_heads, n_layers, d_ff, seq_length, epochs, batch_size, lr, tokenizer,
optimizer, muon_lr, lr_schedule, use_qk_norm, sliding_window, seed,
amp_dtype, chat_template, ...).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

DEFAULT_TRAIN_PARAMS: dict = {
    "d_model": 128, "n_heads": 4, "n_layers": 4, "d_ff": 256,
    "seq_length": 64, "epochs": 10, "batch_size": 16,
    "tokenizer": "bpe", "val_split": 0.1,
}


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


def _progress(epoch: int, total: int, train_loss: float, val_loss: float | None,
              info: dict | None = None) -> None:
    info = info or {}
    phase = info.get("phase", "train")
    # Quiet setup spam — one line is enough
    if phase == "setup":
        msg = info.get("message") or "setup"
        print(f"  setup | {msg}", flush=True)
        return
    # Live batch lines (same epoch) — overwrite-friendly single line
    if phase == "train" and not info.get("is_epoch_end"):
        batch = info.get("batch")
        batches = info.get("batches")
        pct = info.get("percent")
        eta = info.get("eta_seconds")
        tps = info.get("tokens_per_sec")
        parts = [f"  epoch {epoch}/{total}"]
        if batch is not None and batches is not None:
            parts.append(f"batch {batch}/{batches}")
        if train_loss:
            parts.append(f"loss={train_loss:.4f}")
        if tps:
            parts.append(f"{float(tps):.0f} tok/s")
        if pct is not None:
            parts.append(f"{float(pct):.1f}%")
        if eta is not None:
            m, s = divmod(int(max(0, float(eta))), 60)
            h, m = divmod(m, 60)
            parts.append(f"ETA {h}h{m:02d}m{s:02d}s" if h else f"ETA {m}m{s:02d}s")
        print(" | ".join(parts), flush=True)
        return
    if phase == "val":
        print(f"  epoch {epoch}/{total} | validating…", flush=True)
        return
    val = f" | val={val_loss:.4f}" if val_loss is not None else ""
    print(f"  epoch {epoch}/{total} done | loss={train_loss:.4f}{val}", flush=True)


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
    # v2.6.5: validate the merged params against the typed config so typos in
    # --params (e.g. "epoch" instead of "epochs") fail loudly instead of being
    # silently ignored by train().
    try:
        from auralite.engine.config import AuraLiteConfig
        AuraLiteConfig(**{k: v for k, v in params.items()})
    except Exception as err:
        print(f"error: invalid training parameters: {err}", file=sys.stderr)
        return 2
    engine = AuraLiteEngine()
    engine.train(text, params,
                 progress_callback=None if getattr(args, "quiet", False) else _progress)
    engine.save_model(args.model)
    print(f"saved: {args.model}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from auralite.engine import AuraLiteEngine
    engine = AuraLiteEngine()
    engine.load_model(args.model)
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


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("error: install serving dependencies: pip install fastapi uvicorn pydantic",
              file=sys.stderr)
        return 2
    os.environ["AURALITE_MODEL"] = os.path.abspath(args.model)
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
            "sliding_window", "dropout", "lora_rank"]
    for k in keys:
        if k in ckpt:
            print(f"{k}: {ckpt[k]}")
    tok = ckpt.get("tokenizer") or {}
    if tok:
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
    p.set_defaults(func=cmd_train)

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
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("serve", help="start the OpenAI-compatible API server")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
