"""AuraLite command line interface.

    auralite train     --text corpus.txt --params params.json --model out.pt
    auralite generate  --model out.pt --prompt "Hello" --length 40
    auralite chat      --model out.pt [--template chatml] [--system "..."]
    auralite serve     --model out.pt [--host 0.0.0.0] [--port 8000]
    auralite info      --model out.pt

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
                     else os.environ.get("AURALITE_LOG_LEVEL", "WARNING"))
    parser.add_argument("--quiet", action="store_true", default=quiet_default,
                        help="suppress progress output (results are still printed)")
    parser.add_argument("--log-level", default=level_default,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="library log level (default: WARNING or $AURALITE_LOG_LEVEL)")


def _configure_logging(args: argparse.Namespace) -> None:
    """Apply --quiet/--log-level (and $AURALITE_LOG_LEVEL) to library loggers.

    The level is also exported through ``AURALITE_LOG_LEVEL`` so that modules
    imported *later* (``model_engine`` reads the variable at import time) keep
    the level requested on the command line instead of resetting it to INFO.
    """
    level = "CRITICAL" if getattr(args, "quiet", False) else str(
        getattr(args, "log_level", "WARNING")).upper()
    os.environ["AURALITE_LOG_LEVEL"] = level
    logger = logging.getLogger("auralite")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    # The library installs its own StreamHandler; without this the same record
    # would also reach the root logger and be printed twice.
    logger.propagate = False
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


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
    from model_engine import AuraLiteEngine
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
    engine = AuraLiteEngine()
    engine.train(text, params,
                 progress_callback=None if getattr(args, "quiet", False) else _progress)
    engine.save_model(args.model)
    print(f"saved: {args.model}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from model_engine import AuraLiteEngine
    engine = AuraLiteEngine()
    engine.load_model(args.model)
    out = engine.generate(args.prompt, args.length,
                          temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    print(out)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from model_engine import AuraLiteEngine
    engine = AuraLiteEngine()
    engine.load_model(args.model)
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


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("error: install serving dependencies: pip install fastapi uvicorn pydantic",
              file=sys.stderr)
        return 2
    os.environ["AURALITE_MODEL"] = os.path.abspath(args.model)
    # Single worker on purpose: engine/KV-cache and the rate limiter live in
    # process memory (see server/openai_server.py module docstring).
    uvicorn.run("server.openai_server:app", host=args.host, port=args.port, workers=1)
    return 0


def _bench(model_path: str, n_tokens: int) -> int:
    """Measure generation speed and weight memory for the loaded checkpoint."""
    from model_engine import AuraLiteEngine
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
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    # v2.6.5: checkpoints are read with the safe unpickler (see safe_io.py).
    from model_engine.safe_io import safe_torch_load
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
        from quantization_backends import describe_backends
        print("quantization backends:")
        for key, value in describe_backends().items():
            print(f"  {key}: {value}")
    if getattr(args, "bench", 0):
        return _bench(args.model, int(args.bench))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auralite", description="AuraLite AI CLI")
    _add_common_flags(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train a model from a text file")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("--text", required=True, help="training text file (UTF-8)")
    p.add_argument("--params", help="JSON file with training params")
    p.add_argument("--model", default="model.pt", help="output checkpoint")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("generate", help="generate text from a checkpoint")
    _add_common_flags(p, suppress_defaults=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
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
    p.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
