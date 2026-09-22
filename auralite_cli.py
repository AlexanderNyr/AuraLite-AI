"""AuraLite command line interface.

    auralite train     --text corpus.txt --params params.json --model out.pt
    auralite generate  --model out.pt --prompt "Привет" --length 40
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
import os
import sys

DEFAULT_TRAIN_PARAMS: dict = {
    "d_model": 128, "n_heads": 4, "n_layers": 4, "d_ff": 256,
    "seq_length": 64, "epochs": 10, "batch_size": 16,
    "tokenizer": "bpe", "val_split": 0.1,
}


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
    engine.train(text, params, progress_callback=_progress)
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


def cmd_info(args: argparse.Namespace) -> int:
    import torch
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
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
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auralite", description="AuraLite AI CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train a model from a text file")
    p.add_argument("--text", required=True, help="training text file (UTF-8)")
    p.add_argument("--params", help="JSON file with training params")
    p.add_argument("--model", default="model.pt", help="output checkpoint")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("generate", help="generate text from a checkpoint")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("chat", help="chat with a checkpoint (interactive or one-shot)")
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
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("info", help="inspect a checkpoint")
    p.add_argument("--model", required=True)
    p.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
