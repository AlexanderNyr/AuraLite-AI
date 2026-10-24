"""Chat fine-tuning benchmark (2.15): synthetic dialogs, training, stop rate, reply loss.

Used for the 2.15 numbers in docs/training.md § 15.

    python benchmarks/chat_bench.py data chats.jsonl --n 3000 --seed 1     # training dialogs
    python benchmarks/chat_bench.py data eval.jsonl --n 300 --seed 99      # other dialogs
    python benchmarks/chat_bench.py text chats.jsonl chats.txt   # the dialogs as plain ChatML text
    python benchmarks/chat_bench.py train --data chats.jsonl --out chat.pt [--loss all]
        [--base base.pt] [--lora 8] [--epochs 40] [--seed 3]
    python benchmarks/chat_bench.py train --data chats.txt --out plain.pt   # as 2.14.2 would
    python benchmarks/chat_bench.py eval --model chat.pt --data eval.jsonl [--temperature 0]

`data` writes dialogs a small model can learn (arithmetic, capitals, animal
sounds, counting, opposites, greetings, a two-turn name recall), in the
OpenAI ``{"messages": [...]}`` format. `eval` answers the last user turn of
every dialog and reports how often the reply ended with the template's end
marker (``stop_rate``: the model stopped by itself, before the token limit),
how often it was free of template markers, how often it was exactly the
expected reply, the cross-entropy over the reply tokens of the dialogs
(``reply_loss``, nats per token of the model's tokenizer; the loss ``train``
minimises; None for a model whose markers are not tokens) and ``reply_bpb``:
the loss of each dialog's last reply and its
end marker given the dialog before it, in bits per UTF-8 byte, which compares
models with different tokenizers (a model trained on the `text` file, whose
markers are several tokens, too).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CAPITALS = {"France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid",
            "Japan": "Tokyo", "Egypt": "Cairo", "Canada": "Ottawa", "Peru": "Lima",
            "Norway": "Oslo", "Kenya": "Nairobi", "Chile": "Santiago", "Greece": "Athens",
            "Poland": "Warsaw", "Cuba": "Havana", "India": "New Delhi", "China": "Beijing"}
SOUNDS = {"cow": "moo", "dog": "woof", "cat": "meow", "duck": "quack", "sheep": "baa",
          "owl": "hoot", "pig": "oink", "horse": "neigh", "frog": "croak", "lion": "roar"}
OPPOSITES = {"hot": "cold", "big": "small", "fast": "slow", "up": "down", "happy": "sad",
             "light": "dark", "old": "young", "early": "late", "full": "empty", "hard": "soft"}
NAMES = ["Anna", "Boris", "Clara", "David", "Elena", "Felix", "Greta", "Hugo", "Irina",
         "Jonas", "Katya", "Leo", "Maria", "Nikolai", "Olga", "Pavel"]
SYSTEMS = ["You are a helpful assistant.", "You are a friendly assistant. Answer briefly.", ""]
#: The model of the § 15 runs (about 0.8 M parameters, a few minutes on 2 CPU threads).
PARAMS = {"d_model": 128, "n_heads": 4, "n_layers": 4, "d_ff": 256, "seq_length": 128,
          "batch_size": 32, "seed": 3, "progress_every_seconds": 1000}


def dialog(rng: random.Random) -> dict:
    """One synthetic dialog (OpenAI format)."""
    kind = rng.choice(["add", "sub", "capital", "sound", "count", "opposite", "greet", "name"])
    if kind == "add":
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        turns = [(rng.choice([f"What is {a} plus {b}?", f"Add {a} and {b}.", f"{a} + {b} = ?"]),
                  f"{a} plus {b} is {a + b}.")]
    elif kind == "sub":
        a, b = rng.randint(20, 99), rng.randint(1, 19)
        turns = [(rng.choice([f"What is {a} minus {b}?", f"Subtract {b} from {a}."]),
                  f"{a} minus {b} is {a - b}.")]
    elif kind == "capital":
        c = rng.choice(list(CAPITALS))
        turns = [(rng.choice([f"What is the capital of {c}?", f"Name the capital of {c}.",
                              f"Which city is the capital of {c}?"]),
                  f"The capital of {c} is {CAPITALS[c]}.")]
    elif kind == "sound":
        a = rng.choice(list(SOUNDS))
        turns = [(rng.choice([f"What sound does a {a} make?", f"How does a {a} sound?"]),
                  f"A {a} says {SOUNDS[a]}.")]
    elif kind == "count":
        a = rng.randint(1, 15)
        b = a + rng.randint(2, 6)
        turns = [(f"Count from {a} to {b}.", ", ".join(str(i) for i in range(a, b + 1)) + ".")]
    elif kind == "opposite":
        w = rng.choice(list(OPPOSITES))
        turns = [(rng.choice([f"What is the opposite of {w}?", f"Give me the opposite of {w}."]),
                  f"The opposite of {w} is {OPPOSITES[w]}.")]
    elif kind == "greet":
        turns = [(rng.choice(["Hello!", "Hi there.", "Good morning!", "Hey, how are you?"]),
                  rng.choice(["Hello! How can I help you today?",
                              "Hi! What can I do for you?"]))]
    else:
        n = rng.choice(NAMES)
        turns = [(rng.choice([f"My name is {n}.", f"Hi, I am {n}."]), f"Nice to meet you, {n}!"),
                 (rng.choice(["What is my name?", "Do you remember my name?"]),
                  f"Your name is {n}.")]
    messages = []
    system = rng.choice(SYSTEMS)
    if system:
        messages.append({"role": "system", "content": system})
    for user, reply in turns:
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
    return {"messages": messages}


def cmd_data(args) -> dict:
    rng = random.Random(args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            f.write(json.dumps(dialog(rng)) + "\n")
    return {"section": "data", "out": args.out, "dialogs": args.n, "seed": args.seed}


def cmd_text(args) -> dict:
    from auralite.chat_interface import ChatMessage, apply_chat_template
    from auralite.engine import load_training_data

    dialogs = load_training_data([args.data]).dialogs
    with open(args.out, "w", encoding="utf-8") as f:
        for messages in dialogs:
            f.write(apply_chat_template([ChatMessage.from_dict(m) for m in messages],
                                        template_name="chatml", add_generation_prompt=False))
    return {"section": "text", "out": args.out, "dialogs": len(dialogs)}


def cmd_train(args) -> dict:
    from auralite.engine import AuraLiteEngine, load_training_data

    data = load_training_data([args.data])
    params = dict(PARAMS, epochs=args.epochs, dialog_loss=args.loss, seed=args.seed)
    if args.lora:
        params["lora_rank"] = args.lora
    engine = AuraLiteEngine()
    if args.base:
        engine.load_model(args.base)
        params["continue_training"] = True
    started = time.time()
    summary = engine.train(data, params)
    engine.save_model(args.out)
    return {"section": "train", "out": args.out, "loss": args.loss, "base": args.base,
            "lora": args.lora, "epochs": summary["epochs_completed"],
            "steps": summary["optimizer_steps"], "val_loss": summary["best_val_loss"],
            "val_bpb": summary["best_val_bpb"], "dialogs": summary.get("dialogs"),
            "seconds": round(time.time() - started, 1)}


def reply_loss(engine, dialogs) -> float:
    """Mean cross-entropy over the reply tokens of `dialogs` (one forward per dialog)."""
    import torch
    import torch.nn.functional as F

    from auralite.engine.dataset import IGNORE_INDEX
    from auralite.engine.dialogs import encode_dialog

    total, count = 0.0, 0
    engine.model.eval()
    with torch.no_grad():
        for messages in dialogs:
            ids, targets = encode_dialog(engine.tokenizer, messages,
                                         engine.default_chat_template)
            logits = engine.model(torch.tensor([ids[:-1]]))
            y = torch.tensor([targets[1:]])
            total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                           ignore_index=IGNORE_INDEX, reduction="sum"))
            count += int((y != IGNORE_INDEX).sum())
    return total / max(1, count)


def reply_bits_per_byte(engine, dialogs) -> float:
    """Bits per byte of each dialog's last reply and end marker, given the dialog before it.

    The text continues with a newline, as in training; the tokens up to the
    one that completes the end marker count (a BPE without special tokens can
    join the marker with that newline).
    """
    import math

    import torch
    import torch.nn.functional as F

    from auralite.chat_interface import ChatMessage, apply_chat_template, get_stop_tokens

    template = engine.default_chat_template
    end = get_stop_tokens(template)[0]
    nats, total_bytes = 0.0, 0
    engine.model.eval()
    for messages in dialogs:
        prompt = apply_chat_template([ChatMessage.from_dict(m) for m in messages[:-1]],
                                     template_name=template, add_generation_prompt=True)
        reply = messages[-1]["content"] + end
        prompt_ids = engine.encode(prompt)
        ids = engine.encode(prompt + reply + "\n")
        k = len(prompt_ids)
        while k < len(ids) and len(engine.decode(ids[len(prompt_ids):k])) < len(reply):
            k += 1
        with torch.no_grad():
            logits = engine.model(torch.tensor([ids[:k - 1]]))[0]
        start = len(prompt_ids) - 1
        nats += float(F.cross_entropy(logits[start:], torch.tensor(ids[1:k])[start:],
                                      reduction="sum"))
        total_bytes += len(engine.decode(ids[len(prompt_ids):k]).encode("utf-8"))
    return nats / max(1, total_bytes) / math.log(2.0)


def cmd_eval(args) -> dict:
    from auralite.chat_interface import get_stop_tokens
    from auralite.engine import AuraLiteEngine, load_training_data

    engine = AuraLiteEngine()
    engine.load_model(args.model)
    dialogs = load_training_data([args.data]).dialogs[:args.n]
    from auralite.engine.dialogs import SFT_TEMPLATES, template_special_tokens

    template = engine.default_chat_template
    stop_ids = engine._stop_token_ids(get_stop_tokens(template))
    markers = template_special_tokens(template) if template in SFT_TEMPLATES else []
    greedy = args.temperature <= 0
    stopped = clean = exact = new_tokens = 0
    seen: list[int] = []

    def record(token: int, _logits) -> None:
        seen.append(token)

    for messages in dialogs:
        seen.clear()
        reply = engine.generate_chat(messages[:-1], max_new_tokens=args.max_new_tokens,
                                     temperature=args.temperature, top_k=1 if greedy else 40,
                                     repetition_penalty=1.0, on_token=record)
        stopped += bool(seen) and seen[-1] in stop_ids
        new_tokens += len(seen)
        clean += not any(m in reply for m in markers)
        exact += reply.strip() == messages[-1]["content"].strip()
    n = max(1, len(dialogs))
    return {"section": "eval", "model": args.model, "template": template, "dialogs": len(dialogs),
            "temperature": args.temperature, "stop_rate": stopped / n, "clean_rate": clean / n,
            "exact": exact / n, "mean_new_tokens": round(new_tokens / n, 2),
            # per token only where the markers are tokens (a plain-text model: None)
            "reply_loss": round(reply_loss(engine, dialogs), 4) if stop_ids else None,
            "reply_bpb": round(reply_bits_per_byte(engine, dialogs), 4),
            "single_token_stops": sorted(stop_ids)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="section", required=True)
    p = sub.add_parser("data", help="write synthetic dialogs (JSONL)")
    p.add_argument("out")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_data)
    p = sub.add_parser("text", help="write dialogs as plain ChatML text (a 2.14.2-style corpus)")
    p.add_argument("data")
    p.add_argument("out")
    p.set_defaults(func=cmd_text)
    p = sub.add_parser("train", help="train (or fine-tune --base) a chat model on dialogs")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--loss", choices=["assistant", "all"], default="assistant")
    p.add_argument("--base", help="fine-tune this checkpoint instead of a new model")
    p.add_argument("--lora", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=PARAMS["seed"])
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("eval", help="stop rate, exact replies and reply loss")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.set_defaults(func=cmd_eval)
    args = parser.parse_args(argv)
    print(json.dumps(args.func(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
