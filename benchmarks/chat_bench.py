"""Chat fine-tuning benchmark (2.15): synthetic dialogs, training, stop rate, reply loss.

Used for the 2.15 numbers in docs/training.md § 15 (2.16: `eval` is the code of
``auralite chat-eval``, `auralite.engine.chat_eval`).

    python benchmarks/chat_bench.py data chats.jsonl --n 3000 --seed 1     # training dialogs
    python benchmarks/chat_bench.py data eval.jsonl --n 300 --seed 99      # other dialogs
    python benchmarks/chat_bench.py text chats.jsonl chats.txt   # the dialogs as plain ChatML text
    python benchmarks/chat_bench.py train --data chats.jsonl --out chat.pt [--loss all]
        [--base base.pt] [--lora 8] [--epochs 40] [--seed 3]
        [--prompt-weight 0.1] [--pack] [--early-stop 3] [--lora-lr 0.01]   (2.16)
    python benchmarks/chat_bench.py train --data chats.txt --out plain.pt   # as 2.14.2 would
    python benchmarks/chat_bench.py eval --model chat.pt --data eval.jsonl [--temperature 0]
    python benchmarks/chat_bench.py arith arith.jsonl --n 3000 --seed 55   # 2.16: sums only
    python benchmarks/chat_bench.py questions prompts.txt --n 1200 --seed 2024   # for a teacher

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

2.16 (docs/training.md § 17-18): `arith` writes the addition and subtraction
dialogs of `data` alone (the preference benchmark: the chat model's own wrong
answers become the rejected replies, ``auralite distill --judge exact
--pairs``), and `questions` writes short, varied questions, one per line, for
a teacher model to answer (``auralite distill --teacher``).
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


def arith_dialog(rng: random.Random) -> dict:
    """An addition or subtraction dialog, as `dialog` writes them (2.16)."""
    if rng.random() < 0.5:
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        user = rng.choice([f"What is {a} plus {b}?", f"Add {a} and {b}.", f"{a} + {b} = ?"])
        reply = f"{a} plus {b} is {a + b}."
    else:
        a, b = rng.randint(20, 99), rng.randint(1, 19)
        user = rng.choice([f"What is {a} minus {b}?", f"Subtract {b} from {a}."])
        reply = f"{a} minus {b} is {a - b}."
    messages = []
    system = rng.choice(SYSTEMS)
    if system:
        messages.append({"role": "system", "content": system})
    messages += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
    return {"messages": messages}


COUNTRIES = ["France", "Germany", "Italy", "Spain", "Japan", "Egypt", "Canada", "Peru", "Norway",
             "Kenya", "Chile", "Greece", "Poland", "Cuba", "India", "China", "Brazil", "Mexico",
             "Sweden", "Finland", "Latvia", "Estonia", "Ireland", "Portugal", "Austria", "Turkey",
             "Iran", "Thailand", "Vietnam", "Argentina", "Australia", "Morocco", "Nigeria",
             "Hungary", "Denmark", "Belgium", "Switzerland", "Ukraine", "Colombia", "Indonesia"]
ANIMALS = ["cat", "dog", "cow", "horse", "spider", "ant", "bird", "fish", "snake", "frog", "lion",
           "elephant", "duck", "sheep", "goat", "bee", "owl", "pig", "bear", "rabbit", "wolf",
           "whale", "crab", "octopus", "penguin", "kangaroo", "camel", "mouse", "chicken", "fox"]
WORDS = ["hot", "big", "fast", "up", "happy", "light", "old", "early", "full", "hard", "open",
         "rich", "strong", "wet", "loud", "near", "high", "clean", "thick", "sweet", "brave",
         "busy", "deep", "easy", "empty", "heavy", "long", "new", "quiet", "young"]
THINGS = ["the sky", "grass", "snow", "a banana", "blood", "coal", "an orange", "the sea",
          "a lemon", "milk", "a tomato", "chocolate", "a carrot", "ice", "the sun", "a cucumber"]
JOBS = ["doctor", "teacher", "baker", "pilot", "farmer", "chef", "nurse", "firefighter",
        "plumber", "dentist", "librarian", "carpenter", "painter", "police officer", "writer",
        "mechanic", "gardener", "programmer", "scientist", "judge"]
TOPICS = ["water", "the moon", "trees", "rain", "the heart", "bread", "music", "a rainbow",
          "the wind", "a volcano", "sleep", "the internet", "a battery", "a bicycle", "salt",
          "a library", "the winter", "a clock", "a bridge", "an island"]


def question(rng: random.Random) -> str:
    """A short question for a teacher model (2.16)."""
    kind = rng.randrange(12)
    if kind == 0:
        return f"What is the capital of {rng.choice(COUNTRIES)}?"
    if kind == 1:
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        return rng.choice([f"What is {a} plus {b}?", f"What is {a} times {b % 12 + 1}?",
                           f"What is {max(a, b)} minus {min(a, b)}?"])
    if kind == 2:
        return f"How many legs does a {rng.choice(ANIMALS)} have?"
    if kind == 3:
        return f"What is the opposite of {rng.choice(WORDS)}?"
    if kind == 4:
        return f"What color is {rng.choice(THINGS)}?"
    if kind == 5:
        return f"What does a {rng.choice(JOBS)} do?"
    if kind == 6:
        return f"Where does a {rng.choice(ANIMALS)} live?"
    if kind == 7:
        return f"Give me a synonym for {rng.choice(WORDS)}."
    if kind == 8:
        return f"Describe {rng.choice(TOPICS)} in one sentence."
    if kind == 9:
        return f"What sound does a {rng.choice(ANIMALS)} make?"
    if kind == 10:
        return f"Is a {rng.choice(ANIMALS)} bigger than a {rng.choice(ANIMALS)}?"
    return f"Why is {rng.choice(TOPICS)} important?"


def cmd_arith(args) -> dict:
    rng = random.Random(args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            f.write(json.dumps(arith_dialog(rng)) + "\n")
    return {"section": "arith", "out": args.out, "dialogs": args.n, "seed": args.seed}


def cmd_questions(args) -> dict:
    rng = random.Random(args.seed)
    seen: list[str] = []
    have: set[str] = set()
    while len(seen) < args.n:
        text = question(rng)
        if text not in have:
            have.add(text)
            seen.append(text)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(seen) + "\n")
    return {"section": "questions", "out": args.out, "questions": args.n, "seed": args.seed}


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
    params = dict(PARAMS, epochs=args.epochs, seed=args.seed)
    # 2.16: only what is given (the engine's defaults otherwise)
    for key, value in (("dialog_loss", args.loss), ("prompt_loss_weight", args.prompt_weight),
                       ("early_stopping_patience", args.early_stop),
                       ("lora_lr", args.lora_lr), ("lora_targets", args.lora_targets),
                       ("lora_embeddings", args.lora_embeddings)):
        if value is not None:
            params[key] = value
    if args.pack:
        params["dialog_packing"] = True
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


def reply_loss(engine, dialogs):
    """Mean cross-entropy over the reply tokens of `dialogs` (2.16: ``chat_eval``)."""
    from auralite.engine.chat_eval import reply_loss as measure
    return measure(engine, dialogs)


def reply_bits_per_byte(engine, dialogs):
    """Bits per byte of each dialog's last reply and end marker (2.16: ``chat_eval``)."""
    from auralite.engine.chat_eval import reply_bits_per_byte as measure
    return measure(engine, dialogs)


def cmd_eval(args) -> dict:
    """2.16: the code of ``auralite chat-eval`` (`auralite.engine.chat_eval`)."""
    from auralite.engine import AuraLiteEngine, load_training_data
    from auralite.engine.chat_eval import evaluate_chat

    engine = AuraLiteEngine()
    engine.load_model(args.model)
    dialogs = load_training_data([args.data]).dialogs[:args.n]
    result = evaluate_chat(engine, dialogs, max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature, samples=0)
    loss, bpb = result["reply_loss"], result["reply_bpb"]
    return {"section": "eval", "model": args.model, "template": result["template"],
            "dialogs": result["dialogs"], "temperature": args.temperature,
            "stop_rate": result["stop_rate"], "clean_rate": result["clean_rate"],
            "exact": result["exact"], "mean_new_tokens": result["mean_new_tokens"],
            # per token only where the markers are tokens (a plain-text model: None)
            "reply_loss": round(loss, 4) if loss is not None else None,
            "reply_bpb": round(bpb, 4) if bpb is not None else None,
            "single_token_stops": result["single_token_stops"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="section", required=True)
    p = sub.add_parser("data", help="write synthetic dialogs (JSONL)")
    p.add_argument("out")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_data)
    p = sub.add_parser("arith", help="2.16: addition and subtraction dialogs only (JSONL)")
    p.add_argument("out")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=55)
    p.set_defaults(func=cmd_arith)
    p = sub.add_parser("questions", help="2.16: short questions for a teacher, one per line")
    p.add_argument("out")
    p.add_argument("--n", type=int, default=1200)
    p.add_argument("--seed", type=int, default=2024)
    p.set_defaults(func=cmd_questions)
    p = sub.add_parser("text", help="write dialogs as plain ChatML text (a 2.14.2-style corpus)")
    p.add_argument("data")
    p.add_argument("out")
    p.set_defaults(func=cmd_text)
    p = sub.add_parser("train", help="train (or fine-tune --base) a chat model on dialogs")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--loss", choices=["assistant", "all"], default=None)
    p.add_argument("--prompt-weight", type=float, default=None, help="2.16")
    p.add_argument("--pack", action="store_true", help="2.16: pack whole dialogs")
    p.add_argument("--early-stop", type=int, default=None,
                   help="early_stopping_patience (2.15 used 3; 2.16: off on dialogs)")
    p.add_argument("--base", help="fine-tune this checkpoint instead of a new model")
    p.add_argument("--lora", type=int, default=0)
    p.add_argument("--lora-lr", type=float, default=None, help="2.16 (default 10 x lr)")
    p.add_argument("--lora-targets", choices=["all", "ffn", "attn"], default=None, help="2.16")
    p.add_argument("--lora-embeddings", choices=["all", "new"], default=None, help="2.16")
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
