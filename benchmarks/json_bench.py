"""JSON mode benchmark (2.17): how often answers parse, with and without the grammar.

Used for the numbers in docs/engine.md § 12.

    python benchmarks/json_bench.py data people.jsonl --n 2000 --seed 1     # training dialogs
    python benchmarks/json_bench.py data people_eval.jsonl --n 300 --seed 99
    python benchmarks/chat_bench.py train --data people.jsonl --out people.pt --epochs 20
    python benchmarks/json_bench.py eval --model people.pt --data people_eval.jsonl
    python benchmarks/json_bench.py noise --n 200          # an untrained model: pure noise

`data` writes dialogs that ask for a person's record in words ("Anna is 34
and lives in Riga.") and answer with it as JSON (``{"name": "Anna", "age":
34, "city": "Riga"}``). `eval` answers every dialog twice: as plain text and
under the record's JSON schema (``response_format``), and reports how many
answers parse, follow the schema and are exactly the expected record, and
the decoding speed. `noise` does the same with a model that has never been
trained (random weights, a character vocabulary of printable ASCII), for
several schemas, JSON mode and tool calls: the worst model there is.
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

NAMES = ["Anna", "Boris", "Clara", "David", "Elena", "Felix", "Greta", "Hugo", "Irina",
         "Jonas", "Katya", "Leo", "Maria", "Nikolai", "Olga", "Pavel", "Rita", "Sven",
         "Tanya", "Viktor", "Wanda", "Yuri", "Zoe", "Arturs", "Liga", "Maris", "Ilze"]
CITIES = ["Riga", "Oslo", "Paris", "Rome", "Berlin", "Madrid", "Tokyo", "Lima", "Cairo",
          "Vilnius", "Tallinn", "Warsaw", "Prague", "Vienna", "Dublin", "Lisbon", "Athens"]
TEMPLATES = ["{name} is {age} and lives in {city}.", "Make a record: {name}, {age}, {city}.",
             "Name {name}, age {age}, city {city}.", "{name} ({age}) is from {city}.",
             "Record {name} from {city}, {age} years old."]
SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                         "city": {"type": "string"}},
          "required": ["name", "age", "city"]}
NOISE_SCHEMAS = {
    "record": SCHEMA,
    "nested": {"type": "object", "required": ["id", "items", "owner"], "properties": {
        "id": {"type": "integer"},
        "items": {"type": "array", "maxItems": 4, "items": {
            "type": "object", "required": ["sku", "qty"],
            "properties": {"sku": {"type": "string", "maxLength": 8},
                           "qty": {"type": "integer"},
                           "kind": {"enum": ["book", "food", "tool"]}}}},
        "owner": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "paid": {"type": "boolean"}}},
}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"},
                                     "unit": {"enum": ["c", "f"]}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "send_mail", "parameters": {
        "type": "object", "properties": {"to": {"type": "string"},
                                         "body": {"type": "string", "maxLength": 40}},
        "required": ["to", "body"]}}}]


def record(rng: random.Random) -> tuple[str, dict]:
    person = {"name": rng.choice(NAMES), "age": rng.randint(1, 99), "city": rng.choice(CITIES)}
    return rng.choice(TEMPLATES).format(**person), person


def cmd_data(args) -> dict:
    rng = random.Random(args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            prompt, person = record(rng)
            f.write(json.dumps({"messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(person)}]}) + "\n")
    return {"section": "data", "out": args.out, "dialogs": args.n}


def _measure(engine, dialogs, grammar, args) -> dict:
    from auralite.engine.structured import validate_json

    parsed = valid = exact = tokens = 0
    started = time.time()
    for d in dialogs:
        expected = json.loads(d[-1]["content"])
        text = engine.generate_chat(d[:-1], args.max_tokens,
                                    temperature=args.temperature, top_k=0, top_p=1.0,
                                    repetition_penalty=1.0, grammar=grammar)
        tokens += len(engine.encode(text))
        try:
            value = json.loads(text)
        except ValueError:
            continue
        parsed += 1
        if not validate_json(value, SCHEMA):
            valid += 1
        exact += value == expected
    seconds = time.time() - started
    n = len(dialogs)
    return {"parsed": round(parsed / n, 4), "schema_valid": round(valid / n, 4),
            "exact": round(exact / n, 4), "tokens_per_second": round(tokens / seconds, 1),
            "seconds": round(seconds, 1)}


def cmd_eval(args) -> dict:
    import torch

    from auralite.engine import AuraLiteEngine, load_training_data
    from auralite.engine.structured import compile_schema

    engine = AuraLiteEngine()
    engine.load_model(args.model)
    dialogs = load_training_data([args.data]).dialogs[: args.limit]
    out: dict = {"section": "eval", "model": args.model, "dialogs": len(dialogs),
                 "temperature": args.temperature}
    for name, grammar in (("plain", None), ("schema", compile_schema(SCHEMA))):
        torch.manual_seed(args.seed)
        out[name] = _measure(engine, dialogs, grammar, args)
    table = engine.__dict__.get("_json_token_table")
    if table is not None:
        out["mask_states"] = table[2].computed
    return out


def cmd_noise(args) -> dict:
    import string

    import torch

    from auralite.engine import AuraLiteEngine
    from auralite.engine import structured as S

    torch.manual_seed(args.seed)
    engine = AuraLiteEngine()
    alphabet = string.printable * 4
    engine.train(alphabet, {"tokenizer": "char", "d_model": 64, "d_ff": 128, "n_heads": 4,
                            "n_layers": 2, "seq_length": 32, "batch_size": 4,
                            "max_steps": 1, "lr": 1e-12, "epochs": 1, "val_split": 0.0,
                            "seed": args.seed})             # one step that changes nothing
    messages = [{"role": "user", "content": "Give me JSON."}]
    tools = S.normalize_tools(TOOLS)
    cases: dict[str, tuple[S.JsonGrammar, dict | None]] = {
        name: (S.compile_schema(schema), schema) for name, schema in NOISE_SCHEMAS.items()}
    cases["json_object"] = (S.json_object_grammar(), {"type": "object"})
    cases["tool_call"] = (S.tool_grammar(tools, "required"), None)
    rng = random.Random(args.seed)
    out: dict = {"section": "noise", "answers_per_case": args.n}
    for name, (grammar, schema) in cases.items():
        least = grammar.cost(grammar.initial)
        parsed = valid = 0
        lengths = []
        started = time.time()
        for _ in range(args.n):
            budget = least + rng.choice([0, 8, 32, 96])
            text = engine.generate_chat(messages, budget, temperature=1.0, top_k=0, top_p=1.0,
                                        repetition_penalty=1.0, grammar=grammar)
            lengths.append(len(engine.encode(text)))
            try:
                value = json.loads(text)
            except ValueError:
                continue
            parsed += 1
            if schema is not None:
                valid += not S.validate_json(value, schema)
            else:
                call = S.parse_tool_call(text, tools)
                valid += call is not None
        seconds = time.time() - started
        plain_ok = 0
        for _ in range(min(args.n, 50)):
            text = engine.generate_chat(messages, least + 32, temperature=1.0, top_k=0,
                                        top_p=1.0, repetition_penalty=1.0)
            try:
                json.loads(text)
                plain_ok += 1
            except ValueError:
                pass
        out[name] = {"parsed": round(parsed / args.n, 4), "valid": round(valid / args.n, 4),
                     "shortest_tokens": least, "mean_tokens": round(sum(lengths) / len(lengths), 1),
                     "tokens_per_second": round(sum(lengths) / seconds, 1),
                     "plain_parsed": round(plain_ok / min(args.n, 50), 4)}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("data", help="person-record dialogs (JSONL)")
    p.add_argument("out")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_data)
    p = sub.add_parser("eval", help="answers as plain text and under the schema")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_eval)
    p = sub.add_parser("noise", help="an untrained model under the grammar")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_noise)
    args = parser.parse_args(argv)
    print(json.dumps(args.func(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
