"""Dialogs from a teacher model (2.16): ``auralite distill``.

A teacher model answers a file of prompts, and its replies become chat
dialogs to fine-tune a smaller model on (``auralite train --data``). The
teacher can be any model AuraLite runs:

* a GGUF file, through llama-cpp-python, with the chat template stored in the
  file;
* a running llama.cpp server (an ``http://`` address);
* a Hugging Face model (a folder or a hub id), with its own chat template;
* an AuraLite checkpoint (``.pt`` / ``.safetensors``), for example the chat
  model itself, to collect its own replies;
* in Python, any function (:class:`CallableTeacher`); the tests use one.

A judge can check the replies:

* ``exact``, ``contains`` or ``number`` compare them with the prompt's
  reference answer;
* ``regex:PATTERN`` accepts the replies the pattern matches;
* ``teacher`` asks the teacher which of two replies is better (twice, with the
  order swapped; only a consistent verdict counts).

With a judge only accepted replies become dialogs. A ``pairs`` file also gets
preference pairs for DPO or SimPO (:mod:`auralite.engine.preference`): an
accepted reply as the chosen one (or the reference answer, when no reply was
right) and a rejected reply.

The work can be resumed. After every prompt the files are flushed and
``<out>.meta.json`` records how far the work got. A run that was interrupted
(Ctrl+C, the Stop button, a crash) continues from there; the same command
does it. The metadata also names the teacher and its license, if the model
file states one. Some licenses forbid using a model's outputs to train other
models, so check the teacher's license before you train on its replies.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from auralite.engine.dialogs import dialog_from_record

logger = logging.getLogger("auralite")

__all__ = ["CallableTeacher", "DistillPrompt", "EngineTeacher", "JUDGES", "Teacher",
           "TeacherReply", "distill_dialogs", "load_teacher", "make_judge", "meta_path",
           "read_prompts"]

#: Judges by name (``regex`` takes a pattern: ``regex:PATTERN``).
JUDGES = ("exact", "contains", "number", "regex", "teacher")
#: Keys that hold a prompt's reference answer, in the order tried.
_REFERENCE_KEYS = ("reference", "answer", "completion", "response", "output", "chosen")
#: Keys that hold a prompt, in the order tried.
_PROMPT_KEYS = ("prompt", "question", "instruction", "input", "text")
_LICENSE_NOTE = ("Check that the teacher's license lets you train other models on its "
                 "replies before you use them.")


# ----------------------------------------------------------------------
#  Prompts
# ----------------------------------------------------------------------
@dataclass
class DistillPrompt:
    """The messages to reply to (the last is the user's) and, if known, the expected reply."""

    messages: list[dict]
    reference: str | None = None


def _prompt_from_record(record: Any) -> DistillPrompt | None:
    if isinstance(record, str):
        return DistillPrompt([{"role": "user", "content": record}]) if record.strip() else None
    if not isinstance(record, dict) and not isinstance(record, list):
        return None
    reference = None
    if isinstance(record, dict):
        reference = next((record[k] for k in _REFERENCE_KEYS if isinstance(record.get(k), str)),
                         None)
        key = next((k for k in _PROMPT_KEYS if record.get(k) is not None), None)
        if key is not None and isinstance(record[key], (str, list)):
            value = record[key]
            extra = record.get("input")
            if key == "instruction" and isinstance(value, str) and isinstance(extra, str) \
                    and extra.strip():
                value = f"{value}\n\n{extra}"
            if isinstance(value, str):
                messages = [{"role": "user", "content": value}]
            else:
                messages = dialog_from_record(value) or []
            system = record.get("system")
            if isinstance(system, str) and system.strip() and \
                    not any(m["role"] == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": system})
            return DistillPrompt(messages, reference) if _asks(messages) else None
    dialog = dialog_from_record(record)
    if not dialog:
        return None
    if dialog[-1]["role"] == "assistant":          # the dialog's own reply: the reference
        reference = dialog[-1]["content"] if reference is None else reference
        dialog = dialog[:-1]
    return DistillPrompt(dialog, reference) if _asks(dialog) else None


def _asks(messages: list[dict]) -> bool:
    return bool(messages) and messages[-1]["role"] == "user" and \
        bool(str(messages[-1]["content"]).strip())


def read_prompts(path: str) -> tuple[list[DistillPrompt], int]:
    """``(prompts, skipped records)`` of a prompts file.

    A ``.txt`` file has one prompt per line. In a ``.jsonl`` / ``.json`` file
    each record is a string, ``{"prompt": "...", "answer": "..."}`` (also
    ``question``, ``instruction``; ``reference``, ``completion``,
    ``response``, ``output``), or a dialog (``{"messages": [...]}``, ShareGPT,
    Alpaca); a dialog that ends with a reply is asked its last question, and
    that reply is the reference answer of the judges.
    """
    prompts: list[DistillPrompt] = []
    skipped = 0
    suffix = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8-sig") as f:
        if suffix == ".json":
            data = json.load(f)
            records: list = data if isinstance(data, list) else [data]
        elif suffix == ".jsonl":
            records = []
            for number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as err:
                    raise ValueError(f"{path}:{number}: not a JSON line ({err.msg})") from err
        else:
            records = [line.strip() for line in f if line.strip()]
    for record in records:
        prompt = _prompt_from_record(record)
        if prompt is None:
            skipped += 1
        else:
            prompts.append(prompt)
    return prompts, skipped


# ----------------------------------------------------------------------
#  Teachers
# ----------------------------------------------------------------------
@dataclass
class TeacherReply:
    """A teacher's reply. `finished`: it ended by itself (False: cut at the
    token limit; None: unknown); `tokens`: its length in the teacher's tokens."""

    text: str
    finished: bool | None = None
    tokens: int | None = None


class Teacher:
    """A model that replies to messages: :meth:`reply`."""

    name: str = "teacher"
    kind: str = "callable"
    path: str = ""
    license: str | None = None

    def reply(self, messages: list[dict], *, max_tokens: int = 256, temperature: float = 0.0,
              top_p: float = 0.95, seed: int | None = None) -> TeacherReply:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "kind": self.kind, "path": self.path, "license": self.license}


class CallableTeacher(Teacher):
    """A teacher made of a function ``fn(messages, max_tokens=, temperature=)``
    that returns the reply's text, ``(text, finished)`` or a :class:`TeacherReply`."""

    def __init__(self, fn: Callable[..., Any], name: str = "callable",
                 license: str | None = None):
        self.fn = fn
        self.name = name
        self.license = license

    def reply(self, messages: list[dict], *, max_tokens: int = 256, temperature: float = 0.0,
              top_p: float = 0.95, seed: int | None = None) -> TeacherReply:
        out = self.fn(messages, max_tokens=max_tokens, temperature=temperature)
        if isinstance(out, TeacherReply):
            return out
        if isinstance(out, tuple):
            text = out[0] if out else ""
            finished = out[1] if len(out) > 1 else None
            return TeacherReply(str(text), None if finished is None else bool(finished))
        return TeacherReply(str(out))


class EngineTeacher(Teacher):
    """The model loaded in an ``AuraLiteEngine``: GGUF, llama.cpp server, Hugging Face or AuraLite."""

    def __init__(self, engine: Any, name: str | None = None, license: str | None = None,
                 path: str = ""):
        self.engine = engine
        self.path = path
        if engine.is_gguf_model():
            self.kind = "llama-server" if getattr(engine.model, "backend", "") == "gguf-server" \
                else "gguf"
            metadata = getattr(engine.model, "metadata", None) or {}
            found = metadata.get("general.license") or metadata.get("general.license.name")
            self.license = license or (str(found) if found else None)
            self.name = name or str(metadata.get("general.name") or "") or \
                os.path.splitext(os.path.basename(path))[0] or "gguf"
        elif engine.is_hf_model():
            self.kind = "hf"
            self.license = license or _readme_license(path)
            self.name = name or os.path.basename(os.path.normpath(path)) or "hf"
        else:
            self.kind = "auralite"
            self.license = license
            self.name = name or os.path.splitext(os.path.basename(path))[0] or "auralite"

    def reply(self, messages: list[dict], *, max_tokens: int = 256, temperature: float = 0.0,
              top_p: float = 0.95, seed: int | None = None) -> TeacherReply:
        engine = self.engine
        greedy = float(temperature) <= 0.0
        if self.kind in ("gguf", "llama-server"):
            extra = {"seed": int(seed)} if seed is not None else {}
            result = engine.model.create_chat_completion(
                messages=messages, max_tokens=int(max_tokens), temperature=float(temperature),
                top_k=1 if greedy else 40, top_p=float(top_p), repeat_penalty=1.0, min_p=0.0,
                stream=False, **extra)
            choice = (result.get("choices") or [{}])[0]
            text = str((choice.get("message") or {}).get("content") or "")
            reason = choice.get("finish_reason")
            tokens = (result.get("usage") or {}).get("completion_tokens")
            return TeacherReply(text.strip(), None if reason is None else reason == "stop",
                                int(tokens) if tokens else None)
        if self.kind == "hf":
            return _hf_reply(engine, messages, int(max_tokens), float(temperature),
                             float(top_p))
        if seed is not None:
            import torch
            torch.manual_seed(int(seed))
        seen: list[int] = []
        stops = engine._stop_token_ids(engine._resolve_stop_strings(None,
                                                                    engine.default_chat_template))
        text = engine.generate_chat(messages, max_new_tokens=int(max_tokens),
                                    temperature=float(temperature), top_k=1 if greedy else 40,
                                    top_p=float(top_p), repetition_penalty=1.0,
                                    on_token=lambda token, _logits: seen.append(int(token)))
        finished = bool(seen) and seen[-1] in stops
        return TeacherReply(text.strip(), finished, len(seen))


def _hf_reply(engine: Any, messages: list[dict], max_tokens: int, temperature: float,
              top_p: float) -> TeacherReply:
    """A Hugging Face model's reply, written with the model's own chat template."""
    proxy = engine.hf_proxy
    tokenizer = proxy.tokenizer
    if not getattr(tokenizer, "chat_template", None):
        text = engine.generate_chat(messages, max_new_tokens=max_tokens,
                                    temperature=temperature, top_p=top_p,
                                    repetition_penalty=1.0)
        return TeacherReply(text.strip())
    import torch
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(proxy.device) for k, v in inputs.items()
              if k in ("input_ids", "attention_mask")}
    sample = temperature > 0.0
    options: dict = {"do_sample": sample, "repetition_penalty": 1.0}
    if sample:
        options.update(temperature=temperature, top_p=top_p)
    with torch.no_grad():
        output = proxy.model.generate(**inputs, max_new_tokens=max_tokens,
                                      pad_token_id=tokenizer.pad_token_id
                                      if tokenizer.pad_token_id is not None
                                      else tokenizer.eos_token_id,
                                      eos_token_id=tokenizer.eos_token_id, **options)
    new = output[0][inputs["input_ids"].shape[1]:]
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    finished = len(new) < max_tokens or (len(new) > 0 and int(new[-1]) in eos_ids)
    return TeacherReply(tokenizer.decode(new, skip_special_tokens=True).strip(), finished,
                        int(len(new)))


def _readme_license(path: str) -> str | None:
    """The ``license:`` of a Hugging Face model card (README.md front matter), if any."""
    readme = os.path.join(path, "README.md") if path and os.path.isdir(path) else ""
    if not readme or not os.path.isfile(readme):
        return None
    with open(readme, "r", encoding="utf-8", errors="replace") as f:
        head = f.read(4096)
    if not head.startswith("---"):
        return None
    match = re.search(r"^license:\s*([^\n]+)$", head.split("\n---", 1)[0], re.M)
    return match.group(1).strip().strip("'\"") if match else None


def load_teacher(spec: str, *, engine: Any = None, license: str | None = None,
                 n_ctx: int = 4096, n_threads: int | None = None) -> EngineTeacher:
    """A teacher from a GGUF file, a server address, a checkpoint or a Hugging Face model.

    `engine` (default: a new ``AuraLiteEngine``) loads it; `license` names
    the teacher's license when the model file does not.
    """
    if engine is None:
        from auralite.engine import AuraLiteEngine
        engine = AuraLiteEngine()
    text = str(spec)
    if text.startswith(("http://", "https://")):
        engine.load_gguf_model(text, n_ctx=n_ctx)
    elif text.lower().endswith(".gguf"):
        if not os.path.isfile(text):
            raise FileNotFoundError(f"teacher model not found: {text}")
        engine.load_gguf_model(text, n_ctx=n_ctx, n_threads=n_threads, use_chat_completion=True)
    elif os.path.isfile(text):
        engine.load_model(text)
    elif os.path.isdir(text) or re.fullmatch(r"[\w.-]+/[\w.-]+", text):
        engine.load_hf_model(text, local_files_only=os.path.isdir(text), verbose=False)
    else:
        raise FileNotFoundError(f"teacher model not found: {text} (a .gguf file, an "
                                f"http:// server, a .pt checkpoint or a Hugging Face model)")
    return EngineTeacher(engine, license=license, path=text)


# ----------------------------------------------------------------------
#  Judges
# ----------------------------------------------------------------------
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def _last_number(text: str) -> str | None:
    found = _NUMBER.findall(text.replace(",", "")) if text else []
    return found[-1].rstrip(".") if found else None


class _Judge:
    """Accepts or rejects a reply (rule judges) or picks the better of two (teacher)."""

    def __init__(self, name: str, pattern: str | None = None, teacher: Teacher | None = None):
        self.name = name
        self.regex = re.compile(pattern) if pattern is not None else None
        self.teacher = teacher
        self.needs_reference = name in ("exact", "contains", "number")
        self.pairwise = name == "teacher"

    def accept(self, reply: str, reference: str | None) -> bool:
        if self.name == "exact":
            return reply.strip() == str(reference).strip()
        if self.name == "contains":
            return str(reference).strip().lower() in reply.lower()
        if self.name == "number":
            return _last_number(reply) is not None and \
                _last_number(reply) == _last_number(str(reference))
        if self.regex is not None:
            return self.regex.search(reply) is not None
        raise ValueError(f"the {self.name} judge compares two replies")

    def prefer(self, messages: list[dict], first: str, second: str) -> int | None:
        """0 or 1 (the better reply), or None when the verdicts disagree."""
        assert self.teacher is not None
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        votes = []
        for a, b in ((first, second), (second, first)):
            ask = [{"role": "system", "content": "You compare two replies to a question. "
                                                 "Answer with one letter: A or B."},
                   {"role": "user", "content": f"Question:\n{question}\n\nReply A:\n{a}\n\n"
                                               f"Reply B:\n{b}\n\nWhich reply is better? "
                                               f"Answer A or B."}]
            verdict = self.teacher.reply(ask, max_tokens=4, temperature=0.0).text
            letter = re.search(r"\b([AB])\b", verdict.upper())
            votes.append(None if letter is None else "AB".index(letter.group(1)))
        if votes[0] is None or votes[1] is None or votes[0] == votes[1]:
            return None                        # no verdict, or it followed the position
        return votes[0]


def make_judge(spec: str | None, teacher: Teacher | None = None) -> _Judge | None:
    """A judge from its name: exact, contains, number, regex:PATTERN or teacher (None: none)."""
    if spec is None or not str(spec).strip() or str(spec).strip().lower() == "none":
        return None
    text = str(spec).strip()
    name, _, pattern = text.partition(":")
    name = name.strip().lower()
    if name == "regex":
        if not pattern:
            raise ValueError("the regex judge needs a pattern: regex:PATTERN")
        try:
            return _Judge("regex", pattern)
        except re.error as err:
            raise ValueError(f"bad regex judge pattern {pattern!r}: {err}") from err
    if name not in JUDGES:
        raise ValueError(f"unknown judge {text!r} (use {', '.join(JUDGES[:3])}, "
                         f"regex:PATTERN or teacher)")
    if name == "teacher" and teacher is None:
        raise ValueError("the teacher judge needs a teacher")
    return _Judge(name, teacher=teacher if name == "teacher" else None)


# ----------------------------------------------------------------------
#  The work
# ----------------------------------------------------------------------
def meta_path(out_path: str) -> str:
    """Where :func:`distill_dialogs` keeps the metadata of `out_path`."""
    return str(out_path) + ".meta.json"


def _prompts_hash(prompts: list[DistillPrompt]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(json.dumps([prompt.messages, prompt.reference], sort_keys=True,
                                 ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def _write_meta(path: str, meta: dict) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _truncate(path: str, size: int) -> None:
    if os.path.exists(path) and os.path.getsize(path) > size:
        with open(path, "r+b") as f:
            f.truncate(size)


def distill_dialogs(teacher: Teacher, prompts: list[DistillPrompt], out_path: str, *,
                    samples: int = 1, system: str | None = None, max_tokens: int = 256,
                    temperature: float | None = None, top_p: float = 0.95,
                    judge: str | None = None, pairs_path: str | None = None,
                    keep_truncated: bool = False, overwrite: bool = False, seed: int = 0,
                    prompts_file: str = "",
                    on_progress: Callable[[dict], None] | None = None,
                    stop_event: Any = None) -> dict:
    """Have `teacher` answer `prompts`; write the replies as dialogs to `out_path`.

    `samples` replies per prompt, sampled at `temperature` (default 0, greedy,
    for one sample and 0.8 for more). `system` is put before prompts that
    have no system message. Replies cut at `max_tokens` are dropped unless
    `keep_truncated`. `judge` (see :func:`make_judge`) keeps only the accepted
    replies; with `pairs_path` the preference pairs go there (TRL format:
    ``{"prompt": [...], "chosen": [...], "rejected": [...]}``).

    The work resumes where an earlier run on the same files, prompts and
    settings stopped; `overwrite` starts again. ``on_progress(info)`` follows
    each prompt; a set `stop_event` ends after the current prompt, ready to
    resume. Returns the metadata written next to `out_path` (``<out>.meta.json``).
    """
    samples = max(1, int(samples))
    if temperature is None:
        temperature = 0.0 if samples == 1 else 0.8
    rule = make_judge(judge, teacher)
    if rule is not None and rule.pairwise and samples < 2:
        raise ValueError("the teacher judge compares two replies: use samples >= 2")
    if pairs_path and rule is None:
        raise ValueError("preference pairs need a judge (exact, contains, number, "
                         "regex:PATTERN or teacher)")
    if pairs_path and os.path.abspath(pairs_path) == os.path.abspath(out_path):
        raise ValueError("the pairs file must not be the dialogs file")
    settings = {"samples": samples, "system": system or None, "max_tokens": int(max_tokens),
                "temperature": float(temperature), "top_p": float(top_p),
                "judge": str(judge).strip() if rule is not None else None,
                "pairs": bool(pairs_path), "keep_truncated": bool(keep_truncated),
                "seed": int(seed)}
    digest = _prompts_hash(prompts)
    meta_file = meta_path(out_path)
    meta: dict | None = None
    if not overwrite and os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            old: dict = json.load(f)
        meta = old
        before = old.get("prompts") or {}
        count = int(before.get("count") or 0)
        same = (old.get("settings") == settings
                and (old.get("teacher") or {}).get("name") == teacher.name
                and (before.get("sha256") == digest
                     # more prompts than before, the same ones first: the work goes on
                     or (count <= len(prompts)
                         and _prompts_hash(prompts[:count]) == before.get("sha256"))))
        if same and before.get("sha256") != digest:
            old["prompts"] = {**before, "sha256": digest, "count": len(prompts)}
            old["progress"] = {**old["progress"], "total": len(prompts), "complete": False}
        if not same:
            raise ValueError(f"{out_path} was made with another teacher, other prompts or "
                             f"other settings ({meta_file}); start again with overwrite "
                             f"(--overwrite) or write to another file")
    elif not overwrite and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        raise ValueError(f"{out_path} already exists (and has no {os.path.basename(meta_file)} "
                         f"to resume from); use overwrite (--overwrite) or another file")
    if meta is None:
        meta = {
            "format": "auralite-distill", "version": 1,
            "teacher": teacher.describe(), "license_note": _LICENSE_NOTE,
            "prompts": {"file": prompts_file, "sha256": digest, "count": len(prompts)},
            "settings": settings,
            "progress": {"done": 0, "total": len(prompts), "complete": False},
            "files": {"dialogs": {"path": os.path.basename(out_path), "bytes": 0, "records": 0},
                      "pairs": {"path": os.path.basename(pairs_path) if pairs_path else None,
                                "bytes": 0, "records": 0}},
            "counts": {"replies": 0, "kept": 0, "truncated": 0, "empty": 0, "rejected": 0,
                       "pairs": 0, "no_reference": 0, "no_verdict": 0},
            "speed": {"seconds": 0.0, "completion_tokens": 0},
        }
        try:
            from auralite import __version__
            meta["auralite"] = __version__
        except Exception:                 # pragma: no cover - always importable in a checkout
            pass
        for path in (out_path, pairs_path):
            if path:
                os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
                open(path, "w", encoding="utf-8").close()
        _write_meta(meta_file, meta)
    else:
        logger.info(f"[AuraLite] Distill: resuming {out_path} at prompt "
                    f"{meta['progress']['done'] + 1:,} of {len(prompts):,}")
    files = meta["files"]
    _truncate(out_path, int(files["dialogs"]["bytes"]))
    if pairs_path:
        _truncate(pairs_path, int(files["pairs"]["bytes"]))
    counts = meta["counts"]
    speed = meta["speed"]
    if teacher.license is None and teacher.kind != "auralite":
        logger.warning("[AuraLite] WARNING: the teacher's license is unknown. " + _LICENSE_NOTE)

    done = int(meta["progress"]["done"])
    first = done
    started = time.time()
    seconds_before = float(speed.get("seconds") or 0.0)
    dialogs_out = open(out_path, "a", encoding="utf-8", newline="\n")
    pairs_out = open(pairs_path, "a", encoding="utf-8", newline="\n") if pairs_path else None
    stopped = False
    try:
        for index in range(done, len(prompts)):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            prompt = prompts[index]
            messages = [dict(m) for m in prompt.messages]
            if system and not any(m["role"] == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": system})
            replies: list[TeacherReply] = []
            for k in range(samples):
                reply = teacher.reply(messages, max_tokens=int(max_tokens),
                                      temperature=float(temperature), top_p=float(top_p),
                                      seed=int(seed) + index * samples + k)
                replies.append(reply)
                counts["replies"] += 1
                speed["completion_tokens"] = int(speed.get("completion_tokens") or 0) + \
                    int(reply.tokens or 0)
            usable: list[str] = []
            for reply in replies:
                if not reply.text.strip():
                    counts["empty"] += 1
                elif reply.finished is False and not keep_truncated:
                    counts["truncated"] += 1
                elif reply.text not in usable:
                    usable.append(reply.text)
            good: list[str] = []
            bad: list[str] = []
            if rule is None:
                good = usable
            elif rule.pairwise:
                if len(usable) >= 2:
                    verdict = rule.prefer(messages, usable[0], usable[1])
                    if verdict is None:
                        counts["no_verdict"] += 1
                    else:
                        good, bad = [usable[verdict]], [usable[1 - verdict]]
            elif rule.needs_reference and prompt.reference is None:
                counts["no_reference"] += 1
            else:
                for text in usable:
                    (good if rule.accept(text, prompt.reference) else bad).append(text)
            counts["rejected"] += len(bad) if rule is not None and not rule.pairwise else 0
            for text in good:
                dialogs_out.write(json.dumps({"messages": messages + [
                    {"role": "assistant", "content": text}]}, ensure_ascii=False) + "\n")
                files["dialogs"]["records"] += 1
                counts["kept"] += 1
            if pairs_out is not None and bad:
                chosen = good[0] if good else (prompt.reference if rule is not None
                                               and rule.needs_reference else None)
                if chosen is not None and chosen.strip() != bad[0].strip():
                    pairs_out.write(json.dumps({
                        "prompt": messages,
                        "chosen": [{"role": "assistant", "content": chosen}],
                        "rejected": [{"role": "assistant", "content": bad[0]}]},
                        ensure_ascii=False) + "\n")
                    files["pairs"]["records"] += 1
                    counts["pairs"] += 1
            dialogs_out.flush()
            files["dialogs"]["bytes"] = dialogs_out.tell()
            if pairs_out is not None:
                pairs_out.flush()
                files["pairs"]["bytes"] = pairs_out.tell()
            done = index + 1
            elapsed = seconds_before + time.time() - started
            speed["seconds"] = round(elapsed, 3)
            speed["prompts_per_hour"] = round(3600.0 * done / max(1e-9, elapsed), 1) \
                if elapsed else None
            speed["tokens_per_second"] = round(int(speed["completion_tokens"])
                                               / max(1e-9, elapsed), 2)
            meta["progress"] = {"done": done, "total": len(prompts),
                                "complete": done == len(prompts)}
            _write_meta(meta_file, meta)
            if on_progress is not None:
                rate = (done - first) / max(1e-9, time.time() - started)
                on_progress({"done": done, "total": len(prompts), "kept": counts["kept"],
                             "pairs": counts["pairs"], "truncated": counts["truncated"],
                             "seconds": elapsed, "prompts_per_second": rate,
                             "eta_seconds": (len(prompts) - done) / rate if rate else None,
                             "last": {"prompt": messages[-1]["content"],
                                      "reply": replies[0].text if replies else ""}})
    finally:
        dialogs_out.close()
        if pairs_out is not None:
            pairs_out.close()
    meta["stopped"] = stopped
    return meta
