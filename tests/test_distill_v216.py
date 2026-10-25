"""2.16: dialogs from a teacher model (`auralite distill`): prompts files, fake
teachers (no network, no llama-cpp), judges, preference pairs, resuming an
interrupted run, the license in the metadata, and the CLI."""
from __future__ import annotations

import json
import os
import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from auralite.engine import (  # noqa: E402
    AuraLiteEngine,
    CallableTeacher,
    DialogSet,
    EngineTeacher,
    PreferenceSet,
    TeacherReply,
    distill_dialogs,
    load_teacher,
    load_training_data,
    read_prompts,
)
from auralite.engine import distill as dist  # noqa: E402


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


def _jsonl(path, rows):
    return _write(path, "".join(json.dumps(r) + "\n" for r in rows))


def _sums(n=6):
    return [dist.DistillPrompt([{"role": "user", "content": f"What is {i} plus {i}?"}],
                               f"{i} plus {i} is {2 * i}.") for i in range(n)]


class Counting(CallableTeacher):
    """A fake teacher: right on even questions, wrong on odd ones."""

    def __init__(self, **kw):
        super().__init__(self._answer, name="fake", **kw)
        self.calls = 0

    def _answer(self, messages, max_tokens, temperature):
        self.calls += 1
        i = int(messages[-1]["content"].split()[2])
        return f"{i} plus {i} is {2 * i + (i % 2)}."


# ----------------------------------------------------------------------
#  Prompts
# ----------------------------------------------------------------------
def test_read_prompts_formats(tmp_path):
    txt = _write(tmp_path / "p.txt", "What is 1 plus 1?\n\n  Hello!  \n")
    prompts, skipped = read_prompts(txt)
    assert [p.messages[-1]["content"] for p in prompts] == ["What is 1 plus 1?", "Hello!"]
    assert skipped == 0 and prompts[0].reference is None
    rows = [
        "Plain string?",
        {"prompt": "Q1?", "answer": "A1."},
        {"question": "Q2?", "reference": "A2.", "system": "Be brief."},
        {"instruction": "Translate", "input": "chat", "output": "cat"},
        {"messages": [{"role": "user", "content": "Q3?"}, {"role": "assistant", "content": "A3."}]},
        {"conversations": [{"from": "human", "value": "Q4?"}]},
        {"messages": [{"role": "user", "content": "Q5?"}, {"role": "assistant", "content": "x"},
                      {"role": "user", "content": "Q6?"}]},
        {"messages": [{"role": "system", "content": "only"}]},
        {"other": 1},
    ]
    prompts, skipped = read_prompts(_jsonl(tmp_path / "p.jsonl", rows))
    assert skipped == 2 and len(prompts) == 7
    assert prompts[1].reference == "A1."
    assert prompts[2].messages[0] == {"role": "system", "content": "Be brief."}
    assert prompts[3].messages[-1]["content"] == "Translate\n\nchat"
    assert prompts[3].reference == "cat"
    assert prompts[4].messages == [{"role": "user", "content": "Q3?"}]
    assert prompts[4].reference == "A3."
    assert prompts[5].messages == [{"role": "user", "content": "Q4?"}]
    assert len(prompts[6].messages) == 3 and prompts[6].reference is None
    with pytest.raises(ValueError, match="not a JSON line"):
        read_prompts(_write(tmp_path / "bad.jsonl", "{oops\n"))


# ----------------------------------------------------------------------
#  The work
# ----------------------------------------------------------------------
def test_dialogs_metadata_and_license(tmp_path):
    out = str(tmp_path / "chats.jsonl")
    teacher = CallableTeacher(lambda m, max_tokens, temperature: "Sure.", name="fake",
                              license="apache-2.0")
    meta = distill_dialogs(teacher, _sums(3), out, system="Be brief.", prompts_file="p.txt")
    data = load_training_data([out])
    assert isinstance(data, DialogSet) and len(data) == 3
    assert data.dialogs[0] == [{"role": "system", "content": "Be brief."},
                               {"role": "user", "content": "What is 0 plus 0?"},
                               {"role": "assistant", "content": "Sure."}]
    on_disk = json.load(open(dist.meta_path(out), encoding="utf-8"))
    assert on_disk == {k: v for k, v in meta.items() if k != "stopped"}
    assert on_disk["teacher"] == {"name": "fake", "kind": "callable", "path": "",
                                  "license": "apache-2.0"}
    assert "license" in on_disk["license_note"]
    assert on_disk["progress"] == {"done": 3, "total": 3, "complete": True}
    assert on_disk["files"]["dialogs"]["records"] == 3
    assert on_disk["files"]["dialogs"]["bytes"] == os.path.getsize(out)
    assert on_disk["settings"]["temperature"] == 0.0 and on_disk["settings"]["samples"] == 1
    assert on_disk["prompts"]["count"] == 3 and on_disk["auralite"]


def test_reply_kinds_and_filters(tmp_path):
    replies = iter([TeacherReply("cut off", finished=False), ("", True), (), ("ok", True),
                    TeacherReply("ok"), ("other", None)])
    teacher = CallableTeacher(lambda m, max_tokens, temperature: next(replies), name="r")
    out = str(tmp_path / "c.jsonl")
    meta = distill_dialogs(teacher, _sums(1), out, samples=6)
    assert meta["counts"]["truncated"] == 1 and meta["counts"]["empty"] == 2
    assert meta["settings"]["temperature"] == 0.8          # several samples: sampled
    assert [d[-1]["content"] for d in load_training_data([out]).dialogs] == ["ok", "other"]
    replies = iter([TeacherReply("cut off", finished=False)])
    meta = distill_dialogs(CallableTeacher(lambda *a, **k: next(replies), name="r"), _sums(1),
                           str(tmp_path / "k.jsonl"), keep_truncated=True)
    assert meta["counts"]["kept"] == 1


def test_exact_judge_makes_on_policy_pairs(tmp_path):
    out, pairs = str(tmp_path / "ok.jsonl"), str(tmp_path / "pairs.jsonl")
    meta = distill_dialogs(Counting(), _sums(6), out, judge="exact", pairs_path=pairs)
    assert meta["counts"]["kept"] == 3 and meta["counts"]["rejected"] == 3
    assert meta["counts"]["pairs"] == 3
    data = load_training_data([pairs])
    assert isinstance(data, PreferenceSet) and len(data) == 3
    first = data.pairs[0]
    assert first.prompt == [{"role": "user", "content": "What is 1 plus 1?"}]
    assert (first.chosen, first.rejected) == ("1 plus 1 is 2.", "1 plus 1 is 3.")


def _judge(spec):
    judge = dist.make_judge(spec)
    assert judge is not None
    return judge


def test_rule_judges():
    exact, contains, number = (_judge(n) for n in ("exact", "contains", "number"))
    assert exact.accept(" 2 is 2. ", "2 is 2.") and not exact.accept("2 is 3.", "2 is 2.")
    assert contains.accept("I think PARIS.", "paris") and not contains.accept("Rome", "Paris")
    assert number.accept("The answer: 1,234", "It is 1234.")
    assert not number.accept("no number", "4") and not number.accept("5", "4")
    regex = _judge("regex:^[A-Z].*\\.$")
    assert regex.accept("Fine.", None) and not regex.accept("fine", None)
    assert dist.make_judge(None) is None and dist.make_judge("none") is None
    with pytest.raises(ValueError, match="pattern"):
        dist.make_judge("regex:")
    with pytest.raises(ValueError, match="bad regex"):
        dist.make_judge("regex:(")
    with pytest.raises(ValueError, match="unknown judge"):
        dist.make_judge("vibes")
    with pytest.raises(ValueError, match="needs a teacher"):
        dist.make_judge("teacher")


def test_judge_needs_a_reference_and_pairs_need_a_judge(tmp_path):
    prompts = [dist.DistillPrompt([{"role": "user", "content": "Hi?"}])]
    meta = distill_dialogs(CallableTeacher(lambda *a, **k: "Hello.", name="t"), prompts,
                           str(tmp_path / "a.jsonl"), judge="exact")
    assert meta["counts"]["no_reference"] == 1 and meta["counts"]["kept"] == 0
    with pytest.raises(ValueError, match="need a judge"):
        distill_dialogs(Counting(), _sums(1), str(tmp_path / "b.jsonl"),
                        pairs_path=str(tmp_path / "p.jsonl"))
    with pytest.raises(ValueError, match="samples >= 2"):
        distill_dialogs(Counting(), _sums(1), str(tmp_path / "c.jsonl"), judge="teacher")
    with pytest.raises(ValueError, match="must not be the dialogs file"):
        distill_dialogs(Counting(), _sums(1), str(tmp_path / "d.jsonl"), judge="exact",
                        pairs_path=str(tmp_path / "d.jsonl"))


def test_teacher_judge(tmp_path):
    state = {"n": 0}

    def answer(messages, max_tokens, temperature):
        text = messages[-1]["content"]
        if text.startswith("Question:"):                 # asked to judge
            a = text.split("Reply A:\n")[1].split("\n\nReply B:")[0]
            return "A" if "good" in a else "B"
        state["n"] += 1
        return "a good reply" if state["n"] % 2 else "a bad reply"

    out, pairs = str(tmp_path / "t.jsonl"), str(tmp_path / "tp.jsonl")
    meta = distill_dialogs(CallableTeacher(answer, name="judge"), _sums(2), out, samples=2,
                           judge="teacher", pairs_path=pairs)
    assert meta["counts"]["pairs"] == 2
    pair = load_training_data([pairs]).pairs[0]
    assert (pair.chosen, pair.rejected) == ("a good reply", "a bad reply")
    # a judge that always says "A" follows the position: no verdict, no pair
    biased = CallableTeacher(lambda m, max_tokens, temperature: "A" if m[-1]["content"]
                             .startswith("Question:") else f"reply {len(m)} {temperature}",
                             name="biased")
    replies = iter(["one", "two", "A", "A"])
    biased.fn = lambda m, max_tokens, temperature: next(replies)
    meta = distill_dialogs(biased, _sums(1), str(tmp_path / "b.jsonl"), samples=2,
                           judge="teacher")
    assert meta["counts"]["no_verdict"] == 1 and meta["counts"]["kept"] == 0


def test_resume_after_a_stop_and_after_a_crash(tmp_path):
    out, pairs = str(tmp_path / "r.jsonl"), str(tmp_path / "rp.jsonl")
    prompts = _sums(10)
    stop = threading.Event()
    seen = []

    def progress(info):
        seen.append(info)
        if info["done"] == 4:
            stop.set()

    teacher = Counting()
    meta = distill_dialogs(teacher, prompts, out, judge="exact", pairs_path=pairs,
                           on_progress=progress, stop_event=stop)
    assert meta["stopped"] and meta["progress"] == {"done": 4, "total": 10, "complete": False}
    assert teacher.calls == 4 and seen[-1]["eta_seconds"] is not None
    # a crash while a record was being written: half a line past the recorded end
    with open(out, "a", encoding="utf-8") as f:
        f.write('{"messages": [{"role": "us')
    with open(pairs, "a", encoding="utf-8") as f:
        f.write('{"prompt"')
    teacher = Counting()
    meta = distill_dialogs(teacher, prompts, out, judge="exact", pairs_path=pairs)
    assert teacher.calls == 6 and not meta["stopped"] and meta["progress"]["complete"]
    lines = open(out, encoding="utf-8").read().splitlines()
    assert len(lines) == 5 and len(set(lines)) == 5          # even questions: right
    assert len(load_training_data([out])) == 5                # every record whole
    assert len(load_training_data([pairs])) == 5
    # done: the same call does nothing
    teacher = Counting()
    distill_dialogs(teacher, prompts, out, judge="exact", pairs_path=pairs)
    assert teacher.calls == 0
    # more prompts, the same ones first: the work goes on
    meta = distill_dialogs(teacher, _sums(12), out, judge="exact", pairs_path=pairs)
    assert teacher.calls == 2 and meta["progress"] == {"done": 12, "total": 12,
                                                       "complete": True}


def test_other_settings_do_not_resume(tmp_path):
    out = str(tmp_path / "s.jsonl")
    distill_dialogs(Counting(), _sums(2), out)
    with pytest.raises(ValueError, match="another teacher, other prompts or other settings"):
        distill_dialogs(Counting(), _sums(2), out, max_tokens=9)
    with pytest.raises(ValueError, match="another teacher"):
        distill_dialogs(Counting(), _sums(3)[1:], out)
    meta = distill_dialogs(Counting(), _sums(2), out, max_tokens=9, overwrite=True)
    assert meta["progress"]["done"] == 2
    assert len(open(out, encoding="utf-8").read().splitlines()) == 2
    plain = tmp_path / "plain.jsonl"
    plain.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        distill_dialogs(Counting(), _sums(1), str(plain))


# ----------------------------------------------------------------------
#  Teachers
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_chat(tmp_path_factory):
    root = tmp_path_factory.mktemp("teacher")
    rows = [{"messages": [{"role": "user", "content": f"Say {i}"},
                          {"role": "assistant", "content": "ok"}]} for i in range(40)]
    engine = AuraLiteEngine()
    engine.train(load_training_data([_jsonl(root / "c.jsonl", rows)]),
                 {"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "seq_length": 32,
                  "batch_size": 8, "epochs": 30, "tokenizer": "char", "seed": 1,
                  "optimizer": "adamw", "lr": 3e-3, "dropout": 0.0,
                  "progress_every_seconds": 1000})
    path = root / "chat.pt"
    engine.save_model(str(path))
    return str(path)


def test_an_auralite_checkpoint_teaches(tiny_chat, tmp_path):
    teacher = load_teacher(tiny_chat)
    assert teacher.kind == "auralite" and teacher.name == "chat" and teacher.license is None
    reply = teacher.reply([{"role": "user", "content": "Say 3"}], max_tokens=12)
    assert reply.finished is True and reply.text == "ok" and reply.tokens == 3
    cut = teacher.reply([{"role": "user", "content": "Say 3"}], max_tokens=1)
    assert cut.finished is False
    meta = distill_dialogs(teacher, [dist.DistillPrompt([{"role": "user", "content": "Say 5"}],
                                                        "ok")],
                           str(tmp_path / "self.jsonl"), judge="exact", max_tokens=12)
    assert meta["counts"]["kept"] == 1 and meta["teacher"]["kind"] == "auralite"


def test_a_gguf_teacher_without_llama_cpp():
    calls = []

    class FakeLlama:
        backend = "gguf"
        metadata = {"general.name": "tiny-instruct", "general.license": "apache-2.0"}

        def create_chat_completion(self, **kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": " Hi! "}, "finish_reason": "length"}],
                    "usage": {"completion_tokens": 7}}

    engine = SimpleNamespace(model=FakeLlama(), is_gguf_model=lambda: True,
                             is_hf_model=lambda: False)
    teacher = EngineTeacher(engine, path="/models/tiny.gguf")
    assert (teacher.kind, teacher.name, teacher.license) == ("gguf", "tiny-instruct",
                                                             "apache-2.0")
    reply = teacher.reply([{"role": "user", "content": "Hi"}], max_tokens=7, temperature=0.0,
                          seed=11)
    assert reply == TeacherReply("Hi!", False, 7)
    assert calls[0]["seed"] == 11 and calls[0]["top_k"] == 1 and calls[0]["max_tokens"] == 7
    assert EngineTeacher(engine, license="mit").license == "mit"


def test_a_hugging_face_teacher_uses_its_own_template(tmp_path):
    (tmp_path / "README.md").write_text("---\nlicense: apache-2.0\ntags: [chat]\n---\n# Card\n",
                                        encoding="utf-8")

    class Tokenizer:
        chat_template = "{{ messages }}"
        eos_token_id = 2
        pad_token_id = None

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "<s>" + messages[-1]["content"] + "</a>"

        def __call__(self, text, return_tensors, add_special_tokens):
            return {"input_ids": torch.tensor([[5, 6, 7]]),
                    "attention_mask": torch.ones(1, 3, dtype=torch.long)}

        def decode(self, ids, skip_special_tokens):
            return " hello"

    class Model:
        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False and kwargs["pad_token_id"] == 2
            return torch.tensor([[5, 6, 7, 9, 2]])

    proxy = SimpleNamespace(tokenizer=Tokenizer(), model=Model(), device="cpu")
    engine = SimpleNamespace(hf_proxy=proxy, is_gguf_model=lambda: False,
                             is_hf_model=lambda: True)
    teacher = EngineTeacher(engine, path=str(tmp_path))
    assert teacher.kind == "hf" and teacher.license == "apache-2.0"
    assert teacher.reply([{"role": "user", "content": "Hi"}], max_tokens=8) == \
        TeacherReply("hello", True, 2)


def test_load_teacher_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="teacher model not found"):
        load_teacher(str(tmp_path / "missing.gguf"))
    with pytest.raises(FileNotFoundError, match="teacher model not found"):
        load_teacher(str(tmp_path / "nothing here"))


def test_llama_server_reports_finish_reason_and_usage():
    from auralite.engine.gguf_proxy import LlamaServerProxy
    proxy = LlamaServerProxy.__new__(LlamaServerProxy)
    proxy._props = {}
    sent = []

    def request(method, path, payload=None):
        sent.append(payload)
        return {"choices": [{"message": {"content": "Hi"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3}}

    proxy._request = request
    result = proxy.create_chat_completion([{"role": "user", "content": "x"}], seed=4)
    assert isinstance(result, dict)
    assert result["choices"][0]["finish_reason"] == "length"
    assert result["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    assert sent[0]["seed"] == 4


def test_gguf_proxy_passes_the_seed():
    from auralite.engine.gguf_proxy import GGUFModelProxy
    proxy = GGUFModelProxy.__new__(GGUFModelProxy)
    proxy._generation_lock = threading.RLock()
    got = {}
    proxy.llama = SimpleNamespace(create_chat_completion=lambda **k: got.update(k) or {})
    proxy.create_chat_completion([{"role": "user", "content": "x"}], seed=9)
    assert got["seed"] == 9
    got.clear()
    proxy.create_chat_completion([{"role": "user", "content": "x"}])
    assert "seed" not in got


# ----------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------
def test_cli_distill(tiny_chat, tmp_path, capsys):
    from auralite.cli import main
    prompts = _jsonl(tmp_path / "p.jsonl", [{"prompt": f"Say {i}", "answer": "ok"}
                                            for i in range(4)])
    out, pairs = str(tmp_path / "o.jsonl"), str(tmp_path / "pairs.jsonl")
    rc = main(["distill", "--teacher", tiny_chat, "--prompts", prompts, "--out", out,
               "--judge", "exact", "--pairs", pairs, "--max-tokens", "12",
               "--system", "Be brief."])
    text = capsys.readouterr().out
    assert rc == 0 and "teacher: chat (auralite)" in text and "saved: " in text
    assert "metadata: " in text and os.path.exists(dist.meta_path(out))
    meta = json.load(open(dist.meta_path(out), encoding="utf-8"))
    assert meta["progress"]["complete"] and meta["settings"]["system"] == "Be brief."
    assert meta["files"]["dialogs"]["records"] == meta["counts"]["kept"]
    rc = main(["distill", "--teacher", tiny_chat, "--prompts", prompts, "--out", out,
               "--judge", "exact", "--pairs", pairs, "--max-tokens", "12",
               "--system", "Be brief.", "--quiet"])
    assert rc == 0
    assert main(["distill", "--teacher", tiny_chat, "--prompts", prompts, "--out", out,
                 "--max-tokens", "13"]) == 2
    assert "--overwrite" in capsys.readouterr().err
    assert main(["distill", "--teacher", tiny_chat, "--prompts", str(tmp_path / "no.txt"),
                 "--out", out]) == 2
    assert main(["distill", "--teacher", str(tmp_path / "none.gguf"), "--prompts", prompts,
                 "--out", str(tmp_path / "z.jsonl")]) == 2
    assert main(["distill", "--teacher", tiny_chat, "--prompts", prompts, "--out",
                 str(tmp_path / "y.jsonl"), "--pairs", str(tmp_path / "yp.jsonl")]) == 2
    empty = _write(tmp_path / "empty.txt", "\n\n")
    assert main(["distill", "--teacher", tiny_chat, "--prompts", empty, "--out",
                 str(tmp_path / "w.jsonl")]) == 2
    assert "no prompts" in capsys.readouterr().err
