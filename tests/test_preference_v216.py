"""2.16: preference pairs (DPO, SimPO): their formats, the losses, training on a
chat model in memory, and the CLI (`auralite train --method`, `chat-eval` on pairs)."""
from __future__ import annotations

import json
import math
import threading

import pytest

torch = pytest.importorskip("torch")

from auralite.engine import (  # noqa: E402
    AuraLiteEngine,
    DataFormatError,
    DialogSet,
    PreferencePair,
    PreferenceSet,
    data_kind,
    evaluate_preferences,
    load_training_data,
    validate_params,
)
from auralite.engine import preference as pref  # noqa: E402
from auralite.engine.dialogs import encode_dialog, encode_reply  # noqa: E402

TINY = {"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "seq_length": 48,
        "batch_size": 8, "tokenizer": "char", "seed": 1, "optimizer": "adamw", "lr": 3e-3,
        "dropout": 0.0, "progress_every_seconds": 1000}


def _question(i: int) -> str:
    return f"Is {i} a number?"


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


@pytest.fixture(scope="module")
def chat_model(tmp_path_factory):
    """A tiny chat model that answers "yes." or "no." about as often (a coin to tilt)."""
    root = tmp_path_factory.mktemp("pref")
    rows = [{"messages": [{"role": "user", "content": _question(i)},
                          {"role": "assistant", "content": "yes." if i % 2 else "no."}]}
            for i in range(80)]
    engine = AuraLiteEngine()
    engine.train(load_training_data([_write(root / "chats.jsonl", rows)]),
                 dict(TINY, epochs=6))
    path = root / "chat.pt"
    engine.save_model(str(path))
    return str(path)


def _pairs(n: int = 40, start: int = 100) -> PreferenceSet:
    return PreferenceSet([PreferencePair([{"role": "user", "content": _question(i)}],
                                         "yes.", "no.") for i in range(start, start + n)])


def _engine(path: str) -> AuraLiteEngine:
    engine = AuraLiteEngine()
    engine.load_model(path)
    return engine


def _weights(engine) -> dict:
    return {k: v.detach().clone() for k, v in engine.model.state_dict().items()}


# ----------------------------------------------------------------------
#  Formats
# ----------------------------------------------------------------------
USER = [{"role": "user", "content": "Hi?"}]


@pytest.mark.parametrize("record", [
    {"prompt": "Hi?", "chosen": "Hello.", "rejected": "Go away."},
    {"prompt": USER, "chosen": [{"role": "assistant", "content": "Hello."}],
     "rejected": [{"role": "assistant", "content": "Go away."}]},
    {"chosen": USER + [{"role": "assistant", "content": "Hello."}],
     "rejected": USER + [{"role": "assistant", "content": "Go away."}]},
    {"messages": USER, "chosen": "Hello.", "rejected": "Go away."},
    {"question": "Hi?", "chosen": "Hello.", "rejected": "Go away."},
    {"input": {"messages": USER, "tools": [], "parallel_tool_calls": True},
     "preferred_output": [{"role": "assistant", "content": "Hello."}],
     "non_preferred_output": [{"role": "assistant", "content": "Go away."}]},
])
def test_pair_formats(record):
    pair = pref.pair_from_record(record)
    assert pair == PreferencePair(USER, "Hello.", "Go away.")
    assert pref.is_preference_record(record)


def test_pair_system_and_instruction_input():
    pair = pref.pair_from_record({"system": "Be kind.", "instruction": "Greet", "input": "Bob",
                                  "chosen": "Hi Bob.", "rejected": "No."})
    assert pair is not None
    assert pair.prompt == [{"role": "system", "content": "Be kind."},
                           {"role": "user", "content": "Greet\n\nBob"}]


@pytest.mark.parametrize("record", [
    {"prompt": "Hi?", "chosen": "Same.", "rejected": " Same. "},          # no preference
    {"prompt": "Hi?", "chosen": "", "rejected": "No."},                   # empty reply
    {"prompt": [{"role": "system", "content": "x"}], "chosen": "a", "rejected": "b"},
    {"chosen": USER + [{"role": "assistant", "content": "a"}],            # different prompts
     "rejected": [{"role": "user", "content": "Other?"}, {"role": "assistant", "content": "b"}]},
    {"prompt": "Hi?", "chosen": [{"role": "assistant", "content": "a"},
                                 {"role": "assistant", "content": "b"}], "rejected": "c"},
    {"chosen": "The sky is blue.", "rejected": "The sky is green."},     # plain text, no chat
])
def test_unusable_pairs(record):
    assert pref.pair_from_record(record) is None


def test_load_training_data_reads_pairs(tmp_path):
    path = _write(tmp_path / "prefs.jsonl", [
        {"prompt": "Hi?", "chosen": "Hello.", "rejected": "Go away."},
        {"prompt": "Hi?", "chosen": "x", "rejected": "x"},
        {"messages": USER, "chosen": "Hello!", "rejected": "No."}])
    data = load_training_data([path])
    assert isinstance(data, PreferenceSet) and len(data) == 2
    assert data.skipped == 1 and data.first_skipped == "prefs.jsonl:2"
    assert data_kind([path]) == "preferences"
    dialogs = data.as_dialogs()
    assert isinstance(dialogs, DialogSet)
    assert dialogs.dialogs[0][-1] == {"role": "assistant", "content": "Hello."}


def test_pairs_do_not_mix_with_dialogs(tmp_path):
    pairs = _write(tmp_path / "a.jsonl", [{"prompt": "Hi?", "chosen": "a", "rejected": "b"}])
    chats = _write(tmp_path / "b.jsonl", [{"messages": USER + [
        {"role": "assistant", "content": "a"}]}])
    with pytest.raises(DataFormatError, match="preference pairs train on their own"):
        load_training_data([pairs, chats])
    empty = _write(tmp_path / "c.jsonl", [{"prompt": "Hi?", "chosen": "a", "rejected": "a"}])
    with pytest.raises(DataFormatError, match="no usable preference pair"):
        load_training_data([empty])


def test_split_holds_out_whole_pairs():
    data = _pairs(50)
    train, val = data.split(0.2)
    assert len(val) == 10 and not set(train) & set(val)
    assert data.split(0.2) == (train, val)


def test_encode_reply_matches_the_dialog_tokens(chat_model):
    engine = _engine(chat_model)
    history = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Is 3 a number?"}]
    prompt, reply = encode_reply(engine.tokenizer, history, "yes.", "chatml")
    ids, targets = encode_dialog(engine.tokenizer, history + [{"role": "assistant",
                                                               "content": "yes."}], "chatml")
    assert ids[:len(prompt) + len(reply)] == prompt + reply
    assert targets[len(prompt):len(prompt) + len(reply)] == reply
    assert engine.tokenizer.decode(reply[-1:]) == "<|im_end|>"


def test_fit_cuts_the_prompt_first():
    prompt, reply, cut = pref._fit(list(range(10)), [7, 8, 9], 8)
    assert (prompt, reply, cut) == ([4, 5, 6, 7, 8, 9], [7, 8, 9], True)
    prompt, reply, cut = pref._fit([1, 2], list(range(20)), 8)
    assert prompt == [2] and reply == list(range(8)) and cut
    assert pref._fit([1, 2], [3], 8) == ([1, 2], [3], False)


def test_batch_masks_the_replies():
    inputs, targets, mask = pref._batch([([1, 2, 3], [4, 5]), ([6], [7, 8, 9])], "cpu")
    assert inputs.tolist() == [[1, 2, 3, 4], [6, 7, 8, 0]]
    assert targets.tolist() == [[2, 3, 4, 5], [7, 8, 9, 0]]
    assert mask.tolist() == [[False, False, True, True], [True, True, True, False]]


def test_losses():
    c, r = torch.tensor([-2.0, -5.0]), torch.tensor([-3.0, -1.0])
    rc, rr = torch.tensor([-2.5, -4.0]), torch.tensor([-2.5, -2.0])
    n_c, n_r = torch.tensor([2.0, 4.0]), torch.tensor([1.0, 2.0])
    loss, cw, rw = pref.preference_loss("dpo", c, r, n_c, n_r, rc, rr, 0.1)
    expected = -torch.nn.functional.logsigmoid(0.1 * ((c - rc) - (r - rr)))
    assert torch.allclose(loss, expected) and torch.allclose(cw - rw, 0.1 * ((c - rc) - (r - rr)))
    # DPO-Positive: a chosen reply less likely than before costs lambda x the drop
    loss_p, _cw, _rw = pref.preference_loss("dpo", c, r, n_c, n_r, rc, rr, 0.1, positive=50.0)
    drop = (rc - c).clamp_min(0.0)
    assert drop.tolist() == [0.0, 1.0]
    assert torch.allclose(loss_p, -torch.nn.functional.logsigmoid(
        0.1 * ((c - rc) - (r - rr)) - 0.1 * 50.0 * drop))
    loss, cw, rw = pref.preference_loss("simpo", c, r, n_c, n_r, None, None, 2.0, 1.0)
    assert torch.allclose(loss, -torch.nn.functional.logsigmoid(2.0 * (c / n_c - r / n_r) - 1.0))
    with pytest.raises(ValueError, match="reference"):
        pref.preference_loss("dpo", c, r, n_c, n_r, None, None, 0.1)


def test_params_are_validated():
    assert validate_params({"method": "dpo", "preference_beta": 0.2, "simpo_gamma": 0.0,
                            "preference_sft_weight": 1.0, "dpo_positive": 50}) == []
    errors = validate_params({"method": "ppo", "preference_beta": 0, "simpo_gamma": -1,
                              "preference_sft_weight": "x", "dpo_positive": -1})
    assert len(errors) == 5
    assert validate_params({"preference_beta": "auto"}) == []


# ----------------------------------------------------------------------
#  Training
# ----------------------------------------------------------------------
def test_dpo_tilts_the_model_towards_the_chosen_replies(chat_model):
    engine = _engine(chat_model)
    before = _weights(engine)
    summary = engine.train(_pairs(), {"epochs": 4, "batch_size": 8, "lr": 3e-3, "seed": 1,
                                      "val_split": 0.25})
    report = summary["preference"]
    assert summary["method"] == "dpo" and report["beta"] == pref.DEFAULT_DPO_BETA
    initial, final = report["initial"], report["final"]
    assert initial["loss"] == pytest.approx(math.log(2.0), abs=1e-6)
    assert initial["accuracy"] == 0.0 and initial["kl"] == pytest.approx(0.0, abs=1e-6)
    assert final["accuracy"] == 1.0 and final["margin"] > 0
    assert final["likelihood_accuracy"] == 1.0 and final["kl"] > 0
    assert final["loss"] < initial["loss"]
    assert report["pairs"] == {"train": 30, "val": 10, "skipped": 0, "dropped": 0, "truncated": 0}
    assert len(report["history"]) == summary["epochs_completed"] == 4
    assert summary["optimizer_steps"] == 16 and summary["stop_reason"] is None
    assert any(not torch.equal(before[k], v) for k, v in engine.model.state_dict().items())
    assert engine.optimizer is None and not engine.model.training
    assert engine.params_used["preference"]["method"] == "dpo"
    assert engine.last_train_summary["val_loss"] == final["loss"]
    assert evaluate_preferences(engine, _pairs(20, start=500))["accuracy"] == 1.0


def test_dropout_is_off_during_the_run(tmp_path):
    from auralite.engine.layers import ResidualDropout
    rows = [{"messages": [{"role": "user", "content": _question(i)},
                          {"role": "assistant", "content": "yes." if i % 2 else "no."}]}
            for i in range(40)]
    engine = AuraLiteEngine()
    engine.train(load_training_data([_write(tmp_path / "c.jsonl", rows)]),
                 dict(TINY, epochs=2, dropout=0.3))
    layers = [m for m in engine.model.modules() if isinstance(m, ResidualDropout)]
    assert layers and all(m.p == 0.3 for m in layers)
    losses = []
    engine.train(_pairs(), {"epochs": 1, "seed": 1, "batch_size": 40, "val_split": 0.0,
                            "progress_every_seconds": 1e-9},
                 progress_callback=lambda e, t, loss, val, info=None: losses.append(loss)
                 if (info or {}).get("phase") == "train" else None)
    # the first batch is scored by the model the reference was computed with
    assert losses[0] == pytest.approx(math.log(2.0), abs=1e-6)
    assert all(m.p == 0.3 for m in layers)                    # put back afterwards


def test_simpo_needs_no_reference(chat_model, monkeypatch):
    engine = _engine(chat_model)
    scored = []
    real = pref._score
    monkeypatch.setattr(pref, "_score", lambda model, encoded, *a, **k: (
        scored.append(len(encoded)), real(model, encoded, *a, **k))[1])
    summary = engine.train(_pairs(), {"method": "simpo", "epochs": 4, "batch_size": 8,
                                      "lr": 3e-3, "seed": 1, "val_split": 0.25})
    report = summary["preference"]
    assert report["beta"] == pref.DEFAULT_SIMPO_BETA
    assert report["gamma"] == pytest.approx(pref.SIMPO_GAMMA_RATIO * pref.DEFAULT_SIMPO_BETA)
    assert report["final"]["accuracy"] == 1.0
    assert 30 not in scored            # the training pairs were never scored for a reference


def test_dpo_scores_the_reference_once(chat_model, monkeypatch):
    engine = _engine(chat_model)
    scored = []
    real = pref._score
    monkeypatch.setattr(pref, "_score", lambda model, encoded, *a, **k: (
        scored.append(len(encoded)), real(model, encoded, *a, **k))[1])
    engine.train(_pairs(), {"epochs": 3, "batch_size": 8, "seed": 1, "val_split": 0.25})
    assert scored.count(30) == 1       # the training pairs: once, before the first step


def test_sft_weight_and_beta(chat_model):
    engine = _engine(chat_model)
    summary = engine.train(_pairs(), {"epochs": 1, "seed": 1, "preference_beta": 0.5,
                                      "preference_sft_weight": 1.0, "val_split": 0.25,
                                      "dpo_positive": 5.0})
    report = summary["preference"]
    assert report["beta"] == 0.5 and report["sft_weight"] == 1.0 and report["positive"] == 5.0
    assert engine.params_used["preference"]["positive"] == 5.0
    # the chosen reply's cross-entropy is part of the loss
    assert report["initial"]["loss"] > math.log(2.0)
    # ... and of the training: the chosen replies become likelier than without it
    runs = [_engine(chat_model).train(_pairs(), {"epochs": 2, "seed": 1, "val_split": 0.25,
                                                 "lr": 1e-3, "preference_sft_weight": w})
            for w in (0.0, 5.0)]
    plain, with_sft = (r["preference"]["final"]["chosen_logp"] for r in runs)
    assert with_sft > plain


def test_defaults_of_a_preference_run(chat_model):
    engine = _engine(chat_model)
    summary = engine.train(_pairs(16), {"seed": 1})
    report = summary["preference"]
    assert summary["epochs_completed"] == pref.DEFAULT_PREFERENCE_EPOCHS == 3
    assert report["lr"] == pref.DEFAULT_PREFERENCE_LR["dpo"] == 3e-4
    assert report["positive"] == pref.DEFAULT_DPO_POSITIVE == 50.0
    summary = _engine(chat_model).train(_pairs(16), {"seed": 1, "method": "simpo"})
    assert summary["preference"]["lr"] == pref.DEFAULT_PREFERENCE_LR["simpo"] == 1e-4
    assert summary["preference"]["positive"] == 0.0
    plain = _engine(chat_model).train(_pairs(16), {"seed": 1, "dpo_positive": 0})
    assert plain["preference"]["positive"] == 0.0


def test_lora_trains_only_the_adapters(chat_model):
    engine = _engine(chat_model)
    before = _weights(engine)
    trainable = []

    def progress(epoch, total, loss, val, info=None):
        if (info or {}).get("phase") == "train":
            trainable.append(sorted(n for n, p in engine.model.named_parameters()
                                    if p.requires_grad and ".lora." not in n))

    summary = engine.train(_pairs(), {"lora_rank": 4, "epochs": 2, "seed": 1,
                                      "val_split": 0.25, "progress_every_seconds": 1e-9},
                           progress_callback=progress)
    assert trainable and all(names == [] for names in trainable)
    assert summary["preference"]["lora"] is True
    assert engine.has_lora()
    after = engine.model.state_dict()
    for key, value in before.items():
        assert torch.equal(after[key], value), key       # the base model is untouched
    assert any(".lora." in key for key in after)
    # a second run trains the adapters the model has
    lora = {k: v.clone() for k, v in after.items() if ".lora." in k}
    engine.train(_pairs(), {"epochs": 1, "seed": 2, "val_split": 0.25})
    changed = engine.model.state_dict()
    assert any(not torch.equal(changed[k], v) for k, v in lora.items())
    assert all(torch.equal(changed[k], v) for k, v in before.items())


def test_sft_method_trains_on_the_chosen_replies(chat_model):
    engine = _engine(chat_model)
    summary = engine.train(_pairs(), dict(TINY, method="sft", epochs=1,
                                          continue_training=True))
    assert summary["dialogs"]["train"] + summary["dialogs"]["val"] == 40


def test_wrong_data_for_a_method(chat_model, tmp_path):
    engine = _engine(chat_model)
    chats = load_training_data([_write(tmp_path / "c.jsonl", [{"messages": USER + [
        {"role": "assistant", "content": "a"}]}] * 3)])
    with pytest.raises(ValueError, match="trains on preference pairs"):
        engine.train(chats, {"method": "dpo"})
    with pytest.raises(ValueError, match="continues a chat model"):
        AuraLiteEngine().train(_pairs(), {"method": "dpo"})


def test_a_plain_text_model_is_not_a_chat_model(tmp_path):
    engine = AuraLiteEngine()
    engine.train("plain text for a tiny model. " * 40, dict(TINY, epochs=1))
    with pytest.raises(ValueError, match="fine-tune it on dialogs first"):
        engine.train(_pairs(8), {"epochs": 1})


def test_stop_before_the_first_step_leaves_the_model(chat_model, monkeypatch):
    engine = _engine(chat_model)
    before = _weights(engine)
    stop = threading.Event()
    stop.set()
    scored = []
    real = pref._score
    monkeypatch.setattr(pref, "_score", lambda model, encoded, *a, **k: (
        scored.append(len(encoded)), real(model, encoded, *a, **k))[1])
    summary = engine.train(_pairs(), {"lora_rank": 4, "epochs": 2, "batch_size": 4},
                           stop_event=stop)
    assert scored == [36]               # stopped in the reference pass, before the held-out pairs
    assert summary["stopped_during"] == "reference" and summary["epochs_completed"] == 0
    assert not engine.has_lora()
    assert all(torch.equal(engine.model.state_dict()[k], v) for k, v in before.items())


def test_stop_and_finish_during_training(chat_model):
    for event_name, reason in (("stop_event", "stopped"), ("finish_event", "finished")):
        engine = _engine(chat_model)
        event = threading.Event()

        def progress(epoch, total, loss, val, info=None, event=event):
            if (info or {}).get("phase") == "train":
                event.set()

        summary = engine.train(_pairs(), {"epochs": 3, "batch_size": 4, "seed": 1,
                                          "val_split": 0.25, "progress_every_seconds": 1e-9},
                               progress_callback=progress, **{event_name: event})
        assert summary["stop_reason"] == reason
        assert summary["optimizer_steps"] == 1 and summary["epochs_completed"] == 1


def test_progress_reports_the_held_out_accuracy(chat_model):
    engine = _engine(chat_model)
    events = []
    engine.train(_pairs(), {"epochs": 2, "seed": 1, "val_split": 0.25},
                 progress_callback=lambda e, t, loss, val, info=None: events.append(
                     (e, t, val, info or {})))
    ends = [e for e in events if e[3].get("phase") == "epoch_end"]
    assert [e[0] for e in ends] == [1, 2] and all(e[1] == 2 for e in ends)
    assert ends[-1][3]["preference"]["accuracy"] is not None and ends[-1][2] is not None
    assert any(e[3].get("stage") == "reference" for e in events)


def test_evaluate_preferences(chat_model):
    engine = _engine(chat_model)
    result = evaluate_preferences(engine, _pairs(20))
    assert result["pairs"] == 20 and 0.0 <= result["accuracy"] <= 1.0
    assert set(result) >= {"mean_accuracy", "margin", "mean_margin", "chosen_logp"}
    with pytest.raises(ValueError, match="no preference pairs"):
        evaluate_preferences(engine, [])


def test_dialog_training_does_not_touch_preference_code(chat_model, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("preference training ran")

    monkeypatch.setattr(pref.PreferenceTrainerMixin, "_train_preferences", boom)
    chats = load_training_data([_write(tmp_path / "c.jsonl", [{"messages": [
        {"role": "user", "content": _question(i)}, {"role": "assistant", "content": "yes."}]}
        for i in range(10)])])
    summary = AuraLiteEngine().train(chats, dict(TINY, epochs=1))
    assert summary["dialogs"]["train"] == 9


# ----------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------
def test_cli_trains_and_measures_pairs(chat_model, tmp_path, capsys):
    from auralite.cli import main
    pairs = _write(tmp_path / "prefs.jsonl", [
        {"prompt": _question(i), "chosen": "yes.", "rejected": "no."} for i in range(40)])
    out = tmp_path / "dpo.pt"
    rc = main(["train", "--data", pairs, "--resume", chat_model, "--method", "dpo",
               "--beta", "0.2", "--model", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and out.exists()
    assert "data: 40 preference pairs" in text and "DPO: held-out accuracy" in text
    assert "accuracy" in text.split("epoch 3/3 done")[1]
    engine = _engine(str(out))
    assert engine.params_used["preference"]["beta"] == 0.2
    rc = main(["chat-eval", "--model", str(out), "--data", pairs, "--json"])
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and result["pairs"] == 40 and 0.0 <= result["accuracy"] <= 1.0
    # the chosen replies gained on the rejected ones (default lr 1e-4, 3 epochs)
    assert result["margin"] > evaluate_preferences(_engine(chat_model),
                                                   load_training_data([pairs]))["margin"]
    rc = main(["chat-eval", "--model", str(out), "--data", pairs])
    assert rc == 0 and "chosen reply likelier" in capsys.readouterr().out


def test_cli_errors(chat_model, tmp_path, capsys):
    from auralite.cli import main
    pairs = _write(tmp_path / "prefs.jsonl", [{"prompt": "a?", "chosen": "b", "rejected": "c"}])
    assert main(["train", "--data", pairs, "--model", str(tmp_path / "x.pt")]) == 2
    assert "--resume" in capsys.readouterr().err
    chats = _write(tmp_path / "c.jsonl", [{"messages": USER + [
        {"role": "assistant", "content": "a"}]}])
    assert main(["train", "--data", chats, "--method", "dpo", "--model",
                 str(tmp_path / "x.pt")]) == 2
    assert "needs preference pairs" in capsys.readouterr().err
