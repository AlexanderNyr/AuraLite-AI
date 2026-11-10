"""2.27: the parts the classic interface's Library, wizard and Train tab share with AuraLite
Studio - and what changed in the Studio with them.

The Library's rows, texts and button states (`studio/libraryview.py`), the wizard's steps
(`studio/wizardplan.py`), the classic Train form as the Studio's settings and back, its
settings files of every version (`studio/trainform.py`), ``auralite train --params`` that
says what is wrong with a file (or takes the JSON itself), and in the Studio: the classic
interface's Train settings files load, and *Train more* of a new model's run keeps the
fine-tune's passes.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import pytest

from auralite import runs as runlib
from auralite.studio import libraryview as view
from auralite.studio import trainform, wizardplan
from auralite.studio.library import LibraryItem
from auralite.studio.recipes import CUSTOM, TrainSettings, pretrain_params
from auralite.studio.settingsfile import SETTINGS_FORMAT, read_settings, write_settings

GB = 1 << 30
TEXT = ("AuraLite trains small language models on CPU. It supports RMSNorm, RoPE, SwiGLU "
        "and grouped-query attention. ") * 30


def _hardware(memory_gb: float = 16, gflops: float = 200.0):
    from auralite.studio.hardware import Hardware
    return Hardware(system="Linux", cpu="Test CPU", cores=4, threads=8, simd="AVX2",
                    x86_64_v2=True, memory_total=int(memory_gb * GB),
                    memory_available=int(memory_gb * 0.75 * GB), train_threads=4,
                    gflops=gflops)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    folder = tmp_path / "home"
    monkeypatch.setenv("AURALITE_HOME", str(folder))
    return folder


def _runs(tmp_path) -> dict[str, runlib.Run]:
    """A finished fine-tune with its model, a failed train, a stopped train that can
    continue, and a preference run of the classic Train tab."""
    chats = tmp_path / "chats.jsonl"
    chats.write_text("{}\n", encoding="utf-8")
    text = tmp_path / "notes.txt"
    text.write_text("hello", encoding="utf-8")
    done = runlib.RunRecorder(kind="finetune", method="lora", name="Chat tune",
                              data={"paths": [str(chats)], "kind": "dialogs", "records": 1},
                              settings={"method": "lora", "epochs": 3})
    done.model_path.write_bytes(b"m" * 3000)
    done.record["result"] = runlib.MODEL_FILE
    done.finish({"stop_reason": "finished", "best_val_loss": 1.25, "seconds": 61.0})
    failed = runlib.RunRecorder(kind="train", data={"paths": [str(text)], "kind": "text"})
    failed.fail(RuntimeError("broken"))
    stopped = runlib.RunRecorder(kind="train", method="new", name="Scratch",
                                 data={"paths": [str(text)], "kind": "text"},
                                 settings=TrainSettings(epochs=100).to_dict())
    stopped.checkpoint_path.write_bytes(b"c" * 500)
    stopped.record["checkpoint"] = stopped.record["result"] = runlib.CHECKPOINT_FILE
    stopped.finish({"stop_reason": "stopped"})
    pairs = runlib.RunRecorder(kind="preference", method="dpo", source="gui",
                               data={"paths": [str(chats)], "kind": "preferences"},
                               settings=trainform.settings_from_form(trainform.FORM_DEFAULTS))
    pairs.finish({"stop_reason": "finished"})
    return {"done": done.run, "failed": failed.run, "stopped": stopped.run, "pairs": pairs.run}


# ----------------------------------------------------------------------
#  The Library's runs
# ----------------------------------------------------------------------
def test_run_rows_search_and_buttons(home, tmp_path):
    runs = _runs(tmp_path)
    done, failed, stopped = runs["done"], runs["failed"], runs["stopped"]
    name, status, kind, best, seconds, started = view.run_row(done)
    assert (name, status, kind, best) == ("Chat tune", "finished", "Fine-tune", "1.2500")
    assert seconds and started
    assert view.run_row(failed)[1:4] == ("failed", "Train", "")
    assert view.run_matches(done, ["chat", "fine-tune"]) and view.run_matches(done, [])
    assert view.run_matches(done, ["chats.jsonl"]) and not view.run_matches(done, ["nope"])
    can = view.run_actions(done, busy=False)
    assert can == {"continue": False, "open": True, "evaluate": True, "more": True,
                   "repeat": True, "files": True, "notes": True, "delete": True}
    assert view.run_actions(done, busy=True)["open"] is False
    assert view.run_actions(done, busy=True)["files"] is True
    assert view.run_actions(stopped, busy=False)["continue"] is True
    assert view.run_actions(stopped, busy=False)["evaluate"] is False      # it trained on text
    assert view.run_actions(failed, busy=False)["continue"] is False
    assert view.run_actions(None, busy=False) == dict.fromkeys(view.RUN_ACTIONS, False)
    running = runlib.RunRecorder(kind="train", data={"paths": []}).run   # this process trains it
    assert running.status == "running"
    assert view.run_actions(running, busy=False)["delete"] is False
    assert view.continue_tip(running) == "It cannot continue: it is training now"
    assert view.continue_tip(stopped) == view.CONTINUE_TIP
    assert view.continue_tip(done).startswith("It cannot continue: it has finished")
    assert view.continue_tip(failed) == "It cannot continue: it saved no checkpoint before it " \
                                        "ended"


def test_where_a_run_repeats_and_what_deleting_says(home, tmp_path):
    runs = _runs(tmp_path)
    assert view.tab_of(runs["stopped"]) == "train"
    assert view.tab_of(runs["done"]) == "finetune"
    assert view.tab_of(runs["pairs"]) == "train"               # the classic Train tab's pairs
    runs["pairs"].record["settings"] = {"method": "lora"}       # the Studio's (Fine-tune tab)
    assert view.tab_of(runs["pairs"]) == "finetune"
    question = view.delete_run_question(runs["done"], files_only=True)
    assert question.startswith("Delete the model files of Chat tune (") and "curve stay" in question
    assert view.delete_run_question(runs["done"], files_only=False).startswith(
        "Delete the run Chat tune and its folder (")
    outside = tmp_path / "elsewhere.pt"
    outside.write_bytes(b"x")
    runs["failed"].record["result"] = str(outside)
    assert view.delete_run_question(runs["failed"], files_only=False).endswith(
        f"Files outside its folder stay: {outside}")


def test_evaluation_and_saved_lines(home, tmp_path):
    assert view.evaluation_line({"dialogs": 12, "stop_rate": 0.5, "exact": 0.25}) == \
        "Evaluated on 12 held-out dialogs: 50% of the replies stop by themselves, 25% are exact."
    assert view.evaluation_line({}) == "Evaluated on 0 held-out dialogs."
    runs = _runs(tmp_path)
    assert view.saved_note(None) == ""
    assert view.saved_note(runs["done"]) == ". Saved in the library."
    assert view.saved_note(runs["stopped"]) == ". Saved in the library: it can continue from there."
    assert view.saved_note(runs["failed"]) == ""
    runs["done"].record["save_error"] = "disk full"
    assert view.saved_note(runs["done"]) == ""


# ----------------------------------------------------------------------
#  The Library's models and datasets
# ----------------------------------------------------------------------
def _item(kind: str, path: str, **kw: Any) -> LibraryItem:
    return LibraryItem(kind=kind, name=os.path.basename(path) or path, path=path, **kw)


def test_what_the_models_buttons_may_do(tmp_path):
    download = _item("download", str(tmp_path / "hf"), owned=True,
                     extra={"spec": "hf:Org/Model"})
    can = view.model_actions(download, busy=False)
    assert can == {"chat": True, "finetune": True, "export": False, "show": True,
                   "forget": False, "delete": True}
    gguf = _item("opened", str(tmp_path / "chat.gguf"))
    assert view.model_actions(gguf, False)["finetune"] is False
    assert view.model_actions(gguf, False)["chat"] is True and view.model_actions(gguf, False)[
        "forget"] is True
    adapter = _item("adapter", str(tmp_path / "adapter.pt"))
    assert view.model_actions(adapter, False)["chat"] is False
    assert view.model_actions(adapter, False)["export"] is False
    saved = _item("saved", str(tmp_path / "mine.pt"))
    assert view.model_actions(saved, False)["export"] is True
    assert view.model_actions(saved, True) == {"chat": False, "finetune": False, "export": False,
                                               "show": True, "forget": True, "delete": False}
    gone = _item("saved", str(tmp_path / "gone.pt"), exists=False)
    assert view.model_actions(gone, False)["show"] is False
    assert not any(view.model_actions(None, False).values())


def test_what_the_datasets_buttons_may_do(tmp_path):
    dialogs = _item("data", str(tmp_path / "chats.jsonl"),
                    extra={"report": {"kind": "dialogs"}, "remembered": True})
    assert view.data_actions(dialogs, False) == {"finetune": True, "train": True, "show": True,
                                                 "forget": True, "delete": True}
    pairs = _item("data", str(tmp_path / "pairs.jsonl"), extra={"report": {"kind": "preferences"}})
    assert view.data_actions(pairs, False)["train"] is False
    assert view.data_actions(pairs, False)["forget"] is False          # found through a run
    shards = _item("shards", str(tmp_path / "datasets" / "web"), owned=True)
    assert view.data_actions(shards, False)["finetune"] is False
    hub = _item("data", "hf:Org/data", exists=False)
    assert view.data_actions(hub, False)["finetune"] is True
    assert view.data_actions(hub, False)["show"] is False
    assert not any(view.data_actions(None, False).values())


def test_deleting_and_the_cells_of_an_item(tmp_path):
    download = _item("download", str(tmp_path / "x"), owned=True, bytes=3 << 20)
    assert view.delete_item_question(download) == \
        "Delete x (3.0 MB)? It is downloaded again when it is used next time."
    result = _item("result", str(tmp_path / "model.pt"), owned=True, bytes=2048)
    assert view.delete_item_question(result).endswith("its record and curve stay.")
    mine = _item("saved", str(tmp_path / "mine.pt"), bytes=10)
    assert view.delete_item_question(mine) == (f"Delete {tmp_path / 'mine.pt'} (10 B) from the "
                                               f"disk? This is your file: it cannot be brought "
                                               f"back.")
    item = _item("data", str(tmp_path / "chats.jsonl"), bytes=2048, modified=time.time(),
                 detail="Dialogs, 3 dialogs", extra={"runs": ["a", "b"]})
    cells = view.item_cells(item, ("kind_label", "size", "changed", "runs", "detail"))
    assert cells[0] == "Data file" and cells[1] == "2.0 KB" and cells[3] == "2"
    assert cells[4] == "Dialogs, 3 dialogs" and cells[2]
    item.exists = False
    assert view.item_cells(item, ("size",)) == [""]


def test_the_disk_line():
    assert view.disk_line({"total": 3 << 30, "runs": 1 << 30, "downloads": 2 << 30,
                           "datasets": 0, "free": 40 << 30}) == \
        "AuraLite's files: 3.0 GB (runs 1.0 GB, downloads 2.0 GB) · 40.0 GB free on the disk"
    assert view.disk_line({"total": 0, "runs": 0, "downloads": 0, "datasets": 0,
                           "free": None}) == "AuraLite's files: 0 B"


# ----------------------------------------------------------------------
#  The wizard's steps
# ----------------------------------------------------------------------
def _report(kind: str, ok: bool = True, paths: tuple[str, ...] = ("data.txt",), **kw: Any):
    from auralite.studio.datacheck import DataReport
    return DataReport(paths=list(paths), kind=kind, records=kw.get("records", 100),
                      tokens=kw.get("tokens", 50_000), bytes=kw.get("bytes", 200_000),
                      chars=kw.get("tokens", 50_000) * 4, longest_tokens=64,
                      problems=[] if ok else ["it cannot be read"])


def test_the_wizard_texts_and_the_data_it_takes():
    assert wizardplan.goal_title("pretrain") == "Train a new model from scratch"
    assert wizardplan.computer_text(None) == "Looking at this computer…"
    hw = _hardware()
    text = wizardplan.computer_text(hw)
    assert text.startswith(hw.summary_lines()[0]) and "Memory a training run may use:" in text
    subtitle, explanation = wizardplan.data_texts("pretrain")
    assert subtitle.startswith("Texts the new model learns from") and "hundreds of MB" in explanation
    assert wizardplan.data_texts("finetune")[1].startswith("Dialogs: a .jsonl file")
    assert wizardplan.data_complete("pretrain", _report("text"))
    assert wizardplan.data_complete("pretrain", _report("dialogs"))
    assert not wizardplan.data_complete("pretrain", _report("preferences"))
    assert wizardplan.data_complete("finetune", _report("preferences"))
    assert not wizardplan.data_complete("finetune", _report("shards"))
    assert not wizardplan.data_complete("finetune", _report("text", ok=False))
    assert not wizardplan.data_complete("continue", None)


def test_the_wizard_table_choice_and_plan(tmp_path):
    hw = _hardware()
    table = wizardplan.model_table("pretrain", hw, _report("text", tokens=2_000_000))
    rec = table.recommendation
    assert rec is not None and table.headers[0] == "Size"
    assert len(table.rows) == len(rec.presets) and 0 <= table.chosen < len(table.rows)
    cells, recommended, usable = table.rows[table.chosen]
    assert recommended and usable and cells[0].endswith(" ★")
    assert sum(1 for _cells, star, _u in table.rows if star) == 1
    choice = wizardplan.choose("pretrain", rec, table.chosen)
    assert choice["base"] == "" and choice["settings"]["preset"] == rec.presets[table.chosen].preset
    assert wizardplan.choose("pretrain", rec, 99)["settings"]["preset"] == rec.presets[-1].preset
    report = _report("text")
    lines = wizardplan.plan_lines("pretrain", choice, report)
    assert lines[0] == "Goal: Train a new model from scratch"
    assert lines[1].startswith(f"Model: a {choice['settings']['preset']} model from scratch")
    assert lines[3:] == report.summary_lines()[:2]
    plan = wizardplan.make_plan("pretrain", choice, _report("text", paths=("c.txt",)), True)
    assert plan == {"goal": "pretrain", "base": "", "settings": choice["settings"],
                    "data": "c.txt", "start": True}
    plan = wizardplan.make_plan("pretrain", choice, None, False)
    assert plan["start"] is False and plan["data"] == ""

    table = wizardplan.model_table("finetune", hw, _report("dialogs", records=500))
    rec = table.recommendation
    assert rec is not None and table.headers == ["Model", "Size", "Method", "Memory", "Fits",
                                                 "Time", "License"]
    choice = wizardplan.choose("finetune", rec, table.chosen)
    fit = rec.options[table.chosen]
    assert choice["base"] == fit.model.spec and choice["settings"]["method"] == fit.method
    assert wizardplan.model_complete("finetune", table.chosen, rec)
    other = wizardplan.choose("finetune", rec, -1, other="SmolLM2-135M-Instruct")
    assert other["base"].startswith("hf:") and other["settings"]["base"] == other["base"]
    folder = tmp_path / "my-model"
    folder.mkdir()
    assert wizardplan.choose("finetune", rec, -1, other=str(folder))["base"] == str(folder)
    assert wizardplan.choose("finetune", rec, -1, other="Org/Thing")["base"] == "hf:Org/Thing"
    assert wizardplan.model_complete("finetune", -1, None, other="Org/Thing")
    assert "Method: " in wizardplan.plan_lines("finetune", choice, None)[2]

    model = tmp_path / "mine.pt"
    model.write_bytes(b"x")
    assert wizardplan.model_table("continue", hw, None).subtitle == "Choose the model to continue."
    assert wizardplan.choose("continue", None, -1, file=f" {model} ") == {
        "base": str(model), "settings": wizardplan.CONTINUE_SETTINGS}
    assert wizardplan.model_complete("continue", -1, None, file=str(model))
    assert not wizardplan.model_complete("continue", -1, None, file=str(tmp_path / "no.pt"))
    waiting = wizardplan.model_table("finetune", None, None)
    assert waiting.recommendation is None and waiting.note.startswith("This computer is still")
    assert not wizardplan.model_complete("finetune", -1, None)


# ----------------------------------------------------------------------
#  The classic Train form as settings
# ----------------------------------------------------------------------
def test_the_default_form_is_the_classic_tabs_and_round_trips():
    settings = trainform.settings_from_form(trainform.FORM_DEFAULTS)
    s = TrainSettings.from_dict(settings)
    assert (s.preset, s.d_model, s.n_layers, s.n_heads, s.n_kv_heads, s.d_ff) == (
        CUSTOM, 128, 4, 4, 0, 256)
    assert (s.seq_length, s.batch_size, s.epochs, s.lr, s.optimizer, s.dropout) == (
        64, 32, 100, 0.001, "muon", None)
    assert s.early_stopping and s.patience == 0 and s.memory_plan == "auto"
    assert settings["lora_rank"] == 0 and settings["quantize_base"] == ""
    assert settings["preference"] == {"method": "dpo", "beta": None, "lr": None, "epochs": None}
    form = trainform.form_from_settings(settings)
    assert trainform.settings_from_form(form) == settings
    params = pretrain_params(s, "text")
    assert (params["d_model"], params["n_layers"], params["seq_length"]) == (128, 4, 64)


def test_every_field_of_the_form_survives_settings(tmp_path):
    tok = tmp_path / "tok.json"
    tok.write_text("{}", encoding="utf-8")
    form = dict(trainform.FORM_DEFAULTS, lr="0.002", epochs="7", d_model="96", d_ff="192",
                n_heads="6", n_layers="3", seq_length="48", batch_size="16", dropout="0.1",
                grad_clip="0.5", tokenizer="file", tokenizer_path=str(tok),
                bpe_vocab_size="1024", val_split="0.2", early_stop="5", time_limit="1.5",
                n_kv_heads="2", accumulation="4", alibi=True, lora_rank="8",
                lora_targets="attn", lora_lr="0.004", qlora=True, adam8=True,
                checkpointing=True, cluster_workers="2", cluster_port="29600",
                rope_type="yarn", rope_factor="2.0", qk_norm=False, optimizer="adamw",
                muon_lr="0.03", lr_schedule="cosine", compile=True, tie=True,
                autosave_every="2", chat_template="llama3", pack_dialogs=True,
                prompt_weight="0.2", pref_method="simpo", pref_beta="2.5", pref_lr="1e-4",
                pref_epochs="2")
    settings = trainform.settings_from_form(form)
    s = TrainSettings.from_dict(settings)
    assert (s.tokenizer, s.tokenizer_path, s.optimizer, s.patience, s.early_stopping) == (
        "file", str(tok), "adamw8bit", 5, True)
    assert (s.rope_scaling, s.rope_factor, s.use_qk_norm, s.dropout, s.prompt_weight) == (
        "yarn", 2.0, False, 0.1, 0.2)
    assert (s.time_limit_min, s.accumulation_steps, s.cluster_workers, s.cluster_port) == (
        1.5, 4, 2, 29600)
    assert (settings["lora_rank"], settings["lora_targets"], settings["lora_lr"],
            settings["quantize_base"]) == (8, "attn", 0.004, "nf4")
    assert settings["preference"] == {"method": "simpo", "beta": 2.5, "lr": 1e-4, "epochs": 2}
    back = trainform.form_from_settings(settings)
    assert back["optimizer"] == "adamw" and back["adam8"] is True and back["qlora"] is True
    assert trainform.settings_from_form(back) == settings
    off = trainform.settings_from_form(dict(form, early_stop="0", rope_type="none",
                                            tokenizer="bpe"))
    assert off["early_stopping"] is False and off["rope_scaling"] == ""
    assert off["tokenizer_path"] == ""                       # only with the tokenizer «file»
    with pytest.raises(ValueError):
        trainform.settings_from_form(dict(form, epochs="many"))


def test_a_studio_size_fills_the_classic_shape():
    from auralite.engine.recipe import DEFAULT_RECIPE
    from auralite.engine.sizing import MODEL_PRESETS
    form = trainform.form_from_settings(TrainSettings(preset="45m", seq_length=128,
                                                      time_limit_min=30).to_dict())
    shape = MODEL_PRESETS["45m"]
    assert (form["d_model"], form["n_layers"], form["n_heads"], form["n_kv_heads"],
            form["d_ff"]) == tuple(str(shape[k]) for k in ("d_model", "n_layers", "n_heads",
                                                            "n_kv_heads", "d_ff"))
    assert form["seq_length"] == "128" and form["time_limit"] == "30"
    assert form["lr"] == f"{DEFAULT_RECIPE['lr']:g}" and form["optimizer"] == "muon"
    assert form["muon_lr"] == f"{DEFAULT_RECIPE['muon_lr']:g}" and form["lr_schedule"] == "wsd"
    assert form["chat_template"] == "chatml" and form["dropout"] == "auto"
    assert form["lora_rank"] == "0" and form["qlora"] is False


def test_a_cli_runs_parameters_fill_a_form():
    form = trainform.form_from_params({
        "lr": 0.002, "epochs": 3, "d_model": 64, "n_layers": 2, "n_heads": 4, "d_ff": 128,
        "seq_length": 32, "batch_size": 8, "dropout": 0.0, "tokenizer": "char",
        "early_stopping_patience": 0, "time_budget_s": 90, "n_kv_heads": None,
        "optimizer": "adamw8bit", "quantize_base": "nf4", "lora_rank": 16,
        "lora_targets": "ffn", "lora_lr": 0.01, "rope_scaling": {"type": "ntk", "factor": 4.0},
        "use_qk_norm": False, "continue_training": True, "chat_template": "llama3",
        "prompt_loss_weight": 0.0, "dialog_packing": True, "method": "simpo",
        "preference_beta": 2.0, "cluster_workers": 1, "autosave_every_minutes": 10.0})
    assert (form["lr"], form["epochs"], form["d_model"], form["tokenizer"]) == (
        "0.002", "3", "64", "char")
    assert (form["early_stop"], form["time_limit"], form["n_kv_heads"]) == ("0", "1.5", "0")
    assert (form["optimizer"], form["adam8"], form["qlora"], form["lora_rank"]) == (
        "adamw", True, True, "16")
    assert (form["rope_type"], form["rope_factor"], form["qk_norm"], form["continue"]) == (
        "ntk", "4", False, True)
    assert (form["chat_template"], form["prompt_weight"], form["pack_dialogs"]) == (
        "llama3", "0", True)
    assert (form["pref_method"], form["pref_beta"], form["cluster_workers"]) == ("simpo", "2", "1")
    assert trainform.form_from_params({}) == trainform.FORM_DEFAULTS


def test_the_classic_files_of_2_26_still_read_and_write():
    config = trainform.config_from_form(trainform.FORM_DEFAULTS)
    assert list(config) == [            # the keys and the order of 2.26's Save Config
        "lr", "epochs", "d_model", "d_ff", "n_heads", "n_layers", "seq_length", "batch_size",
        "dropout", "grad_clip", "tokenizer", "tokenizer_path", "bpe_vocab_size", "val_split",
        "early_stopping_patience", "time_budget_s", "use_compile", "continue_training",
        "autosave_every", "n_kv_heads", "accumulation_steps", "use_alibi", "lora_rank",
        "lora_targets", "prompt_loss_weight", "dialog_packing", "use_gradient_checkpointing",
        "use_ddp", "rope_scaling", "use_qk_norm", "optimizer", "muon_lr", "lr_schedule",
        "tie_word_embeddings"]
    assert config["dropout"] == "auto" and config["n_kv_heads"] is None
    assert config["early_stopping_patience"] == "auto" and config["rope_scaling"] is None
    form = trainform.form_from_config(dict(config, lr=0.005, rope_scaling={"type": "linear",
                                                                            "factor": 2.0},
                                           optimizer="lion", lr_schedule="odd",
                                           time_budget_s=120.0, lora_lr=0.003))
    assert (form["lr"], form["rope_type"], form["rope_factor"], form["optimizer"],
            form["lr_schedule"], form["time_limit"], form["lora_lr"]) == (
        "0.005", "linear", "2.0", "muon", "wsd", "2", "0.003")
    assert trainform.settings_from_form(trainform.form_from_config(config)) == \
        trainform.settings_from_form(trainform.FORM_DEFAULTS)


def test_one_settings_file_for_both_interfaces_and_older_versions(tmp_path):
    data = tmp_path / "corpus.txt"
    data.write_text(TEXT, encoding="utf-8")
    form = dict(trainform.FORM_DEFAULTS, epochs="9", adam8=True, cluster_workers="3",
                chat_template="mistral")
    path = tmp_path / "train.json"
    path.write_text(json.dumps(trainform.train_settings_doc(form, [str(data)])), encoding="utf-8")
    studio = read_settings(str(path), "train")               # the Studio's reader takes it
    assert studio["format"] == SETTINGS_FORMAT and studio["data"] == [str(data)]
    assert TrainSettings.from_dict(studio["settings"]).epochs == 9
    assert studio["epochs"] == 9 and studio["optimizer"] == "muon"   # 2.26 reads these
    doc = trainform.read_train_settings(str(path))
    assert doc["data"] == [str(data)] and doc["form"]["epochs"] == "9"
    assert doc["form"]["adam8"] is True and doc["form"]["cluster_workers"] == "3"
    assert doc["form"]["chat_template"] == "mistral"
    edited = json.loads(path.read_text(encoding="utf-8"))
    edited["epochs"] = 11                                     # by hand: the classic key wins
    path.write_text(json.dumps(edited), encoding="utf-8")
    doc = trainform.read_train_settings(str(path))
    assert doc["form"]["epochs"] == "11" and doc["settings"]["epochs"] == 11
    assert doc["form"]["adam8"] is True                      # the rest from "settings"

    old = tmp_path / "old.json"                               # 2.26's Save Config
    old.write_text(json.dumps(trainform.config_from_form(dict(form, epochs="4"))),
                   encoding="utf-8")
    doc = trainform.read_train_settings(str(old))
    assert doc["form"]["epochs"] == "4" and doc["data"] == []
    assert TrainSettings.from_dict(doc["settings"]).preset == CUSTOM

    studio_file = write_settings(str(tmp_path / "studio.json"), "train",
                                 TrainSettings(preset="25m", epochs=5).to_dict(), [])
    doc = trainform.read_train_settings(studio_file)
    assert doc["settings"]["preset"] == "25m" and doc["settings"]["epochs"] == 5
    assert doc["form"]["d_model"] == "256"

    finetune = write_settings(str(tmp_path / "ft.json"), "finetune", {"method": "lora"}, [])
    with pytest.raises(ValueError, match="not of the train tab"):
        trainform.read_train_settings(finetune)
    for name, text in (("broken.json", "{oops"), ("list.json", "[1]"), ("other.json", "{}")):
        bad = tmp_path / name
        bad.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            trainform.read_train_settings(str(bad))
    with pytest.raises(OSError):
        trainform.read_train_settings(str(tmp_path / "missing.json"))


# ----------------------------------------------------------------------
#  auralite train --params, and smaller fixes
# ----------------------------------------------------------------------
def test_params_say_what_is_wrong_or_take_the_json_itself(tmp_path, capsys, monkeypatch):
    from auralite.cli import _read_params, main
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AURALITE_NO_RUNS", "1")
    corpus = tmp_path / "c.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    (tmp_path / "list.json").write_text("[1, 2]", encoding="utf-8")
    (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")
    (tmp_path / "good.json").write_text('{"epochs": 2}', encoding="utf-8")
    assert _read_params(str(tmp_path / "good.json")) == {"epochs": 2}
    assert _read_params(' {"epochs": 3} ') == {"epochs": 3}
    for value, said in ((str(tmp_path / "missing.json"), "--params: no such file: "),
                        (str(tmp_path / "bad.json"), "is not JSON: "),
                        (str(tmp_path / "list.json"), "holds a list, not settings"),
                        ("{nope", "the JSON given is not JSON")):
        code = main(["train", "--text", str(corpus), "--model", str(tmp_path / "m.pt"),
                     "--params", value])
        err = capsys.readouterr().err
        assert code == 2 and said in err and "Traceback" not in err
    params = json.dumps({"epochs": 1, "d_model": 32, "n_layers": 1, "n_heads": 2, "d_ff": 64,
                         "seq_length": 16, "batch_size": 4, "tokenizer": "char",
                         "val_split": 0, "optimizer": "adamw"})
    assert main(["train", "--text", str(corpus), "--model", str(tmp_path / "m.pt"),
                 "--params", params]) == 0
    assert (tmp_path / "m.pt").is_file()
    assert main(["plan", "--params", str(tmp_path / "missing.json")]) == 2
    assert "no such file" in capsys.readouterr().err
    assert main(["plan", "--params", '{"d_model": 64, "n_layers": 2}', "--memory", "4GB"]) == 0
    images = tmp_path / "images"
    images.mkdir()
    assert main(["image-train", "--images", str(images), "--params",
                 str(tmp_path / "nope.json")]) == 2
    assert "--params: no such file" in capsys.readouterr().err


def test_a_run_of_the_model_in_memory_says_so_and_errors_point_to_tools(home):
    from auralite.studio.advisor import size_text
    from auralite.studio.errors import explain_error
    assert [size_text(n) for n in (36_000, 99_000, 100_000, 800_000)] == [
        "36K", "99K", "0.1M", "0.8M"]                            # tiny models: not "0.0M"
    run = runlib.RunRecorder(kind="train", method="more", data={"paths": []}).run
    assert dict(runlib.describe_run(run))["Method"] == "the model in memory, trained further"
    assert explain_error(RuntimeError("odd"), action="training").hints == [
        "The log has the details (Tools → Open the log folder)"]


# ----------------------------------------------------------------------
#  The Studio: the classic Train files load, Train more keeps the fine-tune's passes
# ----------------------------------------------------------------------
def _qt():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from tests._integration import require
    return require("PySide6.QtWidgets"), require("PySide6.QtCore")


@pytest.fixture()
def studio(tmp_path, monkeypatch):
    QtWidgets, QtCore = _qt()
    from auralite.engine import AuraLiteEngine
    from auralite.studio.qt import common
    from auralite.studio.qt.window import MainWindow
    from auralite.studio.settings import StudioSettings
    seen: dict[str, Any] = {"error": [], "open": ""}
    common.DIALOGS.update(error=lambda friendly: seen["error"].append(friendly),
                          inform=lambda title, text: None, confirm=lambda title, text: True,
                          open=lambda title: seen["open"])
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    window = MainWindow(engine=AuraLiteEngine(), settings=StudioSettings(first_run_done=True),
                        detect=False, capture_streams=False, save_on_close=False,
                        auto_speed_test=False)
    window.hardware = _hardware()
    window.hardware_changed.emit(window.hardware)
    window.seen = seen                                       # type: ignore[attr-defined]
    window.qapp = app                                        # type: ignore[attr-defined]
    yield window
    common.DIALOGS.clear()
    window.model_dirty = False
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()


def test_the_studio_loads_the_classic_train_tabs_files(studio, tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    old = tmp_path / "old.json"
    old.write_text(json.dumps(trainform.config_from_form(dict(
        trainform.FORM_DEFAULTS, d_model="96", n_heads="6", n_layers="3", d_ff="192",
        epochs="12", lr="0.003"))), encoding="utf-8")
    studio.seen["open"] = str(old)
    tab = studio.train_tab
    tab.load_settings()
    assert studio.seen["error"] == [], [e.message for e in studio.seen["error"]]
    s = tab.settings()
    assert (s.preset, s.d_model, s.n_heads, s.n_layers, s.d_ff, s.epochs, s.lr) == (
        CUSTOM, 96, 6, 3, 192, 12, 0.003)
    new = tmp_path / "new.json"
    new.write_text(json.dumps(trainform.train_settings_doc(
        dict(trainform.FORM_DEFAULTS, epochs="5", time_limit="20"), [str(corpus)])),
        encoding="utf-8")
    studio.seen["open"] = str(new)
    tab.load_settings()
    s = tab.settings()
    assert (s.epochs, s.time_limit_min) == (5, 20.0) and tab.data.paths() == [str(corpus)]
    studio.seen["open"] = str(tmp_path / "missing.json")
    tab.load_settings()
    assert studio.seen["error"] and studio.seen["error"][-1].kind == "not_found"


def test_train_more_of_a_new_models_run_keeps_the_finetunes_passes(studio, tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    run = runlib.RunRecorder(kind="train", method="new", data={"paths": [str(corpus)],
                                                               "kind": "text"},
                             settings=TrainSettings(epochs=100, seq_length=256).to_dict())
    run.model_path.write_bytes(b"m")
    run.record["result"] = runlib.MODEL_FILE
    run.finish({"stop_reason": "finished"})
    tab = studio.finetune_tab
    before = tab.settings()
    tab.apply_run(run.run, base=str(run.model_path))
    after = tab.settings()
    assert (after.epochs, after.seq_length) == (before.epochs, before.seq_length)
    assert tab.source() == "file" and tab.base_spec() == str(run.model_path)
    tuned = runlib.RunRecorder(kind="finetune", method="lora", data={"paths": [str(corpus)]},
                               settings={"method": "lora", "epochs": 3, "lora_rank": 8})
    tuned.finish({"stop_reason": "finished"})
    tab.apply_run(tuned.run)
    assert tab.settings().epochs == 3 and tab.settings().lora_rank == 8
