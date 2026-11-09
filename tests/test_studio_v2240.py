"""2.24: the classic interface's Train, Generate and Export features in the Studio.

Part 1 of moving everything the tkinter interface does into AuraLite Studio: a model of
one's own shape, a tokenizer file, the tokenizer trained alone, a prepared dataset, several
computers, the model in memory trained further, the dialogs' options and settings files
(Train); the Generate tab; .pt2, ONNX and TorchScript (Export). And the advisor calibrated
on the i5-2400's benchmark, the language groundwork.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from typing import Any

import pytest

from auralite.studio.recipes import (CUSTOM, FineTuneSettings, SettingsError, TrainSettings,
                                     custom_params, finetune_params, pretrain_params)

GB = 1 << 30
TEXT = ("AuraLite trains small language models on CPU. It supports RMSNorm, RoPE, SwiGLU "
        "and grouped-query attention. ") * 30
TINY = {"d_model": 32, "n_heads": 2, "n_layers": 1, "d_ff": 64, "seq_length": 16,
        "batch_size": 8, "epochs": 1, "tokenizer": "char", "val_split": 0.1,
        "progress_every_seconds": 100.0}


# ----------------------------------------------------------------------
#  Settings -> params (no Qt)
# ----------------------------------------------------------------------
def test_the_default_settings_train_as_before():
    """2.23's params for the default settings, key for key: the new fields stay out of
    them until they are set."""
    assert pretrain_params(TrainSettings(), "text") == {
        "d_model": 256, "n_layers": 8, "n_heads": 4, "n_kv_heads": 2, "d_ff": 768,
        "seq_length": 256, "batch_size": 32, "epochs": 100, "tokenizer": "bpe",
        "bpe_vocab_size": 512, "val_split": 0.1, "memory_plan": "auto",
        "continue_training": False, "use_gradient_checkpointing": False,
        "early_stopping_patience": "auto", "progress_every_seconds": 0.5}
    dialogs = pretrain_params(TrainSettings(), "dialogs")
    assert not {"chat_template", "prompt_loss_weight", "dialog_packing"} & set(dialogs)


def test_a_shape_of_ones_own():
    s = TrainSettings(preset=CUSTOM, d_model=96, n_layers=3, n_heads=6, n_kv_heads=2,
                      d_ff=256)
    params = pretrain_params(s, "text")
    assert {k: params[k] for k in ("d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff")} \
        == {"d_model": 96, "n_layers": 3, "n_heads": 6, "n_kv_heads": 2, "d_ff": 256}
    assert custom_params(TrainSettings(preset=CUSTOM, n_kv_heads=0))["n_kv_heads"] is None
    with pytest.raises(SettingsError, match="divides into the heads"):
        custom_params(TrainSettings(preset=CUSTOM, d_model=100, n_heads=3))
    with pytest.raises(SettingsError, match="key/value heads"):
        custom_params(TrainSettings(preset=CUSTOM, n_heads=4, n_kv_heads=3))
    with pytest.raises(SettingsError, match="above 0"):
        custom_params(TrainSettings(preset=CUSTOM, n_layers=0))
    with pytest.raises(SettingsError, match="unknown model size"):
        pretrain_params(TrainSettings(preset="3m"), "text")


def test_the_classic_settings_reach_the_trainer():
    s = TrainSettings(use_alibi=True, rope_scaling="linear", rope_factor=2.0,
                      use_qk_norm=False, tie_word_embeddings=True, dropout=0.1, grad_clip=0.5,
                      accumulation_steps=4, lr_schedule="cosine", muon_lr=0.01, patience=3,
                      use_compile=True, cluster_workers=2, cluster_port=29600,
                      continue_model=True)
    p = pretrain_params(s, "text")
    assert p["use_alibi"] is True and p["rope_scaling"] == {"type": "linear", "factor": 2.0}
    assert p["use_qk_norm"] is False and p["tie_word_embeddings"] is True
    assert p["dropout"] == 0.1 and p["grad_clip"] == 0.5 and p["accumulation_steps"] == 4
    assert p["lr_schedule"] == "cosine" and p["muon_lr"] == 0.01
    assert p["early_stopping_patience"] == 3 and p["use_compile"] is True
    assert (p["cluster_workers"], p["cluster_port"]) == (2, 29600)
    assert p["continue_training"] is True
    assert pretrain_params(TrainSettings(early_stopping=False, patience=3), "text")[
        "early_stopping_patience"] == 0
    with pytest.raises(SettingsError, match="this computer only"):
        pretrain_params(TrainSettings(cluster_workers=1), "dialogs")
    with pytest.raises(SettingsError, match="RoPE scaling"):
        pretrain_params(TrainSettings(rope_scaling="dynamic"), "text")
    with pytest.raises(SettingsError, match="schedule"):
        pretrain_params(TrainSettings(lr_schedule="linear"), "text")
    d = pretrain_params(TrainSettings(chat_template="gemma", prompt_weight=0.0,
                                      pack_dialogs=True), "dialogs")
    assert (d["chat_template"], d["prompt_loss_weight"], d["dialog_packing"]) == \
        ("gemma", 0.0, True)


def test_a_tokenizer_file(tmp_path):
    with pytest.raises(SettingsError, match="Own file"):
        pretrain_params(TrainSettings(tokenizer="file"), "text")
    with pytest.raises(SettingsError, match="missing"):
        pretrain_params(TrainSettings(tokenizer="file",
                                      tokenizer_path=str(tmp_path / "none.json")), "text")
    path = tmp_path / "tok.json"
    path.write_text("{}", encoding="utf-8")
    p = pretrain_params(TrainSettings(tokenizer="file", tokenizer_path=str(path)), "text")
    assert (p["tokenizer"], p["tokenizer_path"]) == ("bpe", str(path))


def test_settings_keep_every_field():
    s = TrainSettings(preset=CUSTOM, d_model=64, use_alibi=True, patience=2,
                      tokenizer_path="t.json", prompt_weight=0.2)
    assert TrainSettings.from_dict(json.loads(json.dumps(s.to_dict()))) == s
    assert TrainSettings.from_dict({"preset": "45m", "unknown": 1}).preset == "45m"


def test_preference_pairs_take_their_beta_and_passes():
    p = finetune_params(FineTuneSettings(preference_beta=0.2, preference_epochs=5),
                        "preferences")
    assert (p["method"], p["preference_beta"], p["epochs"]) == ("dpo", 0.2, 5)
    assert "epochs" not in finetune_params(FineTuneSettings(), "preferences")   # their own 3


def test_lora_trains_the_layers_chosen():
    p = finetune_params(FineTuneSettings(method="lora", lora_targets="attn"), "dialogs")
    assert p["lora_targets"] == "attn"
    assert finetune_params(FineTuneSettings(), "dialogs")["lora_targets"] == "all"
    with pytest.raises(SettingsError, match="LoRA trains"):
        finetune_params(FineTuneSettings(lora_targets="mlp"), "dialogs")


def test_a_dataset_of_characters_is_prepared(tmp_path):
    """2.24: prepare_dataset(tokenizer_kind="char") called CharTokenizer.fit, which it
    lacks - the classic interface's «Prepare Dataset» with «Char-level» always failed."""
    pytest.importorskip("torch")
    from auralite.engine.shards import TokenShards, prepare_dataset
    source = tmp_path / "corpus.txt"
    source.write_text(TEXT, encoding="utf-8")
    shards = prepare_dataset([str(source)], tmp_path / "shards", tokenizer_kind="char",
                             workers=1)
    again = TokenShards.open(str(tmp_path / "shards"))
    # one token per character, and the document's separators
    assert shards.complete and len(TEXT) <= again.train_tokens + again.val_tokens < len(TEXT) + 8


# ----------------------------------------------------------------------
#  The advisor
# ----------------------------------------------------------------------
def test_the_advisor_is_calibrated_on_the_i5_2400():
    """2.24: 50 % of the measured product speed. The i5-2400 (76 GFLOP/s) measured
    Qwen3.5-0.8B LoRA at 16.1 tokens/s (the estimate stays below) and new models of 6.6M
    and 19M parameters at 923 and 353 tokens/s (the estimates within 10 %)."""
    from auralite.studio import advisor
    from auralite.studio.catalog import find_model
    from auralite.studio.hardware import Hardware
    assert advisor.EFFICIENCY == 0.50
    i5 = Hardware(system="Windows 10", cores=4, threads=4, simd="AVX", memory_total=16 * GB,
                  train_threads=4, gflops=76.0)
    qwen = find_model("hf:Qwen/Qwen3.5-0.8B")
    assert qwen is not None
    speed = advisor.tokens_per_second(qwen.compute_params, i5)
    assert speed is not None and 11.4 < speed < 16.1
    from auralite.engine.sizing import preset_params
    for preset, measured in (("25m", 923.0), ("45m", 353.0)):
        shape = preset_params(preset)
        shape.pop("seq_length")
        fit = advisor.shape_fit(shape, i5, vocab=512)
        assert fit.tokens_per_second is not None
        assert abs(fit.tokens_per_second / measured - 1) < 0.10, (preset, fit)


def test_a_shape_is_measured_as_the_presets_are():
    from auralite.engine.sizing import preset_params
    from auralite.studio import advisor
    from auralite.studio.hardware import Hardware
    hw = Hardware(system="Linux", cores=4, threads=8, simd="AVX2", memory_total=16 * GB,
                  train_threads=4, gflops=200.0)
    preset = preset_params("45m")
    seq = preset.pop("seq_length")
    by_name = advisor._preset_fit("45m", hw, advisor.memory_budget(hw), 512, 1_000_000)
    by_shape = advisor.shape_fit(preset, hw, vocab=512, tokens=1_000_000, seq_length=seq,
                                 name="45m")
    assert by_shape == by_name
    tied = advisor.shape_fit(dict(preset, tie_word_embeddings=True), hw, vocab=32000)
    assert tied.params < advisor.shape_fit(preset, hw, vocab=32000).params


# ----------------------------------------------------------------------
#  Languages (the groundwork)
# ----------------------------------------------------------------------
def test_tr_speaks_english_and_falls_back(monkeypatch):
    from auralite.studio import i18n
    assert i18n.set_language("xx") == "en" and i18n.language() == "en"
    assert i18n.tr("Open a model…") == "Open a model…"
    assert i18n.tr("{n} passes", n=3) == "3 passes"
    assert i18n.catalogue("zz") == {}
    monkeypatch.setitem(i18n.LANGUAGES, "eo", "Esperanto")
    monkeypatch.setitem(i18n._catalogues, "eo", {"{n} passes": "{n} pasoj"})
    try:
        assert i18n.set_language("eo") == "eo"
        assert i18n.tr("{n} passes", n=2) == "2 pasoj"
        assert i18n.tr("Not translated") == "Not translated"
    finally:
        i18n.set_language("en")
    monkeypatch.setenv("AURALITE_LANG", "eo")
    assert i18n.set_language() == "eo"
    i18n.set_language("en")


# ----------------------------------------------------------------------
#  The window
# ----------------------------------------------------------------------
def _qt():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from tests._integration import require
    return require("PySide6.QtWidgets"), require("PySide6.QtCore")


@pytest.fixture()
def dialogs():
    _qt()
    from auralite.studio.qt import common
    seen: dict[str, Any] = {"error": [], "inform": [], "confirm": []}
    answers = {"confirm": True, "save": "", "open": "", "dir": ""}
    common.DIALOGS.update(
        error=lambda friendly: seen["error"].append(friendly),
        inform=lambda title, text: seen["inform"].append((title, text)),
        confirm=lambda title, text: seen["confirm"].append((title, text)) or answers["confirm"],
        save=lambda title: answers["save"], open=lambda title: answers["open"],
        dir=lambda title: answers["dir"])
    seen["answers"] = answers
    yield seen
    common.DIALOGS.clear()


@pytest.fixture()
def studio(tmp_path, monkeypatch, dialogs):
    QtWidgets, QtCore = _qt()
    from auralite.engine import AuraLiteEngine
    from auralite.studio.hardware import Hardware
    from auralite.studio.qt.window import MainWindow
    from auralite.studio.settings import StudioSettings
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    window = MainWindow(engine=AuraLiteEngine(), settings=StudioSettings(first_run_done=True),
                        detect=False, capture_streams=False, save_on_close=False,
                        auto_speed_test=False)
    window.hardware = Hardware(system="Linux", cpu="Test CPU", cores=4, threads=8,
                               simd="AVX2", x86_64_v2=True, memory_total=16 * GB,
                               memory_available=12 * GB, train_threads=4, gflops=200.0)
    window.hardware_changed.emit(window.hardware)
    window.resize(1000, 760)
    window.show()
    app.processEvents()
    window.qapp = app                                        # type: ignore[attr-defined]
    yield window
    window.model_dirty = False
    for job in window.running_jobs():
        job.stop_event.set()
        if job.thread is not None:
            job.thread.join(timeout=30)
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()


def wait(app, condition, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    while not condition():
        if time.time() > deadline:
            raise AssertionError("timed out waiting for the window")
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def _text_file(tmp_path, name: str = "corpus.txt") -> str:
    path = tmp_path / name
    path.write_text(TEXT, encoding="utf-8")
    return str(path)


def _tiny_model(studio) -> None:
    """A model in memory, as if trained or opened (the Generate and Export tabs')."""
    studio.engine.train(TEXT, dict(TINY))
    studio.model_trained("new")
    studio.model_changed.emit()
    studio.qapp.processEvents()


def _pick(tab, path: str) -> None:
    tab.data.set_path(path)
    wait(tab.studio.qapp, lambda: tab.data.report is not None)
    assert tab.data.report.ok, tab.data.report.problems


def test_the_train_tab_has_a_shape_of_ones_own(studio, dialogs):
    tab = studio.train_tab
    tab.size.setCurrentIndex(tab.size.findData("45m"))
    tab.size.setCurrentIndex(tab.size.findData(CUSTOM))
    studio.qapp.processEvents()
    assert not tab.shape_box.isHidden()
    # it starts from the size chosen before
    assert (tab.layers.value(), tab.width.value(), tab.heads.value(), tab.kv_heads.value(),
            tab.ffn.value()) == (12, 384, 6, 2, 1024)
    assert "19M parameters" in tab.size_info.text() and "of memory" in tab.size_info.text()
    tab.width.setValue(385)                              # 385 / 6 heads
    assert "divides into the heads" in tab.size_info.text()
    tab.width.setValue(96)
    tab.layers.setValue(2)
    s = tab.settings()
    assert (s.preset, s.d_model, s.n_layers, s.n_heads) == (CUSTOM, 96, 2, 6)
    # applying settings keeps their shape (the choice of "custom" does not refill it)
    tab.size.setCurrentIndex(tab.size.findData("25m"))
    tab.apply_settings(s)
    assert tab.size.currentData() == CUSTOM and (tab.width.value(), tab.layers.value()) == \
        (96, 2)
    assert tab.run_base_label(CUSTOM).startswith("a new ")


def test_train_a_model_of_ones_own_shape_then_train_it_further(studio, tmp_path, dialogs):
    tab = studio.train_tab
    _pick(tab, _text_file(tmp_path))
    tab.size.setCurrentIndex(tab.size.findData(CUSTOM))
    tab.layers.setValue(2)
    tab.width.setValue(32)
    tab.heads.setValue(2)
    tab.kv_heads.setValue(0)
    tab.ffn.setValue(64)
    tab.alibi.setChecked(True)
    tab.tokenizer.setCurrentIndex(tab.tokenizer.findData("char"))
    tab.seq.setValue(16)
    tab.epochs.setValue(1)
    tab.batch.setValue(8)
    tab.accumulation.setValue(2)
    tab.start()
    wait(studio.qapp, lambda: tab.job is None, timeout=300)
    assert dialogs["error"] == [], [f.message for f in dialogs["error"]]
    model = studio.engine.model
    assert (model.d_model, model.n_layers, model.n_heads) == (32, 2, 2)
    assert model.use_alibi
    # the model in memory, trained further: the same model, not a new one
    from auralite import runs as runlib
    tab.continue_model.setChecked(True)
    tab.epochs.setValue(1)
    tab.start()
    assert studio.busy == "training"
    wait(studio.qapp, lambda: tab.job is None, timeout=300)
    assert dialogs["error"] == [], [f.message for f in dialogs["error"]]
    assert studio.engine.model is model
    latest = sorted(runlib.list_runs(), key=lambda r: r.id)[-1]
    assert latest.record["method"] == "more" and latest.name.startswith("Train the model in memory")
    assert studio.model_origin == latest.name             # the run saved in the library


def test_further_training_needs_a_model(studio, tmp_path, dialogs):
    tab = studio.train_tab
    _pick(tab, _text_file(tmp_path))
    tab.continue_model.setChecked(True)
    tab.start()
    assert tab.job is None and "no model of AuraLite's own" in dialogs["error"][-1].message


def test_the_tokenizer_trained_alone_then_used(studio, tmp_path, dialogs):
    tab = studio.train_tab
    _pick(tab, _text_file(tmp_path))
    tab.vocab.setValue(80)
    path = tmp_path / "corpus_bpe80_tokenizer.json"
    dialogs["answers"]["save"] = str(path)
    tab.train_tokenizer()
    assert tab.task == "tokenizer"
    wait(studio.qapp, lambda: tab.job is None)
    assert dialogs["error"] == [], [f.message for f in dialogs["error"]]
    assert path.is_file() and tab.tokenizer.currentData() == "file"
    assert tab.tokenizer_path.text() == str(path) and tab.tokenizer_vocab > 0
    assert "Tokenizer trained and saved" in tab.panel.status.text()
    assert not tab.tokenizer_file_row.isHidden() and not tab.vocab.isEnabled()
    # the next training uses it
    tab.size.setCurrentIndex(tab.size.findData(CUSTOM))
    for spin, value in ((tab.layers, 1), (tab.width, 32), (tab.heads, 2), (tab.kv_heads, 0),
                        (tab.ffn, 64)):
        spin.setValue(value)
    tab.seq.setValue(16)
    tab.epochs.setValue(1)
    tab.batch.setValue(8)
    tab.start()
    wait(studio.qapp, lambda: tab.job is None, timeout=300)
    assert dialogs["error"] == [], [f.message for f in dialogs["error"]]
    assert studio.engine.tokenizer.vocab_size == tab.tokenizer_vocab
    # ... and the model's tokenizer saved for other models
    saved = tmp_path / "saved-tokenizer.json"
    dialogs["answers"]["save"] = str(saved)
    assert tab.save_tokenizer_button.isEnabled()
    tab.save_tokenizer()
    assert saved.is_file() and tab.use_tokenizer_file(str(saved))


def test_a_prepared_dataset(studio, tmp_path, dialogs):
    tab = studio.train_tab
    _pick(tab, _text_file(tmp_path))
    tab.tokenizer.setCurrentIndex(tab.tokenizer.findData("char"))
    out = tmp_path / "shards"
    dialogs["answers"]["dir"] = str(out)
    tab.prepare_dataset()
    assert tab.task == "prepare" and not tab.panel.finish_button.isEnabled()
    wait(studio.qapp, lambda: tab.job is None)
    assert dialogs["error"] == [], [f.message for f in dialogs["error"]]
    wait(studio.qapp, lambda: tab.data.report is not None and tab.data.report.kind == "shards")
    assert tab.data.paths() == [str(out)] and "The dataset is ready" in tab.panel.status.text()
    tab.prepare_dataset()                       # prepared already
    assert "prepared dataset already" in dialogs["error"][-1].message


def test_passes_are_suggested_for_the_text(studio, tmp_path, dialogs):
    tab = studio.train_tab
    tab.suggest_epochs()
    assert dialogs["error"] and tab.suggest_hint.isHidden()      # no text yet
    _pick(tab, _text_file(tmp_path))
    tab.epochs.setValue(1)
    tab.suggest_epochs()
    assert tab.epochs.value() >= 5 and not tab.suggest_hint.isHidden()
    assert "passes" in tab.suggest_hint.text()


def test_settings_files(studio, tmp_path, dialogs):
    tab = studio.train_tab
    data = _text_file(tmp_path)
    _pick(tab, data)
    tab.size.setCurrentIndex(tab.size.findData(CUSTOM))
    tab.width.setValue(64)
    tab.grad_clip.setValue(0.7)
    tab.schedule.setCurrentIndex(tab.schedule.findData("cosine"))
    tab.dropout.setText("0.05")
    path = tmp_path / "train-settings.json"
    dialogs["answers"]["save"] = str(path)
    tab.save_settings()
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["format"] == "auralite-studio-settings" and doc["tab"] == "train"
    assert doc["settings"]["d_model"] == 64 and doc["data"] == [data]
    tab.width.setValue(128)
    tab.grad_clip.setValue(1.0)
    tab.schedule.setCurrentIndex(0)
    tab.dropout.setText("auto")
    tab.data.set_path("")
    dialogs["answers"]["open"] = str(path)
    tab.load_settings()
    assert tab.width.value() == 64 and tab.grad_clip.value() == pytest.approx(0.7)
    assert tab.schedule.currentData() == "cosine" and tab.dropout.text() == "0.05"
    assert tab.size.currentData() == CUSTOM and tab.data.paths() == [data]
    # the Fine-tune tab's file is not the Train tab's
    other = tmp_path / "finetune-settings.json"
    dialogs["answers"]["save"] = str(other)
    studio.finetune_tab.lora_targets.setCurrentIndex(
        studio.finetune_tab.lora_targets.findData("ffn"))
    studio.finetune_tab.save_settings()
    assert json.loads(other.read_text(encoding="utf-8"))["settings"]["lora_targets"] == "ffn"
    dialogs["answers"]["open"] = str(other)
    tab.load_settings()
    assert "settings of the finetune tab" in dialogs["error"][-1].message
    studio.finetune_tab.lora_targets.setCurrentIndex(0)
    studio.finetune_tab.load_settings()
    assert studio.finetune_tab.lora_targets.currentData() == "ffn"
    ft = studio.finetune_tab
    ft.beta.setText("0.3")
    ft.pref_epochs.setValue(4)
    s = ft.settings()
    assert (s.preference_beta, s.preference_epochs) == (0.3, 4)
    ft.beta.setText("x")
    with pytest.raises(SettingsError, match="beta"):
        ft.settings()
    ft.apply_settings(s)
    assert ft.beta.text() == "0.3" and ft.pref_epochs.value() == 4


def test_bad_numbers_are_explained(studio, tmp_path, dialogs):
    tab = studio.train_tab
    _pick(tab, _text_file(tmp_path))
    tab.dropout.setText("lots")
    tab.start()
    assert tab.job is None and "numbers" in dialogs["error"][-1].message


def test_helping_another_computer_joins_and_leaves(studio, dialogs):
    tab = studio.train_tab
    tab.workers.setValue(2)
    assert ":29500" in tab.address_hint.text() and "Join" in tab.address_hint.text()
    tab.toggle_worker()
    assert tab.worker_job is None and "address of the main computer" in \
        dialogs["error"][-1].message
    tab.join_address.setText("127.0.0.1:9")              # nothing listens there
    tab.toggle_worker()
    assert tab.worker_job is not None and studio.busy and tab.join_button.text() == "Leave"
    assert tab.worker_job in studio.running_jobs()
    # the worker's own words ("Connecting to …", then "Waiting for the main computer at …")
    wait(studio.qapp, lambda: "127.0.0.1:9" in tab.worker_status.text()
         and not tab.worker_status.text().startswith("Joining"), timeout=60)
    tab.toggle_worker()                                   # Leave
    wait(studio.qapp, lambda: tab.worker_job is None, timeout=60)
    assert tab.join_button.text() == "Join" and studio.busy == ""
    assert tab.worker_status.text().startswith("Left")


def test_the_dialog_options_show_for_dialogs(studio, tmp_path):
    tab = studio.train_tab
    path = tmp_path / "chats.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(20):
            f.write(json.dumps({"messages": [{"role": "user", "content": f"hi {i}"},
                                             {"role": "assistant", "content": f"hello {i}"}]})
                    + "\n")
    assert tab.dialog_box.isHidden()
    _pick(tab, str(path))
    assert not tab.dialog_box.isHidden()
    tab.template.setCurrentIndex(tab.template.findData("gemma"))
    tab.prompt_weight.setText("0.2")
    tab.pack.setChecked(True)
    p = pretrain_params(tab.settings(), "dialogs")
    assert (p["chat_template"], p["prompt_loss_weight"], p["dialog_packing"]) == \
        ("gemma", 0.2, True)


# ---- Generate -------------------------------------------------------------
def test_the_generate_tab_writes_on_from_a_prompt(studio, tmp_path, dialogs, monkeypatch):
    tab = studio.generate_tab
    assert not tab.generate_button.isEnabled()          # no model yet
    _tiny_model(studio)
    assert tab.generate_button.isEnabled() and "trained length" in tab.context_hint.text()
    tab.prompt.setPlainText("AuraLite trains")
    tab.length.setValue(12)
    tab.generate()                                      # token by token
    assert studio.busy == "writing"
    wait(studio.qapp, lambda: tab.job is None)
    text = tab.output.toPlainText()
    assert dialogs["error"] == [] and text.startswith("AuraLite trains") and len(text) > 15
    tab.stream.setChecked(False)                        # at once
    tab.top_k.setValue(5)
    tab.repetition.setValue(1.2)
    seen = {}
    real = studio.engine.generate

    def spy(prompt, length, **options):
        seen.update(options, length=length)
        return real(prompt, length, **options)

    monkeypatch.setattr(studio.engine, "generate", spy)
    tab.generate()
    wait(studio.qapp, lambda: tab.job is None)
    assert seen["top_k"] == 5 and seen["repetition_penalty"] == pytest.approx(1.2)
    assert seen["length"] == 12 and tab.output.toPlainText().startswith("AuraLite trains")
    monkeypatch.undo()
    tab.batch.setChecked(True)                          # one batch
    tab.prompt.setPlainText("AuraLite\n\nIt supports")
    tab.generate()
    wait(studio.qapp, lambda: tab.job is None)
    out = tab.output.toPlainText()
    assert "Prompt 1: AuraLite" in out and "Prompt 2: It supports" in out
    assert "Prompt 3" not in out
    tab.batch.setChecked(False)
    # a suggested length, the whole history
    tab.length.setValue(1)
    tab.suggest_length()
    assert tab.length.value() > 1
    tab.full_context.setChecked(True)
    assert "whole history" in tab.context_hint.text()
    tab.prompt.setPlainText("RoPE")
    tab.generate()
    wait(studio.qapp, lambda: tab.job is None)
    assert studio.engine.full_context
    tab.full_context.setChecked(False)
    tab.prompt.setPlainText("   ")
    tab.generate()
    assert tab.job is None and "write a prompt" in dialogs["error"][-1].message


def test_thinking_and_the_web_search(studio, dialogs, monkeypatch):
    tab = studio.generate_tab
    _tiny_model(studio)
    import auralite.web_tools as web_tools
    monkeypatch.setattr(web_tools, "build_web_context",
                        lambda query, max_results=4: f"Snippet about {query}.")
    tab.prompt.setPlainText("AuraLite")
    tab.length.setValue(8)
    tab.thinking.setChecked(True)
    tab.web.setChecked(True)
    assert not tab.web_query.isHidden()
    tab.web_query.setText("small models")
    tab.generate()
    wait(studio.qapp, lambda: tab.job is None)
    out = tab.output.toPlainText()
    assert "WEB CONTEXT" in out and "Snippet about small models." in out
    assert "THINKING (the draft)" in out and "ANSWER" in out
    tab.thinking.setChecked(False)

    def offline(query, max_results=4):
        raise OSError("no network")

    monkeypatch.setattr(web_tools, "build_web_context", offline)
    tab.generate()
    wait(studio.qapp, lambda: tab.job is None)
    out = tab.output.toPlainText()
    assert dialogs["error"] == [] and "WEB CONTEXT" not in out and "ANSWER" in out


def test_generated_text_is_saved_appended_copied_and_stopped(studio, tmp_path, dialogs):
    tab = studio.generate_tab
    tab.save()
    assert "no text to save" in dialogs["error"][-1].message
    _tiny_model(studio)
    tab.prompt.setPlainText("AuraLite")
    tab.length.setValue(6)
    tab.generate()
    wait(studio.qapp, lambda: tab.job is None)
    path = tmp_path / "out.txt"
    dialogs["answers"]["save"] = str(path)
    tab.save()
    first = path.read_text(encoding="utf-8")
    assert first == tab.output.toPlainText()
    tab.save(append=True)
    assert path.read_text(encoding="utf-8") == first + "\n\n" + first
    tab.copy()
    from PySide6.QtWidgets import QApplication
    assert QApplication.clipboard().text() == first
    tab.length.setValue(4000)                           # long: stopped on the way
    tab.generate()
    tab.stop()
    wait(studio.qapp, lambda: tab.job is None)
    assert tab.speed.text().startswith(("Stopped", "Done"))
    assert studio.busy == ""


# ---- Export -----------------------------------------------------------------
def test_export_a_pytorch_2_program_and_torchscript(studio, tmp_path, dialogs):
    import torch
    tab = studio.export_tab
    _tiny_model(studio)
    tab.format_buttons["pt2"].setChecked(True)
    assert tab.dynamic.isEnabled() and not tab.quant.isEnabled()
    path = tmp_path / "model.pt2"
    tab.path.setText(str(path))
    tab.export()
    wait(studio.qapp, lambda: tab.job is None)
    assert dialogs["error"] == [], [f.message for f in dialogs["error"]]
    assert path.is_file() and tab.result.text().startswith("Saved:")
    program = torch.export.load(str(path)).module()
    ids = torch.tensor([studio.engine.encode("AuraLite")])
    studio.engine.model.eval()
    with torch.no_grad():
        assert torch.allclose(program(ids), studio.engine.model(ids), atol=1e-5)
    tab.format_buttons["torchscript"].setChecked(True)
    assert tab.path.text().endswith("model-torchscript.pt") and tab.script.isEnabled()
    tab.export()
    wait(studio.qapp, lambda: tab.job is None)
    assert dialogs["error"] == [], [f.message for f in dialogs["error"]]
    assert (tmp_path / "model-torchscript.pt").is_file()
    assert "deprecated" in tab.result.text()


def test_onnx_needs_its_package_or_exports(studio, tmp_path, dialogs):
    tab = studio.export_tab
    _tiny_model(studio)
    tab.format_buttons["onnx"].setChecked(True)
    assert tab.opset.isEnabled()
    if importlib.util.find_spec("onnx") is None:
        assert not tab.needs.isHidden() and "«onnx»" in tab.needs.text()
        assert "pip install" in tab.needs.text() and not tab.export_button.isEnabled()
        tab.format_buttons["all"].setChecked(True)
        assert not tab.export_button.isEnabled()
        tab.format_buttons["pt2"].setChecked(True)
        assert tab.needs.isHidden() and tab.export_button.isEnabled()
        return
    path = tmp_path / "model.onnx"
    tab.path.setText(str(path))
    tab.export()
    wait(studio.qapp, lambda: tab.job is None)
    assert dialogs["error"] == [], [f.message for f in dialogs["error"]]
    assert path.is_file() and "exporter" in tab.result.text()


def test_the_library_lists_what_the_studio_opens_again(studio, tmp_path, dialogs):
    from auralite.studio import library
    tab = studio.export_tab
    _tiny_model(studio)
    tab.format_buttons["pt2"].setChecked(True)
    tab.path.setText(str(tmp_path / "m.pt2"))
    tab.export()
    wait(studio.qapp, lambda: tab.job is None)
    assert not any(str(item.path).endswith(".pt2") for item in library.list_models())


# ---- The log ------------------------------------------------------------------
def test_the_log_colours_its_lines_and_saves_them(studio, tmp_path, dialogs):
    from auralite.studio.qt.tab_log import line_level
    assert line_level("Traceback (most recent call last):") == "error"
    assert line_level("WARNING the data is small") == "warn"
    assert line_level("[AuraLite] Epoch 2/5") == "epoch"
    assert line_level("[AuraLite] Training done: 1 epochs") == "ok"
    assert line_level("plain words") == ""
    log = studio.log_tab
    log.view.clear()
    log.append("an error <happened> & more")
    log.append("plain words")
    assert log.view.toPlainText() == "an error <happened> & more\nplain words"
    path = tmp_path / "log.txt"
    dialogs["answers"]["save"] = str(path)
    log.save()
    assert path.read_text(encoding="utf-8") == "an error <happened> & more\nplain words\n"
    assert "errors" in log.legend.text()


# ---- The window -------------------------------------------------------------
def test_the_window_opens_where_it_was_left(tmp_path, monkeypatch, dialogs):
    QtWidgets, _QtCore = _qt()
    from auralite.engine import AuraLiteEngine
    from auralite.studio.qt.window import MainWindow
    from auralite.studio.settings import StudioSettings, load_settings
    if QtWidgets.QApplication.instance() is None:
        QtWidgets.QApplication([])
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    window = MainWindow(engine=AuraLiteEngine(),
                        settings=StudioSettings(first_run_done=True, last_tab="generate"),
                        detect=False, capture_streams=False, save_on_close=True,
                        auto_speed_test=False)
    assert window.tabs.currentWidget() is window.generate_tab
    window.goto("export")
    window.close()
    assert load_settings().last_tab == "export"
