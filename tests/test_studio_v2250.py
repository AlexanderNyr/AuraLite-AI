"""2.25: the classic interface's Quant, Eval and Model tabs in the Studio.

Part 2 of moving everything the tkinter interface does into AuraLite Studio: the
Quantize tab (every method, its settings, calibration text, the benchmark, every method
compared, the quantized weights saved, the original back), the Evaluate tab (a chat model
on held-out dialogs, lm-evaluation-harness) and the Model tab (what the model is, GGUF
with llama.cpp's settings, Hugging Face models imported or run by transformers with PEFT
adapters, a fine-tune and the Hub). The toolkit-free parts are shared with the classic
interface.
"""
from __future__ import annotations

import json
import os
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from auralite.studio.evaluate import chat_eval_rows, lm_eval_lines, parse_tasks
from auralite.studio.quantize import (BITS, CALIBRATED, METHODS, QuantSettings, bits_for,
                                      calibration_text, describe_method, read_calibration_file,
                                      uses)

GB = 1 << 30
TEXT = ("AuraLite trains small language models on CPU. It supports RMSNorm, RoPE, SwiGLU "
        "and grouped-query attention. ") * 30
TINY = {"d_model": 32, "n_heads": 2, "n_layers": 1, "d_ff": 64, "seq_length": 16,
        "batch_size": 8, "epochs": 1, "tokenizer": "char", "val_split": 0.1,
        "progress_every_seconds": 100.0}


def _dialogs(n: int = 24) -> list[list[dict]]:
    out = []
    for i in range(n):
        a, b = i % 7 + 1, i % 5 + 2
        out.append([{"role": "user", "content": f"What is {a} plus {b}?"},
                    {"role": "assistant", "content": f"{a} plus {b} is {a + b}."}])
    return out


def _write_dialogs(path, dialogs) -> str:
    path.write_text("".join(json.dumps({"messages": d}) + "\n" for d in dialogs),
                    encoding="utf-8")
    return str(path)


# ----------------------------------------------------------------------
#  Toolkit-free: the Quantize tab's data
# ----------------------------------------------------------------------
def test_every_method_and_its_bits():
    from auralite.quantization import METHOD_SUPPORTED_BITS, QuantMethod
    assert [m for m, _ in METHODS] == ["dynamic", "static", "qat", "gptq", "awq", "hqq", "half"]
    assert {m.value for m in METHOD_SUPPORTED_BITS} == {m for m, _ in METHODS}
    for method, _label in METHODS:
        assert bits_for(method) == [b.value for b in METHOD_SUPPORTED_BITS[QuantMethod(method)]]
        assert set(bits_for(method)) <= set(BITS) and describe_method(method)
    assert bits_for("half") == ["fp16", "bf16", "fp8_e4m3", "fp8_e5m2"]   # FP8 as well
    assert "int2" in bits_for("hqq") and CALIBRATED == {"static", "qat", "gptq", "awq"}
    assert uses("percdamp", "gptq") and not uses("percdamp", "awq")
    assert uses("group_size", "hqq") and not uses("group_size", "dynamic")
    assert uses("per_channel", "qat") and not uses("unknown", "qat")


def test_the_settings_make_the_engines_config():
    s = QuantSettings(method="gptq", bits="int4", group_size=64, percdamp=0.02,
                      calibration_samples=16, calibration_seq=32, awq_alpha=0.7, qat_epochs=2,
                      qat_lr=3e-5, symmetric=False, per_channel=False)
    c = s.config("calibration")
    assert (c.method.value, c.bits.value, c.gptq_group_size, c.gptq_percdamp) == \
        ("gptq", "int4", 64, 0.02)
    assert (c.calibration_samples, c.awq_n_calibration, c.calibration_seq_length) == (16, 16, 32)
    assert (c.awq_alpha, c.qat_epochs, c.qat_lr, c.symmetric, c.per_channel) == \
        (0.7, 2, 3e-5, False, False)
    assert c.calibration_text == "calibration" and c.validate() == []
    tight = QuantSettings(group_size=0, calibration_samples=0, calibration_seq=1,
                          qat_epochs=0).config()
    assert (tight.gptq_group_size, tight.calibration_samples, tight.calibration_seq_length,
            tight.qat_epochs) == (1, 1, 8, 1)
    with pytest.raises(ValueError):
        QuantSettings(method="nope").config()
    assert QuantSettings.from_dict(json.loads(json.dumps(s.to_dict()))) == s
    assert QuantSettings.from_dict({"method": "awq", "unknown": 1}).method == "awq"
    assert QuantSettings.from_dict(None).method == "dynamic"         # type: ignore[arg-type]


def test_awq_searches_the_balance_of_the_settings():
    """2.25: ``awq_alpha`` was a field the search ignored; now it joins the grid."""
    from auralite.quantization import QuantConfig, awq_alphas
    assert awq_alphas(QuantConfig()) == (0.25, 0.5, 0.75, 1.0)          # in the grid already
    assert awq_alphas(QuantConfig(awq_alpha=0.6)) == (0.25, 0.5, 0.75, 1.0, 0.6)
    assert awq_alphas(QuantConfig(awq_grid_alphas=(), awq_alpha=0.3)) == (0.3,)


def test_calibration_text_from_training_data(tmp_path):
    text = tmp_path / "corpus.txt"
    text.write_text(TEXT, encoding="utf-8")
    whole = calibration_text([str(text)])
    assert whole.startswith("AuraLite trains") and len(whole) >= len(TEXT) - 2
    assert len(calibration_text([str(text)], limit=50)) == 50
    dialogs = _write_dialogs(tmp_path / "d.jsonl", _dialogs(3))
    out = calibration_text([dialogs])
    assert out.split("\n\n")[0] == "What is 1 plus 2?\n1 plus 2 is 3."
    assert "{" not in out and "role" not in out                  # the words, not the JSON
    assert len(calibration_text([dialogs], limit=20)) == 20
    pairs = tmp_path / "p.jsonl"
    pairs.write_text(json.dumps({"prompt": "Name a colour.", "chosen": "Blue.",
                                 "rejected": "No."}) + "\n", encoding="utf-8")
    assert calibration_text([str(pairs)]) == "Name a colour.\nBlue."
    raw = tmp_path / "bytes.txt"
    raw.write_bytes(b"abc\xff\xfedef")
    assert read_calibration_file(str(raw)) == "abcdef"
    assert read_calibration_file(str(text), limit=8) == "AuraLite"


def test_data_without_text_cannot_calibrate(monkeypatch, tmp_path):
    import auralite.engine.corpus as corpus
    monkeypatch.setattr(corpus, "load_training_data", lambda paths, stream=None: object())
    with pytest.raises(ValueError, match="no text to calibrate on"):
        calibration_text([str(tmp_path)])


# ----------------------------------------------------------------------
#  Toolkit-free: what the evaluation and the model show
# ----------------------------------------------------------------------
def test_the_rows_of_a_chat_evaluation():
    rows = chat_eval_rows({"stop_rate": 0.5, "exact": 0.25, "clean_rate": 1.0,
                           "reply_loss": 1.23456, "reply_bpb": None, "mean_new_tokens": 7.25,
                           "samples": [{"user": "u" * 50, "reply": "r", "exact": True},
                                       {"user": "q", "reply": "x" * 200, "stopped": True},
                                       {"user": "w", "reply": "cut"}]})
    assert rows[:6] == [("Stopped by itself", "50.0%"), ("Exact replies", "25.0%"),
                        ("No stray template markers", "100.0%"),
                        ("Reply loss (nats per token)", "1.2346"),
                        ("Reply bits per byte", "n/a"), ("New tokens per reply", "7.2")]
    assert rows[6] == ("✓ " + "u" * 40, "r") and rows[7] == ("· q", "x" * 120)
    assert rows[8] == ("… w", "cut") and len(chat_eval_rows({})) == 6


def test_the_lines_of_lm_eval_and_its_tasks():
    lines = lm_eval_lines({"results": {"arc_easy": {"acc,none": 0.61234, "alias": "arc_easy",
                                                    "n": 3, "ok": True}}})
    assert lines == ["=== arc_easy ===", "  acc,none: 0.6123", "  alias: arc_easy",
                     "  n: 3.0000", "  ok: True", ""]
    assert lm_eval_lines("plain") == ["plain"] and lm_eval_lines({"x": 1}) == ["{'x': 1}"]
    assert parse_tasks(" arc_easy, hellaswag,,arc_easy ") == ["arc_easy", "hellaswag"]
    assert parse_tasks("") == []


def test_the_model_info_of_each_kind():
    from auralite.engine import AuraLiteEngine
    from auralite.studio.modelinfo import model_info_lines
    engine = AuraLiteEngine()
    assert model_info_lines(engine) == ["No model in memory."]
    engine.train(TEXT, dict(TINY))
    lines = model_info_lines(engine)
    assert lines[0].startswith("Parameters      : ") and "Tokenizer       : char" in lines
    assert any(line.startswith("Chat template   : chatml") for line in lines)
    assert "d_model         : 32" in lines and "n_kv_heads      : 2 (MHA)" in lines
    assert any(line.startswith("last val loss") for line in lines)
    assert "Last training params:" in lines
    gguf = SimpleNamespace(
        model=SimpleNamespace(path="m.gguf", max_seq_len=4096, n_threads=4, n_gpu_layers=0,
                              n_batch=512, use_mmap=True, use_mlock=False,
                              use_chat_completion=True, chat_format=None,
                              metadata={"general.name": "tiny"}),
        tokenizer=None, vocab_size=0, is_gguf_model=lambda: True, is_hf_model=lambda: False)
    lines = model_info_lines(gguf)
    assert lines[0] == "Backend         : GGUF / llama.cpp" and "Parameters      : —" in lines
    assert "Chat completion : Yes" in lines and "  general.name = tiny" in lines
    proxy = SimpleNamespace(count_parameters=lambda: 1000, count_trainable_parameters=lambda: 10,
                            get_info=lambda: {"model": "org/tiny", "is_peft": True,
                                              "lora_config": "r=16", "max_seq_len": 128})
    hf = SimpleNamespace(model=proxy, hf_proxy=proxy, hf_path="org/tiny", tokenizer=None,
                         vocab_size=50, device="cpu", is_gguf_model=lambda: False,
                         is_hf_model=lambda: True)
    lines = model_info_lines(hf)
    assert lines[:3] == ["Backend         : Hugging Face / transformers",
                         "Model           : org/tiny", "Parameters      : 1,000"]
    assert "LoRA / PEFT     : Yes" in lines and "LoRA config     : r=16" in lines


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
    answers: dict[str, Any] = {"confirm": True, "save": "", "open": "", "dir": "", "text": ""}
    common.DIALOGS.update(
        error=lambda friendly: seen["error"].append(friendly),
        inform=lambda title, text: seen["inform"].append((title, text)),
        confirm=lambda title, text: seen["confirm"].append((title, text)) or answers["confirm"],
        save=lambda title: answers["save"], open=lambda title: answers["open"],
        dir=lambda title: answers["dir"], text=lambda title: answers["text"])
    seen["answers"] = answers
    yield seen
    common.DIALOGS.clear()


def _window(tmp_path, monkeypatch):
    QtWidgets, _QtCore = _qt()
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
    return window


def _close(window) -> None:
    _QtWidgets, QtCore = _qt()
    window.model_dirty = False
    for job in window.running_jobs():
        job.stop_event.set()
        if job.thread is not None:
            job.thread.join(timeout=30)
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    window.qapp.processEvents()


@pytest.fixture()
def studio(tmp_path, monkeypatch, dialogs):
    window = _window(tmp_path, monkeypatch)
    yield window
    _close(window)


def wait(app, condition, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    while not condition():
        if time.time() > deadline:
            raise AssertionError("timed out waiting for the window")
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def _tiny_model(studio, data: Any = TEXT) -> None:
    """A model in memory, as if trained or opened."""
    studio.engine.train(data, dict(TINY))
    studio.model_trained("new")
    studio.model_changed.emit()
    studio.qapp.processEvents()


def _errors(dialogs) -> list[str]:
    return [f.message for f in dialogs["error"]]


def test_the_new_tabs_are_in_the_window(studio):
    from auralite.studio.qt.window import TAB_TITLES
    titles = [studio.tabs.tabText(i) for i in range(studio.tabs.count())]
    assert titles == [title for _key, title in TAB_TITLES]
    assert titles[6:9] == ["Model", "Quantize", "Evaluate"]      # 2.26: Images before them
    for key in ("model", "quantize", "evaluate"):
        studio.goto(key)
        assert studio.tabs.currentIndex() == [k for k, _ in TAB_TITLES].index(key)
    assert studio.pages["quantize"] is studio.quantize_tab
    assert "No model in memory." in studio.model_tab.info.toPlainText()
    assert not studio.quantize_tab.quantize_button.isEnabled()
    assert not studio.evaluate_tab.chat_button.isEnabled()


def test_quantize_benchmark_save_and_go_back(studio, tmp_path, dialogs):
    import torch
    tab = studio.quantize_tab
    _tiny_model(studio)
    original = studio.engine.model
    assert tab.quantize_button.isEnabled() and not tab.benchmark_button.isEnabled()
    assert not tab.fields["group_size"].isEnabled()             # dynamic takes no settings
    studio.model_dirty = False
    tab.quantize()                                               # dynamic INT8
    assert studio.busy == "quantizing"
    wait(studio.qapp, lambda: tab.job is None)
    assert _errors(dialogs) == []
    assert tab.original is not None and studio.engine.model is not original
    assert studio.model_dirty and "Quantized:" in tab.status.text()
    assert "dynamic" in tab.output.toPlainText().lower()
    assert not tab.quantize_button.isEnabled() and tab.revert_button.isEnabled()
    assert not tab.benchmark_button.isEnabled()                 # nothing to measure on yet
    tab.set_calibration(TEXT, "corpus.txt")
    assert f"corpus.txt: {len(TEXT):,} characters" in tab.calib_label.text()
    tab.benchmark()
    wait(studio.qapp, lambda: tab.job is None)
    assert _errors(dialogs) == [] and tab.output.toPlainText().startswith("BENCHMARK")
    assert "perplexity" in tab.status.text()
    path = tmp_path / "small.pt"
    dialogs["answers"]["save"] = str(tmp_path / "small")
    tab.save()
    saved = torch.load(str(path), map_location="cpu", weights_only=False)
    assert saved["is_quantized"] and "model_state" in saved
    assert str(path) in tab.status.text()
    ids = torch.tensor([studio.engine.encode("AuraLite")])
    tab.revert()
    assert studio.engine.model is not original and tab.original is None
    assert not studio.model_dirty and tab.quantize_button.isEnabled()
    studio.engine.model.eval()
    original.eval()
    with torch.no_grad():
        assert torch.allclose(studio.engine.model(ids), original(ids))   # the copy, untouched


def test_methods_that_calibrate_need_text(studio, tmp_path, dialogs):
    tab = studio.quantize_tab
    _tiny_model(studio)
    tab.method.setCurrentIndex(tab.method.findData("gptq"))
    assert tab.fields["percdamp"].isEnabled() and tab.fields["group_size"].isEnabled()
    assert not tab.fields["qat_epochs"].isEnabled()
    assert [tab.bits.itemData(i) for i in range(tab.bits.count())] == bits_for("gptq")
    assert not tab.quantize_button.isEnabled() and "needs calibration" in \
        tab.quantize_button.toolTip()
    tab.quantize()
    assert tab.job is None and "GPTQ needs calibration text" in _errors(dialogs)[-1]
    # the training data: the Train tab's, else the Fine-tune tab's
    tab.calibrate_from_data()
    assert "choose data" in _errors(dialogs)[-1]
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    studio.finetune_tab.data.set_path(str(corpus))
    assert tab.training_paths() == [str(corpus)]
    tab.calibrate_from_data()
    assert tab.calibration.startswith("AuraLite") and "corpus.txt" in tab.calib_label.text()
    tab.set_calibration("", "")
    dialogs["answers"]["open"] = str(corpus)
    tab.calibrate_from_file()
    assert tab.calibration == TEXT and tab.quantize_button.isEnabled()
    empty = tmp_path / "empty.txt"
    empty.write_text("  \n", encoding="utf-8")
    dialogs["answers"]["open"] = str(empty)
    tab.calibrate_from_file()
    assert "holds no text" in _errors(dialogs)[-1] and tab.calibration == TEXT
    tab.bits.setCurrentIndex(tab.bits.findData("int4"))
    tab.group_size.setValue(16)
    tab.samples.setValue(8)
    tab.seq.setValue(16)
    tab.quantize()
    wait(studio.qapp, lambda: tab.job is None)
    assert len(dialogs["error"]) == 3                            # no new one
    assert tab.config.gptq_group_size == 16 and tab.config.bits.value == "int4"
    assert tab.original is not None and "gptq" in tab.output.toPlainText().lower()
    tab.revert()
    tab.qat_lr.setText("fast")
    tab.method.setCurrentIndex(tab.method.findData("qat"))
    tab.quantize()
    assert "QAT learning rate" in _errors(dialogs)[-1]
    tab.qat_lr.setText("1e-5")
    s = tab.settings()
    tab.apply_settings(QuantSettings(method="awq", bits="int4", group_size=32, qat_lr=2e-5))
    assert tab.method_key() == "awq" and tab.bits.currentData() == "int4"
    assert tab.group_size.value() == 32 and tab.qat_lr.text() == "2e-05"
    assert s.method == "qat"


def test_every_method_compared_and_stopped(studio, dialogs, monkeypatch):
    tab = studio.quantize_tab
    _tiny_model(studio)
    model = studio.engine.model
    assert not tab.compare_button.isEnabled()
    tab.set_calibration(TEXT, "text")
    tab.seq.setValue(16)
    tab.compare()
    wait(studio.qapp, lambda: tab.job is None)
    out = tab.output.toPlainText()
    assert _errors(dialogs) == [] and out.startswith("EVERY METHOD")
    assert "gptq" in out and "dynamic" in out and "Compared" in tab.status.text()
    assert studio.engine.model is model and tab.original is None     # as it was
    import auralite.quantization as quantization

    def slow(model, tokenizer, text, device, seq_length=64, progress_callback=None, **_kw):
        assert progress_callback is not None
        for i in range(10_000):
            progress_callback(i, 10_000, "method")
            time.sleep(0.01)
        return []

    monkeypatch.setattr(quantization, "compare_quantizations", slow)
    tab.compare()
    wait(studio.qapp, lambda: tab.stop_button.isEnabled())
    assert tab.job in studio.running_jobs()                  # closing the window stops it
    tab.stop()
    wait(studio.qapp, lambda: tab.job is None)
    assert tab.status.text() == "Stopped." and _errors(dialogs) == []
    assert not studio.busy


def test_fp8_is_for_storage(studio, tmp_path, dialogs):
    """PyTorch keeps float8 weights, but a processor cannot multiply them: an FP8 model is
    saved, not run - the tab says so and does not offer the benchmark."""
    import torch

    from auralite.studio.quantize import FP8_NOTE, bits_note, runnable
    assert bits_note("fp8_e5m2") == FP8_NOTE and bits_note("int4") == ""
    assert not runnable("fp8_e4m3") and runnable("bf16")
    tab = studio.quantize_tab
    _tiny_model(studio)
    tab.set_calibration(TEXT, "text")
    tab.method.setCurrentIndex(tab.method.findData("half"))
    assert FP8_NOTE not in tab.description.text()
    tab.bits.setCurrentIndex(tab.bits.findData("fp8_e4m3"))
    assert FP8_NOTE in tab.description.text()
    tab.quantize()
    wait(studio.qapp, lambda: tab.job is None)
    assert FP8_NOTE in tab.status.text() and not tab.benchmark_button.isEnabled()
    dialogs["answers"]["save"] = str(tmp_path / "fp8.pt")
    tab.save()
    saved = torch.load(str(tmp_path / "fp8.pt"), map_location="cpu", weights_only=False)
    assert saved["is_quantized"]
    tab.revert()
    tab.bits.setCurrentIndex(tab.bits.findData("bf16"))
    tab.quantize()
    wait(studio.qapp, lambda: tab.job is None)
    assert FP8_NOTE not in tab.status.text() and tab.benchmark_button.isEnabled()
    assert _errors(dialogs) == []


def test_a_failed_quantization_leaves_the_model(studio, dialogs, monkeypatch):
    import torch

    from auralite.quantization import QuantizationEngine, QuantResult
    tab = studio.quantize_tab
    _tiny_model(studio)
    ids = torch.tensor([studio.engine.encode("AuraLite")])
    studio.engine.model.eval()
    with torch.no_grad():
        before = studio.engine.model(ids)

    def broken(self, model, config, tokenizer=None, device=None, progress_callback=None):
        torch.nn.init.zeros_(model.embedding.weight)             # a method half-way through
        result = QuantResult()
        result.errors.append("not today")
        return model, result

    monkeypatch.setattr(QuantizationEngine, "quantize", broken)
    tab.quantize()
    wait(studio.qapp, lambda: tab.job is None)
    assert "failed" in tab.status.text() and tab.original is None
    assert "not today" in tab.output.toPlainText() and not tab.save_button.isEnabled()
    studio.engine.model.eval()
    with torch.no_grad():
        assert torch.allclose(studio.engine.model(ids), before)


def test_another_model_drops_the_original(studio, dialogs):
    tab = studio.quantize_tab
    _tiny_model(studio)
    tab.quantize()
    wait(studio.qapp, lambda: tab.job is None)
    assert tab.original is not None
    studio.model_changed.emit()                                # a reply, a save: still it
    assert tab.original is not None
    _tiny_model(studio)                                          # a new model
    assert tab.original is None and tab.quantize_button.isEnabled()


def test_evaluate_a_chat_model_on_dialogs(studio, tmp_path, dialogs):
    from auralite.engine.dialogs import DialogSet
    tab = studio.evaluate_tab
    _tiny_model(studio, DialogSet(_dialogs()))
    assert tab.chat_button.isEnabled() and not tab.chat_save.isEnabled()
    tab.evaluate_chat()                                          # no file: asks, cancelled
    assert tab.job is None and not studio.busy
    tab.dialogs.setText(str(tmp_path / "none.jsonl"))
    tab.evaluate_chat()
    assert "not found" in _errors(dialogs)[-1]
    tab.dialogs.setText(_write_dialogs(tmp_path / "held.jsonl", _dialogs(6)))
    tab.limit.setValue(4)
    tab.reply_tokens.setValue(4)
    tab.evaluate_chat()
    assert studio.busy == "an evaluation"
    wait(studio.qapp, lambda: tab.job is None)
    assert tab.chat_result is not None and tab.chat_result["dialogs"] == 4
    rows = [tab.table.item(i, 0).text() for i in range(tab.table.rowCount())]
    assert rows[:3] == ["Stopped by itself", "Exact replies", "No stray template markers"]
    assert "4 dialogs, the chatml template" in tab.chat_status.text()
    out = tmp_path / "eval.json"
    dialogs["answers"]["save"] = str(out)
    tab.save("chat")
    assert json.loads(out.read_text(encoding="utf-8"))["dialogs"] == 4
    tab.limit.setValue(0)                                        # all of them, stopped
    tab.evaluate_chat()
    tab.stop()
    wait(studio.qapp, lambda: tab.job is None)
    assert len(dialogs["error"]) == 1                            # the missing file only


def test_benchmarks_need_lm_eval_or_run(studio, tmp_path, dialogs, monkeypatch):
    import importlib.util
    tab = studio.evaluate_tab
    tab.add_task("gsm8k")
    tab.add_task("arc_easy")
    assert tab.tasks.text() == "arc_easy,hellaswag,winogrande,gsm8k"
    if importlib.util.find_spec("lm_eval") is None:
        assert not tab.needs.isHidden() and "lm_eval" in tab.needs.text()
        assert not tab.bench_button.isEnabled() and not tab.task_buttons["mmlu"].isEnabled()
        tab.run_benchmarks()
        assert "lm_eval" in _errors(dialogs)[-1]
    monkeypatch.setattr(tab, "lm_eval_note", lambda: "")       # as if installed
    tab._update_buttons()
    assert tab.needs.isHidden() and not tab.bench_button.isEnabled()   # no model yet
    _tiny_model(studio)
    assert tab.bench_button.isEnabled() and tab.task_buttons["mmlu"].isEnabled()
    seen: dict[str, Any] = {}

    def fake(tasks, num_fewshot=0, batch_size=1, limit=None):
        seen.update(tasks=tasks, fewshot=num_fewshot, batch=batch_size, limit=limit)
        return {"results": {t: {"acc,none": 0.5} for t in tasks}}

    monkeypatch.setattr(studio.engine, "evaluate_model", fake)
    tab.tasks.setText("arc_easy, mmlu")
    tab.fewshot.setValue(5)
    tab.batch.setValue(4)
    tab.bench_limit.setValue(20)
    tab.run_benchmarks()
    wait(studio.qapp, lambda: tab.job is None)
    assert seen == {"tasks": ["arc_easy", "mmlu"], "fewshot": 5, "batch": 4, "limit": 20}
    assert "=== mmlu ===" in tab.output.toPlainText() and "Done in" in tab.bench_status.text()
    dialogs["answers"]["save"] = str(tmp_path / "bench.json")
    tab.save("bench")
    assert json.loads((tmp_path / "bench.json").read_text(encoding="utf-8"))[
        "results"]["mmlu"] == {"acc,none": 0.5}
    tab.tasks.setText(" , ")
    tab.run_benchmarks()
    assert "at least one task" in _errors(dialogs)[-1]
    tab.bench_limit.setValue(0)
    tab.tasks.setText("arc_easy")

    def broken(**_kw):
        raise RuntimeError("no network")

    monkeypatch.setattr(studio.engine, "evaluate_model", broken)
    tab.run_benchmarks()
    wait(studio.qapp, lambda: tab.job is None)
    assert "no network" in dialogs["error"][-1].text() + dialogs["error"][-1].details
    assert not studio.busy and tab.bench_status.text().startswith("Failed")


def test_the_model_tab_shows_the_model_and_its_adapters(studio, tmp_path, dialogs):
    tab = studio.model_tab
    assert not tab.save_button.isEnabled() and not tab.merge_button.isEnabled()
    _tiny_model(studio)
    assert "d_model         : 32" in tab.info.toPlainText()
    assert tab.save_button.isEnabled() and tab.load_adapter_button.isEnabled()
    assert not tab.merge_button.isEnabled()
    studio.engine.model.enable_lora(rank=2)
    studio.model_changed.emit()
    assert tab.merge_button.isEnabled() and tab.save_adapter_button.isEnabled()
    assert "Trainable (LoRA)" in tab.info.toPlainText()
    dialogs["answers"]["save"] = str(tmp_path / "adapter.pt")
    tab.save_adapter_button.click()
    assert (tmp_path / "adapter.pt").is_file()
    tab.merge_button.click()
    assert not studio.engine.has_lora() and dialogs["inform"][-1][0] == "Adapters merged"
    assert not tab.merge_button.isEnabled()


def test_gguf_files_open_with_the_settings_of_the_tab(tmp_path, monkeypatch, dialogs):
    monkeypatch.setenv("AURALITE_GGUF_N_CTX", "8192")
    monkeypatch.setenv("AURALITE_GGUF_USE_MLOCK", "1")
    studio = _window(tmp_path, monkeypatch)
    try:
        tab = studio.model_tab
        assert tab.n_ctx.value() == 8192 and tab.mlock.isChecked() and tab.mmap.isChecked()
        tab.threads.setValue(3)
        tab.gpu_layers.setValue(0)
        tab.chat_format.setText("chatml")
        tab.chat_completion.setChecked(True)
        seen: dict[str, Any] = {}
        engine = studio.engine

        def fake_load(path, **options):
            seen.update(options, path=path)
            engine.backend = "gguf"
            engine.gguf_path = path
            engine.model = SimpleNamespace(path=path, max_seq_len=options["n_ctx"],
                                           n_threads=options["n_threads"], n_gpu_layers=0,
                                           n_batch=512, use_mmap=True, use_mlock=True,
                                           use_chat_completion=True, chat_format="chatml",
                                           metadata={})

        monkeypatch.setattr(engine, "load_gguf_model", fake_load)
        model = tmp_path / "tiny.gguf"
        model.write_bytes(b"GGUF")
        dialogs["answers"]["open"] = str(model)
        tab.open_gguf()
        wait(studio.qapp, lambda: not studio.busy)
        assert seen == {"path": str(model), "n_ctx": 8192, "n_gpu_layers": 0, "n_threads": 3,
                        "n_batch": 512, "chat_format": "chatml", "use_chat_completion": True,
                        "use_mmap": True, "use_mlock": True}
        assert "Backend         : GGUF / llama.cpp" in tab.info.toPlainText()
        assert studio.model_path == str(model) and not tab.save_button.isEnabled()
        assert "GGUF model tiny.gguf" in studio.model_description()
        tab.n_ctx.setValue(2048)                             # File -> Open: the same settings
        seen.clear()
        studio.open_model(str(model))
        wait(studio.qapp, lambda: not studio.busy)
        assert seen["n_ctx"] == 2048 and seen["n_threads"] == 3 and seen["use_mlock"]
    finally:
        studio.engine.model = None
        studio.engine.backend = "torch"
        _close(studio)


class FakeProxy:
    """What the Model tab reads of a transformers model (no transformers needed)."""

    def __init__(self) -> None:
        self.is_peft = False
        self.tokenizer = None

    def count_parameters(self) -> int:
        return 1234

    def count_trainable_parameters(self) -> int:
        return 12 if self.is_peft else 1234

    def get_info(self) -> dict:
        return {"model": "org/tiny", "is_peft": self.is_peft}


def _fake_transformers(studio, monkeypatch) -> dict[str, Any]:
    import auralite.studio.qt.tab_model as tab_model
    calls: dict[str, Any] = {}
    engine = studio.engine
    monkeypatch.setattr(tab_model, "package_note", lambda module: "")

    def load(name, **options):
        calls["load"] = dict(options, name=name)
        engine.hf_proxy = FakeProxy()
        engine.hf_proxy.is_peft = bool(options.get("apply_lora"))
        engine.model = engine.hf_proxy
        engine.backend = "huggingface"
        engine.hf_path = name

    def apply_lora(rank=16, **_kw):
        calls["lora"] = rank
        engine.hf_proxy.is_peft = True

    def finetune(texts, output_dir, progress_callback=None, stop_event=None, **options):
        calls["finetune"] = dict(options, texts=texts, output_dir=output_dir)
        assert progress_callback is not None
        for epoch in (1, 2):
            progress_callback(epoch, 2, 0.5 / epoch)
        return output_dir

    monkeypatch.setattr(engine, "load_hf_model", load)
    monkeypatch.setattr(engine, "apply_lora_to_hf", apply_lora)
    monkeypatch.setattr(engine, "save_hf_lora", lambda path: calls.update(save=path))
    monkeypatch.setattr(engine, "load_hf_lora", lambda path: calls.update(adapter=path))
    monkeypatch.setattr(engine, "push_hf_model_to_hub",
                        lambda repo, commit_message="", private=False:
                        calls.update(push=(repo, private)))
    monkeypatch.setattr(engine, "finetune_hf", finetune)
    return calls


def test_transformers_models_load_tune_and_go_to_the_hub(studio, tmp_path, dialogs,
                                                          monkeypatch):
    tab = studio.model_tab
    if tab.hf_needs.isHidden():
        assert tab.hf_load_button.isEnabled()
    else:
        assert "transformers" in tab.hf_needs.text() and not tab.hf_load_button.isEnabled()
    calls = _fake_transformers(studio, monkeypatch)
    tab._update_buttons()
    assert tab.hf_load_button.isEnabled() and not tab.push_button.isEnabled()
    tab.load_transformers()
    assert "enter a Hugging Face model" in _errors(dialogs)[-1]
    tab.hf_name.setText("org/tiny")
    tab.trust_remote.setChecked(True)
    dialogs["answers"]["confirm"] = False                      # not trusted: not loaded
    tab.load_transformers()
    assert "load" not in calls and "trust_remote_code" in dialogs["confirm"][-1][1]
    dialogs["answers"]["confirm"] = True
    tab.trust_remote.setChecked(False)
    tab.precision["8bit"].setChecked(True)
    tab.apply_lora.setChecked(False)
    tab.lora_rank.setValue(8)
    tab.load_transformers()
    wait(studio.qapp, lambda: tab.job is None)
    assert calls["load"] == {"name": "org/tiny", "load_in_4bit": False, "load_in_8bit": True,
                             "apply_lora": False, "lora_rank": 8, "local_files_only": False,
                             "trust_remote_code": False, "verbose": True}
    assert "Loaded org/tiny: 1,234 parameters" in tab.status.text()
    assert "Backend         : Hugging Face / transformers" in tab.info.toPlainText()
    assert studio.model_description() == "transformers model org/tiny"
    assert tab.hf_lora_button.isEnabled() and not tab.hf_save_lora_button.isEnabled()
    tab.hf_lora_button.click()
    assert calls["lora"] == 8 and tab.hf_save_lora_button.isEnabled()
    assert studio.model_description() == "transformers model org/tiny with LoRA adapters"
    dialogs["answers"]["dir"] = str(tmp_path / "adapter")
    tab.save_lora()
    tab.load_lora()
    assert calls["save"] == calls["adapter"] == str(tmp_path / "adapter")
    # a fine-tune on a text file, the classic tab's paragraphs
    text = tmp_path / "notes.txt"
    text.write_text("Short.\n\n" + "A paragraph long enough to learn from, surely.\n\n" * 3,
                    encoding="utf-8")
    dialogs["answers"]["open"] = str(text)
    dialogs["answers"]["dir"] = str(tmp_path / "tuned")
    tab.ft_epochs.setValue(2)
    tab.ft_lr.setText("1e-4")
    tab.finetune()
    wait(studio.qapp, lambda: tab.job is None)
    tuned = calls["finetune"]
    assert tuned["texts"] == ["A paragraph long enough to learn from, surely."] * 3
    assert (tuned["epochs"], tuned["learning_rate"], tuned["batch_size"], tuned["max_length"],
            tuned["gradient_accumulation_steps"]) == (2, 1e-4, 2, 512, 8)
    assert tuned["output_dir"] == str(tmp_path / "tuned")
    assert dialogs["inform"][-1][0] == "Fine-tuning complete" and studio.model_dirty
    tab.ft_lr.setText("quick")
    tab.finetune()
    assert "learning rate" in _errors(dialogs)[-1]
    tab.ft_lr.setText("2e-4")
    # the Hub: pushed privately, then loaded from it (online)
    dialogs["answers"]["text"] = "me/tiny"
    tab.push_to_hub()
    wait(studio.qapp, lambda: tab.job is None)
    assert calls["push"] == ("me/tiny", True) and "huggingface.co/me/tiny" in tab.status.text()
    tab.local_only.setChecked(True)
    studio.model_dirty = False
    dialogs["answers"]["text"] = "org/other"
    tab.load_from_hub()
    wait(studio.qapp, lambda: tab.job is None)
    assert calls["load"]["name"] == "org/other" and calls["load"]["local_files_only"] is False
    folder = tmp_path / "downloaded"
    folder.mkdir()
    dialogs["answers"]["dir"] = str(folder)
    tab.local_only.setChecked(False)
    tab.hf_folder_button.click()
    assert tab.hf_name.text() == str(folder) and tab.local_only.isChecked()
    studio.engine.model = None
    studio.engine.backend = "torch"
    studio.engine.hf_proxy = None


def test_a_transformers_fine_tune_on_the_training_data_stops(studio, tmp_path, dialogs,
                                                           monkeypatch):
    tab = studio.model_tab
    calls = _fake_transformers(studio, monkeypatch)
    tab.hf_name.setText("org/tiny")
    tab.load_transformers()
    wait(studio.qapp, lambda: tab.job is None)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("First paragraph of the training data, long enough.\n\nSecond one, "
                      "also long enough to count.", encoding="utf-8")
    studio.train_tab.data.set_path(str(corpus))
    started = threading.Event()

    def slow(texts, output_dir, progress_callback=None, stop_event=None, **options):
        calls["texts"] = texts
        started.set()
        assert stop_event is not None
        while not stop_event.is_set():
            time.sleep(0.01)
        return output_dir

    monkeypatch.setattr(studio.engine, "finetune_hf", slow)
    dialogs["answers"]["dir"] = str(tmp_path / "out")
    tab.finetune()
    assert "the data chosen for training" in dialogs["confirm"][-1][1]
    wait(studio.qapp, lambda: started.is_set() and tab.ft_stop_button.isEnabled())
    tab.stop()
    wait(studio.qapp, lambda: tab.job is None)
    assert calls["texts"] == ["First paragraph of the training data, long enough.",
                              "Second one, also long enough to count."]
    assert dialogs["inform"][-1][0] == "Fine-tuning stopped" and "stopped" in tab.status.text()
    studio.engine.model = None
    studio.engine.backend = "torch"
    studio.engine.hf_proxy = None


def test_import_into_auralite_shows_its_progress(studio, tmp_path, dialogs, monkeypatch):
    tab = studio.model_tab
    tab.import_native()
    assert "enter a Hugging Face model" in _errors(dialogs)[-1]
    engine = studio.engine
    seen: dict[str, Any] = {}

    def fake_import(source, quantize=None, progress_callback=None, stop_event=None, **_kw):
        seen.update(source=source, quantize=quantize)
        assert progress_callback is not None
        progress_callback({"stage": "download", "file": "model.safetensors",
                           "all_bytes": 5e6, "all_total": 10e6})
        progress_callback({"stage": "convert", "done": 3, "total": 4})
        engine.train(TEXT, dict(TINY))
        return {"architecture": "LlamaForCausalLM", "params": 1_500_000, "dtype": "float32",
                "context": 2048, "chat_template": "chatml", "memory_bytes": 6e6,
                "seconds": 1.0}

    monkeypatch.setattr(engine, "import_hf_model", fake_import)
    progress: list[str] = []
    shown = tab._import_progress
    monkeypatch.setattr(tab, "_import_progress",
                        lambda info: (shown(info), progress.append(tab.status.text())))
    tab.hf_name.setText("org/small")
    tab.import_native()
    assert tab.job is not None and studio.busy == "importing a model"
    wait(studio.qapp, lambda: tab.job is None)
    assert seen == {"source": "hf:org/small", "quantize": "nf4"}       # 4 bits: NF4
    assert progress == ["Downloading model.safetensors: 5 / 10 MB",
                        "Converting the weights 3/4…"]
    assert tab.status.text().startswith("Imported: LlamaForCausalLM 2M params")
    assert studio.model_origin == "org/small" and not studio.model_dirty
    tab.precision["full"].setChecked(True)
    folder = tmp_path / "local-model"
    folder.mkdir()
    tab.hf_name.setText(str(folder))
    tab.import_native()
    wait(studio.qapp, lambda: tab.job is None)
    assert seen == {"source": str(folder), "quantize": None}
    from auralite.studio.qt.tab_model import import_text
    assert import_text({"stage": "download", "file": "a", "bytes": 2e6}) == \
        "Downloading a: 2 MB"
    assert import_text({"stage": "convert", "done": 1, "total": 2}) == \
        "Converting the weights 1/2…"


def test_a_failed_load_says_so(studio, dialogs, monkeypatch):
    tab = studio.model_tab
    _fake_transformers(studio, monkeypatch)

    def broken(name, **options):
        raise OSError("no such model")

    monkeypatch.setattr(studio.engine, "load_hf_model", broken)
    _tiny_model(studio)
    studio.model_dirty = False
    tab.hf_name.setText("org/none")
    tab.load_transformers()
    wait(studio.qapp, lambda: tab.job is None)
    assert tab.status.text().startswith("Failed") and not studio.busy
    assert studio.engine.model is None and studio.model_origin == ""
    assert dialogs["error"]
