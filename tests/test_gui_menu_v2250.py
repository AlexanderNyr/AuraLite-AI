"""2.25: what the Studio has, in the classic interface - part 1.

The menu bar (File with the recent models, View with the dark theme remembered, Tools with
the benchmark, the speed test, the AuraLite folder and the Studio, Help with the guide,
updates and about), the memory and the speed in the footer, HQQ in the Quant tab and the
words of dialogs as calibration text.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tkinter", reason="tkinter not available (headless environment)")
import importlib as _importlib  # noqa: E402

_importlib.import_module("tkinter.filedialog")
_importlib.import_module("tkinter.messagebox")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._gui_harness import InlineThread, build_gui_app, make_mock, wait_for  # noqa: E402

TEXT = ("AuraLite trains small language models on CPU. It supports RMSNorm, RoPE, SwiGLU "
        "and grouped-query attention. ") * 30
TINY = {"d_model": 32, "n_heads": 2, "n_layers": 1, "d_ff": 64, "seq_length": 16,
        "batch_size": 8, "epochs": 1, "tokenizer": "char", "val_split": 0.1,
        "progress_every_seconds": 100.0}


@pytest.fixture()
def app(tmp_path, monkeypatch):
    with build_gui_app(tmp_path, monkeypatch) as instance:
        yield instance


def _messagebox():
    return sys.modules["tkinter.messagebox"]


def _labels(menu_mock, method: str = "add_command") -> list[str]:
    return [c.kwargs.get("label") for c in getattr(menu_mock, method).call_args_list]


def _saved_model(tmp_path) -> str:
    from auralite.engine import AuraLiteEngine
    engine = AuraLiteEngine()
    engine.train(TEXT, dict(TINY))
    path = str(tmp_path / "tiny.pt")
    engine.save_model(path)
    return path


def test_the_menu_bar_has_the_studios_menus(app):
    assert _labels(app.menubar, "add_cascade") == ["File", "View", "Tools", "Help"]
    assert _labels(app.tools_menu) == ["Benchmark…", "Speed test", "Setup wizard…",  # 2.27
                                       "Library: runs, models, datasets",
                                       "Open the AuraLite folder", "Open the log folder",
                                       "AuraLite Studio (Qt)"]
    loads = []
    app.load_model = lambda path=None: loads.append(path)
    file_menu = app.menubar.add_cascade.call_args_list[0].kwargs["menu"]
    commands = {c.kwargs["label"]: c.kwargs["command"] for c in
                file_menu.add_command.call_args_list}
    assert list(commands) == ["Open a model…", "Save the model as…", "Quit"]
    commands["Open a model…"]()
    assert loads == [None]                                   # the file dialog
    view = app.menubar.add_cascade.call_args_list[1].kwargs["menu"]
    assert view.add_checkbutton.call_args.kwargs["variable"] is app.dark_mode


def test_recent_models_are_remembered_and_opened(app, tmp_path):
    path = _saved_model(tmp_path)
    app.load_model(path)                                     # File -> Recent models' way
    assert app.engine.model is not None and app.is_trained
    assert app.gui_config.recent_models == [os.path.abspath(path)]
    assert app.gui_config.last_model == os.path.abspath(path)
    saved = json.loads((tmp_path / "gui_config.json").read_text(encoding="utf-8"))
    assert saved["recent_models"] == [os.path.abspath(path)]
    other = tmp_path / "other.pt"
    app._remember_model(str(other))
    app._remember_model(path)                                # back to the front, once
    assert app.gui_config.recent_models == [os.path.abspath(path), os.path.abspath(other)]
    for _ in range(10):
        app._remember_model(str(tmp_path / f"m{_}.pt"))
    assert len(app.gui_config.recent_models) == 8
    app.gui_config = app.gui_config.update(recent_models=[path, str(other)])
    app.recent_menu = MagicMock()
    app._fill_recent_menu()
    entries = app.recent_menu.add_command.call_args_list
    assert [c.kwargs["label"] for c in entries] == [path, str(other)]
    loads = []
    app.load_model = lambda p=None: loads.append(p)
    entries[0].kwargs["command"]()
    assert loads == [path]
    warn = _messagebox().showwarning
    warn.reset_mock()
    entries[1].kwargs["command"]()                           # gone: said, and forgotten
    assert "gone" in warn.call_args.args[1] and app.gui_config.recent_models == [path]
    app.gui_config = app.gui_config.update(recent_models=[])
    app.recent_menu = MagicMock()
    app._fill_recent_menu()
    assert app.recent_menu.add_command.call_args.kwargs["label"] == "(none yet)"


def test_saving_a_model_remembers_it(app, tmp_path):
    app.engine.train(TEXT, dict(TINY))
    target = str(tmp_path / "saved.pt")
    with patch.object(sys.modules["tkinter.filedialog"], "asksaveasfilename",
                      MagicMock(return_value=target)):
        app.save_model()
    assert os.path.isfile(target) and app.gui_config.recent_models[0] == os.path.abspath(target)


def test_the_dark_theme_is_remembered(tmp_path, monkeypatch):
    with build_gui_app(tmp_path, monkeypatch) as app:
        assert not app.dark_mode.get()
        app.dark_mode.set(True)
        app._toggle_dark_mode()
        assert app.gui_config.theme == "dark"
        app._on_close()                                      # saved with the settings
    assert json.loads((tmp_path / "gui_config.json").read_text(encoding="utf-8"))[
        "theme"] == "dark"
    with build_gui_app(tmp_path, monkeypatch) as app:      # read back from the file
        assert app.dark_mode.get() and app.gui_config.theme == "dark"
    from auralite.gui.config import GuiConfig
    with build_gui_app(tmp_path, monkeypatch, config=GuiConfig(theme="light")) as app:
        assert not app.dark_mode.get()


def test_memory_and_speed_in_the_footer(app, monkeypatch):
    import auralite.studio.hardware as hardware
    text = app.memory_label.config.call_args.kwargs["text"]
    assert text.startswith("RAM: ") and text != "RAM: —"     # this process, now
    reads = []
    monkeypatch.setattr(hardware, "process_memory", lambda: reads.append(1) or 3 << 30)
    app._tick_memory()               # the test root runs after() at once: one update, no loop
    assert app.memory_label.config.call_args.kwargs["text"] == "RAM: 3.0 GB"
    assert len(reads) == 1
    monkeypatch.setattr(threading, "Thread", InlineThread)
    monkeypatch.setattr(hardware, "measure_gflops", lambda device="cpu": 123.4)
    app._run_speed_test()
    assert app.gui_config.gflops == 123.4 and app.gui_config.gflops_threads >= 1
    assert app.speed_label.config.call_args.kwargs["text"] == "Speed: 123 GFLOP/s"

    def broken(device="cpu"):
        raise RuntimeError("no BLAS")

    monkeypatch.setattr(hardware, "measure_gflops", broken)
    errors = _messagebox().showerror
    errors.reset_mock()
    app._run_speed_test()
    # 2.27: explained with what to try (a test's root has no window: the plain box)
    title, text = errors.call_args.args
    assert title == "Speed test" and text.startswith("The speed test failed\n\nno BLAS")


def test_the_last_speed_test_shows_at_the_start(tmp_path, monkeypatch):
    from auralite.gui.config import GuiConfig
    labels = []
    real = make_mock

    def label(*args, **kwargs):
        widget = real()
        labels.append(kwargs.get("text"))
        return widget

    labelled = patch("tkinter.ttk.Label", label)             # over the harness's double
    with build_gui_app(tmp_path, monkeypatch, config=GuiConfig(gflops=76.4),
                       prepare=labelled.start) as app:
        labelled.stop()
        assert app._speed_text() == "Speed: 76 GFLOP/s"
    assert "Speed: 76 GFLOP/s" in labels


def _dialog(app, monkeypatch):
    import tkinter
    monkeypatch.setattr(tkinter, "Toplevel", make_mock)
    assert app._ask_benchmark() is None                     # nothing clicked
    return app._bench_dialog


def test_the_benchmark_dialog_checks_the_model(app, tmp_path, monkeypatch):
    from auralite import bench
    from auralite.studio.benchchoice import BenchChoice, check_choice, describe
    assert check_choice(BenchChoice(seconds=60.0, method="full")).startswith("unknown method")
    assert describe(BenchChoice(seconds=30.0, finetune=bench.FINETUNE_MODEL, method="lora")) \
        == "30 s per size, fine-tuning Qwen3.5 0.8B (lora)"
    dialog = _dialog(app, monkeypatch)
    values = dialog["vars"]
    assert values["seconds"].get() == str(int(bench.DEFAULT_SECONDS))
    assert values["where"].get() == "catalog" and not values["finetune"].get()
    assert "Qwen3.5" in values["catalog"].get()
    assert dialog["read"]() == BenchChoice(seconds=60.0)
    values["seconds"].set("5")
    assert dialog["read"]().seconds == 15.0                   # the dialog's range
    values["seconds"].set("ninety")
    assert dialog["read"]().seconds == bench.DEFAULT_SECONDS
    values["seconds"].set("90")
    values["finetune"].set(True)
    assert dialog["read"]() == BenchChoice(seconds=90.0, finetune=bench.FINETUNE_MODEL)
    values["where"].set("disk")
    values["path"].set(str(tmp_path / "nothing"))
    dialog["start"]()
    assert "choice" not in dialog["result"]
    assert dialog["problem"].config.call_args.kwargs["text"]   # why not
    values["path"].set("")
    dialog["start"]()
    assert dialog["problem"].config.call_args.kwargs["text"] == \
        "Choose the model: a folder or a checkpoint file"
    checkpoint = _saved_model(tmp_path)
    values["path"].set(checkpoint)
    values["method"].set("QLoRA (a 4-bit base: less memory)")
    dialog["start"]()
    assert dialog["result"]["choice"] == BenchChoice(seconds=90.0, finetune=checkpoint,
                                                     method="qlora")
    # the next time the dialog starts from the last choice
    app.bench_choice = dialog["result"]["choice"]
    again = _dialog(app, monkeypatch)["vars"]
    assert (again["where"].get(), again["path"].get(), again["seconds"].get()) == \
        ("disk", checkpoint, "90")


def test_the_benchmark_runs_reports_and_stops(app, tmp_path, monkeypatch):
    from auralite import bench
    from auralite.studio.benchchoice import BenchChoice
    seen = {}

    def fake_run(*, sizes, seconds, finetune, method, progress, stop, **_kw):
        seen.update(sizes=sizes, seconds=seconds, finetune=finetune, method=method)
        progress("tiny: 4,466 tokens/s")
        return {"sizes": sizes}

    monkeypatch.setattr(bench, "run_bench", fake_run)
    monkeypatch.setattr(bench, "save_report",
                        lambda report: (tmp_path / "bench.md", tmp_path / "bench.json"))
    monkeypatch.setattr(bench, "summary_lines", lambda report: ["tiny: 4,466 tokens/s"])
    info = _messagebox().showinfo
    info.reset_mock()
    monkeypatch.setattr(app, "_ask_benchmark", lambda start=None: BenchChoice(seconds=30.0))
    app._benchmark_clicked()
    wait_for(lambda: info.called, timeout=30)
    assert seen == {"sizes": list(bench.SIZES), "seconds": 30.0, "finetune": None,
                    "method": ""}
    title, text = info.call_args.args
    assert title == "Benchmark" and str(tmp_path / "bench.md") in text
    assert "tiny: 4,466 tokens/s" in text and app._bench_stop is None
    labels = [c.kwargs.get("label") for c in app.tools_menu.entryconfig.call_args_list]
    assert labels[-2:] == ["Stop the benchmark", "Benchmark…"]
    started = threading.Event()

    def slow_run(*, stop, progress, **_kw):
        started.set()
        while not stop.is_set():
            time.sleep(0.01)
        raise InterruptedError("stopped")

    monkeypatch.setattr(bench, "run_bench", slow_run)
    app._benchmark_clicked()
    assert started.wait(10) and app._bench_stop is not None
    app._benchmark_clicked()                                 # the item stops it
    wait_for(lambda: app._bench_stop is None, timeout=30)
    assert app.last_bench_text == "The benchmark was stopped."

    def broken(**_kw):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(bench, "run_bench", broken)
    errors = _messagebox().showerror
    errors.reset_mock()
    app._run_benchmark(BenchChoice(seconds=15.0))
    wait_for(lambda: errors.called, timeout=30)
    title, text = errors.call_args.args                         # 2.27: explained
    assert title == "Benchmark" and text.startswith("Not enough memory")
    assert "during the benchmark" in text and "What to try:" in text


def test_updates_about_and_the_studio(app, monkeypatch, tmp_path):
    import auralite.gui.menu as menu
    import auralite.studio.updates as updates
    from auralite import __version__
    monkeypatch.setattr(threading, "Thread", InlineThread)
    monkeypatch.setattr(updates, "check_for_update",
                        lambda current: updates.UpdateInfo(current=current, latest="99.0.0"))
    info = _messagebox().showinfo
    info.reset_mock()
    app._check_updates()
    assert info.call_args.args[0] == "Updates" and "99.0.0 is out" in info.call_args.args[1]
    text = app.about_text()
    assert f"AuraLite {__version__}" in text and "tkinter" in text and "PyTorch" in text
    started = []
    monkeypatch.setattr(menu.subprocess, "Popen", lambda args, **kw: started.append(args))
    import importlib.util
    app._open_studio()
    if importlib.util.find_spec("PySide6") is not None:
        assert started == [[sys.executable, "-m", "auralite.studio"]]
    else:
        assert "auralite-ai[studio]" in info.call_args.args[1]
    started.clear()
    folder = tmp_path / "new" / "folder"
    assert menu.open_folder(str(folder)) and folder.is_dir()
    if sys.platform.startswith("linux"):
        assert started == [["xdg-open", str(folder)]]
    assert not menu.open_folder("")


def test_hqq_and_fp8_in_the_quant_tab(app):
    from auralite.studio.quantize import bits_for
    app.q_method_var.set("hqq")
    app._on_quant_method_changed()
    assert app.q_bits_combo.config.call_args.kwargs["values"] == bits_for("hqq")
    app.q_method_var.set("half")
    app._on_quant_method_changed()
    assert "fp8_e4m3" in app.q_bits_combo.config.call_args.kwargs["values"]
    from auralite.studio.quantize import FP8_NOTE
    assert FP8_NOTE not in app.q_desc_label.config.call_args.kwargs["text"]   # fp16 first
    app.q_bits_var.set("fp8_e4m3")
    app._update_quant_description()
    assert app.q_desc_label.config.call_args.kwargs["text"].endswith(FP8_NOTE)


def test_dialogs_calibrate_with_their_words(app, tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"messages": [{"role": "user", "content": "Hi there."},
                                             {"role": "assistant", "content": "Hello!"}]})
                    + "\n", encoding="utf-8")
    app.selected_file_path = str(path)
    app._use_training_file_for_calib()
    assert app.q_calib_text == "Hi there.\nHello!"


def test_the_model_info_is_the_studios(app):
    from auralite.studio.modelinfo import model_info_lines
    app.engine.train(TEXT, dict(TINY))
    app._refresh_model_info()
    assert app.model_info.get("1.0", "end").strip() == "\n".join(model_info_lines(app.engine))
