"""2.22: ``auralite bench`` - a computer's numbers for training and writing in one command.

Real (short) runs of the benchmark: the built-in text, the sizes, what a size measures,
the report in Markdown and JSON and its file names, the command line, a fine-tune of a
(fake, local) Hugging Face model, the Studio's *Tools -> Benchmark*, the app's smoke step
and README.txt - and the memory plan's margin that the benchmark's numbers set.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from auralite import bench

REPO = Path(__file__).resolve().parents[1]
APP_DIR = REPO / "packaging" / "app"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_bench_{name}", APP_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    """One real benchmark of the tiny size, for the tests that read a report."""
    pytest.importorskip("torch")
    home = tmp_path_factory.mktemp("home")
    old = os.environ.get("AURALITE_HOME")
    os.environ["AURALITE_HOME"] = str(home)
    lines: list[str] = []
    try:
        result = bench.run_bench(sizes=["tiny"], seconds=2.0, generate_tokens=8,
                                 progress=lines.append)
    finally:
        if old is None:
            os.environ.pop("AURALITE_HOME", None)
        else:
            os.environ["AURALITE_HOME"] = old
    result["_lines"] = lines
    return result


# ----------------------------------------------------------------------
#  What it trains
# ----------------------------------------------------------------------
def test_the_built_in_text_is_the_same_everywhere():
    text = bench.bench_text()
    assert text == bench.bench_text() and text.isascii()
    assert 200_000 <= len(text) < 202_000 and text.count("\n") > 300
    assert bench.bench_text(chars=5_000, seed=1) != bench.bench_text(chars=5_000, seed=2)


def test_the_sizes_are_the_studios():
    from auralite.engine.sizing import MODEL_PRESETS
    assert list(bench.SIZES) == ["tiny", "small", "medium"]
    assert bench.size_dims("tiny") == {"d_model": 128, "n_layers": 4, "n_heads": 4,
                                       "n_kv_heads": 2, "d_ff": 384}
    for name, preset in (("small", "25m"), ("medium", "45m")):
        dims = bench.size_dims(name)
        assert all(dims[k] == MODEL_PRESETS[preset][k] for k in dims)
    with pytest.raises(bench.BenchError, match="unknown size 'huge'"):
        bench.size_dims("huge")
    assert bench.QUICK["sizes"] == ("tiny", "small") and bench.DEFAULT_SECONDS == 60.0
    assert 4.0 < bench.estimated_minutes(list(bench.SIZES), 60.0) < 6.0
    assert bench.estimated_minutes(["tiny"], 60.0, finetune=True) > \
        bench.estimated_minutes(["tiny"], 60.0)


def test_steady_speed_skips_the_first_batch_and_the_epochs_ends():
    # (time, batches done, epoch): 4 batches a second, then a 10 s validation, then again
    events = [(0.0, 1, 1), (0.25, 2, 1), (0.5, 3, 1), (0.75, 4, 1),
              (10.75, 1, 2), (11.0, 2, 2), (11.25, 3, 2)]
    assert bench._steady_speed(events, 1000.0) == pytest.approx(4000.0)
    assert bench._steady_speed([(0.0, 1, 1)], 1000.0) == 0.0
    assert bench._steady_speed(events, 0.0) == 0.0


# ----------------------------------------------------------------------
#  A real run
# ----------------------------------------------------------------------
def test_a_size_trains_writes_and_says_its_memory(report):
    from auralite import __version__
    (row,) = report["train"]
    assert row["name"] == "tiny" and row["label"] == "tiny: 4 layers x 128"
    assert 800_000 < row["params"] < 1_000_000                  # 0.9M with BPE 512
    assert row["tokens_per_second"] > 0 and row["optimizer_steps"] >= 2
    assert row["night_tokens"] == int(row["tokens_per_second"] * 8 * 3600)
    assert row["generate_tokens"] == 8 and row["generate_tokens_per_second"] > 0
    assert row["memory_estimate"] > 0 and row["memory_peak"] >= row["memory_start"]
    assert row["memory_used"] == row["memory_peak"] - row["memory_start"]
    assert row["memory_estimate"] < 300 << 20             # the run, not the framework too
    assert row["micro_batch"] * row["accumulation_steps"] >= bench.BATCH
    # 2.23: the Studio's 32 windows a step, in batches of 8
    assert row["micro_batch"] * row["accumulation_steps"] == bench.BATCH * bench.ACCUMULATION
    assert report["settings"]["accumulation_steps"] == bench.ACCUMULATION == 4
    assert "batches of 8 windows, 4 of them a step, Muon." in bench.to_markdown(report)
    assert report["settings"]["text"] == {"source": "built-in",
                                          "chars": len(bench.bench_text())}
    assert report["hardware"]["gflops"] > 0 and report["hardware"]["threads"] >= 1
    assert report["environment"]["auralite"] == __version__
    assert report["finetune"] is None and report["seconds"] > 0
    assert report["_lines"][0].startswith("The computer: ")
    assert any(line.startswith("tiny: 4 layers x 128: building the model")
               for line in report["_lines"])


def test_the_report_in_markdown(report):
    text = bench.to_markdown(report)
    hw = report["hardware"]
    assert text.startswith(f"# AuraLite benchmark: {hw['cpu']}\n")
    assert "| Computer | |\n|---|---|\n| Processor | " in text
    assert f"| Speed test | {hw['gflops']:,.0f} GFLOP/s" in text
    assert "## Training a new model on the processor" in text
    assert ("| Model | Parameters | Training, tokens/s | In 8 hours | Memory used (estimate) "
            "| Writing, tokens/s |") in text
    row = report["train"][0]
    assert f"| tiny: 4 layers x 128 | 0.9M | {row['tokens_per_second']:,.0f} | " in text
    assert "One pass over the text" not in text and "Fine-tuning" not in text
    assert text.isascii() or not hw["cpu"].isascii()
    assert bench.summary_lines(report) == [
        f"tiny: 4 layers x 128: 0.9M parameters - {row['tokens_per_second']:,.0f} tokens/s "
        f"training, {row['generate_tokens_per_second']:,.0f} tokens/s writing"]


def test_the_report_files(report, tmp_path, monkeypatch):
    md, js = bench.save_report(report, tmp_path / "out")
    assert md.parent == js.parent == tmp_path / "out"
    assert md.name == bench.file_stem(report) + ".md" and js.suffix == ".json"
    assert md.read_text(encoding="utf-8") == bench.to_markdown(report)
    assert json.loads(js.read_text(encoding="utf-8")) == json.loads(json.dumps(report))
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    assert bench.default_folder() == tmp_path / "home" / "bench"
    md2, _ = bench.save_report(report)
    assert md2.parent == tmp_path / "home" / "bench"


@pytest.mark.parametrize("cpu, node, stem", [
    ("Intel(R) Core(TM) i5-2400 CPU @ 3.10GHz", "PC", "bench-i5-2400-2026-11-08-14-05"),
    ("Intel(R) Core(TM) i3-2100 CPU @ 3.10GHz", "PC", "bench-i3-2100-2026-11-08-14-05"),
    ("AMD Ryzen 5 5600X 6-Core Processor", "PC", "bench-ryzen-5-5600x-2026-11-08-14-05"),
    ("Apple M2 Pro", "mac", "bench-apple-m2-pro-2026-11-08-14-05"),
    ("Intel(R) Xeon(R) Processor @ 2.60GHz", "build.box", "bench-build-box-2026-11-08-14-05"),
    ("", "\u041f\u041a", "bench-computer-2026-11-08-14-05"),     # a name in Cyrillic
])
def test_a_reports_file_is_named_by_its_processor(cpu, node, stem):
    report = {"hardware": {"cpu": cpu}, "environment": {"node": node},
              "date": "2026-11-08 14:05"}
    assert bench.file_stem(report) == stem


def test_a_text_of_ones_own(tmp_path):
    pytest.importorskip("torch")
    story = tmp_path / "story.txt"
    story.write_text(bench.bench_text(chars=30_000, seed=7), encoding="utf-8")
    result = bench.run_bench(sizes=["tiny"], seconds=1.5, generate_tokens=4, data=str(story),
                             isolate=False)
    assert result["settings"]["text"]["source"] == "story.txt"
    assert result["settings"]["isolated"] is False
    assert result["train"][0]["text_tokens"] > 1000
    text = bench.to_markdown(result)
    assert "on `story.txt`" in text and "| One pass over the text |" in text
    short = tmp_path / "short.txt"
    short.write_text("Too short.", encoding="utf-8")
    with pytest.raises(bench.BenchError, match="too short"):
        bench.run_bench(sizes=["tiny"], seconds=1.0, data=str(short))
    dialogs = tmp_path / "chat.jsonl"
    dialogs.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"},
                                                {"role": "assistant", "content": "hello"}]})
                       + "\n", encoding="utf-8")
    with pytest.raises(bench.BenchError, match="plain text file"):
        bench.run_bench(sizes=["tiny"], seconds=1.0, data=str(dialogs))
    with pytest.raises(bench.BenchError, match="unknown size"):
        bench.run_bench(sizes=["tiny", "giant"], seconds=1.0)
    with pytest.raises(bench.BenchError, match="above 0"):
        bench.run_bench(sizes=["tiny"], seconds=0)


def test_a_stop_ends_the_benchmark():
    pytest.importorskip("torch")
    stop = threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        bench.run_bench(sizes=["tiny"], seconds=1.0, stop=stop)
    # a stop while a size trains in its own process ends that process
    stop = threading.Event()

    def stop_when_it_trains(line: str) -> None:
        if "building the model" in line:
            stop.set()

    started = time.monotonic()
    with pytest.raises(InterruptedError):
        bench.run_bench(sizes=["tiny"], seconds=60.0, progress=stop_when_it_trains, stop=stop)
    assert time.monotonic() - started < 50


def test_a_stage_in_a_process_of_its_own(tmp_path, capsys):
    pytest.importorskip("torch")
    from auralite.engine import tokenizer_io
    text = bench.bench_text(chars=20_000)
    tokenizer, info = tokenizer_io.train_tokenizer(text, vocab_size=300, val_split=0.0)
    (tmp_path / "text.txt").write_text(text, encoding="utf-8")
    tok = tokenizer_io.save_tokenizer(tokenizer, tmp_path / "tok.json", training=info)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"kind": "train", "name": "tiny", "text": str(tmp_path / "text.txt"),
                                "tokenizer": str(tok), "seconds": 1.0, "generate_tokens": 4,
                                "threads": 1}), encoding="utf-8")
    import torch
    threads = torch.get_num_threads()
    try:
        assert bench.child_main(spec) == 0
        assert torch.get_num_threads() == 1                  # the stage's threads
    finally:
        torch.set_num_threads(threads)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == bench.PROGRESS_MARK + "tiny: 4 layers x 128: building the model"
    result = json.loads(lines[-1][len(bench.RESULT_MARK):])
    assert lines[-1].startswith(bench.RESULT_MARK) and result["name"] == "tiny"
    assert result["generate_tokens"] == 4
    assert result["text_tokens"] == len(tokenizer.encode(text))     # the tokenizer it was given
    spec.write_text(json.dumps({"kind": "train", "name": "huge", "text": str(tmp_path / "text.txt"),
                                "tokenizer": str(tok), "seconds": 1.0, "generate_tokens": 4}),
                    encoding="utf-8")
    assert bench.child_main(spec) == 1
    assert capsys.readouterr().out.splitlines()[-1] == (
        bench.ERROR_MARK + "BenchError: unknown size 'huge': tiny, small, medium")
    # the parent: a stage that ends without a result is reported, the others go on
    row = bench._run_child({"kind": "train", "name": "huge", "text": str(tmp_path / "text.txt"),
                            "tokenizer": str(tok), "seconds": 1.0, "generate_tokens": 4},
                           tmp_path, lambda _line: None, None)
    assert row == {"error": "BenchError: unknown size 'huge': tiny, small, medium"}


def test_what_ended_a_stage_without_a_result():
    assert bench._child_error(-9, ["Killed"]) == "out of memory"
    assert bench._child_error(137, []) == "out of memory"
    assert bench._child_error(1, ["Traceback", "MemoryError"]) == "out of memory"
    assert bench._child_error(1, [bench.ERROR_MARK + "ValueError: no"]) == "ValueError: no"
    assert bench._child_error(3, ["", "boom", ""]) == "exit code 3: boom"
    assert bench._child_error(3, []) == "exit code 3"


def test_a_size_that_ran_out_of_memory_is_said_so():
    report = dict(FAKE, train=[dict(ROW), {"name": "medium",
                                           "label": "medium: 12 layers x 384 (preset 45m)",
                                           "error": "out of memory"}],
                  finetune={"model": "hf:Qwen/Qwen3.5-0.8B", "method": "qlora",
                            "error": "out of memory"})
    text = bench.to_markdown(report)
    assert "| medium: 12 layers x 384 (preset 45m) | - | out of memory | - | - | - |" in text
    assert "| hf:Qwen/Qwen3.5-0.8B | qlora | out of memory | - | - |" in text
    assert bench.summary_lines(report)[1:] == [
        "medium: 12 layers x 384 (preset 45m): out of memory",
        "Fine-tuning hf:Qwen/Qwen3.5-0.8B (qlora): out of memory"]


def test_fine_tuning_a_ready_model(tmp_path):
    pytest.importorskip("torch")
    from tests.test_hf_v2181 import write_hf_folder
    write_hf_folder(tmp_path / "tiny-llama", "llama")
    result = bench.run_bench(sizes=["tiny"], seconds=1.5, generate_tokens=4,
                             finetune=str(tmp_path / "tiny-llama"), method="lora",
                             isolate=False)
    ft = result["finetune"]
    assert ft["method"] == "lora" and ft["architecture"] == "LlamaForCausalLM"
    assert ft["tokens_per_second"] > 0 and ft["optimizer_steps"] >= 1
    assert ft["seconds_per_step"] is not None and ft["memory_used"] >= 0
    text = bench.to_markdown(result)
    assert "## Fine-tuning a ready model" in text and "| tiny-llama (" in text
    assert bench.summary_lines(result)[-1].startswith("Fine-tuning tiny-llama (lora): ")
    assert bench._model_name("hf:Qwen/Qwen3.5-0.8B") == "hf:Qwen/Qwen3.5-0.8B"
    # the same in a process of its own
    spec = tmp_path / "ft.json"
    spec.write_text(json.dumps({"kind": "finetune", "model": str(tmp_path / "tiny-llama"),
                                "method": "lora", "seconds": 1.0}), encoding="utf-8")
    row = bench._run_child(json.loads(spec.read_text(encoding="utf-8")), tmp_path,
                           lambda _line: None, None)
    assert row["method"] == "lora" and row["tokens_per_second"] > 0


# ----------------------------------------------------------------------
#  2.23: a model already on this computer
# ----------------------------------------------------------------------
def test_finetune_source_names_what_it_was_given(tmp_path):
    assert bench.finetune_source("hf:Qwen/Qwen3.5-0.8B") == "hub"
    assert bench.finetune_source(" HF:org/model@main ") == "hub"
    folder = tmp_path / "Qwen3.5-0.8B"
    folder.mkdir()
    with pytest.raises(bench.BenchError, match="no config.json - not a downloaded Hugging Face"):
        bench.finetune_source(str(folder))
    (folder / "config.json").write_text("{}", encoding="utf-8")
    assert bench.finetune_source(str(folder)) == "folder"
    for name in ("chat.pt", "chat.safetensors", "CHAT.PT"):
        (tmp_path / name).write_bytes(b"x")
        assert bench.finetune_source(str(tmp_path / name)) == "checkpoint"
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(bench.BenchError, match="notes.txt: the benchmark fine-tunes a Hugging"):
        bench.finetune_source(str(tmp_path / "notes.txt"))
    with pytest.raises(bench.BenchError, match="no such folder or file"):
        bench.finetune_source(str(tmp_path / "missing"))
    with pytest.raises(bench.BenchError, match="--finetune needs a model"):
        bench.finetune_source("  ")


def test_a_wrong_model_path_fails_before_the_sizes(monkeypatch, tmp_path, capsys):
    pytest.importorskip("torch")
    from auralite import cli
    from auralite.studio import hardware
    stages: list[dict] = []
    monkeypatch.setattr(hardware, "measure_gflops", lambda *a, **k: 10.0)
    monkeypatch.setattr(bench, "_run_child",
                        lambda spec, folder, say, stop: stages.append(spec) or {"error": "x"})
    with pytest.raises(bench.BenchError, match="no such folder or file"):
        bench.run_bench(sizes=["tiny"], seconds=1.0, finetune=str(tmp_path / "missing"))
    assert stages == []                                  # not after minutes of the sizes
    assert cli.main(["bench", "--sizes", "tiny", "--finetune", str(tmp_path / "missing"),
                     "--out", str(tmp_path)]) == 2
    assert "no such folder or file" in capsys.readouterr().err and stages == []


def test_the_command_line_takes_a_models_path(monkeypatch, tmp_path, capsys):
    from auralite import cli
    seen: list[dict] = []
    monkeypatch.setattr(bench, "run_bench", lambda **options: seen.append(options) or dict(FAKE))
    folder = tmp_path / "Qwen3.5-0.8B"
    assert cli.main(["bench", "--quick", "--out", str(tmp_path), "--finetune", str(folder),
                     "--method", "qlora"]) == 0
    assert seen[0]["finetune"] == str(folder) and seen[0]["method"] == "qlora"
    assert f"then fine-tuning {folder}" in capsys.readouterr().err


def test_fine_tuning_an_auralite_checkpoint(tmp_path):
    """A model trained or imported before: its .pt file, loaded as the Studio opens it."""
    pytest.importorskip("torch")
    from auralite.engine import AuraLiteEngine
    from tests.test_hf_v2181 import write_hf_folder
    write_hf_folder(tmp_path / "tiny-llama", "llama")
    engine = AuraLiteEngine()
    engine.import_hf_model(str(tmp_path / "tiny-llama"))
    engine.save_model(str(tmp_path / "tiny-chat.pt"))
    lines: list[str] = []
    ft = bench._finetune(str(tmp_path / "tiny-chat.pt"), "lora", 1.5, lines.append, None)
    assert ft["architecture"] == "AuraLite checkpoint" and ft["method"] == "lora"
    assert ft["tokens_per_second"] > 0 and ft["optimizer_steps"] >= 1 and ft["params"] > 0
    assert lines[0] == f"Fine-tuning: loading {tmp_path / 'tiny-chat.pt'} (lora)"


def test_fine_tuning_a_downloaded_qwen35_folder(tmp_path, monkeypatch):
    """Qwen3.5 as Hugging Face ships it (the vision tower beside the text): QLoRA in the
    benchmark, its linear attention included - imported in 4 bits (never whole in float32)."""
    pytest.importorskip("torch")
    from auralite.engine import AuraLiteEngine
    from tests.test_qwen35_v2230 import _write_qwen35
    _write_qwen35(tmp_path / "Qwen3.5-tiny", multimodal=True)
    imports: list[dict] = []
    real_import = AuraLiteEngine.import_hf_model

    def import_hf_model(self, source, **options):
        imports.append(options)
        return real_import(self, source, **options)

    monkeypatch.setattr(AuraLiteEngine, "import_hf_model", import_hf_model)
    ft = bench._finetune(str(tmp_path / "Qwen3.5-tiny"), "qlora", 2.0, lambda _line: None, None)
    assert imports[0]["quantize"] == "nf4"
    assert ft["architecture"] == "Qwen3_5ForConditionalGeneration" and ft["method"] == "qlora"
    assert ft["tokens_per_second"] > 0 and ft["optimizer_steps"] >= 1


# ----------------------------------------------------------------------
#  The command line
# ----------------------------------------------------------------------
def test_auralite_bench(tmp_path, capsys, monkeypatch):
    pytest.importorskip("torch")
    from auralite import cli
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    out_dir = tmp_path / "reports"
    assert cli.main(["bench", "--quick", "--sizes", "tiny", "--seconds", "1.5",
                     "--out", str(out_dir)]) == 0
    captured = capsys.readouterr()
    assert captured.err.startswith("AuraLite bench: tiny - 1.5 s of training each; about ")
    assert "The tokenizer: BPE 512 on the text, once for every size" in captured.err
    assert "# AuraLite benchmark: " in captured.out and "| tiny: 4 layers x 128 | 0.9M |" in \
        captured.out
    assert "[AuraLite]" not in captured.out                  # the engine's lines stay out
    (md,) = out_dir.glob("bench-*.md")
    assert f"Saved: {md}" in captured.out and md.with_suffix(".json").is_file()
    assert cli.main(["bench", "--quick", "--sizes", "tiny", "--seconds", "1", "--json",
                     "--quiet", "--out", str(out_dir)]) == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["train"][0]["name"] == "tiny" and data["settings"]["seconds"] == 1.0
    assert captured.err.startswith("saved: ")
    assert cli.main(["bench", "--sizes", "huge"]) == 2
    assert "error: unknown size huge" in capsys.readouterr().err


def test_the_command_lines_defaults():
    from auralite import cli
    parser = cli.build_parser()
    args = parser.parse_args(["bench", "--finetune"])
    assert args.finetune == bench.FINETUNE_MODEL and args.func is cli.cmd_bench
    args = parser.parse_args(["bench"])
    assert args.finetune is None and not args.quick and args.method is None


# ----------------------------------------------------------------------
#  The Studio: Tools -> Benchmark
# ----------------------------------------------------------------------
@pytest.fixture()
def studio(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from tests._integration import require
    QtWidgets = require("PySide6.QtWidgets")
    QtCore = require("PySide6.QtCore")
    from auralite.engine import AuraLiteEngine
    from auralite.studio.qt import common
    from auralite.studio.qt.window import MainWindow
    from auralite.studio.settings import StudioSettings
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    seen: dict[str, list] = {"inform": [], "confirm": [], "error": [], "bench": []}
    answers: dict = {"confirm": True, "bench": lambda start: start}
    common.DIALOGS.update(error=lambda friendly: seen["error"].append(friendly),
                          inform=lambda title, text: seen["inform"].append((title, text)),
                          confirm=lambda title, text: seen["confirm"].append((title, text))
                          or answers["confirm"],
                          bench=lambda start: seen["bench"].append(start)
                          or answers["bench"](start),
                          save=lambda title: "", open=lambda title: "", dir=lambda title: "")
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    window = MainWindow(engine=AuraLiteEngine(), settings=StudioSettings(first_run_done=True),
                        detect=False, capture_streams=False, save_on_close=False,
                        auto_speed_test=False)
    window.seen, window.answers, window.qapp = seen, answers, app   # type: ignore[attr-defined]
    yield window
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()
    common.DIALOGS.clear()


def _wait(app, condition, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while not condition():
        if time.time() > deadline:
            raise AssertionError("timed out waiting for the window")
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


ROW = {"name": "small", "label": "small: 8 layers x 256 (preset 25m)", "preset": "25m",
       "params": 6_561_024, "layers": 8, "width": 256, "tokens_per_second": 1061.0,
       "tokens": 63_488, "optimizer_steps": 31, "batches": 31, "seconds": 60.0,
       "micro_batch": 8, "accumulation_steps": 1, "gradient_checkpointing": False,
       "memory_used": 515 << 20, "memory_estimate": 522 << 20, "memory_start": 800 << 20,
       "memory_peak": 1315 << 20, "night_tokens": 30_556_800, "text_tokens": 56_485,
       "generate_tokens": 128, "generate_tokens_per_second": 95.0}
FAKE = {"format": 1, "date": "2026-11-08 14:05",
        "hardware": {"cpu": "Intel(R) Core(TM) i5-2400 CPU @ 3.10GHz", "cores": 4, "threads": 4,
                     "simd": "AVX", "memory_total": 16 << 30, "memory_available": 12 << 30,
                     "system": "Windows 10 (build 19045)", "train_threads": 4, "gflops": 74.0},
        "environment": {"auralite": "2.22.0", "python": "3.13.9", "torch": "2.15.0+cpu",
                        "torch_threads": 4, "app": True, "app_platform": "windows-x64",
                        "node": "pc"},
        "settings": {"seconds": 60.0, "seq_length": 256, "batch_size": 8, "vocab": 512,
                     "generate_tokens": 128, "text": {"source": "built-in", "chars": 200_000}},
        "train": [ROW], "finetune": None, "seconds": 290.0}


def test_a_report_of_a_pc_from_2011():
    text = bench.to_markdown(FAKE)
    assert text.startswith("# AuraLite benchmark: Intel(R) Core(TM) i5-2400 CPU @ 3.10GHz\n\n"
                           "2026-11-08 14:05 - AuraLite 2.22.0 (the app for windows-x64), "
                           "Python 3.13.9, PyTorch 2.15.0+cpu\n")
    assert "| Processor | Intel(R) Core(TM) i5-2400 CPU @ 3.10GHz - 4 cores / 4 threads, AVX |" \
        in text
    assert "| Memory | 16 GB (12 GB free at the start) |" in text
    assert "| Speed test | 74 GFLOP/s (float32 matrix product) |" in text
    assert ("| small: 8 layers x 256 (preset 25m) | 7M | 1,061 | 30.6M tokens | "
            "0.50 GB (0.51 GB) | 95 |") in text
    assert "in 4.8 min" in text and text.isascii()


def test_tools_benchmark_runs_it_and_says_the_result(studio, monkeypatch, tmp_path):
    app = studio.qapp
    seen = studio.seen
    calls: list[dict] = []

    def fake_run(**options):
        calls.append(options)
        options["progress"]("small: 8 layers x 256 (preset 25m): building the model")
        return dict(FAKE)

    monkeypatch.setattr(bench, "run_bench", fake_run)
    monkeypatch.setattr(bench, "default_folder", lambda: tmp_path / "bench")
    studio.answers["bench"] = lambda start: None            # Cancel
    studio.bench_action.trigger()
    assert seen["bench"][-1].seconds == 60.0 and seen["bench"][-1].finetune is None
    assert not calls and studio.busy == ""
    studio.answers["bench"] = lambda start: start           # Start with the defaults
    studio.bench_action.trigger()
    assert studio.busy == "benchmark" and studio.bench_action.text() == "Stop the benchmark"
    _wait(app, lambda: studio.bench_job is None)
    assert calls[0]["sizes"] == list(bench.SIZES) and calls[0]["stop"] is not None
    assert calls[0]["seconds"] == 60.0 and calls[0]["finetune"] is None
    assert calls[0]["method"] == ""
    title, text = seen["inform"][-1]
    assert title == "Benchmark" and text.startswith(
        "small: 8 layers x 256 (preset 25m): 7M parameters - 1,061 tokens/s training, "
        "95 tokens/s writing")
    md = tmp_path / "bench" / "bench-i5-2400-2026-11-08-14-05.md"
    assert f"The report, a table for an article or a README: {md}" in text and md.is_file()
    assert studio.busy == "" and studio.bench_action.text() == "Benchmark…"


def test_tools_benchmark_stops(studio, monkeypatch, tmp_path):
    app = studio.qapp
    started = threading.Event()

    def slow_run(**options):
        started.set()
        if not options["stop"].wait(30):
            raise AssertionError("never stopped")
        raise InterruptedError("the benchmark was stopped")

    monkeypatch.setattr(bench, "run_bench", slow_run)
    studio.bench_action.trigger()
    assert started.wait(10)
    studio.bench_action.trigger()                        # "Stop the benchmark"
    _wait(app, lambda: studio.bench_job is None)
    assert studio.seen["error"] == [] and studio.busy == ""
    assert studio.statusBar().currentMessage() == "The benchmark was stopped"
    assert studio.bench_action.text() == "Benchmark…"


def test_tools_benchmark_fine_tunes_a_model_already_downloaded(studio, monkeypatch, tmp_path):
    """2.23: the dialog's model - a downloaded folder here - reaches the benchmark, and the
    next benchmark starts from the same settings."""
    from auralite.studio.qt.bench_dialog import BenchChoice
    app = studio.qapp
    calls: list[dict] = []
    monkeypatch.setattr(bench, "run_bench", lambda **options: calls.append(options)
                        or dict(FAKE, finetune={"model": str(tmp_path / "Qwen3.5-0.8B"),
                                                "architecture": "Qwen3_5ForConditionalGeneration",
                                                "params": 752_393_024, "method": "qlora",
                                                "tokens_per_second": 12.5, "tokens": 750,
                                                "optimizer_steps": 3, "seconds_per_step": 10.0,
                                                "seconds": 30.0, "micro_batch": 8,
                                                "accumulation_steps": 1,
                                                "memory_used": 3 << 30}))
    monkeypatch.setattr(bench, "default_folder", lambda: tmp_path / "bench")
    chosen = BenchChoice(seconds=30.0, finetune=str(tmp_path / "Qwen3.5-0.8B"), method="qlora")
    studio.answers["bench"] = lambda start: chosen
    studio.bench_action.trigger()
    _wait(app, lambda: studio.bench_job is None)
    assert calls[0]["seconds"] == 30.0 and calls[0]["method"] == "qlora"
    assert calls[0]["finetune"] == str(tmp_path / "Qwen3.5-0.8B")
    assert studio.seen["inform"][-1][1].split("\n")[1] == \
        "Fine-tuning Qwen3.5-0.8B (qlora): 12 tokens/s, 3.00 GB"
    studio.answers["bench"] = lambda start: None
    studio.bench_action.trigger()
    assert studio.seen["bench"][-1] == chosen               # the last settings come back


def test_the_benchmark_dialog(studio, tmp_path):
    """2.23: the catalog's Qwen3.5 0.8B or a model on this computer; Start only for a path
    that is a model."""
    from auralite.studio.qt import common
    from auralite.studio.qt.bench_dialog import BenchChoice, BenchDialog, describe
    dialog = BenchDialog(studio)
    ok = dialog.buttons.button(dialog.buttons.StandardButton.Ok)
    assert dialog.seconds.value() == 60 and not dialog.finetune.isChecked()
    assert not dialog.group.isEnabled() and dialog.model_spec() is None and ok.isEnabled()
    assert dialog.choice() == BenchChoice(seconds=60.0)
    assert dialog.estimate.text() == f"About {bench.estimated_minutes(list(bench.SIZES), 60.0):,.0f} minutes"
    dialog.finetune.setChecked(True)
    assert dialog.from_catalog.isChecked() and dialog.catalog.isEnabled()
    assert dialog.model_spec() == bench.FINETUNE_MODEL == "hf:Qwen/Qwen3.5-0.8B"
    assert dialog.catalog.currentText().startswith("Qwen3.5 0.8B - 752M, 1.8 GB to download")
    assert "(and the download, the first time)" in dialog.estimate.text()
    assert not dialog.path.isEnabled() and ok.isEnabled()
    # a path: typing it chooses "already downloaded"
    dialog.path.setText(str(tmp_path / "nothing-here"))
    assert dialog.from_disk.isChecked() and not dialog.catalog.isEnabled()
    assert "no such folder or file" in dialog.problem.text() and not ok.isEnabled()
    folder = tmp_path / "Qwen3.5-0.8B"
    folder.mkdir()
    dialog.path.setText(str(folder))
    assert "no config.json" in dialog.problem.text() and not ok.isEnabled()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    dialog.seconds.setValue(30)
    assert dialog.problem.text() == "" and ok.isEnabled()
    dialog.method.setCurrentIndex(dialog.method.findData("qlora"))
    assert dialog.choice() == BenchChoice(seconds=30.0, finetune=str(folder), method="qlora")
    assert describe(dialog.choice()) == "30 s per size, fine-tuning Qwen3.5-0.8B (qlora)"
    # Folder... and File... ask through the Studio's dialogs
    checkpoint = tmp_path / "my-model.pt"
    checkpoint.write_bytes(b"x")
    common.DIALOGS["open"] = lambda title: str(checkpoint)
    dialog.from_catalog.setChecked(True)
    dialog.pick_file.setEnabled(True)
    dialog.pick_file.click()
    assert dialog.from_disk.isChecked() and dialog.model_spec() == str(checkpoint)
    assert dialog.problem.text() == ""
    common.DIALOGS["dir"] = lambda title: str(folder)
    dialog.pick_folder.click()
    assert dialog.model_spec() == str(folder)
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    dialog.path.setText(str(tmp_path / "notes.txt"))
    assert "a Hugging Face model folder or an AuraLite checkpoint" in dialog.problem.text()
    dialog.close()
    # the dialog opens with the settings it was given
    again = BenchDialog(studio, BenchChoice(seconds=45.0, finetune="hf:Qwen/Qwen3.5-2B",
                                            method="lora"))
    assert again.finetune.isChecked() and again.from_catalog.isChecked()
    assert again.model_spec() == "hf:Qwen/Qwen3.5-2B" and again.seconds.value() == 45
    assert again.choice().method == "lora"
    again.close()
    local = BenchDialog(studio, BenchChoice(seconds=60.0, finetune=str(folder)))
    assert local.from_disk.isChecked() and local.path.text() == str(folder)
    local.close()
    assert describe(BenchChoice(seconds=60.0, finetune="hf:Qwen/Qwen3.5-0.8B")) == \
        "60 s per size, fine-tuning Qwen3.5 0.8B"


# ----------------------------------------------------------------------
#  The app, the memory plan
# ----------------------------------------------------------------------
def test_the_apps_readme_and_smoke_test_know_the_benchmark():
    build = _load("build_app")
    info = {"app": "AuraLite Studio", "version": "2.22.0", "platform": "windows-x64",
            "python": "3.13.9", "torch": "2.15.0+cpu", "qt": "6.9.3", "built": "2026-11-08",
            "commit": "abc1234"}
    assert ".\\auralite.exe bench                        how fast this computer trains" in \
        build.readme_text(info, "win32")
    smoke = _load("smoke_app")
    assert smoke.STEPS[-1] == "bench" and callable(smoke.Smoke.step_bench)


def test_the_memory_plan_keeps_a_margin(monkeypatch):
    """The field case: the Studio's 7M run (batches of 32 windows of 256) with 1.15 GB free
    planned batches of 16 at 85 % of it, and a computer with 2 GB killed it for want of
    memory; at 70 % it plans batches of 8 x 4 steps (0.54 GB, measured 0.51 GB)."""
    pytest.importorskip("torch")
    from auralite.engine import AuraLiteEngine, trainer
    from auralite.engine import sizing
    assert trainer._MEMORY_MARGIN == 0.7
    monkeypatch.setattr(sizing, "available_memory", lambda device="cpu": int(1.15 * (1 << 30)))
    engine = AuraLiteEngine()
    params = dict(bench.size_dims("small"), seq_length=256, batch_size=32, epochs=1,
                  max_steps=1, val_split=0.0, memory_plan="auto", tokenizer="char",
                  progress_every_seconds=100, seed=0)
    summary = engine.train(bench.bench_text(chars=60_000), params)
    assert (summary["batch_size"], summary["accumulation_steps"]) == (8, 4)


def test_python_m_auralite_bench_is_the_command(monkeypatch):
    seen = []
    from auralite import cli
    monkeypatch.setattr(cli, "main", lambda argv: seen.append(argv) or 0)
    assert bench._main(["--quick"]) == 0 and seen == [["bench", "--quick"]]
    monkeypatch.setattr(sys, "argv", ["bench.py", "--sizes", "tiny"])
    assert bench._main() == 0 and seen[-1] == ["bench", "--sizes", "tiny"]


def test_the_docs_say_how_to_measure():
    doc = (REPO / "docs" / "benchmark.md").read_text(encoding="utf-8")
    assert "auralite bench --quick" in doc and "--finetune" in doc
    assert "| tiny | 4 × 128 | 0.9M |" in doc and "| medium | 12 × 384 | 19M | 45m |" in doc
    assert "auralite bench" in (REPO / "README.md").read_text(encoding="utf-8")
    assert ".\\auralite.exe bench" in (REPO / "docs" / "windows-app.md").read_text(
        encoding="utf-8")
    assert "Tools → Benchmark…" in (REPO / "docs" / "studio.md").read_text(encoding="utf-8")
    assert "## 26. How fast is my computer for AuraLite?" in (
        REPO / "docs" / "faq.md").read_text(encoding="utf-8")


def test_a_size_that_fails_does_not_end_the_benchmark(monkeypatch):
    pytest.importorskip("torch")
    stages: list[dict] = []

    def fake_child(spec, folder, say, stop):
        stages.append(spec)
        return {"error": "out of memory"} if spec["kind"] == "train" else {"error": "no network"}

    monkeypatch.setattr(bench, "_run_child", fake_child)
    lines: list[str] = []
    report = bench.run_bench(sizes=["tiny", "small"], seconds=1.0, finetune="hf:Org/Model",
                             method="qlora", progress=lines.append)
    assert [row["error"] for row in report["train"]] == ["out of memory", "out of memory"]
    assert report["train"][1] == {"name": "small", "label": "small: 8 layers x 256 (preset 25m)",
                                  "error": "out of memory"}
    assert report["finetune"] == {"model": "hf:Org/Model", "method": "qlora",
                                  "error": "no network"}
    assert "  tiny: out of memory" in lines and "  fine-tuning: no network" in lines
    assert [s["kind"] for s in stages] == ["train", "train", "finetune"]
    assert stages[0]["tokenizer"].endswith("bench-tokenizer.json")


@pytest.mark.parametrize("memory, method", [(4 << 30, "qlora"), (16 << 30, "lora")])
def test_fine_tuning_takes_qlora_below_8_gb(monkeypatch, memory, method):
    pytest.importorskip("torch")
    from auralite.studio import hardware
    real = hardware.detect_hardware

    def detect(**kwargs):
        hw = real(**kwargs)
        hw.memory_total = memory
        return hw

    seen: list[dict] = []
    monkeypatch.setattr(hardware, "detect_hardware", detect)
    monkeypatch.setattr(hardware, "measure_gflops", lambda *a, **k: 10.0)
    monkeypatch.setattr(bench, "_run_child",
                        lambda spec, folder, say, stop: seen.append(spec) or {"error": "x"})
    bench.run_bench(sizes=["tiny"], seconds=1.0, finetune="hf:Org/Model")
    assert seen[-1]["kind"] == "finetune" and seen[-1]["method"] == method


def test_the_benchmark_leaves_the_threads_as_they_were(monkeypatch):
    torch = pytest.importorskip("torch")
    before = torch.get_num_threads()
    seen = []
    monkeypatch.setattr(bench, "_run", lambda *args: seen.append(torch.get_num_threads()) or {})
    assert bench.run_bench(sizes=["tiny"], threads=1) == {}
    assert seen == [1] and torch.get_num_threads() == before


def test_quick_is_two_sizes_for_15_seconds(monkeypatch, tmp_path, capsys):
    from auralite import cli
    seen: list[dict] = []
    monkeypatch.setattr(bench, "run_bench", lambda **options: seen.append(options) or dict(FAKE))
    assert cli.main(["bench", "--quick", "--out", str(tmp_path)]) == 0
    assert seen[0]["sizes"] == ["tiny", "small"] and seen[0]["seconds"] == 15.0
    assert seen[0]["generate_tokens"] == 32 and seen[0]["finetune"] is None
    assert cli.main(["bench", "--out", str(tmp_path), "--finetune", "--method", "lora"]) == 0
    assert seen[1]["sizes"] == ["tiny", "small", "medium"] and seen[1]["seconds"] == 60.0
    assert seen[1]["finetune"] == "hf:Qwen/Qwen3.5-0.8B" and seen[1]["method"] == "lora"
    assert "then fine-tuning hf:Qwen/Qwen3.5-0.8B" in capsys.readouterr().err


def test_a_child_keeps_the_engines_lines_to_itself(tmp_path, monkeypatch, capsys):
    """`auralite bench --child SPEC` prints marked lines only: the engine's own (model built,
    memory, ...) would bury them in the parent's reading."""
    import logging

    from auralite import cli
    seen = []
    monkeypatch.setattr(bench, "child_main",
                        lambda spec: seen.append(logging.getLogger("auralite").level) or 0)
    level = logging.getLogger("auralite").level
    try:
        assert cli.main(["bench", "--child", str(tmp_path / "spec.json")]) == 0
        assert seen == [logging.WARNING]
    finally:
        logging.getLogger("auralite").setLevel(level)
