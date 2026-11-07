"""2.19: AuraLite Studio's toolkit-free core (no Qt needed).

What the computer can do (hardware), which base models fit it and how long a
run takes (catalog, advisor), what a data file holds (datacheck), training
parameters from a few choices (recipes), errors explained (errors), progress
lines, settings, the update check, the log file, `auralite doctor` and the
Studio's start-up check.
"""
from __future__ import annotations

import errno
import io
import json
import logging
import os
import platform
import struct
import subprocess
import sys
import threading
import urllib.error

import pytest

from auralite.studio import (CATALOG, DataReport, FineTuneSettings, Hardware, SettingsError,
                             TrainSettings, detect_hardware, estimate_finetune, explain_error,
                             find_model, finetune_params, inspect_data, measure_gflops,
                             memory_budget, pretrain_params, recommend_finetune,
                             recommend_pretrain, tokens_per_second)
from auralite.studio.advisor import (REFERENCE_TOKENS, TIGHT, suggested_epochs,
                                     suggested_seq_length)
from auralite.studio.datacheck import estimate_tokens, text_script
from auralite.studio.progress import ProgressTracker, format_duration, summary_text
from auralite.studio.settings import StudioSettings, load_settings, save_settings
from auralite.studio.updates import UpdateInfo, check_for_update, version_tuple

GB = 1 << 30
#: "Hello, world! " in Russian (written as escapes: the repository stays ASCII-only)
HELLO_RU = "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! "


def _hw(memory_gb: float, *, system: str = "Linux", gflops: float | None = 250.0,
        gpu_gb: float = 0.0, cores: int = 4) -> Hardware:
    return Hardware(system=system, cpu="test CPU", cores=cores, threads=cores, simd="AVX2",
                    x86_64_v2=True, memory_total=int(memory_gb * GB),
                    memory_available=int(memory_gb * GB * 0.8), train_threads=cores,
                    accelerator="cuda" if gpu_gb else "cpu", gpu="Test GPU" if gpu_gb else "",
                    gpu_memory=int(gpu_gb * GB), gflops=gflops)


def _model(repo: str):
    model = find_model(repo)
    assert model is not None, repo
    return model


def _speed(params: int, hw: Hardware, **kwargs) -> float:
    speed = tokens_per_second(params, hw, **kwargs)
    assert speed is not None
    return speed


def _dialogs(path, n: int, text: str = "What is {a} plus {b}?") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            a, b = i % 7, i % 5
            f.write(json.dumps({"messages": [
                {"role": "user", "content": text.format(a=a, b=b)},
                {"role": "assistant", "content": f"{a} plus {b} is {a + b}."}]}) + "\n")


# ----------------------------------------------------------------------
#  Catalog
# ----------------------------------------------------------------------
#: Parameter counts from the models' cards / safetensors metadata.
PUBLISHED = {"HuggingFaceTB/SmolLM2-135M-Instruct": 134.5e6,
             "HuggingFaceTB/SmolLM2-360M-Instruct": 361.8e6,
             "HuggingFaceTB/SmolLM2-1.7B-Instruct": 1.71e9,
             "Qwen/Qwen2.5-0.5B-Instruct": 0.494e9, "Qwen/Qwen2.5-1.5B-Instruct": 1.54e9,
             "Qwen/Qwen2.5-3B-Instruct": 3.09e9, "Qwen/Qwen2.5-7B-Instruct": 7.62e9,
             "Qwen/Qwen3-0.6B": 0.596e9, "Qwen/Qwen3-1.7B": 1.72e9, "Qwen/Qwen3-4B": 4.02e9,
             "Qwen/Qwen3-8B": 8.19e9, "meta-llama/Llama-3.2-1B-Instruct": 1.24e9,
             "meta-llama/Llama-3.2-3B-Instruct": 3.21e9,
             "meta-llama/Llama-3.1-8B-Instruct": 8.03e9,
             "mistralai/Mistral-7B-Instruct-v0.3": 7.25e9}


def test_catalog_dimensions_give_the_published_sizes():
    assert {m.repo for m in CATALOG} == set(PUBLISHED)
    for model in CATALOG:
        assert abs(model.params - PUBLISHED[model.repo]) / PUBLISHED[model.repo] < 0.01, model
        assert model.family in ("llama", "mistral", "qwen2", "qwen3")
        assert 0 < model.hidden_params < model.params
        assert model.compute_params >= model.hidden_params
        assert model.spec == f"hf:{model.repo}" and model.download_gb > 0
    sizes = [m.params for m in CATALOG]
    assert sizes == sorted(sizes)                    # offered from small to big
    gated = {m.repo for m in CATALOG if m.gated}
    assert gated == {r for r in PUBLISHED if r.startswith("meta-llama/")}
    assert not _model("Qwen/Qwen2.5-3B-Instruct").commercial
    assert not _model("mistralai/Mistral-7B-Instruct-v0.3").gguf


def test_find_model_takes_specs_in_any_case():
    model = find_model("Qwen/Qwen2.5-0.5B-Instruct")
    assert model is not None
    assert find_model("hf:qwen/qwen2.5-0.5b-instruct@main") is model
    assert find_model(" HF:Qwen/Qwen2.5-0.5B-Instruct ") is model
    assert find_model("someone/unknown") is None and find_model("") is None


# ----------------------------------------------------------------------
#  Hardware
# ----------------------------------------------------------------------
def test_detect_hardware_sees_this_computer():
    hw = detect_hardware()
    assert hw.threads >= 1 and 1 <= hw.cores <= hw.threads and hw.train_threads >= 1
    assert hw.accelerator in ("cpu", "cuda", "mps")
    if sys.platform.startswith("linux"):
        assert hw.memory_total > 0 and hw.cpu and hw.simd
    lines = hw.summary_lines()
    assert any(line.startswith("Memory:") for line in lines)
    assert Hardware.from_dict({**hw.to_dict(), "unknown": 1}) == hw


def test_cpu_features_and_gpu_kernels():
    import platform
    from types import SimpleNamespace

    from auralite.studio.hardware import _cuda_arch_supported, _simd
    if platform.machine().lower() in ("x86_64", "amd64", "i386", "i686"):
        assert _simd("flags\t: fpu sse4_2 popcnt avx avx2 avx512f\n") == ("AVX512", True)
        assert _simd("flags\t: fpu sse4_2 popcnt avx\n") == ("AVX", True)
        assert _simd("flags\t: fpu sse4_1 ssse3\n") == ("basic", False)
        assert _simd("flags\t: fpu sse4_2\n") == ("SSE4.2", False)        # no POPCNT
    torch = SimpleNamespace(cuda=SimpleNamespace(get_arch_list=lambda: ["sm_50", "sm_86",
                                                                       "compute_90"]))
    assert not _cuda_arch_supported(torch, 3, 5)                       # a Kepler card
    assert _cuda_arch_supported(torch, 8, 6) and _cuda_arch_supported(torch, 8, 9)
    assert _cuda_arch_supported(torch, 12, 0)                          # through the PTX
    assert not _cuda_arch_supported(SimpleNamespace(cuda=SimpleNamespace(
        get_arch_list=lambda: ["sm_70", "sm_80"])), 6, 1)


def test_measure_gflops_is_quick_and_positive():
    assert measure_gflops(seconds=0.1, size=96) > 0


def test_memory_budget_leaves_room_for_the_system():
    def budget(hw: Hardware) -> float:
        return pytest.approx(memory_budget(hw), abs=2)

    assert budget(_hw(16, system="Windows 10")) == 16 * GB * 0.8      # a fifth for Windows
    assert budget(_hw(4, system="Windows 10")) == 4 * GB - 1536 * (1 << 20)   # at least 1.5 GB
    assert budget(_hw(2)) == 2 * GB - 450 * (1 << 20)
    assert budget(_hw(64)) == 64 * GB * 0.9
    assert budget(_hw(16, gpu_gb=12)) == 12 * GB * 0.92


# ----------------------------------------------------------------------
#  Advisor
# ----------------------------------------------------------------------
def test_speed_follows_the_measured_speed_and_the_method():
    hw = _hw(16, gflops=200)
    lora = _speed(500_000_000, hw)
    assert lora == pytest.approx(0.45 * 200e9 / (4 * 500e6))
    assert _speed(500_000_000, hw, checkpointing=True) == pytest.approx(lora * 4 / 6)
    assert _speed(500_000_000, hw, lora=False) == pytest.approx(lora * 4 / 6)
    assert _speed(500_000_000, hw, qlora=True) < lora
    assert _speed(500_000_000, _hw(16, gflops=400)) == pytest.approx(2 * lora)
    guessed = _hw(16, gflops=None)
    assert _speed(500_000_000, guessed) > 0                      # a guess from the CPU
    assert tokens_per_second(500_000_000, _hw(16, gflops=None, gpu_gb=8)) is None


def test_the_calibration_run_is_reproduced():
    """2.18.1: Qwen2.5-0.5B QLoRA with checkpointing ran at 39 tokens/s on a 254 GFLOP/s
    CPU and peaked at 1.48 GB of 1.98 GB - the estimate must call that a (tight) fit."""
    qwen = _model("Qwen/Qwen2.5-0.5B-Instruct")
    sandbox = Hardware(system="Linux", cores=2, threads=2, simd="AVX512",
                       memory_total=int(1.98 * GB), train_threads=2, gflops=254.0)
    fit = estimate_finetune(qwen, "qlora", sandbox, seq_length=128)
    assert fit.checkpointing and fit.usable and fit.tokens_per_second is not None
    assert 30 <= fit.tokens_per_second <= 48
    assert 1.3 * GB < fit.memory < 1.8 * GB


def test_recommendations_grow_with_the_computer():
    small = recommend_finetune(_hw(4, system="Windows 10", gflops=70))
    mid = recommend_finetune(_hw(16, system="Windows 10", gflops=150))
    big = recommend_finetune(_hw(32, gflops=700))
    gpu = recommend_finetune(_hw(32, gpu_gb=24, gflops=150_000))
    chosen = [r.choice for r in (small, mid, big, gpu)]
    assert all(c is not None and c.usable for c in chosen)
    sizes = [c.model.params for c in chosen if c is not None]
    assert sizes == sorted(sizes) and sizes[0] < 1e9 and sizes[-1] > 7e9
    assert mid.choice is not None and mid.choice.model.params < 1e9
    assert mid.choice.method == "lora"              # a small model on a float base: it fits easily
    assert small.choice is not None and small.choice.method == "qlora"
    for rec in (small, mid, big, gpu):
        assert rec.choice is not None and rec.choice.memory <= TIGHT * rec.budget
        assert rec.settings["base"] == rec.choice.model.spec
        assert rec.settings["method"] in ("qlora", "lora")
        assert rec.choice.model.commercial and not rec.choice.model.gated
        assert len(rec.options) == len(CATALOG) and rec.lines()


def test_data_in_another_language_prefers_a_multilingual_model():
    hw = _hw(16, system="Windows 10", gflops=150)
    russian = DataReport(paths=["x"], kind="dialogs", records=1500, tokens=225_000,
                         longest_tokens=300, script="cyrillic")
    english = DataReport(paths=["x"], kind="dialogs", records=1500, tokens=225_000,
                         longest_tokens=300, script="latin")
    rec = recommend_finetune(hw, russian)
    assert rec.choice is not None and rec.choice.model.multilingual
    assert any("cyrillic" in w for w in rec.warnings)
    assert rec.settings["seq_length"] == 512 and rec.settings["epochs"] == 2
    assert rec.choice.tokens == 2 * 225_000
    assert recommend_finetune(hw, english).choice is not None


def test_a_bigger_english_model_loses_to_a_multilingual_one_for_cyrillic_data():
    russian = DataReport(paths=["x"], kind="dialogs", records=1500, tokens=225_000,
                         longest_tokens=300, script="cyrillic")
    rec = recommend_finetune(_hw(16, system="Windows 10", gflops=300), russian)
    smol = next(f for f in rec.options if f.model.repo == "HuggingFaceTB/SmolLM2-1.7B-Instruct")
    assert smol.usable and smol.hours is not None and smol.hours <= 8      # it would fit...
    assert rec.choice is not None and rec.choice.model.repo == "Qwen/Qwen2.5-1.5B-Instruct"
    assert smol.model.params > rec.choice.model.params                  # ...and is bigger


def test_a_comfortable_fit_beats_a_bigger_tight_one():
    sandbox = Hardware(system="Linux", cores=2, threads=2, simd="AVX512",
                       memory_total=int(1.98 * GB), train_threads=2, gflops=254.0)
    short = DataReport(paths=["x"], kind="dialogs", records=2000, tokens=70_000,
                       longest_tokens=60, script="latin")
    rec = recommend_finetune(sandbox, short)
    qwen = next(f for f in rec.options if f.model.repo == "Qwen/Qwen2.5-0.5B-Instruct")
    assert qwen.tight and not qwen.fits and qwen.usable
    assert rec.choice is not None and rec.choice.fits
    assert rec.choice.model.repo == "HuggingFaceTB/SmolLM2-360M-Instruct"


def test_a_computer_too_small_for_every_model_is_told_so():
    rec = recommend_finetune(_hw(0.9))
    assert rec.choice is None and rec.settings == {}
    assert rec.warnings and "needs about" in rec.warnings[0]
    assert any("none of the catalog" in line for line in rec.lines())


def test_a_tight_time_budget_picks_the_quickest_model():
    rec = recommend_finetune(_hw(16, gflops=150), max_hours=0.01)
    preferred = [f for f in rec.options if f.usable and f.model.commercial and not f.model.gated
                 and f.model.family != "qwen3" and f.model.gguf]
    assert rec.choice is not None and rec.choice.hours is not None
    assert rec.choice.hours == min(f.hours for f in preferred if f.hours is not None)
    assert any("longer than" in w for w in rec.warnings)


def test_window_and_passes_follow_the_data():
    dialogs = DataReport(paths=["x"], kind="dialogs", records=500, tokens=50_000,
                         longest_tokens=300)
    assert suggested_seq_length(None) == 512
    assert suggested_seq_length(dialogs) == 512
    assert suggested_seq_length(DataReport(paths=["x"], kind="dialogs", records=5,
                                           longest_tokens=90)) == 128
    long = DataReport(paths=["x"], kind="dialogs", records=5, longest_tokens=9000)
    assert suggested_seq_length(long) == 1024 and suggested_seq_length(long, "cuda") == 2048
    assert suggested_epochs(None) == 2 and suggested_epochs(dialogs) == 3
    assert suggested_epochs(DataReport(paths=["x"], kind="dialogs", records=5000)) == 2
    assert suggested_epochs(DataReport(paths=["x"], kind="dialogs", records=50_000)) == 1
    assert suggested_epochs(DataReport(paths=["x"], kind="text", records=3)) == 1


def test_pretrain_sizes_follow_memory_data_and_time():
    tiny_data = DataReport(paths=["t"], kind="text", records=1, tokens=200_000)
    rec = recommend_pretrain(_hw(32, gflops=700), tiny_data)
    assert rec.preset is not None and rec.preset.preset == "25m"
    lots = DataReport(paths=["t"], kind="text", records=1, tokens=500_000_000)
    big = recommend_pretrain(_hw(64, gflops=5000), lots, max_hours=1000)
    assert big.preset is not None and big.preset.params > rec.preset.params
    assert big.settings["preset"] == big.preset.preset and big.settings["epochs"] == 100
    no_data = recommend_pretrain(_hw(8))
    assert any("choose the training text" in w for w in no_data.warnings)


# ----------------------------------------------------------------------
#  Data check
# ----------------------------------------------------------------------
def test_dialogs_are_counted_and_judged(tmp_path):
    path = tmp_path / "chats.jsonl"
    _dialogs(path, 3)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"messages": [{"role": "user", "content": "no reply"}]}) + "\n")
    report = inspect_data(str(path))
    assert report.ok and report.for_finetune and report.kind == "dialogs"
    assert report.records == 3 and report.skipped == 1 and "chats.jsonl:4" in report.first_skipped
    assert report.turns == 2.0 and report.script == "latin" and report.tokens > 0
    assert report.longest_tokens >= estimate_tokens(len("What is 1 plus 1?"), "latin")
    assert any("only 3 dialogs" in w for w in report.warnings)
    assert any("skipped" in w for w in report.warnings)
    assert report.sample.startswith("user: What is 0 plus 0?")
    assert any(line.startswith("Data: chat dialogs - 3 dialogs") for line in
               report.summary_lines())


def test_pairs_and_text_and_problems(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("\n".join(json.dumps({"prompt": f"q{i}", "chosen": "good",
                                           "rejected": "bad"}) for i in range(4)) + "\n",
                     encoding="utf-8")
    report = inspect_data(str(pairs))
    assert report.kind == "preferences" and report.records == 4
    assert any("fine-tune it on dialogs first" in w for w in report.warnings)
    text = tmp_path / "book.txt"
    text.write_text(HELLO_RU * 300, encoding="utf-8")
    report = inspect_data(str(text))
    assert report.kind == "text" and report.script == "cyrillic"
    assert report.tokens == estimate_tokens(len(HELLO_RU * 300), "cyrillic")
    assert any("plain text" in w for w in report.warnings)
    assert not inspect_data([]).ok and "choose" in inspect_data([]).problems[0]
    missing = inspect_data(str(tmp_path / "nope.jsonl"))
    assert not missing.ok and "not found" in missing.problems[0]
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    (mixed / "a.txt").write_text("some text", encoding="utf-8")
    _dialogs(mixed / "b.jsonl", 2)
    report = inspect_data(str(mixed))
    assert not report.ok and "mixes" in report.problems[0]
    assert report.summary_lines()[0] == "Data: cannot be used"


def test_scripts_and_token_estimates():
    assert text_script("hello world") == "latin"
    assert text_script("\u043f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440") == "cyrillic"
    assert text_script("你好世界") == "cjk"
    assert text_script("hello \u043c\u0438\u0440") == "mixed"
    assert text_script("1234 !!") == ""
    assert estimate_tokens(4000, "latin") == 1000 and estimate_tokens(3000, "cyrillic") == 1000


# ----------------------------------------------------------------------
#  Recipes
# ----------------------------------------------------------------------
def test_finetune_params_for_each_method_and_data():
    qlora = finetune_params(FineTuneSettings(), "dialogs")
    assert qlora["quantize_base"] == "nf4" and qlora["lora_rank"] == 16
    assert qlora["lora_lr"] == 1e-3 and qlora["lora_targets"] == "all"
    assert qlora["continue_training"] and qlora["optimizer"] == "adamw8bit"
    assert qlora["prompt_loss_weight"] == "auto" and qlora["dialog_packing"] is True
    assert "chat_template" not in qlora and "lr" not in qlora
    lora = finetune_params(FineTuneSettings(method="lora", chat_template="llama3",
                                            prompt_weight=0.2, max_steps=7,
                                            time_limit_min=30), "dialogs")
    assert "quantize_base" not in lora and lora["chat_template"] == "llama3"
    assert lora["prompt_loss_weight"] == 0.2 and lora["max_steps"] == 7
    assert lora["time_budget_s"] == 1800.0
    full = finetune_params(FineTuneSettings(method="full", lr=2e-5), "text")
    assert full["lr"] == 2e-5 and "lora_rank" not in full and "dialog_packing" not in full
    dpo = finetune_params(FineTuneSettings(preference_beta=0.2), "preferences")
    assert dpo["method"] == "dpo" and dpo["preference_beta"] == 0.2 and "epochs" not in dpo
    sft = finetune_params(FineTuneSettings(objective="sft"), "preferences")
    assert "method" not in sft and sft["epochs"] == 2
    assert FineTuneSettings.from_dict({**FineTuneSettings().to_dict(), "extra": 1}) == \
        FineTuneSettings()


@pytest.mark.parametrize("settings, kind, message", [
    (FineTuneSettings(method="magic"), "dialogs", "unknown method"),
    (FineTuneSettings(), "shards", "a new model"),
    (FineTuneSettings(lora_rank=0), "dialogs", "rank"),
    (FineTuneSettings(optimizer="sgd"), "dialogs", "optimizer"),
    (FineTuneSettings(objective="dpo"), "dialogs", "preference pairs"),
    (FineTuneSettings(objective="ppo"), "preferences", "objective"),
    (FineTuneSettings(seq_length=0), "dialogs", "seq_length"),
])
def test_finetune_settings_that_cannot_work(settings, kind, message):
    with pytest.raises(SettingsError, match=message):
        finetune_params(settings, kind)


def test_pretrain_params():
    params = pretrain_params(TrainSettings(preset="45m", time_limit_min=10,
                                           early_stopping=False, optimizer="adamw",
                                           lr=0.002), "text")
    assert params["d_model"] == 384 and params["n_layers"] == 12
    assert params["time_budget_s"] == 600.0 and params["early_stopping_patience"] == 0
    assert params["optimizer"] == "adamw" and params["lr"] == 0.002
    assert not params["continue_training"]
    assert pretrain_params(TrainSettings(), "shards")["early_stopping_patience"] == "auto"
    with pytest.raises(SettingsError, match="Fine-tune tab"):
        pretrain_params(TrainSettings(), "preferences")
    with pytest.raises(SettingsError, match="unknown model size"):
        pretrain_params(TrainSettings(preset="9t"), "text")
    with pytest.raises(SettingsError, match="tokenizer"):
        pretrain_params(TrainSettings(tokenizer="words"), "text")
    # 2.20: a copy every N epochs; where it goes is the run's (its folder in the library)
    params = pretrain_params(TrainSettings(autosave_every=2), "text")
    assert params["autosave_every"] == 2 and "autosave_path" not in params
    assert "autosave_every" not in pretrain_params(TrainSettings(), "text")


# ----------------------------------------------------------------------
#  Errors explained
# ----------------------------------------------------------------------
def test_errors_are_explained_with_something_to_try():
    from auralite.engine.corpus import DataFormatError
    from auralite.engine.hf_models import HFModelError

    def kind(exc: BaseException) -> str:
        return explain_error(exc).kind

    assert kind(MemoryError()) == "memory"
    assert kind(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")) == "memory"
    assert kind(RuntimeError("[enforce fail] DefaultCPUAllocator: not enough memory")) == "memory"
    missing = ModuleNotFoundError("No module named 'transformers'", name="transformers")
    friendly = explain_error(missing)
    assert friendly.kind == "missing_package" and 'auralite-ai[hf]' in friendly.hints[0]
    gated = HFModelError("meta-llama/X is gated or private: accept its license on "
                         "huggingface.co and set HF_TOKEN (HTTP 401)")
    assert kind(gated) == "gated" and "HF_TOKEN" in " ".join(explain_error(gated).hints)
    assert kind(urllib.error.URLError("name or service not known")) == "network"
    assert kind(HFModelError("AuraLite imports LlamaForCausalLM ...")) == "model"
    assert kind(DataFormatError("the data mixes plain text with dialogs")) == "data"
    assert kind(SettingsError("seq_length must be positive")) == "settings"
    assert kind(PermissionError("denied")) == "permission"
    assert kind(FileNotFoundError("model.pt")) == "not_found"
    assert kind(OSError(errno.ENOSPC, "No space left on device")) == "disk"
    assert kind(KeyboardInterrupt()) == "stopped"
    # libraries that would not load (Windows' loader, Linux's), with what fixes them
    torch_dll = explain_error(OSError(
        '[WinError 1114] A dynamic link library (DLL) initialization routine failed. Error '
        'loading "C:\\py\\Lib\\site-packages\\torch\\lib\\c10.dll" or one of its '
        'dependencies.'))
    assert torch_dll.kind == "library" and "vc_redist.x64.exe" in torch_dll.hints[0]
    assert any('PySide6-Essentials>=6.9.3' in hint for hint in torch_dll.hints)
    assert any("force-reinstall torch" in hint for hint in torch_dll.hints)
    qt_dll = explain_error(ImportError("DLL load failed while importing QtCore: The specified "
                                       "procedure could not be found."))
    assert qt_dll.kind == "library" and any("Windows 10" in hint for hint in qt_dll.hints)
    assert not any("torch" in hint for hint in qt_dll.hints)
    linux = explain_error(ImportError("libxkbcommon.so.0: cannot open shared object file: No "
                                      "such file or directory"))
    assert linux.kind == "library" and "libxkbcommon0" in linux.hints[0]
    assert kind(ImportError("libfoo.so.3: cannot open shared object file")) == "library"
    assert kind(ImportError("cannot import name 'x' from 'y'")) == "unknown"
    try:
        try:
            raise MemoryError()
        except MemoryError as inner:
            raise RuntimeError("the step failed") from inner
    except RuntimeError as outer:
        assert kind(outer) == "memory"
    unknown = explain_error(ValueError("strange"), action="the export")
    assert unknown.kind == "unknown" and unknown.title == "The export failed"
    assert "log" in unknown.hints[0] and "ValueError: strange" in unknown.details
    assert "What to try:" in explain_error(MemoryError()).text()


# ----------------------------------------------------------------------
#  Progress, settings, updates, logs
# ----------------------------------------------------------------------
def test_progress_lines_from_the_engine_callbacks():
    tracker = ProgressTracker()
    view = tracker.update(0, 2, None, None, {"phase": "setup", "message": "Training the "
                                                                          "tokenizer",
                                             "stage_label": "Tokenizer", "stage_done": 5,
                                             "stage_total": 10, "stage_unit": "merges"})
    assert view.status == "Training the tokenizer" and "5/10 merges (50%)" in view.detail
    view = tracker.update(1, 2, 2.5, None, {"phase": "train", "percent": 25.0, "batch": 3,
                                            "batches": 12, "tokens_per_sec": 150.0,
                                            "eta_seconds": 75, "elapsed": 10})
    assert view.percent == 25.0 and view.status.startswith("Epoch 1/2 · step 3/12")
    assert "loss 2.5000" in view.status and "150 tokens/s" in view.detail
    assert "about 1 min 15 s left" in view.detail
    view = tracker.update(1, 2, 2.0, 2.2, {"phase": "epoch_end", "percent": 50.0,
                                           "is_epoch_end": True})
    assert view.status.startswith("Epoch 1/2 done") and "val 2.2000" in view.status
    assert tracker.losses == [(25.0, 2.5), (50.0, 2.0)] and tracker.val_losses == [(50.0, 2.2)]
    tracker.update(2, 2, 0.0, None, {"phase": "train", "percent": 51.0, "batch": 0})
    tracker.update(2, 2, 3.0, None, {"phase": "train", "percent": 51.0, "batch": 0})
    assert tracker.losses[-1] == (50.0, 2.0)            # no loss before a batch has run
    assert format_duration(3725) == "1 h 02 min" and format_duration(None) == "—"
    assert format_duration(55 * 3600) == "2.3 days" and format_duration(42) == "42 s"
    assert summary_text({"stop_reason": "time_budget"}, 60).startswith("Time limit reached")
    assert summary_text({"stop_reason": "early_stopping", "best_val_loss": 1.5}, 5) \
        .startswith("Stopped early")
    assert summary_text({}, 5, stopped=True).startswith("Stopped:")
    assert summary_text({"stopped_during": "tokenizer"}, 1).startswith("Stopped before")
    assert summary_text({}, 3).startswith("Training complete")


def test_settings_survive_broken_files(tmp_path, monkeypatch):
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path))
    assert load_settings() == StudioSettings()
    settings = StudioSettings(dark=True, gflops=123.0)
    for i in range(10):
        settings.add_recent(str(tmp_path / f"m{i}.pt"))
    settings.add_recent(str(tmp_path / "m3.pt"))
    assert len(settings.recent_models) == 8 and settings.recent_models[0].endswith("m3.pt")
    assert len(set(settings.recent_models)) == 8                    # moved to the top, not twice
    assert save_settings(settings) and load_settings() == settings
    (tmp_path / "studio.json").write_text("{not json", encoding="utf-8")
    assert load_settings() == StudioSettings()
    (tmp_path / "studio.json").write_text(json.dumps({"dark": "yes", "gflops": "fast",
                                                      "recent_models": ["a", 3],
                                                      "geometry": 7, "nope": 1}),
                                          encoding="utf-8")
    loaded = load_settings()
    assert loaded.dark is False and loaded.gflops == 0.0 and loaded.recent_models == ["a"]


def test_update_check_compares_versions_and_survives_being_offline():
    assert version_tuple("2.19.0") == (2, 19, 0) and version_tuple("2.20.0rc1") == (2, 20, 0)
    newer = check_for_update("2.19.0", fetch=lambda url, t: {"info": {"version": "2.20.0"}})
    assert newer.newer and "pip install -U" in newer.message()
    same = check_for_update("2.19.0", fetch=lambda url, t: {"info": {"version": "2.19.0"}})
    assert not same.newer and "latest" in same.message()

    def offline(url, timeout):
        raise urllib.error.URLError("offline")

    unknown = check_for_update("2.19.0", fetch=offline)
    assert unknown.error == "URLError" and not unknown.newer
    assert "Could not check" in unknown.message()
    assert not UpdateInfo("2.19.0", "2.18.1").newer


def test_log_file_and_uncaught_errors(tmp_path, monkeypatch):
    from auralite.studio.logs import install_excepthooks, remove_logging, setup_logging
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    path = setup_logging("test-app")
    try:
        assert setup_logging("test-app") == path
        root = logging.getLogger()
        assert sum(1 for h in root.handlers
                   if getattr(h, "baseFilename", "") == str(path)) == 1
        seen = []
        install_excepthooks(seen.append)
        try:
            raise ValueError("boom")
        except ValueError as err:
            sys.excepthook(type(err), err, err.__traceback__)
        worker = threading.Thread(target=lambda: (_ for _ in ()).throw(KeyError("in a thread")))
        worker.start()
        worker.join()
        assert [type(e) for e in seen] == [ValueError, KeyError]
        for handler in root.handlers:
            handler.flush()
        text = path.read_text(encoding="utf-8")
        assert "started" in text and "ValueError: boom" in text and "in a thread" in text
    finally:
        remove_logging("test-app")


# ----------------------------------------------------------------------
#  auralite doctor, auralite studio
# ----------------------------------------------------------------------
def test_doctor_recommends_and_writes_the_params(tmp_path, capsys):
    from auralite.cli import main
    data = tmp_path / "chats.jsonl"
    _dialogs(data, 30)
    plan = tmp_path / "plan.json"
    assert main(["doctor", "--no-speed", "--data", str(data), "--write-params", str(plan),
                 "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["version"] and out["hardware"]["threads"] >= 1
    assert out["data"]["kind"] == "dialogs" and out["data"]["records"] == 30
    assert len(out["options"]) == len(CATALOG) and "PySide6" in out["packages"]
    if out["choice"] is not None:
        params = json.loads(plan.read_text(encoding="utf-8"))
        assert params["epochs"] == 3 and params["seq_length"] == 128
        assert out["commands"][0].startswith(f"auralite train --data {data} --resume hf:")
        assert f"--params {plan}" in out["commands"][0]
    assert main(["doctor", "--no-speed", "--goal", "pretrain"]) == 0
    text = capsys.readouterr().out
    assert "Recommended: a" in text and "auralite train --data DIALOGS.jsonl --preset" in text


def test_studio_explains_what_it_needs(monkeypatch, tmp_path):
    from auralite.studio import app
    from auralite.studio.logs import remove_logging
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path))
    monkeypatch.setitem(sys.modules, "PySide6", None)            # not installed
    monkeypatch.setattr(app, "_qt_version", lambda: "")
    text = app.qt_problem()
    # the command installs into this very Python, whatever PyPI's AuraLite has
    assert app.install_command() in text and '"PySide6-Essentials>=6.9.3"' in text
    monkeypatch.setattr(app, "_qt_version", lambda: "6.12.0")    # installed, does not load
    with pytest.raises(ImportError):
        app.qt_problem()
    monkeypatch.delitem(sys.modules, "PySide6")
    pytest.importorskip("PySide6")
    monkeypatch.setattr("auralite.studio.hardware._simd", lambda _info: ("SSE4.1", False))
    assert "PySide6-Essentials<6.10" in app.qt_problem()
    monkeypatch.setattr(app, "_qt_version", lambda: "6.10.0")
    assert "PySide6-Essentials<6.10" in app.qt_problem()
    monkeypatch.setattr(app, "_qt_version", lambda: "6.9.3")
    assert app.qt_problem() == ""
    monkeypatch.setattr(app, "_qt_version", lambda: "6.12.0")    # nothing can tell: start
    monkeypatch.setattr("auralite.studio.hardware._simd", lambda _info: ("", None))
    assert app.qt_problem() == ""
    told = []
    monkeypatch.setattr(app, "qt_problem", lambda: "needs Qt")
    monkeypatch.setattr(app, "_tell", told.append)
    try:
        assert app.main([]) == 2 and told == ["needs Qt"]
    finally:
        remove_logging("studio")


def test_install_command_names_this_python(monkeypatch, tmp_path):
    from auralite.studio import app
    folder = tmp_path / "Python 3.12"
    folder.mkdir()
    (folder / "python.exe").write_bytes(b"")
    monkeypatch.setattr(sys, "executable", str(folder / "pythonw.exe"))   # no console
    assert app.install_command() == (f'"{folder / "python.exe"}" -m pip install '
                                     f'"PySide6-Essentials>=6.9.3"')
    monkeypatch.setattr(sys, "executable", "/usr/bin/python3")
    assert app.install_command("x<1", upgrade=True) == '/usr/bin/python3 -m pip install -U "x<1"'


def test_installation_notices_an_install_older_than_the_code(monkeypatch):
    """2.19.1: an install from a folder made before 2.19 keeps its old metadata - no
    `auralite-studio` command, no `studio` extra (pip then only warns)."""
    import types

    from auralite import __version__
    from auralite.studio import app

    def installed(version, *commands):
        dist = types.SimpleNamespace(version=version, entry_points=[
            types.SimpleNamespace(name=name) for name in ("auralite", *commands)])
        monkeypatch.setattr(app, "_installed", lambda: dist)

    monkeypatch.setattr(sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(app, "_source_folder", lambda: "/src/AuraLite AI")
    installed(__version__, "auralite-gui", "auralite-studio")
    assert app.installation() == (f"auralite-ai {__version__}, the code in /src/AuraLite AI", "")
    installed("2.18.1", "auralite-gui")                       # `pip install -e .` in 2.18.1
    text, problem = app.installation()
    assert text == "auralite-ai 2.18.1, the code in /src/AuraLite AI"
    assert problem == (f"the installed package is 2.18.1 but the code is {__version__}, and it "
                       f"has no auralite-studio command. Install it again: /usr/bin/python3 "
                       f'-m pip install -e "/src/AuraLite AI[studio]"')
    installed("2.19.0", "auralite-gui", "auralite-studio")   # the command, an older version
    monkeypatch.setattr(app, "_source_folder", lambda: "")    # installed from PyPI
    text, problem = app.installation()
    assert text == "auralite-ai 2.19.0" and "auralite-studio command" not in problem
    assert problem == (f"the installed package is 2.19.0 but the code is {__version__}. Install "
                       f'it again: /usr/bin/python3 -m pip install -U "auralite-ai[studio]"')
    installed(__version__, "auralite-gui")                    # this version, yet no command
    assert app.installation()[1].startswith(f"the installed package is {__version__} but the "
                                            f"code is {__version__}, and it has no auralite-")

    monkeypatch.setattr(app, "_installed", lambda: None)      # a source tree, not installed
    text, problem = app.installation()
    assert text.startswith("not installed, the code is in ") and problem == ""
    monkeypatch.undo()
    folder = app._source_folder()                             # the tests run from the clone
    assert folder and os.path.isfile(os.path.join(folder, "auralite", "studio", "app.py"))


def test_self_test_names_an_install_older_than_the_code(monkeypatch, capsys):
    from auralite.studio import app
    stale = "the installed package is 2.18.1 but the code is 2.19.1. Install it again: x"
    monkeypatch.setattr(app, "installation", lambda: ("auralite-ai 2.18.1", stale))
    monkeypatch.setattr(app, "qt_problem", lambda: "AuraLite Studio needs PySide6 (Qt 6).")
    assert app.self_test() == 2                                # stops before Qt
    out = capsys.readouterr().out
    assert f"\n  Installed: auralite-ai 2.18.1\n  ! {stale}\n" in out
    assert out.rstrip().endswith("RESULT: AuraLite Studio cannot start here (see above)")


def test_the_install_counts_not_the_egg_info_in_the_source_folder(monkeypatch, tmp_path):
    """setuptools leaves `auralite_ai.egg-info` in the source folder at every build. Run
    from that folder, importlib finds it first - and it would hide an older install."""
    from auralite.studio import app
    source, site = tmp_path / "AuraLite-AI", tmp_path / "site-packages"
    for folder, name, version in ((source, "auralite_ai.egg-info", "2.19.1"),
                                  (site, "auralite_ai-2.18.1.dist-info", "2.18.1")):
        (folder / name).mkdir(parents=True)
        (folder / name / ("PKG-INFO" if name.endswith("egg-info") else "METADATA")).write_text(
            f"Metadata-Version: 2.1\nName: auralite-ai\nVersion: {version}\n", encoding="utf-8")
    monkeypatch.setattr(app, "_source_folder", lambda: str(source.resolve()))
    monkeypatch.setattr(sys, "path", [str(source), str(site)])
    assert app._installed().version == "2.18.1"               # the install, not the build
    monkeypatch.setattr(sys, "path", [str(source)])           # `setup.py develop`: only that
    assert app._installed().version == "2.19.1"
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    assert app._installed() is None


def test_a_failed_start_is_logged_and_shown(monkeypatch, tmp_path):
    from auralite.studio import app
    from auralite.studio.logs import remove_logging
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path))
    shown: list[str] = []
    monkeypatch.setattr(app, "_message_box", lambda title, text: shown.append(text))
    monkeypatch.setattr(app, "_enable_faulthandler", lambda: None)
    printed = io.StringIO()
    monkeypatch.setattr(sys, "stderr", printed)       # not a terminal: the box shows too

    def broken() -> str:
        raise OSError('[WinError 1114] A dynamic link library (DLL) initialization routine '
                      'failed. Error loading "C:\\py\\torch\\lib\\c10.dll" or one of its '
                      'dependencies.')
    monkeypatch.setattr(app, "qt_problem", broken)
    try:
        assert app.main([]) == 1
        for handler in logging.getLogger().handlers:
            handler.flush()
        log = (tmp_path / "logs" / "studio.log").read_text(encoding="utf-8")
    finally:
        remove_logging("studio")
    assert "could not start" in log and "Traceback" in log and "WinError 1114" in log
    assert len(shown) == 1 and shown[0].startswith("A library would not load")
    assert "vc_redist.x64.exe" in shown[0] and "PySide6-Essentials>=6.9.3" in shown[0]
    assert f"The details are in {tmp_path / 'logs' / 'studio.log'}" in shown[0]
    assert shown[0] in printed.getvalue()
    unknown = app.failure_text(ValueError("strange"))
    assert "Starting AuraLite Studio failed" in unknown and "strange" in unknown
    assert "auralite doctor --studio" in unknown and "python -m auralite.studio" in unknown


def test_the_runtime_is_prepared_before_qt_is_imported(monkeypatch, tmp_path):
    """Windows: the newest msvcp140.dll must be loaded before PySide6 loads its own."""
    from auralite.studio import app, msvc_runtime
    from auralite.studio.logs import remove_logging
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path))
    monkeypatch.setattr(app, "_enable_faulthandler", lambda: None)
    order: list[str] = []
    monkeypatch.setattr(msvc_runtime, "prepare_runtime", lambda: order.append("runtime"))
    monkeypatch.setattr(app, "qt_problem", lambda: order.append("qt") or "no Qt here")
    monkeypatch.setattr(app, "_tell", lambda message: order.append("told"))
    try:
        assert app.main([]) == 2
    finally:
        remove_logging("studio")
    assert order == ["runtime", "qt", "told"]


@pytest.mark.parametrize("child, result, expected", [
    ("import sys; print('closed'); sys.exit(0)", 0, None),
    ("import sys; sys.exit(2)", 2, None),              # it told the user itself
    ("import os, sys; print('Incompatible processor. This Qt build requires the following "
     "features:'); print('    sse4.2 popcnt'); sys.stdout.flush(); os._exit(3)", 3,
     "PySide6-Essentials<6.10"),
    ("import os, sys; print('[WinError 1114] A dynamic link library (DLL) initialization "
     "routine failed.'); sys.stdout.flush(); os._exit(-1073741819)", 3, "vc_redist.x64.exe"),
])
def test_started_without_a_console_a_crash_is_shown(monkeypatch, tmp_path, child, result,
                                                    expected):
    """`auralite-studio` on Windows runs under pythonw: no stderr at all. The app runs as a
    child process; a crash nobody reported is shown with its last output."""
    from auralite.studio import app
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path))
    monkeypatch.delenv("QT_FORCE_STDERR_LOGGING", raising=False)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "studio-faults.log").write_text("File \"old.py\", line 1\n",
                                                         encoding="utf-8")
    shown: list[str] = []
    monkeypatch.setattr(app, "_message_box", lambda title, text: shown.append(text))
    commands: list[list[str]] = []

    def command(argv: list[str]) -> list[str]:
        commands.append(argv)
        return [sys.executable, "-c", "import os; assert os.environ['QT_FORCE_STDERR_LOGGING']"
                " == '1' and os.environ['PYTHONIOENCODING']\n" + child]
    monkeypatch.setattr(app, "_child_command", command)
    monkeypatch.setattr(sys, "stderr", None)
    assert app.main(["--x"]) == result
    assert commands == [["--x"]]
    if expected is None:
        assert shown == []
        return
    assert len(shown) == 1
    text = shown[0]
    assert text.startswith("AuraLite Studio stopped unexpectedly (")
    assert expected in text and f"The logs are in {tmp_path / 'logs'}" in text
    assert "Its last output:" in text and "old.py" not in text   # not an earlier run's
    assert (tmp_path / "logs" / "studio-output.log").read_text(encoding="utf-8").strip()


def test_the_child_of_a_console_less_start_opens_the_window(monkeypatch):
    from auralite.studio import app
    started: list[list[str]] = []
    monkeypatch.setattr(app, "_start", lambda argv: started.append(argv) or 0)
    monkeypatch.setattr(app, "_watch", lambda argv: pytest.fail("watched twice"))
    monkeypatch.setattr(sys, "stderr", None)
    assert app.main([app.CHILD, "--x"]) == 0 and started[-1][1:] == ["--x"]


def test_how_a_process_ended_in_words():
    from auralite.studio.app import _ending
    assert _ending(0xC0000005) == "exit code 0xC0000005: access violation"
    assert _ending(-1073741819) == "exit code 0xC0000005: access violation"     # signed
    assert _ending(3) == "exit code 3: aborted" and _ending(7) == "exit code 7"
    assert _ending(-11) == "signal SIGSEGV" and _ending(-100) == "signal 100"
    assert _ending(0xC0000409).startswith("exit code 0xC0000409: fast fail")


def test_crash_messages_name_the_known_causes(tmp_path):
    from auralite.studio.app import crash_text
    output, faults = tmp_path / "out.log", tmp_path / "faults.log"
    output.write_text("qt.qpa.plugin: Could not find the Qt platform plugin \"windows\"\n\n"
                      "This application failed to start because no Qt platform plugin could "
                      "be initialized.\n", encoding="utf-8")
    faults.write_text("Windows fatal exception: access violation\n\n"
                      '  File "window.py", line 9 in run\n', encoding="utf-8")
    text = crash_text(0xC0000409, 2.4, output, faults)
    assert "(exit code 0xC0000409: fast fail" in text and "after 2 s" in text
    assert "QT_PLUGIN_PATH" in text and "-m pip install -U" in text
    assert "Where it stopped:" in text and "window.py" in text
    output.write_text("", encoding="utf-8")
    faults.unlink()
    bare = crash_text(-11, 0.2, output, faults)
    assert "Its last output" not in bare and "Where it stopped" not in bare
    assert "QT_PLUGIN_PATH" not in bare and "vc_redist" not in bare
    assert "python -m auralite.studio" in bare


def test_log_tail_reads_the_end(tmp_path):
    from auralite.studio.logs import tail
    path = tmp_path / "out.log"
    path.write_text("\n".join(f"line {i}" for i in range(5000)) + "\n\n  \n",
                    encoding="utf-8")
    assert tail(path, 3) == ["line 4997", "line 4998", "line 4999"]
    assert tail(path, 2, limit=40) == ["line 4998", "line 4999"]
    assert 2 <= len(tail(path, 50, limit=40)) < 5            # only the end is read
    assert tail(tmp_path / "missing.log") == []


def test_a_crash_leaves_the_stack_in_the_faults_log(tmp_path):
    if not hasattr(__import__("faulthandler"), "_sigsegv"):
        pytest.skip("this Python has no faulthandler._sigsegv")
    code = ("import os, sys\n"
            "os.environ['AURALITE_HOME'] = sys.argv[1]\n"
            "from auralite.studio import app\n"
            "app._enable_faulthandler()\n"
            "def open_the_window():\n"
            "    import faulthandler\n"
            "    faulthandler._sigsegv()\n"
            "open_the_window()\n")
    proc = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True,
                          timeout=120, check=False)
    assert proc.returncode != 0
    text = (tmp_path / "logs" / "studio-faults.log").read_text(encoding="utf-8")
    assert "open_the_window" in text, text


def _fake_dll(path, version: tuple[int, int, int, int]) -> None:
    a, b, c, d = version
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MZ" + bytes(98) + struct.pack("<I", 0xFEEF04BD) + bytes(8)
                     + struct.pack("<IIII", 0xFEEF04BD, 0x00010000, (a << 16) | b,
                                   (c << 16) | d) + bytes(40))


def test_the_newest_visual_cpp_runtime_is_loaded_first(tmp_path, monkeypatch):
    """PyTorch 2.9+ cannot use msvcp140 older than 14.40, which PySide6 before 6.9.3 brings
    and loads first: the Studio loads the newest copy before Qt and PyTorch."""
    from auralite.studio import msvc_runtime as rt
    system, qt, shiboken = tmp_path / "System32", tmp_path / "PySide6", tmp_path / "shiboken6"
    _fake_dll(system / rt.DLL, (14, 40, 33810, 0))
    _fake_dll(shiboken / rt.DLL, (14, 39, 33520, 0))
    _fake_dll(qt / rt.DLL, (14, 39, 33520, 0))
    assert rt.file_version(system / rt.DLL) == (14, 40, 33810, 0)
    assert rt.file_version(tmp_path / "missing.dll") is None
    (tmp_path / "plain.dll").write_bytes(b"MZ" + struct.pack("<I", 0xFEEF04BD) + bytes(30))
    assert rt.file_version(tmp_path / "plain.dll") is None          # no version record
    monkeypatch.setattr(rt, "_package_dir", {"PySide6": qt, "shiboken6": shiboken}.get)
    copies = rt.runtime_copies(system)
    assert [(c.origin, c.text) for c in copies] == [
        ("system", "14.40.33810.0"), ("shiboken6", "14.39.33520.0"), ("PySide6", "14.39.33520.0")]
    best = rt.choose_runtime(copies)
    assert best is not None and best.origin == "system"             # Qt's copy is too old
    _fake_dll(shiboken / rt.DLL, (14, 44, 35211, 0))
    _fake_dll(qt / rt.DLL, (14, 44, 35211, 0))
    best = rt.choose_runtime(rt.runtime_copies(system))
    assert best is not None and best.origin == "shiboken6" and best.text == "14.44.35211.0"
    assert best.version[:2] >= rt.TORCH_NEEDS
    _fake_dll(system / rt.DLL, (14, 44, 35211, 0))                   # a tie: Windows' own
    best = rt.choose_runtime(rt.runtime_copies(system))
    assert best is not None and best.origin == "system"
    assert rt.choose_runtime(copies, loaded="C:\\Windows\\System32\\msvcp140.dll") is None
    assert rt.choose_runtime([]) is None
    (qt / rt.DLL).unlink()
    assert [c.origin for c in rt.runtime_copies(system)] == ["system", "shiboken6"]
    if sys.platform != "win32":
        assert rt.prepare_runtime() is None and rt.loaded_runtime() == ""


def test_package_folders_are_found_without_importing_them():
    from auralite.studio.msvc_runtime import _package_dir
    found = _package_dir("json")
    assert found is not None and (found / "__init__.py").is_file()
    assert _package_dir("auralite_no_such_package") is None
    assert _package_dir("os") is None                                 # a module, no folder


def test_old_windows_is_not_mistaken_for_an_old_processor(monkeypatch):
    """IsProcessorFeaturePresent knows SSE4.2 / AVX only from Windows 10 2004 on: before
    that it says "no" on any processor - which stopped the Studio on Windows 10 LTSC."""
    from auralite.studio import hardware
    if platform.machine().lower() not in ("x86_64", "amd64", "i386", "i686"):
        pytest.skip("x86 feature names")
    asked: list[int] = []
    monkeypatch.setattr(hardware, "_windows_feature", lambda f: asked.append(f) or f in (38, 39))
    monkeypatch.setattr(hardware, "_numpy_cpu_features", lambda: {})
    monkeypatch.setitem(sys.modules, "torch", None)                  # no fallback either
    monkeypatch.setattr(hardware, "_windows_build", lambda: 17763)  # Windows 10 LTSC 2019
    assert hardware._simd("", system="win32") == ("", None) and asked == []
    monkeypatch.setattr(hardware, "_windows_build", lambda: 19045)  # Windows 10 22H2
    assert hardware._simd("", system="win32") == ("AVX", True) and 38 in asked
    # NumPy's CPUID answers first, on any Windows
    monkeypatch.setattr(hardware, "_windows_build", lambda: 17763)
    monkeypatch.setattr(hardware, "_numpy_cpu_features", lambda: {
        "SSE2": True, "SSE41": True, "POPCNT": True, "SSE42": True, "AVX": True, "AVX2": False})
    assert hardware._simd("", system="win32") == ("AVX", True)      # the i5-2400
    monkeypatch.setattr(hardware, "_numpy_cpu_features", lambda: {"SSE2": True, "SSE41": True})
    assert hardware._simd("", system="win32") == ("basic", False)
    monkeypatch.setattr(hardware, "_numpy_cpu_features", lambda: {
        "SSE2": True, "SSE42": True, "POPCNT": False})
    assert hardware._simd("", system="win32") == ("SSE4.2", False)
    monkeypatch.undo()
    features = hardware._numpy_cpu_features()
    assert features.get("SSE2") is True                              # the real CPUID
    assert hardware._simd("", system="win32")[1] == bool(features.get("SSE42")
                                                         and features.get("POPCNT"))


def test_doctor_runs_the_studio_check_in_a_fresh_process(monkeypatch, capsys):
    from auralite import cli
    monkeypatch.setattr(cli, "_studio_check", lambda timeout=300.0: (
        -11, "AuraLite Studio self-test: ...\n  Qt: PySide6 6.12.0\n"))
    assert cli.main(["doctor", "--no-speed", "--studio"]) == 1
    text = capsys.readouterr().out
    assert "checked in a fresh process" in text and "    AuraLite Studio self-test" in text
    assert "=> it crashed (signal SIGSEGV)" in text
    monkeypatch.setattr(cli, "_studio_check", lambda timeout=300.0: (0, "RESULT: OK\n"))
    assert cli.main(["doctor", "--no-speed", "--studio", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["studio"]["check"] == {"code": 0, "result": "the window starts",
                                      "output": "RESULT: OK\n"}
    assert cli._studio_verdict(2).startswith("it cannot start here")
    assert cli._studio_verdict(1).startswith("it failed")
    assert cli._studio_verdict(None) == "the check did not finish"
    assert cli.main(["doctor", "--no-speed"]) == 0
    text = capsys.readouterr().out
    assert "auralite doctor --studio" in text or "Studio: needs PySide6" in text
    # 2.19.1: an install older than the code is named, with the command that renews it
    stale = "the installed package is 2.18.1 but the code is 2.19.1. Install it again: x"
    monkeypatch.setattr("auralite.studio.app.installation",
                        lambda: ("auralite-ai 2.18.1", stale))
    assert cli.main(["doctor", "--no-speed"]) == 0
    assert f"\n  ! {stale}\n" in capsys.readouterr().out
    assert cli.main(["doctor", "--no-speed", "--json"]) == 0
    studio = json.loads(capsys.readouterr().out)["studio"]
    assert studio["installed"] == "auralite-ai 2.18.1" and studio["install_problem"] == stale
    assert cli.main(["doctor", "--no-speed", "--studio"]) == 0   # the self-test names it
    assert f"  ! {stale}" not in capsys.readouterr().out


def test_self_test_explains_a_qt_that_does_not_load(tmp_path):
    """A fresh process, as `auralite doctor --studio` runs it, with a PySide6 whose DLLs
    do not load."""
    fake = tmp_path / "fake" / "PySide6"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(
        'raise ImportError("DLL load failed while importing QtCore: The specified procedure '
        'could not be found.")\n', encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(tmp_path / "fake"), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep),
        AURALITE_HOME=str(tmp_path / "home"), PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, "-m", "auralite.studio", "--self-test"],
                          capture_output=True, text=True, encoding="utf-8", env=env,
                          timeout=180, check=False)
    from importlib import metadata
    try:
        metadata.version("PySide6-Essentials")
        installed = True
    except metadata.PackageNotFoundError:
        installed = False
    assert "  Installed: " in proc.stdout
    if installed:                          # installed, but it does not load: a library
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "A library would not load" in proc.stdout and "RESULT: FAILED" in proc.stdout
        assert "vc_redist.x64.exe" in proc.stdout and "Traceback" in proc.stdout
    else:
        assert proc.returncode == 2, proc.stdout + proc.stderr
        assert "needs PySide6" in proc.stdout and "cannot start here" in proc.stdout


def test_reference_size_is_a_thousand_dialogs():
    assert REFERENCE_TOKENS == 150_000


# ----------------------------------------------------------------------
#  The library: models and datasets on the disk (2.20)
# ----------------------------------------------------------------------
def _shelf(home, tmp_path):
    """A download (and a tokenizer-only folder and an unfinished download), two
    runs, remembered files and a prepared dataset (finished and not)."""
    from auralite import runs as runlib
    from auralite.studio import library
    for name, files, complete in (("HuggingFaceTB__SmolLM2-135M-Instruct/main",
                                   {"config.json": b"{}", "model.safetensors": b"w" * 5000},
                                   True),
                                  ("Qwen__Qwen2.5-3B-Instruct/main",
                                   {"config.json": b"{}",
                                    "model-00001-of-00002.safetensors.part": b"p" * 300},
                                   False),
                                  ("HuggingFaceTB__SmolLM2-360M/main",
                                   {"tokenizer.json": b"{}"}, True)):
        folder = home / "hf" / name
        folder.mkdir(parents=True)
        for file, data in files.items():
            (folder / file).write_bytes(data)
        if complete:
            (folder / ".auralite-complete.json").write_text("{}", encoding="utf-8")
    chats = tmp_path / "chats.jsonl"
    chats.write_text("{}\n", encoding="utf-8")
    done = runlib.RunRecorder(kind="finetune", data={"paths": [str(chats)], "kind": "dialogs",
                                                     "records": 1})
    done.model_path.write_bytes(b"m" * 3000)
    done.record["result"] = runlib.MODEL_FILE
    done.finish({"stop_reason": "finished", "best_val_loss": 1.25})
    runlib.RunRecorder(kind="train", data={"paths": [str(chats)]}).fail("broken")
    saved = tmp_path / "mine.pt"
    saved.write_bytes(b"s" * 100)
    library.remember_model(saved, "saved")
    adapter = tmp_path / "adapter.pt"
    adapter.write_bytes(b"a" * 10)
    library.remember_model(adapter, "adapter")
    gguf = tmp_path / "chat.gguf"
    gguf.write_bytes(b"g" * 20)
    library.remember_model(gguf, "export", note="SmolLM2 (fine-tuned)")
    library.remember_model(tmp_path / "gone.pt", "saved")
    report = DataReport(paths=[str(chats)], kind="dialogs", records=1, tokens=12, bytes=3)
    library.remember_dataset(chats, report)
    texts = tmp_path / "notes.txt"
    texts.write_text("hello", encoding="utf-8")
    library.remember_dataset(texts, DataReport(paths=[str(texts)], kind="text", records=1))
    for name, complete in (("fineweb-1", True), ("wiki-2", False)):
        folder = home / "datasets" / name
        folder.mkdir(parents=True)
        (folder / "auralite_dataset.json").write_text(json.dumps({
            "complete": complete, "stats": {"tokens": 1000},
            "sources": [{"spec": f"hf:Org/{name}"}],
            "tokenizer": {"description": "BPE, 512 tokens"}}), encoding="utf-8")
        (folder / "train_00000.bin").write_bytes(b"t" * 2000)
    return done.run, saved, adapter, gguf, chats, texts


def test_the_library_lists_models_and_datasets(tmp_path, monkeypatch):
    from auralite.studio import library
    home = tmp_path / "home"
    monkeypatch.setenv("AURALITE_HOME", str(home))
    run, saved, adapter, gguf, chats, texts = _shelf(home, tmp_path)
    models = {m.name: m for m in library.list_models(recent=[str(saved), "hf:Org/Model"])}
    assert set(models) == {"HuggingFaceTB/SmolLM2-135M-Instruct", "Qwen/Qwen2.5-3B-Instruct",
                           run.name, "mine.pt", "adapter.pt", "chat.gguf", "gone.pt"}
    smol = models["HuggingFaceTB/SmolLM2-135M-Instruct"]
    assert smol.kind == "download" and smol.owned and smol.bytes >= 5000
    assert smol.extra["spec"] == "hf:HuggingFaceTB/SmolLM2-135M-Instruct"
    assert smol.detail == "SmolLM2 135M Instruct"
    qwen = models["Qwen/Qwen2.5-3B-Instruct"]
    assert "not for commercial use" in qwen.detail and "not finished" in qwen.detail
    result = models[run.name]
    assert result.kind == "result" and result.run_id == run.id and result.owned
    assert "best validation loss 1.2500" in result.detail
    assert models["mine.pt"].kind == "saved" and models["mine.pt"].remembered
    assert models["adapter.pt"].kind_label == "LoRA adapters"
    assert models["chat.gguf"].detail == "GGUF file - SmolLM2 (fine-tuned)"
    assert not models["gone.pt"].exists and models["gone.pt"].detail == "not found any more"
    data = {d.name: d for d in library.list_datasets()}
    assert set(data) == {"chats.jsonl", "notes.txt", "fineweb-1", "wiki-2"}
    assert data["chats.jsonl"].detail == "Dialogs, 1 dialogs, about 12 tokens"
    assert len(data["chats.jsonl"].extra["runs"]) == 2 and data["chats.jsonl"].remembered
    assert data["notes.txt"].detail == "Text, 1 documents"
    assert data["fineweb-1"].kind == "shards" and data["fineweb-1"].owned
    assert data["fineweb-1"].detail == "1,000 tokens, BPE, 512 tokens"
    assert "not finished" in data["wiki-2"].detail
    disk = library.disk_summary()
    assert disk["downloads"] >= 5300 and disk["datasets"] >= 4000 and disk["runs"] >= 3000
    assert disk["total"] == disk["downloads"] + disk["runs"] + disk["datasets"]
    assert disk["free"] > 0
    assert library.format_size(1536) == "1.5 KB" and library.format_size(5 << 30) == "5.0 GB"
    assert library.format_size(150 << 20) == "150 MB" and library.format_size(12) == "12 B"


def test_the_library_deletes_and_forgets(tmp_path, monkeypatch):
    from auralite import runs as runlib
    from auralite.studio import library
    home = tmp_path / "home"
    monkeypatch.setenv("AURALITE_HOME", str(home))
    run, saved, _adapter, _gguf, chats, _texts = _shelf(home, tmp_path)
    models = {m.name: m for m in library.list_models()}
    assert library.delete_item(models["HuggingFaceTB/SmolLM2-135M-Instruct"]) >= 5000
    assert not (home / "hf" / "HuggingFaceTB__SmolLM2-135M-Instruct").exists()
    assert library.delete_item(models[run.name]) == 3000          # the run's files
    kept = runlib.find_run(run.id)
    assert kept.result_path is None and run.folder.exists() and kept.record["files_deleted"]
    assert library.delete_item(models["mine.pt"]) == 100 and not saved.exists()
    assert "mine.pt" not in {m.name for m in library.list_models()}
    data = {d.name: d for d in library.list_datasets()}
    assert library.delete_item(data["fineweb-1"]) > 2000
    assert not (home / "datasets" / "fineweb-1").exists()
    assert library.forget(chats) and not library.forget(chats)
    assert chats.exists()                                         # the file stays
    outside = library.LibraryItem(kind="download", name="x", path=str(tmp_path))
    with pytest.raises(ValueError, match="is not under"):
        library.delete_item(outside)
    with pytest.raises(ValueError, match="cannot remember"):
        library.remember_model(saved, "download")


def test_the_library_index_is_forgiving_and_bounded(tmp_path, monkeypatch):
    from auralite.studio import library
    home = tmp_path / "home"
    monkeypatch.setenv("AURALITE_HOME", str(home))
    home.mkdir()
    (home / "library.json").write_text("{not json", encoding="utf-8")
    assert library.load_index() == {"models": [], "datasets": []}
    (home / "library.json").write_text('{"models": [1, {"kind": "saved"}], "datasets": 3}',
                                       encoding="utf-8")
    assert library.load_index() == {"models": [], "datasets": []}
    monkeypatch.setattr(library, "MAX_ITEMS", 3)
    for i in range(5):
        library.remember_model(tmp_path / f"m{i}.pt")
    library.remember_model(tmp_path / "m2.pt")                    # to the top, once
    paths = [os.path.basename(e["path"]) for e in library.load_index()["models"]]
    assert paths == ["m2.pt", "m4.pt", "m3.pt"]
    assert library.list_models() and library.list_datasets() == []
