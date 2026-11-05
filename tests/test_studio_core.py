"""2.19: AuraLite Studio's toolkit-free core (no Qt needed).

What the computer can do (hardware), which base models fit it and how long a
run takes (catalog, advisor), what a data file holds (datacheck), training
parameters from a few choices (recipes), errors explained (errors), progress
lines, settings, the update check, the log file, `auralite doctor` and the
Studio's start-up check.
"""
from __future__ import annotations

import errno
import json
import logging
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


def test_studio_explains_what_it_needs(monkeypatch):
    from auralite.studio import app
    monkeypatch.setitem(sys.modules, "PySide6", None)            # not installed
    assert 'pip install "auralite-ai[studio]"' in app.qt_problem()
    monkeypatch.delitem(sys.modules, "PySide6")
    pytest.importorskip("PySide6")
    monkeypatch.setattr(app, "_qt_version", lambda: "6.12.0")
    monkeypatch.setattr("auralite.studio.hardware._simd", lambda _info: ("SSE4.1", False))
    assert "PySide6-Essentials<6.10" in app.qt_problem()
    monkeypatch.setattr(app, "_qt_version", lambda: "6.10.0")
    assert "PySide6-Essentials<6.10" in app.qt_problem()
    monkeypatch.setattr(app, "_qt_version", lambda: "6.9.3")
    assert app.qt_problem() == ""
    told = []
    monkeypatch.setattr(app, "qt_problem", lambda: "needs Qt")
    monkeypatch.setattr(app, "_tell", told.append)
    assert app.main([]) == 2 and told == ["needs Qt"]


def test_reference_size_is_a_thousand_dialogs():
    assert REFERENCE_TOKENS == 150_000
