"""2.22: ``auralite bench`` - what a computer does for AuraLite, measured in one command.

Offline and in a few minutes: the processor and a one-second speed test, then three sizes of
a new model, each trained for a minute on a built-in text (training tokens per second, the
memory the run took against the estimate the memory plan works with) and timed while it
writes. ``--finetune`` adds a LoRA fine-tune of a ready model (downloaded once). The result
is a Markdown table for an article or a README and the same numbers as JSON, in
``~/.auralite/bench/bench-<processor>-<date>.md`` / ``.json``.

The Studio runs the same measurement from *Tools -> Benchmark*.
"""
from __future__ import annotations

import gc
import json
import os
import platform
import random
import re
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["DEFAULT_SECONDS", "FINETUNE_MODEL", "QUICK", "SIZES", "BenchError", "bench_text",
           "finetune_source",
           "default_folder", "estimated_minutes", "file_stem", "run_bench", "save_report",
           "size_dims", "summary_lines", "to_markdown"]

#: The sizes of a new model: their dims, or the preset they come from. With the Studio's
#: BPE vocabulary of 512: 0.9M, 7M and 19M parameters.
SIZES: dict[str, dict[str, Any]] = {
    "tiny": {"d_model": 128, "n_layers": 4, "n_heads": 4, "n_kv_heads": 2, "d_ff": 384},
    "small": {"preset": "25m"},
    "medium": {"preset": "45m"},
}
#: What every size trains with: the Studio's tokenizer and window; batches of 8 windows, so
#: that even the medium size reports often in a minute on a PC from 2011, and (2.23) 4 of
#: them per optimizer step - the Studio's 32 windows a step: what an optimizer step costs
#: (Muon's) is shared by as many tokens as in the Studio.
SEQ_LENGTH = 256
BATCH = 8
ACCUMULATION = 4
VOCAB = 512
DEFAULT_SECONDS = 60.0
GENERATE_TOKENS = 128
#: ``--quick``: two sizes, 15 seconds each (a check that it all runs, not numbers to quote).
QUICK = {"sizes": ("tiny", "small"), "seconds": 15.0, "generate_tokens": 32}
#: 2.23: Qwen3.5 0.8B (1.7 GB, downloaded once); a folder of a downloaded Hugging Face model
#: or an AuraLite checkpoint works as well (``--finetune PATH``)
FINETUNE_MODEL = "hf:Qwen/Qwen3.5-0.8B"
NIGHT_SECONDS = 8 * 3600

ProgressFn = Callable[[str], None]


class BenchError(ValueError):
    """A benchmark that cannot run as asked (an unknown size, an empty text)."""


# ----------------------------------------------------------------------
#  What it trains on
# ----------------------------------------------------------------------
_WORDS = (
    "the of and to in a is that for it as was with be by on not he this are or his from at "
    "which but have an they you were her she there one all we their can has more will would "
    "if been so what out about who up time when them some into only its other new could "
    "people like than may first then these two any after most also made did many over such "
    "where years way well even back much good see how our just work world man life day long "
    "great through down own still between last should might us being under never know same "
    "while use place old year house water light small large hand part child eye woman night "
    "word number side home head point thing city tree story river window morning voice door "
    "road table letter music garden winter summer field mountain friend family question "
    "answer reason idea moment heart paper face school money market train boat village "
    "forest bread fire stone horse snow wind rain sun moon star sea ship king queen soldier "
    "doctor teacher machine engine model language computer program number speed memory "
    "walked looked said asked found gave took came went thought knew told called turned "
    "began seemed felt brought kept heard left stood ran wrote read spoke held opened "
    "quickly slowly quietly again always often soon almost already together perhaps "
    "beautiful dark bright cold warm quiet strange simple clear early late young happy"
).split()


def bench_text(chars: int = 200_000, seed: int = 2026) -> str:
    """The built-in training text: plain English sentences from a fixed list of words, the
    same on every computer (a seeded generator), so that runs compare. Its content does not
    change the speed; the size of a model and its window do."""
    rng = random.Random(seed)
    out: list[str] = []
    total = 0
    while total < chars:
        sentences = []
        for _ in range(rng.randint(3, 7)):
            words = [rng.choice(_WORDS) for _ in range(rng.randint(4, 16))]
            if rng.random() < 0.3:
                words.insert(rng.randint(1, len(words) - 1), ",")
            sentence = " ".join(words).replace(" ,", ",")
            sentences.append(sentence[0].upper() + sentence[1:] + rng.choice(".....?!"))
        paragraph = " ".join(sentences)
        out.append(paragraph)
        total += len(paragraph) + 1
    return "\n".join(out)


def size_dims(name: str) -> dict[str, Any]:
    """The model dims of the size `name`."""
    if name not in SIZES:
        raise BenchError(f"unknown size {name!r}: {', '.join(SIZES)}")
    spec = dict(SIZES[name])
    preset = spec.pop("preset", "")
    if preset:
        from auralite.engine.sizing import MODEL_PRESETS
        spec = {k: MODEL_PRESETS[preset][k]
                for k in ("d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff")}
    return spec


def _size_label(name: str) -> str:
    dims = size_dims(name)
    preset = SIZES[name].get("preset")
    return (f"{name}: {dims['n_layers']} layers x {dims['d_model']}"
            + (f" (preset {preset})" if preset else ""))


def estimated_minutes(sizes: list[str] | tuple[str, ...], seconds: float, *,
                      finetune: bool = False) -> float:
    """About how long a benchmark takes: each size's training, its setup and its writing."""
    per_size = seconds + 25.0
    return (10.0 + per_size * len(sizes) + ((seconds + 120.0) if finetune else 0.0)) / 60.0


# ----------------------------------------------------------------------
#  Measuring
# ----------------------------------------------------------------------
class _PeakMemory:
    """The highest memory the process uses while a stage runs (sampled 5 times a second)."""

    def __init__(self) -> None:
        from auralite.studio.hardware import process_memory
        self._read = process_memory
        self.start = self.peak = int(process_memory())
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, name="auralite-bench-memory",
                                        daemon=True)

    def _watch(self) -> None:
        while not self._stop.wait(0.2):
            self.peak = max(self.peak, int(self._read()))

    def __enter__(self) -> _PeakMemory:
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.peak = max(self.peak, int(self._read()))

    @property
    def used(self) -> int:
        """Bytes the stage added to what the process held when it began."""
        return max(0, self.peak - self.start)


def _steady_speed(events: list[tuple[float, int, int]], tokens_per_batch: float) -> float:
    """Training tokens per second from the progress events ``(time, batches done, epoch)``:
    from the second batch on (the first one allocates and warms up), the same epoch only
    (the end of an epoch can validate)."""
    if tokens_per_batch <= 0:
        return 0.0
    usable = [e for e in events if e[1] >= 1]
    best_time = 0.0
    best_batches = 0
    for epoch in sorted({e[2] for e in usable}):
        rows = [e for e in usable if e[2] == epoch]
        if len(rows) < 2:
            continue
        dt = rows[-1][0] - rows[0][0]
        db = rows[-1][1] - rows[0][1]
        if dt > 0 and db > 0:
            best_time += dt
            best_batches += db
    return best_batches * tokens_per_batch / best_time if best_time > 0 else 0.0


def _train_size(name: str, text: str, seconds: float, generate_tokens: int,
                progress: ProgressFn, stop: threading.Event | None,
                tokenizer_path: str = "") -> dict[str, Any]:
    from auralite.engine import AuraLiteEngine
    from auralite.engine import sizing as _sizing

    dims = size_dims(name)
    params: dict[str, Any] = dict(dims)
    params.update(seq_length=SEQ_LENGTH, batch_size=BATCH, accumulation_steps=ACCUMULATION,
                  tokenizer="bpe", bpe_vocab_size=VOCAB, epochs=100_000,
                  time_budget_s=float(seconds),
                  early_stopping_patience=0, val_split=0.0, memory_plan="auto",
                  progress_every_seconds=0.5, progress_every_batches=1, seed=0)
    if tokenizer_path:                  # trained once for every size (run_bench)
        params["tokenizer_path"] = tokenizer_path
    events: list[tuple[float, int, int]] = []
    last = [0.0]

    def on_progress(epoch: int, total: int, loss: float, val_loss: Any,
                    info: dict | None = None) -> None:
        info = info or {}
        if info.get("phase") != "train":
            return
        now = time.perf_counter()
        events.append((now, int(info.get("batch") or 0), int(epoch)))
        speed = float(info.get("tokens_per_sec") or 0)
        if now - last[0] >= 5.0 and speed > 0:
            last[0] = now
            progress(f"  {name}: training, {speed:,.0f} tokens/s, loss {float(loss):.3f}")

    progress(f"{_size_label(name)}: building the model")
    engine = AuraLiteEngine()
    with _PeakMemory() as memory:
        summary = engine.train(text, params, progress_callback=on_progress, stop_event=stop)
    if summary.get("stop_reason") == "stopped":
        raise InterruptedError("the benchmark was stopped")
    native = engine.model
    count = int(native.count_parameters())
    batches = max((e[1] for e in events), default=0)
    total_batches = sum(max((e[1] for e in events if e[2] == ep), default=0)
                        for ep in {e[2] for e in events})
    tokens = int(summary.get("tokens") or 0)
    tokens_per_batch = tokens / total_batches if total_batches else 0.0
    speed = _steady_speed(events, tokens_per_batch)
    if speed <= 0 and summary.get("seconds"):
        speed = tokens / float(summary["seconds"])
    micro = int(summary.get("batch_size") or BATCH)
    checkpointing = bool(getattr(native, "use_gradient_checkpointing", False))
    estimate = _sizing.estimate_training_memory(
        batch_size=micro, seq_length=SEQ_LENGTH, gradient_checkpointing=checkpointing,
        vocab_size=int(native.head.out_features), optimizer="muon",
        tie_word_embeddings=bool(native.tie_word_embeddings),
        dropout=float(getattr(native, "dropout", 0) or 0), **dims)
    # what the run needs on top of the framework, as the memory plan counts it
    need = int(estimate["total"] - estimate["overhead"])
    progress(f"  {name}: writing {generate_tokens} tokens")
    written = [0]

    def on_token(_token: int, _logits: Any) -> None:
        written[0] += 1

    engine.generate("The", length=8, temperature=0.8, top_k=40)        # warm up
    start = time.perf_counter()
    engine.generate("The house by the river", length=generate_tokens, temperature=0.8,
                    top_k=40, on_token=on_token)
    write_seconds = time.perf_counter() - start
    speed = round(speed, 1)
    result = {
        "name": name, "label": _size_label(name), "preset": SIZES[name].get("preset", ""),
        "params": count, "layers": int(dims["n_layers"]), "width": int(dims["d_model"]),
        "tokens_per_second": speed, "tokens": tokens,
        "optimizer_steps": int(summary.get("optimizer_steps") or 0), "batches": batches,
        "seconds": round(float(summary.get("seconds") or 0.0), 1),
        "micro_batch": micro, "accumulation_steps": int(summary.get("accumulation_steps") or 1),
        "gradient_checkpointing": checkpointing,
        "memory_used": int(memory.used), "memory_estimate": need,
        "memory_start": int(memory.start), "memory_peak": int(memory.peak),
        "night_tokens": int(speed * NIGHT_SECONDS),
        "text_tokens": int(summary.get("train_tokens") or 0),
        "generate_tokens": int(written[0]),
        "generate_tokens_per_second": round(written[0] / write_seconds, 1)
        if write_seconds > 0 else 0.0,
    }
    del engine, native
    gc.collect()
    return result


def _warm_up(text: str, tokenizer_path: str) -> None:
    """Two steps of a very small model, so that the first size's memory and speed do not
    carry what PyTorch sets up on its first training step in a process."""
    from auralite.engine import AuraLiteEngine
    engine = AuraLiteEngine()
    engine.train(text[:20_000], {"d_model": 32, "n_layers": 1, "n_heads": 2, "d_ff": 64,
                                 "seq_length": 64, "batch_size": 4, "epochs": 1, "max_steps": 2,
                                 "tokenizer": "bpe", "tokenizer_path": tokenizer_path,
                                 "val_split": 0.0, "progress_every_seconds": 100, "seed": 0})
    del engine
    gc.collect()


def _bench_dialogs(n: int = 48) -> list[list[dict[str, str]]]:
    out = []
    for i in range(n):
        a, b = i % 9 + 2, i % 7 + 3
        out.append([{"role": "user", "content": f"What is {a} times {b}? Answer in a sentence."},
                    {"role": "assistant", "content": f"{a} times {b} is {a * b}. Multiplying "
                                                     f"{a} by {b} gives {a * b}."}])
    return out


def finetune_source(spec: str) -> str:
    """What `spec` names for ``--finetune`` (2.23): ``"hub"`` (``hf:ORG/NAME``, downloaded
    once into ~/.auralite/hf), ``"folder"`` (a downloaded Hugging Face model: config.json and
    the weights) or ``"checkpoint"`` (an AuraLite .pt / .safetensors file); BenchError for a
    path that is neither - before the benchmark spends minutes on the other stages."""
    text = str(spec or "").strip()
    if not text:
        raise BenchError("--finetune needs a model: hf:ORG/NAME, a model folder or a checkpoint")
    if text.lower().startswith("hf:"):
        return "hub"
    path = Path(text).expanduser()
    if path.is_dir():
        if not (path / "config.json").is_file():
            raise BenchError(f"{path}: no config.json - not a downloaded Hugging Face model "
                             f"folder (Qwen3.5, Qwen2/3, Llama, Mistral)")
        return "folder"
    if path.is_file():
        if path.suffix.lower() not in (".pt", ".safetensors"):
            raise BenchError(f"{path.name}: the benchmark fine-tunes a Hugging Face model "
                             f"folder or an AuraLite checkpoint (.pt, .safetensors)")
        return "checkpoint"
    raise BenchError(f"{path}: no such folder or file")


def _finetune(spec: str, method: str, seconds: float, progress: ProgressFn,
              stop: threading.Event | None) -> dict[str, Any]:
    from auralite.engine import AuraLiteEngine
    from auralite.engine.dialogs import DialogSet
    from auralite.studio.recipes import FineTuneSettings, finetune_params

    progress(f"Fine-tuning: loading {spec} ({method})")
    engine = AuraLiteEngine()
    last = [0.0]

    def on_transfer(info: Any = None, *args: Any) -> None:
        now = time.perf_counter()
        if isinstance(info, dict) and now - last[0] >= 5.0 and info.get("all_total"):
            last[0] = now
            progress(f"  downloading: {int(info.get('all_bytes') or 0) / 1e6:,.0f} of "
                     f"{int(info['all_total']) / 1e6:,.0f} MB")

    kind = finetune_source(spec)
    with _PeakMemory() as load_memory:
        if kind == "checkpoint":            # 2.23: a model trained or imported before
            engine.load_model(str(Path(spec).expanduser()))
            info: dict = {"architecture": "AuraLite checkpoint"}
        else:                               # 4 bits while it imports: less memory at once
            info = engine.import_hf_model(str(Path(spec).expanduser()) if kind == "folder"
                                          else spec,
                                          quantize="nf4" if method == "qlora" else None,
                                          progress_callback=on_transfer, stop_event=stop)
    settings = FineTuneSettings(method=method, seq_length=SEQ_LENGTH, epochs=1000,
                                time_limit_min=float(seconds) / 60.0, val_split=0.0,
                                batch_size=BATCH)
    params = finetune_params(settings, "dialogs")
    params.update(progress_every_seconds=0.5, progress_every_batches=1, seed=0)
    events: list[tuple[float, int, int]] = []

    def on_progress(epoch: int, total: int, loss: float, val_loss: Any,
                    info: dict | None = None) -> None:
        info = info or {}
        if info.get("phase") == "train":
            now = time.perf_counter()
            events.append((now, int(info.get("batch") or 0), int(epoch)))
            speed = float(info.get("tokens_per_sec") or 0)
            if now - last[0] >= 5.0 and speed > 0:
                last[0] = now
                progress(f"  fine-tuning: {speed:,.0f} tokens/s, loss {float(loss):.3f}")

    with _PeakMemory() as memory:
        summary = engine.train(DialogSet(_bench_dialogs()), params, progress_callback=on_progress,
                               stop_event=stop)
    if summary.get("stop_reason") == "stopped":
        raise InterruptedError("the benchmark was stopped")
    total_batches = sum(max((e[1] for e in events if e[2] == ep), default=0)
                        for ep in {e[2] for e in events})
    tokens = int(summary.get("tokens") or 0)
    speed = _steady_speed(events, tokens / total_batches if total_batches else 0.0)
    if speed <= 0 and summary.get("seconds"):
        speed = tokens / float(summary["seconds"])
    steps = int(summary.get("optimizer_steps") or 0)
    result = {
        "model": spec, "architecture": str((info or {}).get("architecture") or ""),
        "params": int(engine.model.count_parameters()), "method": method,
        "tokens_per_second": round(speed, 1), "tokens": tokens, "optimizer_steps": steps,
        "seconds_per_step": round(float(summary.get("seconds") or 0.0) / steps, 2)
        if steps else None,
        "seconds": round(float(summary.get("seconds") or 0.0), 1),
        "micro_batch": int(summary.get("batch_size") or BATCH),
        "accumulation_steps": int(summary.get("accumulation_steps") or 1),
        "memory_used": int(max(memory.peak, load_memory.peak) - load_memory.start),
    }
    del engine
    gc.collect()
    return result


def _environment(threads: int) -> dict[str, Any]:
    import torch

    from auralite import __version__, bundle
    info = bundle.app_info() if bundle.packaged() else None
    return {"auralite": __version__, "python": platform.python_version(),
            "torch": str(torch.__version__), "torch_threads": int(threads),
            "app": info is not None, "app_platform": info.platform if info is not None else "",
            "machine": platform.machine(), "node": platform.node()}


#: The lines a child process prints for its parent (``auralite bench --child SPEC``).
PROGRESS_MARK = "AURALITE-BENCH-PROGRESS "
RESULT_MARK = "AURALITE-BENCH-RESULT "
ERROR_MARK = "AURALITE-BENCH-ERROR "


def child_main(spec_file: str | Path) -> int:
    """One stage of a benchmark in a process of its own (``auralite bench --child SPEC``):
    its memory is its own to measure, and running out of it ends this process only. Prints
    progress and the result as marked lines (:data:`RESULT_MARK`) for :func:`_run_child`."""
    def say(line: str) -> None:
        print(PROGRESS_MARK + line, flush=True)

    try:
        spec = json.loads(Path(spec_file).read_text(encoding="utf-8"))
        if spec.get("threads"):
            import torch

            import auralite.engine._legacy  # noqa: F401  (it sets its threads on import)
            torch.set_num_threads(int(spec["threads"]))
        if spec.get("kind") == "finetune":
            result = _finetune(str(spec["model"]), str(spec["method"]), float(spec["seconds"]),
                               say, None)
        else:
            text = Path(spec["text"]).read_text(encoding="utf-8")
            _warm_up(text, str(spec["tokenizer"]))
            result = _train_size(str(spec["name"]), text, float(spec["seconds"]),
                                 int(spec["generate_tokens"]), say, None,
                                 tokenizer_path=str(spec["tokenizer"]))
    except Exception as exc:                 # the parent reports it as the stage's result
        print(ERROR_MARK + f"{type(exc).__name__}: {exc}", flush=True)
        return 1
    print(RESULT_MARK + json.dumps(result), flush=True)
    return 0


def _child_error(code: int | None, tail: list[str]) -> str:
    """What ended a child process that left no result, in a few words."""
    text = "\n".join(tail)
    marked = [line[len(ERROR_MARK):] for line in tail if line.startswith(ERROR_MARK)]
    if (code in (-9, 137, 3221225495) or "MemoryError" in text or "out of memory" in text
            or "DefaultCPUAllocator: not enough memory" in text):
        return "out of memory"
    if marked:
        return marked[-1]
    last = next((line for line in reversed(tail) if line.strip()), "")
    return f"exit code {code}" + (f": {last.strip()[:200]}" if last else "")


def _run_child(spec: dict[str, Any], folder: Path, say: ProgressFn,
               stop: threading.Event | None) -> dict[str, Any]:
    """Run one stage in a child process (:func:`child_main`); its result, or
    ``{"error": ...}`` when it ended without one."""
    import subprocess
    from collections import deque

    from auralite import bundle
    spec_file = folder / f"stage-{spec.get('name') or spec.get('kind')}.json"
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    command = bundle.cli_command("bench", "--child", str(spec_file))
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    if spec.get("threads"):              # what the engine reads when it is imported
        env["AURALITE_NUM_THREADS"] = str(int(spec["threads"]))
    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    done = threading.Event()

    def watch() -> None:                     # Stop: end the child at once
        while not done.wait(0.2):
            if stop is not None and stop.is_set():
                proc.kill()
                return

    watcher = threading.Thread(target=watch, name="auralite-bench-stop", daemon=True)
    watcher.start()
    result: dict[str, Any] | None = None
    tail: deque[str] = deque(maxlen=40)
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith(RESULT_MARK):
                result = json.loads(line[len(RESULT_MARK):])
            elif line.startswith(PROGRESS_MARK):
                say(line[len(PROGRESS_MARK):])
            else:
                tail.append(line)
        code = proc.wait()
    finally:
        done.set()
        watcher.join(timeout=2.0)
    if stop is not None and stop.is_set():
        raise InterruptedError("the benchmark was stopped")
    if result is None:
        return {"error": _child_error(code, list(tail))}
    return result


def run_bench(*, sizes: list[str] | tuple[str, ...] = ("tiny", "small", "medium"),
              seconds: float = DEFAULT_SECONDS, generate_tokens: int = GENERATE_TOKENS,
              data: str | None = None, finetune: str | None = None, method: str = "",
              threads: int | None = None, progress: ProgressFn | None = None,
              stop: threading.Event | None = None, isolate: bool = True) -> dict[str, Any]:
    """Measure this computer: the report as a dict (see :func:`to_markdown`).

    `sizes` from :data:`SIZES`; `seconds` of training each; `data` a text file to train on
    instead of the built-in text; `finetune` a model (``hf:ORG/NAME``, a folder) to
    fine-tune with LoRA (`method` "lora" or "qlora"; "": QLoRA below 8 GB of memory).
    `progress` receives a line at every stage; `stop` ends it (InterruptedError).
    `isolate` (default) runs every size and the fine-tune in a process of its own: what
    each takes in memory is measured alone, and one that runs out of memory is reported as
    such while the others go on.
    """
    import torch

    from auralite.studio.hardware import detect_hardware, measure_gflops

    say: ProgressFn = progress or (lambda _line: None)
    unknown = [s for s in sizes if s not in SIZES]
    if unknown or not sizes:
        raise BenchError(f"unknown size {', '.join(unknown) or '(none)'}: the sizes are "
                         f"{', '.join(SIZES)}")
    if float(seconds) <= 0:
        raise BenchError("the seconds of training must be above 0")
    previous_threads = torch.get_num_threads()
    if threads:
        torch.set_num_threads(int(threads))
    try:
        return _run(sizes, float(seconds), int(generate_tokens), data, finetune, method,
                    threads, say, stop, isolate, detect_hardware, measure_gflops)
    finally:
        torch.set_num_threads(previous_threads)


def _run(sizes: Any, seconds: float, generate_tokens: int, data: str | None,
         finetune: str | None, method: str, threads: int | None, say: ProgressFn,
         stop: threading.Event | None, isolate: bool, detect_hardware: Any,
         measure_gflops: Any) -> dict[str, Any]:
    started = time.time()
    say("The computer: looking at it and a one-second speed test")
    hardware = detect_hardware()
    hardware.gflops = measure_gflops("cpu", threads=threads or hardware.train_threads)
    text_info: dict[str, Any] = {"source": "built-in", "chars": 0}
    if data:
        from auralite.engine import load_training_data
        from auralite.engine.corpus import TextCorpus
        loaded = load_training_data([str(data)], stream=False)
        if isinstance(loaded, TextCorpus):
            loaded = loaded.text
        if not isinstance(loaded, str):
            raise BenchError(f"{Path(str(data)).name}: the benchmark trains on a plain text "
                             "file (.txt, .md)")
        text = loaded
        text_info = {"source": Path(str(data)).name, "chars": len(text)}
    else:
        text = bench_text()
        text_info["chars"] = len(text)
    if len(text.strip()) < 2000:
        raise BenchError("the text is too short to measure with (at least 2,000 characters)")
    if finetune:
        finetune_source(str(finetune))      # a wrong path fails now, not after the sizes
    report: dict[str, Any] = {
        "format": 1, "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "hardware": hardware.to_dict(),
        "environment": _environment(int(threads or hardware.train_threads)),
        "settings": {"seconds": seconds, "seq_length": SEQ_LENGTH, "batch_size": BATCH,
                     "accumulation_steps": ACCUMULATION, "vocab": VOCAB, "generate_tokens": generate_tokens, "text": text_info,
                     "isolated": bool(isolate)},
        "train": [], "finetune": None,
    }
    say(f"The tokenizer: BPE {VOCAB} on the text, once for every size")
    from auralite.engine import tokenizer_io
    tokenizer, tokenizer_info = tokenizer_io.train_tokenizer(text, kind="bpe", vocab_size=VOCAB,
                                                             val_split=0.0)
    with tempfile.TemporaryDirectory(prefix="auralite-bench-") as name:
        folder = Path(name)
        tokenizer_path = str(tokenizer_io.save_tokenizer(
            tokenizer, folder / "bench-tokenizer.json", training=tokenizer_info))
        text_path = folder / "bench-text.txt"
        text_path.write_text(text, encoding="utf-8")
        if not isolate:
            _warm_up(text, tokenizer_path)
        for size in sizes:
            if stop is not None and stop.is_set():
                raise InterruptedError("the benchmark was stopped")
            if isolate:
                row = _run_child({"kind": "train", "name": size, "text": str(text_path),
                                  "tokenizer": tokenizer_path, "seconds": seconds,
                                  "generate_tokens": generate_tokens, "threads": threads},
                                 folder, say, stop)
            else:
                row = _train_size(size, text, seconds, generate_tokens, say, stop,
                                  tokenizer_path=tokenizer_path)
            if "error" in row:
                row = {"name": size, "label": _size_label(size), "error": row["error"]}
                say(f"  {size}: {row['error']}")
            else:
                say(f"  {size}: {row['tokens_per_second']:,.0f} tokens/s training, "
                    f"{row['generate_tokens_per_second']:,.0f} tokens/s writing")
            report["train"].append(row)
        if finetune:
            chosen = method or ("lora" if hardware.memory_total >= 8 << 30 else "qlora")
            if isolate:
                ft = _run_child({"kind": "finetune", "model": str(finetune), "method": chosen,
                                 "seconds": seconds, "threads": threads}, folder, say, stop)
            else:
                ft = _finetune(str(finetune), chosen, seconds, say, stop)
            if "error" in ft:
                ft = {"model": str(finetune), "method": chosen, "error": ft["error"]}
                say(f"  fine-tuning: {ft['error']}")
            report["finetune"] = ft
    report["seconds"] = round(time.time() - started, 1)
    return report


# ----------------------------------------------------------------------
#  The report
# ----------------------------------------------------------------------
def _gb(n: float) -> str:
    """Bytes as the Studio writes memory (GB of 1024 MB): ``"0.27 GB"``, ``"16 GB"``."""
    gb = float(n) / (1 << 30)
    return f"{gb:,.2f} GB" if gb < 10 else f"{gb:,.0f} GB"


def _millions(n: float) -> str:
    from auralite.studio.advisor import size_text
    return size_text(n)


def _tokens(n: float) -> str:
    return f"{n / 1e6:,.1f}M" if n >= 1e6 else f"{n / 1e3:,.0f}K"


def _model_name(spec: str) -> str:
    """``hf:Qwen/Qwen3.5-0.8B`` as it is; a folder or a file by its name."""
    text = str(spec)
    return text if text.lower().startswith("hf:") else (Path(text).name or text)


def summary_lines(report: dict[str, Any]) -> list[str]:
    """The result in a few lines (the Studio's message, the end of ``auralite bench``)."""
    lines = []
    for row in report.get("train") or []:
        if row.get("error"):
            lines.append(f"{row['label']}: {row['error']}")
            continue
        lines.append(f"{row['label']}: {_millions(row['params'])} parameters - "
                     f"{row['tokens_per_second']:,.0f} tokens/s training, "
                     f"{row['generate_tokens_per_second']:,.0f} tokens/s writing")
    ft = report.get("finetune")
    if ft and ft.get("error"):
        lines.append(f"Fine-tuning {_model_name(ft['model'])} ({ft['method']}): {ft['error']}")
    elif ft:
        lines.append(f"Fine-tuning {_model_name(ft['model'])} ({ft['method']}): "
                     f"{ft['tokens_per_second']:,.0f} tokens/s, {_gb(ft['memory_used'])}")
    return lines


def to_markdown(report: dict[str, Any]) -> str:
    """The report as Markdown: the computer, then a table for training and writing."""
    hw = report.get("hardware") or {}
    env = report.get("environment") or {}
    settings = report.get("settings") or {}
    cpu = hw.get("cpu") or "an unknown processor"
    where = (f"the app for {env.get('app_platform')}" if env.get("app")
             else "installed with pip")
    out = [f"# AuraLite benchmark: {cpu}", "",
           f"{report.get('date', '')} - AuraLite {env.get('auralite', '?')} ({where}), "
           f"Python {env.get('python', '?')}, PyTorch {env.get('torch', '?')}", "",
           "| Computer | |", "|---|---|",
           f"| Processor | {cpu} - {hw.get('cores', '?')} cores / {hw.get('threads', '?')} "
           f"threads{', ' + hw['simd'] if hw.get('simd') else ''} |",
           f"| Memory | {_gb(hw.get('memory_total') or 0)} "
           f"({_gb(hw.get('memory_available') or 0)} free at the start) |"]
    if hw.get("gpu"):
        out.append(f"| GPU | {hw['gpu']} (the benchmark trains on the processor) |")
    out += [f"| System | {hw.get('system') or platform.platform()} |",
            f"| Training threads | {env.get('torch_threads') or hw.get('train_threads', '?')} |"]
    if hw.get("gflops"):
        out.append(f"| Speed test | {hw['gflops']:,.0f} GFLOP/s (float32 matrix product) |")
    text = settings.get("text") or {}
    source = ("the built-in text" if text.get("source") == "built-in"
              else f"`{text.get('source')}`")
    show_epoch = text.get("source") not in (None, "built-in")
    out += ["", "## Training a new model on the processor", "",
            f"Each size trained for {settings.get('seconds', 0):,.0f} s on {source}: BPE "
            f"{settings.get('vocab')}, a window of {settings.get('seq_length')} tokens, "
            f"batches of {settings.get('batch_size')} windows"
            + (f", {settings['accumulation_steps']} of them a step"
               if int(settings.get("accumulation_steps") or 1) > 1 else "")
            + ", Muon. Tokens/s counts the training steps after the first one.", "",
            "| Model | Parameters | Training, tokens/s | In 8 hours | "
            + ("One pass over the text | " if show_epoch else "")
            + "Memory used (estimate) | Writing, tokens/s |",
            "|---|---|---|---|" + ("---|" if show_epoch else "") + "---|---|"]
    for row in report.get("train") or []:
        if row.get("error"):
            out.append(f"| {row['label']} | - | {row['error']} | - | "
                       + ("- | " if show_epoch else "") + "- | - |")
            continue
        epoch = ""
        if show_epoch:
            minutes = (row["text_tokens"] / row["tokens_per_second"] / 60.0
                       if row["tokens_per_second"] else 0.0)
            epoch = (f"{minutes:,.1f} min | " if minutes < 120 else f"{minutes / 60:,.1f} h | ")
        memory = f"{_gb(row['memory_used'])} ({_gb(row['memory_estimate'])})"
        planned = int(settings.get("batch_size") or BATCH)
        if row.get("gradient_checkpointing") or int(row.get("micro_batch") or planned) < planned:
            memory += f", batches of {row['micro_batch']}"     # the memory plan split them
        out.append(f"| {row['label']} | {_millions(row['params'])} | "
                   f"{row['tokens_per_second']:,.0f} | {_tokens(row['night_tokens'])} tokens | "
                   f"{epoch}{memory} | {row['generate_tokens_per_second']:,.0f} |")
    ft = report.get("finetune")
    if ft:
        out += ["", "## Fine-tuning a ready model", "",
                "| Model | Method | Tokens/s | Seconds per step | Memory used |",
                "|---|---|---|---|---|"]
        if ft.get("error"):
            out.append(f"| {_model_name(ft['model'])} | {ft['method']} | {ft['error']} | - | - |")
        else:
            out.append(f"| {_model_name(ft['model'])} ({_millions(ft['params'])}) | "
                       f"{ft['method']} | {ft['tokens_per_second']:,.0f} | "
                       f"{ft['seconds_per_step'] or '-'} | {_gb(ft['memory_used'])} |")
    out += ["", f"Measured with `auralite bench` in {report.get('seconds', 0) / 60:,.1f} min. "
            "Writing: one prompt, sampling (temperature 0.8, top-k 40), after a warm-up.", ""]
    return "\n".join(out)


def file_stem(report: dict[str, Any]) -> str:
    """``bench-i5-2400-2026-11-08-1405``: the processor's model, else the computer's name."""
    cpu = str((report.get("hardware") or {}).get("cpu") or "")
    patterns = (r"\b(i[3579]-\d{3,5}[a-z]{0,2})\b", r"\b(ryzen \d+ (?:pro )?\d{3,5}[a-z0-9]{0,3})\b",
                r"\b(xeon\(r\)?\s*(?:cpu\s*)?[a-z]?\d?-?\d{4}[a-z0-9]{0,3})\b",
                r"\b(apple m\d(?: pro| max| ultra)?)\b")
    name = ""
    for pattern in patterns:
        match = re.search(pattern, cpu.lower())
        if match:
            name = match.group(1)
            break
    if not name:
        name = str((report.get("environment") or {}).get("node") or "computer")
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower().replace("(r)", "")).strip("-") or "computer"
    date = re.sub(r"[^0-9]+", "-", str(report.get("date") or "")).strip("-")
    return f"bench-{slug[:40]}-{date}" if date else f"bench-{slug[:40]}"


def default_folder() -> Path:
    from auralite.studio.hardware import auralite_home
    return auralite_home() / "bench"


def save_report(report: dict[str, Any], folder: str | Path | None = None) -> tuple[Path, Path]:
    """Write ``<stem>.md`` and ``<stem>.json`` into `folder` (default ``~/.auralite/bench``)."""
    target = Path(folder) if folder else default_folder()
    target.mkdir(parents=True, exist_ok=True)
    stem = file_stem(report)
    md, js = target / f"{stem}.md", target / f"{stem}.json"
    md.write_text(to_markdown(report), encoding="utf-8")
    js.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return md, js


def _main(argv: list[str] | None = None) -> int:          # python -m auralite.bench
    from auralite.cli import main
    return main(["bench", *(argv if argv is not None else sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(_main())
