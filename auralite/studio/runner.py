"""Training runs as a window starts them, without a toolkit (2.26): the Studio's tabs and the
classic interface's run the same steps - the base model loaded, the data read, the run
recorded in the library with a checkpoint every 10 minutes, its result saved there.

They run on a worker thread. A window follows them through ``progress_callback`` (the
engine's training progress) and ``on_event(kind, ...)``:

* ``("unloaded",)`` - the model in memory is gone (a new one loads);
* ``("stage", text, percent or None)`` - a step before or after training;
* ``("import", info)`` - a Hugging Face model downloads or converts;
* ``("loaded", spec)`` - the base model is in memory; ``("continued", name)`` - a run's
  checkpoint is;
* ``("recorder", recorder)`` - the run's :class:`~auralite.runs.RunRecorder`.
"""
from __future__ import annotations

import os
from typing import Any, Callable

from auralite import runs

__all__ = ["base_record", "continue_run", "finetune", "machine_record", "train_recorded"]

Event = Callable[..., None]


def _quiet(*_event: Any) -> None:
    return None


def train_recorded(engine: Any, recorder: Any, data: Any, params: dict,
                   progress_callback: Callable[..., Any] | None = None, *,
                   stop_event: Any = None, finish_event: Any = None,
                   on_event: Event | None = None) -> Any:
    """``engine.train`` with the run recorded (2.20): its curve, its checkpoint, and at the
    end its model (in the run's folder) or its error. Returns the training summary."""
    emit = on_event or _quiet
    try:
        summary = engine.train(data, params, progress_callback=recorder.callback(progress_callback),
                               stop_event=stop_event, finish_event=finish_event)
    except BaseException as exc:
        recorder.fail(exc)
        raise
    if not (isinstance(summary, dict) and summary.get("stopped_during")):
        emit("stage", "Saving the run in the library…", None)
    recorder.save_and_finish(summary, engine)
    return summary


def machine_record(engine: Any, hardware: Any = None) -> dict[str, Any]:
    """This computer, as a run's record keeps it (`hardware`: a detected
    :class:`~auralite.studio.hardware.Hardware`, when there is one)."""
    info = runs.machine_summary(engine)
    if hardware is not None:
        info.update(cpu=hardware.cpu or info.get("cpu", ""), cores=hardware.cores,
                    threads=hardware.threads,
                    memory_gb=round(hardware.memory_total / (1 << 30), 1)
                    if hardware.memory_total else None,
                    accelerator=hardware.accelerator, system=hardware.system)
    return info


def base_record(source: str, spec: str, memory_path: str = "",
                memory_label: str = "") -> dict[str, str]:
    """The ``base`` of a fine-tune's record: what it started from."""
    if source == "memory":
        return {"spec": memory_path, "label": memory_label or "the model in memory"}
    return {"spec": spec, "label": runs.base_label(spec)}


def finetune(engine: Any, *, source: str, spec: str, settings: Any, params: dict,
             paths: list[str], record: dict[str, Any],
             progress_callback: Callable[..., Any] | None = None, stop_event: Any = None,
             finish_event: Any = None, on_event: Event | None = None) -> Any:
    """A fine-tune as the Fine-tune tabs start it: the base (`source` "catalog" or "hub":
    imported from Hugging Face - 4-bit for QLoRA; "file": an AuraLite file; "memory": the
    model loaded now), the data of `paths` (preference pairs as dialogs for SFT), the run
    recorded (`record`: its base, settings, hardware and data) with checkpoints - not for
    preference runs, which are short and start anew."""
    from auralite.engine import PreferenceSet, load_training_data
    emit = on_event or _quiet
    if source != "memory":
        engine.unload_model()             # one model in memory, not two
        emit("unloaded")
    if source in ("catalog", "hub"):
        emit("stage", f"Getting {spec}…", 0.0)
        engine.import_hf_model(spec, quantize="nf4" if settings.method == "qlora" else None,
                               progress_callback=lambda info: emit("import", dict(info)),
                               stop_event=stop_event)
        emit("loaded", spec)
    elif source == "file":
        emit("stage", f"Loading {os.path.basename(spec)}…", None)
        engine.load_model(spec)
        emit("loaded", spec)
    if stop_event is not None and stop_event.is_set():
        return {"stopped_during": "loading the base model"}
    emit("stage", "Reading the data…", None)
    data = load_training_data(paths)
    as_dialogs = isinstance(data, PreferenceSet) and settings.objective == "sft"
    if as_dialogs:
        data = data.as_dialogs()
    pairs = isinstance(data, PreferenceSet)
    data_info = dict(record.get("data") or {}, options={"as_dialogs": True} if as_dialogs else {})
    recorder = runs.RunRecorder(
        kind="preference" if pairs else "finetune",
        method=str(params.get("method", "dpo")) if pairs else settings.method,
        params=params, **{**record, "data": data_info})
    emit("recorder", recorder)
    run_params = params if pairs else recorder.checkpoint_params(params)
    return train_recorded(engine, recorder, data, run_params, progress_callback,
                          stop_event=stop_event, finish_event=finish_event, on_event=emit)


def continue_run(engine: Any, run: Any, progress_callback: Callable[..., Any] | None = None,
                 *, stop_event: Any = None, finish_event: Any = None,
                 on_event: Event | None = None) -> Any:
    """Continue a stopped, failed or interrupted run of the library from its checkpoint,
    exactly where it stood - a new model's run or a fine-tune alike. ValueError (it cannot
    continue) comes before the model in memory is touched."""
    emit = on_event or _quiet
    path = runs.continue_point(run)
    emit("stage", f"Loading {run.name}…", None)
    engine.unload_model()
    emit("unloaded")
    engine.load_model(str(path))
    emit("continued", run.name)
    emit("stage", "Reading the data…", None)
    data = runs.load_run_data(run)
    recorder = runs.RunRecorder.resume(run)
    emit("recorder", recorder)
    params: dict[str, Any] = {"continue_training": True, "continue_run": True}
    if run.record.get("files") == "folder":
        params = recorder.checkpoint_params(params)
    return train_recorded(engine, recorder, data, params, progress_callback,
                          stop_event=stop_event, finish_event=finish_event, on_event=emit)
