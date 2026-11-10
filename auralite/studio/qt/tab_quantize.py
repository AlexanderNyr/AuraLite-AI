"""The Quantize tab: the model in memory, smaller and faster (2.25).

The classic interface's Quant tab: dynamic, static, QAT, GPTQ, AWQ and HQQ integers,
half precision and FP8, with their settings; calibration text from a file or from the
training data; the quantized model measured against the original (perplexity and
speed), every method compared, the quantized weights saved, and the original back.
"""
from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, Any

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QVBoxLayout,
                               QWidget)

from ..i18n import N_, number, tr
from ..quantize import (BITS, CALIBRATED, FP8_NOTE, METHODS, QuantSettings, bits_for, bits_note,
                        calibration_text, describe_method, read_calibration_file, runnable,
                        uses)
from ..recipes import SettingsError
from .common import Job, ask_open, ask_save, compact_spins, heading, hint, primary, show_error

if TYPE_CHECKING:
    from .window import MainWindow


class QuantizeTab(QWidget):
    def __init__(self, window: MainWindow):
        super().__init__()
        self.studio = window
        self.job: Job | None = None
        self.original: Any = None              # the model before quantizing (Back to it)
        self.dirty_before = False
        self.config: Any = None                # the last quantization's QuantConfig
        self.result: Any = None                # its QuantResult (the benchmark adds to it)
        self.qengine: Any = None
        self.calibration = ""
        self._quantized: Any = None            # the quantized model in memory
        self._own_change = False               # this tab changes the model: keep the state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        top = QHBoxLayout()
        top.addWidget(heading(tr("Quantize the model")))
        self.model_label = hint("")
        top.addWidget(self.model_label, 1)
        layout.addLayout(top)
        layout.addWidget(hint(tr("Fewer bits per weight: a smaller model in memory, often a "
                                 "faster one, at some cost in quality - the benchmark "
                                 "measures both. For llama.cpp, Ollama or LM Studio, export "
                                 "GGUF with Q8_0 or Q4 instead (the Export tab).")))

        method_box = QGroupBox(tr("Method"))
        form = QFormLayout(method_box)
        self.method = QComboBox()
        for key, label in METHODS:
            self.method.addItem(tr(label), key)
        form.addRow(tr("Method:"), self.method)
        self.bits = QComboBox()
        form.addRow(tr("Bits:"), self.bits)
        self.description = hint("")
        form.addRow(self.description)
        layout.addWidget(method_box)

        settings_box = QGroupBox(tr("Settings"))
        grid = QGridLayout(settings_box)
        self.group_size = _spin(1, 65536, 128)
        self.group_size.setToolTip(tr("Weights that share one scale (GPTQ, AWQ, HQQ): smaller "
                                      "groups keep more quality and take more memory"))
        self.percdamp = QDoubleSpinBox()
        self.percdamp.setRange(0.0, 1.0)
        self.percdamp.setDecimals(3)
        self.percdamp.setSingleStep(0.005)
        self.percdamp.setValue(0.01)
        self.percdamp.setToolTip(tr("GPTQ's dampening: a share of the mean that steadies "
                                    "the importance of the inputs"))
        self.samples = _spin(1, 100_000, 128)
        self.samples.setToolTip(tr("How many windows of the calibration text are read"))
        self.seq = _spin(8, 32768, 64)
        self.seq.setToolTip(tr("Tokens per window of the calibration text (the benchmark "
                               "and the comparison read windows of this length too)"))
        self.awq_alpha = QDoubleSpinBox()
        self.awq_alpha.setRange(0.0, 1.0)
        self.awq_alpha.setSingleStep(0.05)
        self.awq_alpha.setValue(0.5)
        self.awq_alpha.setToolTip(tr("How much the activations count against the weights when "
                                     "AWQ picks the important ones; tried along with 0.25, "
                                     "0.5, 0.75 and 1"))
        self.qat_epochs = _spin(1, 1000, 5)
        self.qat_lr = QLineEdit("1e-5")
        self.qat_lr.setMaximumWidth(110)
        self.symmetric = QCheckBox(tr("Symmetric"))
        self.symmetric.setChecked(True)
        self.symmetric.setToolTip(tr("Scales around zero (QAT, GPTQ, AWQ); off: a zero point "
                                     "as well"))
        self.per_channel = QCheckBox(tr("Per channel"))
        self.per_channel.setChecked(True)
        self.per_channel.setToolTip(tr("A scale per output channel (QAT)"))
        self.fields: dict[str, QWidget] = {
            "group_size": self.group_size, "percdamp": self.percdamp,
            "calibration_samples": self.samples, "calibration_seq": self.seq,
            "awq_alpha": self.awq_alpha, "qat_epochs": self.qat_epochs, "qat_lr": self.qat_lr,
            "symmetric": self.symmetric, "per_channel": self.per_channel}
        cells = ((tr("Group size:"), self.group_size), (tr("Dampening:"), self.percdamp),
                 (tr("Calibration windows:"), self.samples),
                 (tr("Window (tokens):"), self.seq), (tr("AWQ balance:"), self.awq_alpha),
                 (tr("QAT passes:"), self.qat_epochs), (tr("QAT learning rate:"), self.qat_lr))
        for i, (label, widget) in enumerate(cells):
            grid.addWidget(QLabel(label), i // 3, (i % 3) * 2)
            grid.addWidget(widget, i // 3, (i % 3) * 2 + 1)
        grid.addWidget(self.symmetric, 3, 0, 1, 2)
        grid.addWidget(self.per_channel, 3, 2, 1, 2)
        layout.addWidget(settings_box)

        calib_box = QGroupBox(tr("Calibration text"))
        calib = QVBoxLayout(calib_box)
        calib_row = QHBoxLayout()
        self.calib_file_button = QPushButton(tr("From a file…"))
        self.calib_data_button = QPushButton(tr("The training data"))
        self.calib_data_button.setToolTip(tr("The data chosen on the Train tab (or else on the "
                                             "Fine-tune tab)"))
        self.calib_clear_button = QPushButton(tr("Forget it"))
        for button in (self.calib_file_button, self.calib_data_button,
                       self.calib_clear_button):
            calib_row.addWidget(button)
        calib_row.addStretch(1)
        calib.addLayout(calib_row)
        self.calib_label = hint(tr("None yet. Static, QAT, GPTQ and AWQ need it; the benchmark "
                                   "and the comparison measure on it."))
        calib.addWidget(self.calib_label)
        layout.addWidget(calib_box)

        actions = QHBoxLayout()
        self.quantize_button = primary(QPushButton(tr("Quantize")))
        self.benchmark_button = QPushButton(tr("Benchmark"))
        self.benchmark_button.setToolTip(tr("Perplexity and speed: the original against the "
                                            "quantized model, on the calibration text"))
        self.compare_button = QPushButton(tr("Compare every method"))
        self.compare_button.setToolTip(tr("Quantize a copy with each method and measure it; "
                                          "the model in memory stays as it is"))
        self.save_button = QPushButton(tr("Save the quantized model…"))
        self.revert_button = QPushButton(tr("Back to the original"))
        self.stop_button = QPushButton(tr("Stop"))
        self.stop_button.setToolTip(tr("Stop the comparison after the method it is at"))
        for button in (self.quantize_button, self.benchmark_button, self.compare_button,
                       self.save_button, self.revert_button, self.stop_button):
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFormat("%p%")
        layout.addWidget(self.bar)
        self.status = QLabel(tr("Ready"))
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.output.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.output.setMinimumHeight(180)
        self.output.setPlaceholderText(tr("The results appear here"))
        layout.addWidget(self.output, 1)
        out_row = QHBoxLayout()
        self.copy_button = QPushButton(tr("Copy"))
        self.clear_button = QPushButton(tr("Clear"))
        out_row.addWidget(self.copy_button)
        out_row.addWidget(self.clear_button)
        out_row.addStretch(1)
        layout.addLayout(out_row)

        self.method.currentIndexChanged.connect(self._method_changed)
        self.bits.currentIndexChanged.connect(self._bits_changed)
        self.calib_file_button.clicked.connect(self.calibrate_from_file)
        self.calib_data_button.clicked.connect(self.calibrate_from_data)
        self.calib_clear_button.clicked.connect(lambda: self.set_calibration("", ""))
        self.quantize_button.clicked.connect(self.quantize)
        self.benchmark_button.clicked.connect(self.benchmark)
        self.compare_button.clicked.connect(self.compare)
        self.save_button.clicked.connect(self.save)
        self.revert_button.clicked.connect(self.revert)
        self.stop_button.clicked.connect(self.stop)
        self.copy_button.clicked.connect(self.copy)
        self.clear_button.clicked.connect(self.output.clear)
        window.model_changed.connect(self._model_changed)
        window.busy_changed.connect(lambda _busy: self._update_buttons())
        compact_spins(self, 120)
        self._method_changed()
        self._model_changed()

    # ---- the form ------------------------------------------------------------------
    def method_key(self) -> str:
        return str(self.method.currentData())

    def _method_changed(self, *_args: Any) -> None:
        method = self.method_key()
        current = str(self.bits.currentData() or "")
        self.bits.blockSignals(True)
        self.bits.clear()
        widths = bits_for(method)
        for width in widths:
            self.bits.addItem(tr(BITS.get(width, width)), width)
        index = self.bits.findData(current)
        if index < 0 and method == "awq":
            index = self.bits.findData("int4")         # AWQ's reason to exist
        self.bits.setCurrentIndex(max(0, index))
        self.bits.blockSignals(False)
        for name, widget in self.fields.items():
            widget.setEnabled(uses(name, method))
        self._bits_changed()

    def _bits_changed(self, *_args: Any) -> None:
        note = bits_note(str(self.bits.currentData() or ""))
        self.description.setText(describe_method(self.method_key())
                                 + (f"\n{tr(note)}" if note else ""))
        self._update_buttons()

    def runs_here(self) -> bool:
        """Whether the quantized model in memory runs (an FP8 one does not)."""
        return self.config is None or runnable(self.config.bits.value)

    def settings(self) -> QuantSettings:
        """The form as :class:`QuantSettings` (SettingsError for an unreadable number)."""
        try:
            qat_lr = float(self.qat_lr.text().strip())
        except ValueError as err:
            raise SettingsError(tr("the QAT learning rate must be a number (like 1e-5)")) \
                from err
        return QuantSettings(
            method=self.method_key(), bits=str(self.bits.currentData() or "int8"),
            group_size=int(self.group_size.value()), percdamp=float(self.percdamp.value()),
            calibration_samples=int(self.samples.value()),
            calibration_seq=int(self.seq.value()), awq_alpha=float(self.awq_alpha.value()),
            qat_epochs=int(self.qat_epochs.value()), qat_lr=qat_lr,
            symmetric=self.symmetric.isChecked(), per_channel=self.per_channel.isChecked())

    def apply_settings(self, s: QuantSettings) -> None:
        index = self.method.findData(s.method)
        if index >= 0:
            self.method.setCurrentIndex(index)
        index = self.bits.findData(s.bits)
        if index >= 0:
            self.bits.setCurrentIndex(index)
        self.group_size.setValue(int(s.group_size))
        self.percdamp.setValue(float(s.percdamp))
        self.samples.setValue(int(s.calibration_samples))
        self.seq.setValue(int(s.calibration_seq))
        self.awq_alpha.setValue(float(s.awq_alpha))
        self.qat_epochs.setValue(int(s.qat_epochs))
        self.qat_lr.setText(f"{s.qat_lr:g}")
        self.symmetric.setChecked(bool(s.symmetric))
        self.per_channel.setChecked(bool(s.per_channel))

    # ---- calibration text ----------------------------------------------------------------
    def set_calibration(self, text: str, source: str) -> None:
        self.calibration = text
        if text:
            self.calib_label.setText(tr("{source}: {n} characters", source=source,
                                        n=number(len(text))))
        else:
            self.calib_label.setText(tr("None yet. Static, QAT, GPTQ and AWQ need it; the "
                                        "benchmark and the comparison measure on it."))
        self._update_buttons()

    def calibrate_from_file(self) -> None:
        path = ask_open(self, tr("Calibration text"), tr("Text (*.txt *.md);;All files (*)"),
                        self.studio.settings.last_data_dir)
        if not path:
            return
        try:
            text = read_calibration_file(path)
        except OSError as err:
            show_error(self, err, tr("reading the calibration text"))
            return
        if not text.strip():
            show_error(self, SettingsError(tr("{name} holds no text",
                                              name=os.path.basename(path))))
            return
        self.set_calibration(text, os.path.basename(path))

    def training_paths(self) -> list[str]:
        """The data of the Train tab, or else of the Fine-tune tab."""
        for tab in (self.studio.train_tab, self.studio.finetune_tab):
            data = getattr(tab, "data", None)
            paths = data.paths() if data is not None else []
            if paths:
                return list(paths)
        return []

    def calibrate_from_data(self) -> None:
        paths = self.training_paths()
        if not paths:
            show_error(self, SettingsError(tr("choose data on the Train or Fine-tune tab "
                                              "first, or a file here")))
            return
        try:
            text = calibration_text(paths)
        except (OSError, ValueError) as err:
            show_error(self, err, tr("reading the training data"))
            return
        if not text.strip():
            show_error(self, SettingsError(tr("the training data holds no text")))
            return
        self.set_calibration(text, os.path.basename(os.path.normpath(paths[0])))

    # ---- state ----------------------------------------------------------------------------
    def native_model(self) -> bool:
        engine = self.studio.engine
        return engine.model is not None and not engine.is_gguf_model() and \
            not engine.is_hf_model()

    def _model_changed(self) -> None:
        engine = self.studio.engine
        if self.original is not None and self.job is None and not self._own_change and \
                engine.model is not self._quantized:
            # another model came in (opened, trained): the original no longer applies
            self.original = None
            self.result = None
            self.config = None
        if engine.model is None:
            text = tr("No model yet: train, fine-tune or open one first.")
        elif not self.native_model():
            text = tr("{model}: only AuraLite's own models (.pt) quantize here",
                      model=self.studio.model_description())
        else:
            text = self.studio.model_description()
        self.model_label.setText(text)
        self._update_buttons()

    def _update_buttons(self) -> None:
        idle = self.job is None and not self.studio.busy
        native = self.native_model()
        method = self.method_key()
        calibrated = bool(self.calibration)
        self.quantize_button.setEnabled(idle and native and self.original is None and
                                        (calibrated or method not in CALIBRATED))
        self.benchmark_button.setEnabled(idle and native and self.original is not None and
                                         calibrated and self.runs_here())
        self.compare_button.setEnabled(idle and native and self.original is None and calibrated)
        ok = self.result is not None and not getattr(self.result, "errors", None)
        self.save_button.setEnabled(idle and native and self.original is not None and ok)
        self.revert_button.setEnabled(idle and self.original is not None)
        self.stop_button.setEnabled(self.job is not None and self.job.name == "compare")
        quantized = self.original is not None
        self.quantize_button.setToolTip(
            tr("Quantized already: go back to the original to quantize it another way")
            if quantized else
            tr("{method} needs calibration text", method=method.upper())
            if method in CALIBRATED and not calibrated else "")

    # ---- the work ----------------------------------------------------------------------------
    def _progress(self, job: Job) -> Any:
        def callback(step: Any, total: Any, message: Any) -> None:
            job.progress.emit((step, total, str(message)))
        return callback

    def _show_progress(self, event: tuple) -> None:
        step, total, message = event
        try:
            percent = 100.0 * float(step) / float(total) if total else 0.0
        except (TypeError, ValueError):
            percent = 0.0
        self.bar.setValue(int(round(max(0.0, min(100.0, percent)) * 10)))
        self.status.setText(message)

    def _start(self, name: str, busy: str, work: Any, done: Any, text: str) -> None:
        if not self.studio.begin(busy):
            return
        self.bar.setValue(0)
        self.status.setText(text)
        self.job = Job(name, work, self)
        self.job.progress.connect(self._show_progress)
        self.job.finished.connect(done)
        self.job.failed.connect(lambda exc, n=name: self._failed(exc, n))
        self._update_buttons()
        self.job.start()

    def quantize(self) -> None:
        if not self.native_model():
            show_error(self, SettingsError(tr("open or train an AuraLite model (.pt) first")))
            return
        try:
            settings = self.settings()
            config = settings.config(self.calibration)
        except (SettingsError, ValueError) as err:
            show_error(self, err if isinstance(err, SettingsError) else SettingsError(str(err)))
            return
        if settings.method in CALIBRATED and not self.calibration:
            show_error(self, SettingsError(tr("{method} needs calibration text: a file, or the "
                                              "training data",
                                              method=settings.method.upper())))
            return
        errors = config.validate()
        if errors:
            show_error(self, SettingsError("; ".join(errors)))
            return
        from auralite.quantization import QuantizationEngine
        engine = self.studio.engine
        self.qengine = qengine = QuantizationEngine()

        def work(job: Job) -> Any:
            job.progress.emit((0, 1, tr("Copying the original…")))
            original = copy.deepcopy(engine.model)
            model, result = qengine.quantize(engine.model, config, tokenizer=engine.tokenizer,
                                             device=engine.device,
                                             progress_callback=self._progress(job))
            return original, model, result

        self.config = config
        self.dirty_before = self.studio.model_dirty
        self._start("quantize", "quantizing", work, self._quantized_done,
                    tr("Quantizing ({method}, {bits})…", method=settings.method,
                       bits=settings.bits))

    def _quantized_done(self, outcome: Any) -> None:
        original, model, result = outcome
        self.job = None
        self.result = result
        engine = self.studio.engine
        self.output.setPlainText(result.summary())
        if result.errors:
            engine.model = original            # whatever a method changed in place is undone
            self.status.setText(tr("The quantization failed: the model is as it was."))
            self.bar.setValue(0)
            self._own_change = True
            try:
                self.studio.end(model_dirty=None)
            finally:
                self._own_change = False
        else:
            self.original = original
            engine.model = model
            self._quantized = model
            try:
                engine.disable_inference_compile()
            except Exception:
                pass
            self.bar.setValue(1000)
            self.status.setText(tr("Quantized: {ratio}× smaller ({before} MB → {after} MB)",
                                   ratio=f"{result.compression_ratio:.2f}",
                                   before=f"{result.original_size_mb:.1f}",
                                   after=f"{result.quantized_size_mb:.1f}")
                                + ("" if self.runs_here() else " " + tr(FP8_NOTE)))
            self._own_change = True
            try:
                self.studio.end(model_dirty=True)
            finally:
                self._own_change = False
        self._update_buttons()

    def benchmark(self) -> None:
        if self.original is None or not self.calibration:
            show_error(self, SettingsError(tr("quantize first, with calibration text to "
                                              "measure on")))
            return
        from auralite.quantization import QuantizationEngine
        engine = self.studio.engine
        qengine = self.qengine or QuantizationEngine()
        original, text, seq = self.original, self.calibration, int(self.seq.value())

        def work(job: Job) -> Any:
            return qengine.benchmark(original, engine.model, engine.tokenizer, text,
                                     engine.device, seq_length=seq,
                                     progress_callback=self._progress(job))

        self._start("benchmark", N_("a benchmark"), work, self._benchmarked,
                    tr("Measuring the original and the quantized model…"))

    def _benchmarked(self, result: Any) -> None:
        self.job = None
        self.result = result
        self.output.setPlainText(tr("BENCHMARK") + "\n" + "-" * 50 + "\n" + result.summary())
        self.bar.setValue(1000)
        self.status.setText(tr("Measured: perplexity {delta}, speed {speed}×",
                               delta=f"{result.perplexity_delta:+.2f}",
                               speed=f"{result.speedup:.2f}"))
        self.studio.end(model_dirty=None)
        self._update_buttons()

    def compare(self) -> None:
        if not self.native_model() or not self.calibration:
            show_error(self, SettingsError(tr("the comparison needs a model and calibration "
                                              "text")))
            return
        from auralite.quantization import compare_quantizations, format_comparison_table
        engine = self.studio.engine
        text, seq = self.calibration, int(self.seq.value())

        def work(job: Job) -> Any:
            job.progress.emit((0, 1, tr("Copying the model…")))
            model = copy.deepcopy(engine.model)
            stop = job.stop_event

            def progress(step: Any, total: Any, message: Any) -> None:
                if stop.is_set():
                    raise InterruptedError("stopped")
                job.progress.emit((step, total, str(message)))

            results = compare_quantizations(model, engine.tokenizer, text, engine.device,
                                            seq_length=seq, progress_callback=progress)
            return results, format_comparison_table(results)

        self._start("compare", N_("comparing the methods"), work, self._compared,
                    tr("Comparing every method…"))

    def _compared(self, outcome: Any) -> None:
        results, table = outcome
        self.job = None
        self.output.setPlainText(tr("EVERY METHOD") + "\n" + table)
        self.bar.setValue(1000)
        self.status.setText(tr("Compared {n} methods: the model in memory is as it was.",
                               n=len(results)))
        self.studio.end(model_dirty=None)
        self._update_buttons()

    def _failed(self, exc: BaseException, name: str) -> None:
        self.job = None
        self.bar.setValue(0)
        self.studio.end(model_dirty=None)
        self._update_buttons()
        if isinstance(exc, InterruptedError):
            self.status.setText(tr("Stopped."))
            return
        self.status.setText(tr("Failed - see the message."))
        self.studio.report_error(exc, {"quantize": "quantizing the model",
                                       "benchmark": "the benchmark",
                                       "compare": "comparing the methods"}.get(name, name))

    def stop(self) -> None:
        if self.job is not None:
            self.job.stop_event.set()
            self.status.setText(tr("Stopping after this method…"))

    # ---- after -------------------------------------------------------------------------------
    def save(self) -> None:
        engine = self.studio.engine
        if engine.model is None or self.original is None:
            return
        start = os.path.join(self.studio.settings.last_model_dir or os.path.expanduser("~"),
                             "model-quantized.pt")
        path = ask_save(self, tr("Save the quantized model"), tr("Quantized model (*.pt)"),
                        start)
        if not path:
            return
        if not path.endswith(".pt"):
            path += ".pt"
        from auralite.quantization import QuantConfig, QuantizationEngine
        try:
            QuantizationEngine.save_quantized(engine.model, path, self.config or QuantConfig(),
                                              tokenizer=engine.tokenizer,
                                              params_used=engine.params_used,
                                              result=self.result)
        except Exception as err:
            self.studio.report_error(err, tr("saving the quantized model"))
            return
        self.status.setText(tr("Saved: {path} - the quantized weights for PyTorch "
                               "(torch.load); AuraLite opens .pt checkpoints and GGUF files",
                               path=path))

    def revert(self) -> None:
        if self.original is None or self.job is not None:
            return
        engine = self.studio.engine
        engine.model = self.original
        try:
            engine.disable_inference_compile()
        except Exception:
            pass
        self.original = None
        self._quantized = None
        self.result = None
        self.config = None
        self.status.setText(tr("Back to the original model."))
        self.bar.setValue(0)
        self._own_change = True
        try:
            self.studio.model_dirty = self.dirty_before
            self.studio.model_changed.emit()
        finally:
            self._own_change = False
        self._update_buttons()

    def copy(self) -> None:
        text = self.output.toPlainText()
        if text.strip():
            QApplication.clipboard().setText(text)
            self.status.setText(tr("The results are copied."))


def _spin(low: int, high: int, value: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(low, high)
    spin.setValue(value)
    return spin
