"""The Generate tab: the model writes on from a prompt, with every sampling setting (2.24).

The classic interface's Generate tab: temperature, top-k, top-p, min-p and the
repetition penalty; the length (or a suggested one); the whole history instead of the
trained window; token by token or at once; several prompts in one batch; a draft
first ("thinking"); snippets of a web search as context; the text saved, appended
to a file or copied.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from ..i18n import N_, tr
from .common import Job, ask_save, compact_spins, heading, hint, primary, show_error

if TYPE_CHECKING:
    from .window import MainWindow

_RULE = "\n" + "-" * 40 + "\n"


class GenerateTab(QWidget):
    def __init__(self, window: MainWindow):
        super().__init__()
        self.studio = window
        self.job: Job | None = None
        self._started = 0.0
        self._pieces: list[str] = []
        self._count = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        top = QHBoxLayout()
        top.addWidget(heading(tr("Generate text")))
        self.model_label = hint("")
        top.addWidget(self.model_label, 1)
        self.open_button = QPushButton(tr("Open a model…"))
        top.addWidget(self.open_button)
        layout.addLayout(top)
        layout.addWidget(hint(tr("The model writes on from your text, with every sampling "
                                 "setting at hand. To talk with a chat model, use the Chat "
                                 "tab.")))

        prompt_box = QGroupBox(tr("Prompt"))
        prompt_layout = QVBoxLayout(prompt_box)
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText(tr("The start of a text: the model writes on from it"))
        self.prompt.setMaximumHeight(110)
        prompt_layout.addWidget(self.prompt)
        self.batch = QCheckBox(tr("One prompt per line, written as one batch"))
        prompt_layout.addWidget(self.batch)
        length_row = QHBoxLayout()
        length_row.addWidget(QLabel(tr("Length (tokens):")))
        self.length = QSpinBox()
        self.length.setRange(1, 16384)
        self.length.setValue(100)
        length_row.addWidget(self.length)
        self.auto_length = QPushButton(tr("Suggest"))
        self.auto_length.setToolTip(tr("A length for the prompt and the model's context"))
        length_row.addWidget(self.auto_length)
        length_row.addStretch(1)
        prompt_layout.addLayout(length_row)
        layout.addWidget(prompt_box)

        sampling = QGroupBox(tr("Sampling"))
        grid = QGridLayout(sampling)
        self.temperature = _double(0.0, 2.0, 0.8, 0.05)
        self.temperature.setToolTip(tr("Lower: safer, more repetitive; higher: freer, more "
                                       "mistakes"))
        self.top_k = QSpinBox()
        self.top_k.setRange(0, 100_000)
        self.top_k.setValue(50)
        self.top_k.setSpecialValueText(tr("off"))
        self.top_p = _double(0.0, 1.0, 0.9, 0.05)
        self.min_p = _double(0.0, 1.0, 0.0, 0.01)
        self.repetition = _double(1.0, 3.0, 1.0, 0.05)
        self.repetition.setToolTip(tr("1.0 = off; 1.1-1.3 fights loops"))
        cells = ((tr("Temperature:"), self.temperature), (tr("Top-k:"), self.top_k),
                 (tr("Top-p:"), self.top_p), (tr("Min-p:"), self.min_p),
                 (tr("Repetition penalty:"), self.repetition))
        for i, (label, widget) in enumerate(cells):
            grid.addWidget(QLabel(label), i // 3, (i % 3) * 2)
            grid.addWidget(widget, i // 3, (i % 3) * 2 + 1)
        layout.addWidget(sampling)

        options = QGroupBox(tr("How"))
        form = QFormLayout(options)
        self.stream = QCheckBox(tr("Show the text as it is written"))
        self.stream.setChecked(True)
        form.addRow(self.stream)
        self.thinking = QCheckBox(tr("Thinking: a draft first, then the answer written with it"))
        form.addRow(self.thinking)
        self.web = QCheckBox(tr("Search the web (DuckDuckGo) and give the model the snippets"))
        form.addRow(self.web)
        self.web_query = QLineEdit()
        self.web_query.setPlaceholderText(tr("What to search (empty: the prompt)"))
        self.web_query.setVisible(False)
        form.addRow(self.web_query)
        self.full_context = QCheckBox(tr("The whole history instead of the trained window"))
        self.context_hint = hint("")
        form.addRow(self.full_context)
        form.addRow(self.context_hint)
        layout.addWidget(options)

        buttons = QHBoxLayout()
        self.generate_button = primary(QPushButton(tr("Generate")))
        self.stop_button = QPushButton(tr("Stop"))
        buttons.addWidget(self.generate_button)
        buttons.addWidget(self.stop_button)
        self.speed = hint("")
        buttons.addWidget(self.speed, 1)
        layout.addLayout(buttons)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(200)
        self.output.setPlaceholderText(tr("The generated text appears here"))
        layout.addWidget(self.output, 1)
        out_buttons = QHBoxLayout()
        self.clear_button = QPushButton(tr("Clear"))
        self.copy_button = QPushButton(tr("Copy"))
        self.save_button = QPushButton(tr("Save…"))
        self.append_button = QPushButton(tr("Append to a file…"))
        for button in (self.clear_button, self.copy_button, self.save_button,
                       self.append_button):
            out_buttons.addWidget(button)
        out_buttons.addStretch(1)
        layout.addLayout(out_buttons)

        self.timer = QTimer(self)
        self.timer.setInterval(70)
        self.timer.timeout.connect(self._flush)
        self.generate_button.clicked.connect(self.generate)
        self.stop_button.clicked.connect(self.stop)
        self.auto_length.clicked.connect(self.suggest_length)
        self.clear_button.clicked.connect(self.output.clear)
        self.copy_button.clicked.connect(self.copy)
        self.save_button.clicked.connect(lambda: self.save(append=False))
        self.append_button.clicked.connect(lambda: self.save(append=True))
        self.open_button.clicked.connect(window.open_model_dialog)
        self.web.toggled.connect(self.web_query.setVisible)
        self.full_context.toggled.connect(lambda _c: self._update_context_hint())
        window.model_changed.connect(self._model_changed)
        window.busy_changed.connect(lambda _busy: self._update_buttons())
        compact_spins(self, 110)
        self._model_changed()

    # ---- state ---------------------------------------------------------------
    def _model_changed(self) -> None:
        engine = self.studio.engine
        self.model_label.setText(self.studio.model_description() if engine.model is not None
                                 else tr("No model yet: train, fine-tune or open one first."))
        self._update_context_hint()
        self._update_buttons()

    def _update_buttons(self) -> None:
        has_model = self.studio.engine.model is not None
        idle = self.job is None
        self.generate_button.setEnabled(has_model and idle and not self.studio.busy)
        self.auto_length.setEnabled(has_model and idle)
        self.stop_button.setEnabled(not idle)

    def _update_context_hint(self) -> None:
        trained = getattr(self.studio.engine.model, "max_seq_len", None)
        if self.full_context.isChecked():
            text = (tr("Every token sees the whole history: past the trained length ({n} "
                       "tokens) the text gets worse.", n=trained) if trained else
                    tr("Every token sees the whole history: past the trained length the "
                       "text gets worse."))
        else:
            text = (tr("Every token sees the newest {n} tokens (the trained length).",
                       n=trained) if trained else
                    tr("Every token sees as many tokens as the model was trained on."))
        self.context_hint.setText(text)

    def sampling(self) -> dict[str, Any]:
        """The sampling settings as the engine's generate() takes them."""
        return {"temperature": float(self.temperature.value()), "top_k": int(self.top_k.value()),
                "top_p": float(self.top_p.value()),
                "repetition_penalty": float(self.repetition.value()),
                "min_p": float(self.min_p.value())}

    def suggest_length(self) -> None:
        """A length for the prompt and the model's context (the classic «Auto»)."""
        from auralite.engine import recommend_gen_length
        engine = self.studio.engine
        model = engine.model
        trained = int(getattr(model, "max_seq_len", 0) or 4096)
        length = recommend_gen_length(self.prompt.toPlainText(), engine.tokenizer,
                                      max_seq_len=trained,
                                      full_context=self.full_context.isChecked())
        self.length.setValue(max(1, min(int(length), self.length.maximum())))

    # ---- generating ----------------------------------------------------------
    def generate(self) -> None:
        text = self.prompt.toPlainText()
        prompts = [line.strip() for line in text.splitlines() if line.strip()] \
            if self.batch.isChecked() else []
        if not text.strip() or (self.batch.isChecked() and not prompts):
            show_error(self, _error(tr("write a prompt first")))
            return
        engine = self.studio.engine
        if engine.model is None:
            return
        try:                                  # between generations: it drops the cache
            engine.set_context_window("full" if self.full_context.isChecked() else "auto")
        except Exception:                     # an engine without the 2.11 API
            pass
        if not self.studio.begin(N_("writing"), quiet=True):
            return
        length = int(self.length.value())
        options = self.sampling()
        thinking, web = self.thinking.isChecked(), self.web.isChecked()
        query = self.web_query.text().strip()
        stream = self.stream.isChecked() and not prompts and not thinking and not web

        def work(job: Job) -> Any:
            if prompts:                       # one batch: every prompt at once
                results = engine.generate_batch(prompts, length, **options)
                return "\n".join(f"{_RULE}{tr('Prompt {n}:', n=i + 1)} {p}{_RULE}{r}\n"
                                 for i, (p, r) in enumerate(zip(prompts, results, strict=True)))
            if thinking or web:
                return _think_and_search(engine, job, text, length, options, thinking,
                                         query if web else None)
            if stream:
                job.progress.emit(text)
                for piece in engine.generate_streaming(text, length, **options):
                    if job.stop_event.is_set():
                        break
                    job.progress.emit(str(piece))
                return None
            return engine.generate(text, length, **options)

        self.output.clear()
        self._pieces = []
        self._count = 0
        self._started = time.time()
        self.speed.setText(tr("Writing…"))
        self.job = Job("generate", work, self)
        self.job.progress.connect(self._piece)
        self.job.finished.connect(self._done)
        self.job.failed.connect(self._failed)
        self._update_buttons()
        self.timer.start()
        self.job.start()

    def _piece(self, piece: str) -> None:
        self._pieces.append(piece)
        self._count += 1

    def _flush(self) -> None:
        if not self._pieces:
            return
        text, self._pieces = "".join(self._pieces), []
        self.output.moveCursor(QTextCursor.MoveOperation.End)
        self.output.insertPlainText(text)
        self.output.moveCursor(QTextCursor.MoveOperation.End)
        seconds = max(1e-6, time.time() - self._started)
        self.speed.setText(tr("{rate} pieces/s", rate=f"{max(0, self._count - 1) / seconds:.1f}"))

    def _done(self, result: Any) -> None:
        self.timer.stop()
        self._flush()
        stopped = self.job is not None and self.job.stop_event.is_set()
        self.job = None
        if isinstance(result, str):
            self.output.setPlainText(result)
        seconds = time.time() - self._started
        self.speed.setText(tr("Stopped after {s} s", s=f"{seconds:.1f}") if stopped else
                           tr("Done in {s} s", s=f"{seconds:.1f}"))
        self.studio.end(model_dirty=None)
        self._update_buttons()

    def _failed(self, exc: BaseException) -> None:
        self.timer.stop()
        self.job = None
        self.speed.setText("")
        self.studio.end(model_dirty=None)
        self._update_buttons()
        self.studio.report_error(exc, tr("the generation"))

    def stop(self) -> None:
        if self.job is not None:
            self.job.stop_event.set()

    # ---- the text --------------------------------------------------------------
    def copy(self) -> None:
        QApplication.clipboard().setText(self.output.toPlainText())
        self.speed.setText(tr("Copied {n} characters", n=f"{len(self.output.toPlainText()):,}"))

    def save(self, append: bool = False) -> None:
        text = self.output.toPlainText()
        if not text.strip():
            show_error(self, _error(tr("there is no text to save yet")))
            return
        path = ask_save(self, tr("Append to a file") if append else tr("Save the text"),
                        tr("Text (*.txt);;All files (*)"), "generated.txt")
        if not path:
            return
        try:
            with open(path, "a" if append else "w", encoding="utf-8") as f:
                f.write(("\n\n" if append else "") + text)
        except OSError as err:
            show_error(self, err, tr("saving the text"))
            return
        self.speed.setText(tr("Appended to {path}", path=path) if append else
                           tr("Saved: {path}", path=path))


def _think_and_search(engine: Any, job: Job, prompt: str, length: int, options: dict,
                      thinking: bool, query: str | None) -> str:
    """The classic interface's thinking and web-search modes: the snippets of a search
    (when `query` is not None), a draft (with `thinking`), then the answer."""
    context = ""
    if query is not None:
        job.progress.emit(tr("Searching the web…") + "\n")
        from auralite.web_tools import build_web_context
        try:
            context = build_web_context(query or prompt, max_results=4) or ""
        except Exception as err:          # no network: the answer is written without it
            context = ""
            job.progress.emit(tr("The search failed ({error}): writing without it.",
                                 error=err) + "\n")
    if thinking:
        job.progress.emit(tr("Thinking…") + "\n")
        options = dict(options)
        thoughts, final = engine.generate_with_thinking(
            prompt, length, options.pop("temperature"), options.pop("top_k"),
            options.pop("top_p"), repetition_penalty=options.pop("repetition_penalty"),
            web_context=context or None, min_p=options.pop("min_p"))
    else:
        conditioned = f"{context}\n{prompt}" if context else prompt
        full = engine.generate(conditioned, length, **options)
        thoughts, final = "", prompt + full[len(conditioned):]
    parts = []
    if context:
        parts.append(tr("WEB CONTEXT") + _RULE + context)
    if thoughts:
        parts.append(tr("THINKING (the draft)") + _RULE + thoughts)
    parts.append(tr("ANSWER") + _RULE + final)
    return "\n\n".join(parts)


def _double(low: float, high: float, value: float, step: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setSingleStep(step)
    spin.setDecimals(2)
    spin.setValue(value)
    return spin


def _error(text: str) -> Exception:
    from ..recipes import SettingsError
    return SettingsError(text)
