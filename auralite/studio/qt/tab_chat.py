"""The Chat tab: talk to the model in memory, reply by reply as it writes (2.19).

2.26 adds the classic Chat tab's agent mode - the model calls tools (Python, commands,
files in a temporary folder, a web search, a calculator) and reads their results before
it answers, with a log of what ran - and its insights: the last reply's most frequent
tokens, and one layer's attention over the last exchange.
"""
from __future__ import annotations

import html
import json
import time
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
                               QScrollArea, QSpinBox, QSplitter, QTextBrowser, QVBoxLayout,
                               QWidget)

from .. import agentmode, insights
from ..i18n import tr
from ..recipes import SettingsError
from .common import (Collapsible, Job, ask_save, confirm, heading, hint, inform, primary,
                     show_error)
from .widgets import InsightView

if TYPE_CHECKING:
    from .window import MainWindow

#: 2.21.3: the Template list's entry that writes on from the message instead of chatting
TEXT_MODE = "text"
#: 2.26: the sandbox log's colors (the classic Chat tab's)
LOG_COLORS = {"ok": "#6a9955", "error": "#f44747", "info": "#4a90c2", "tool": "#b5a642"}
#: 2.26: the insights' bars show this many tokens
TOP_TOKENS = 20


class _EnterSends(QObject):
    """Enter sends the message; Shift+Enter starts a new line."""

    def __init__(self, tab: ChatTab):
        super().__init__(tab)
        self.tab = tab

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt's name
        if event.type() == QEvent.Type.KeyPress:
            key = getattr(event, "key", lambda: None)()
            modifiers = getattr(event, "modifiers", lambda: Qt.KeyboardModifier.NoModifier)()
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and \
                    not (modifiers & Qt.KeyboardModifier.ShiftModifier):
                self.tab.send()
                return True
        return False


class ChatTab(QWidget):
    def __init__(self, window: MainWindow):
        super().__init__()
        self.studio = window
        self.messages: list[dict[str, str]] = []
        self.reply = ""
        self.job: Job | None = None
        self._started = 0.0
        self._pieces = 0
        self._dirty = False
        self._fit_pending = False
        self.agent: Any = None                  # 2.26: the agent answering now
        self.sandbox: Any = None                # its temporary folder, kept for the chat
        self._agent_accepted = False            # the warning was read and accepted

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        top = QHBoxLayout()
        top.addWidget(heading("Chat"))
        self.model_label = hint("")
        top.addWidget(self.model_label, 1)
        self.open_button = QPushButton("Open a model…")
        top.addWidget(self.open_button)
        layout.addLayout(top)
        self.slow_hint = hint("This model's weights are in 4 bits, which makes every reply "
                              "slow: merge the adapters and export to GGUF (Export tab) for fast "
                              "chat in llama.cpp, Ollama or LM Studio.")
        self.slow_hint.setVisible(False)
        layout.addWidget(self.slow_hint)
        # 2.21.3: a model trained on plain text continues text (text_mode, TEXT_MODE)
        self.text_hint = hint("This model learned plain text, not dialogs: it continues what "
                              "you write instead of answering - type the start of a text. To "
                              "chat, train or fine-tune a model on dialogs (.jsonl).")
        self.text_hint.setVisible(False)
        layout.addWidget(self.text_hint)

        settings = QHBoxLayout()
        self.system = QLineEdit()
        self.system.setPlaceholderText("System prompt (optional): who the assistant is, how it "
                                       "answers")
        settings.addWidget(self.system, 3)
        settings.addWidget(QLabel("Temperature:"))
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(0.7)
        settings.addWidget(self.temperature)
        settings.addWidget(QLabel("Max tokens:"))
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(8, 8192)
        self.max_tokens.setValue(256)
        settings.addWidget(self.max_tokens)
        settings.addWidget(QLabel("Template:"))
        self.template = QComboBox()
        self.template.addItem("auto", "")
        self.template.addItem("continue the text", TEXT_MODE)
        from auralite.engine import CHAT_TEMPLATES
        for name in CHAT_TEMPLATES:
            self.template.addItem(name, name)
        self.template.setToolTip("How messages reach the model: auto - its own chat template, "
                                 "or text it continues for a model trained on plain text; "
                                 "continue the text - the message is the start of a text the "
                                 "model writes on; or a chat template by name")
        settings.addWidget(self.template)
        layout.addLayout(settings)

        # 2.26: agent mode (the classic Chat tab's)
        agent_row = QHBoxLayout()
        self.agent_mode = QCheckBox(tr("Agent mode: the model may use tools"))
        self.agent_mode.setToolTip(tr("Python, commands, files in a temporary folder, a web "
                                      "search, a calculator: the model calls them and reads "
                                      "their results before it answers"))
        agent_row.addWidget(self.agent_mode)
        agent_row.addStretch(1)
        layout.addLayout(agent_row)

        # 2.26: the chat above, agent mode and the insights below: a splitter between them
        chat_area = QWidget()
        chat_layout = QVBoxLayout(chat_area)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(True)
        self.view.setMinimumHeight(160)
        chat_layout.addWidget(self.view, 1)

        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("Write a message - Enter sends, Shift+Enter starts a new "
                                      "line")
        self.input.setMaximumHeight(90)
        self.input.installEventFilter(_EnterSends(self))
        chat_layout.addWidget(self.input)
        buttons = QHBoxLayout()
        self.send_button = primary(QPushButton("Send"))
        self.stop_button = QPushButton("Stop")
        self.new_button = QPushButton("New chat")
        self.save_button = QPushButton("Save the chat…")
        for button in (self.send_button, self.stop_button, self.new_button, self.save_button):
            buttons.addWidget(button)
        self.speed = hint("")
        buttons.addWidget(self.speed, 1)
        chat_layout.addLayout(buttons)
        extras = QWidget()
        extras_layout = QVBoxLayout(extras)
        extras_layout.setContentsMargins(0, 0, 0, 0)
        self._build_agent_section(extras_layout)
        self._build_insights_section(extras_layout)
        extras_layout.addStretch(1)
        self.extras = QScrollArea()
        self.extras.setWidgetResizable(True)
        self.extras.setFrameShape(QFrame.Shape.NoFrame)
        self.extras.setWidget(extras)
        self.extras.setMinimumHeight(56)
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(chat_area)
        self.splitter.addWidget(self.extras)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        layout.addWidget(self.splitter, 1)
        self.agent_section.button.toggled.connect(lambda _open: self._fit_extras())
        self.insights_section.button.toggled.connect(lambda _open: self._fit_extras())

        self.timer = QTimer(self)
        self.timer.setInterval(70)
        self.timer.timeout.connect(self._flush)
        self.send_button.clicked.connect(self.send)
        self.stop_button.clicked.connect(self.stop)
        self.new_button.clicked.connect(self.new_chat)
        self.save_button.clicked.connect(self.save_chat)
        self.open_button.clicked.connect(window.open_model_dialog)
        window.model_changed.connect(self._model_changed)
        window.busy_changed.connect(lambda _busy: self._update_buttons())
        self.template.currentIndexChanged.connect(lambda _i: self._update_mode())
        self.agent_mode.toggled.connect(self._agent_toggled)
        self.clear_log_button.clicked.connect(self.sandbox_log.clear)
        self.tokens_button.clicked.connect(self.show_token_frequencies)
        self.attention_button.clicked.connect(self.show_attention)
        self._model_changed()
        self.render()

    def _fit_extras(self) -> None:
        """2.26: room for an opened section (half the height), little for closed ones; on
        a tab not shown yet, once it shows (its splitter has no height before)."""
        total = sum(self.splitter.sizes())
        if total <= 0 or not self.splitter.isVisible():
            self._fit_pending = True
            return
        self._fit_pending = False
        opened = self.agent_section.button.isChecked() or \
            self.insights_section.button.isChecked()
        below = total // 2 if opened else self.extras.minimumHeight()
        self.splitter.setSizes([max(1, total - below), below])

    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        super().showEvent(event)
        if self._fit_pending:
            QTimer.singleShot(0, self._fit_extras)

    def _build_agent_section(self, layout: QVBoxLayout) -> None:
        """2.26: the tools the agent may call, its limits and the log of what ran."""
        self.agent_section = Collapsible(tr("Agent mode: tools and the sandbox log"))
        body = self.agent_section.body_layout
        tools = QGridLayout()
        self.tool_boxes: dict[str, QCheckBox] = {}
        for i, row in enumerate(agentmode.tool_rows()):
            box = QCheckBox(f"{row.name} - {row.description}")
            box.setChecked(row.available)
            box.setEnabled(row.available)
            if row.note:
                box.setToolTip(row.note)
                box.setText(f"{box.text()} ({row.note})")
            self.tool_boxes[row.name] = box
            tools.addWidget(box, i // 2, i % 2)
        body.addLayout(tools)
        limits = QHBoxLayout()
        limits.addWidget(QLabel(tr("Steps at most:")))
        self.agent_steps = QSpinBox()
        self.agent_steps.setRange(*agentmode.STEPS)
        steps, seconds = agentmode.form_defaults()      # the AURALITE_AGENT_* variables
        self.agent_steps.setValue(steps)
        self.agent_steps.setToolTip(tr("How many times the model may call tools before it "
                                       "must answer"))
        limits.addWidget(self.agent_steps)
        limits.addWidget(QLabel(tr("Seconds per tool:")))
        self.agent_timeout = QSpinBox()
        self.agent_timeout.setRange(*agentmode.TIMEOUTS)
        self.agent_timeout.setValue(max(1, round(seconds)))
        limits.addWidget(self.agent_timeout)
        limits.addWidget(hint(tr("Each step writes at least {n} tokens.",
                                 n=agentmode.DEFAULTS["max_tokens_per_step"])))
        limits.addStretch(1)
        self.clear_log_button = QPushButton(tr("Clear the log"))
        limits.addWidget(self.clear_log_button)
        body.addLayout(limits)
        self.sandbox_log = QPlainTextEdit()
        self.sandbox_log.setReadOnly(True)
        self.sandbox_log.setMaximumBlockCount(500)
        self.sandbox_log.setMaximumHeight(130)
        self.sandbox_log.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.sandbox_log.setPlaceholderText(tr("The tools the agent calls, and what they "
                                               "return, appear here"))
        body.addWidget(self.sandbox_log)
        layout.addWidget(self.agent_section)

    def _build_insights_section(self, layout: QVBoxLayout) -> None:
        """2.26: the last reply's tokens and one layer's attention (the classic Insights)."""
        self.insights_section = Collapsible(tr("Insights: the reply's tokens and the "
                                               "model's attention"))
        body = self.insights_section.body_layout
        row = QHBoxLayout()
        self.tokens_button = QPushButton(tr("Token frequencies"))
        self.tokens_button.setToolTip(tr("The most frequent tokens of the model's last reply"))
        self.attention_button = QPushButton(tr("Attention map"))
        self.attention_button.setToolTip(tr("Where each token of the last exchange looked, in "
                                            "one layer (AuraLite's own models)"))
        row.addWidget(self.tokens_button)
        row.addWidget(self.attention_button)
        row.addWidget(QLabel(tr("Layer:")))
        self.layer = QSpinBox()
        self.layer.setRange(0, 0)
        row.addWidget(self.layer)
        row.addWidget(QLabel(tr("Head:")))
        self.head = QSpinBox()
        self.head.setRange(0, 256)
        self.head.setSpecialValueText(tr("all (average)"))
        row.addWidget(self.head)
        row.addWidget(QLabel(tr("Tokens:")))
        self.attention_tokens = QSpinBox()
        self.attention_tokens.setRange(*insights.TOKENS_RANGE)
        self.attention_tokens.setValue(insights.DEFAULT_TOKENS)
        self.attention_tokens.setToolTip(tr("The map covers the last tokens of the exchange: "
                                            "up to 48 are labelled"))
        row.addWidget(self.attention_tokens)
        row.addStretch(1)
        body.addLayout(row)
        self.insight_view = InsightView(empty_text=tr("Token frequencies and attention maps "
                                                      "appear here"))
        body.addWidget(self.insight_view)
        self.insight_status = hint("")
        body.addWidget(self.insight_status)
        layout.addWidget(self.insights_section)

    # ---- state ---------------------------------------------------------------
    def _model_changed(self) -> None:
        engine = self.studio.engine
        self.model_label.setText(self.studio.model_description() if engine.model is not None
                                 else "no model yet: train, fine-tune or open one")
        quantized = False
        model = engine.model
        if model is not None:
            try:
                from auralite.engine.nf4 import nf4_layers
                quantized = bool(nf4_layers(getattr(model, "module", model)))
            except Exception:
                quantized = False
        self.slow_hint.setVisible(quantized)
        self.layer.setRange(0, max(0, insights.layer_count(engine) - 1))
        self._update_mode()
        self._update_buttons()

    def text_mode(self) -> bool:
        """2.21.3: whether a message is the start of a text the model writes on: "continue
        the text" chosen, or "auto" with a model trained on plain text only (it has never
        seen a chat template: wrapped in ChatML its replies began mid-word and ran to the
        token limit)."""
        choice = str(self.template.currentData() or "")
        if choice == TEXT_MODE:
            return True
        engine = self.studio.engine
        if choice or engine.model is None:
            return False
        try:
            return bool(engine.continues_text())
        except Exception:
            return False

    def _update_mode(self) -> None:
        text = self.text_mode()
        self.text_hint.setVisible(text and self.studio.engine.model is not None)
        self.system.setEnabled(not text)
        self.input.setPlaceholderText(
            "Write the start of a text - the model continues it; Enter sends" if text else
            "Write a message - Enter sends, Shift+Enter starts a new line")
        if not self.messages and self.job is None:
            self.render()

    def _update_buttons(self) -> None:
        chatting = self.job is not None
        can_send = self.studio.engine.model is not None and not chatting and not self.studio.busy
        self.send_button.setEnabled(can_send)
        self.input.setEnabled(not chatting)
        self.stop_button.setEnabled(chatting)
        self.new_button.setEnabled(not chatting)
        self.save_button.setEnabled(bool(self.messages) and not chatting)
        has_model = self.studio.engine.model is not None
        replied = bool(insights.last_reply(self.messages))
        self.tokens_button.setEnabled(has_model and replied and not chatting)
        self.attention_button.setEnabled(has_model and replied and not chatting
                                         and not insights.inspectable(self.studio.engine))

    # ---- talking -------------------------------------------------------------
    def send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text or not self.send_button.isEnabled():
            return
        agent = None
        if self.agent_mode.isChecked():
            try:
                agent = self._make_agent()
            except ValueError as err:
                show_error(self, SettingsError(str(err)))
                return
        if not self.studio.begin("chat", quiet=True):
            if agent is not None:
                agent.cleanup()
            return
        self.input.clear()
        self.messages.append({"role": "user", "content": text})
        self.reply = ""
        self._pieces = 0
        self._started = time.time()
        engine = self.studio.engine
        history = [dict(m) for m in self.messages]
        options = {"max_new_tokens": int(self.max_tokens.value()),
                   "temperature": float(self.temperature.value()),
                   "chat_template": str(self.template.currentData()) or None,
                   "system_prompt": self.system.text().strip() or None}
        text_mode = self.text_mode()

        def work(job: Job) -> Any:
            if agent is not None:
                # 2.26: the agent gets the message itself; the history is what came before
                agent.on_step = lambda step: job.progress.emit(("step",
                                                                agentmode.step_lines(step)))
                try:
                    for piece in agent.run_streaming(text, history=history[:-1]):
                        if job.stop_event.is_set():
                            break
                        job.progress.emit(("piece", str(piece)))
                finally:
                    agent.cleanup()
                return None
            if text_mode:
                # 2.21.3: the message is the start of a text; the model writes on from it
                # (each message starts anew: the model knows no turns), at the chat's
                # sampling settings
                pieces = engine.generate_streaming(
                    text, length=options["max_new_tokens"], temperature=options["temperature"],
                    top_k=40, top_p=0.9, repetition_penalty=1.1)
            else:
                pieces = engine.generate_chat_streaming(history, **options)
            for piece in pieces:
                if job.stop_event.is_set():
                    break
                job.progress.emit(str(piece))
            return None

        self.agent = agent
        if agent is not None:
            self.log_line(tr("You: {text}", text=text[:200]), "info")
        self.job = Job("chat", work, self)
        self.job.progress.connect(self._progress)
        self.job.finished.connect(self._done)
        self.job.failed.connect(self._failed)
        self._update_buttons()
        self.render()
        self.timer.start()
        self.job.start()

    def _make_agent(self) -> Any:
        """2.26: an agent for this message, in the chat's sandbox (ValueError: no tool)."""
        options = agentmode.options_from_form(
            self.agent_tools(), self.agent_steps.value(), self.agent_timeout.value(),
            max_tokens_per_step=max(int(self.max_tokens.value()),
                                    int(agentmode.DEFAULTS["max_tokens_per_step"])),
            temperature=float(self.temperature.value()),
            chat_template=str(self.template.currentData() or "") or "chatml")
        if self.sandbox is None:
            self.sandbox = agentmode.make_sandbox(options.timeout)
        return agentmode.make_agent(self.studio.engine, options, sandbox=self.sandbox)

    def agent_tools(self) -> list[str]:
        return [name for name, box in self.tool_boxes.items()
                if box.isChecked() and box.isEnabled()]

    def _agent_toggled(self, on: bool) -> None:
        if on and not self._agent_accepted:
            if not confirm(self, tr("Turn agent mode on?"), tr(agentmode.WARNING)):
                self.agent_mode.blockSignals(True)
                self.agent_mode.setChecked(False)
                self.agent_mode.blockSignals(False)
                return
            self._agent_accepted = True
        if on:
            self.agent_section.set_expanded(True)
            self.log_line(tr("Agent mode on: the tools work in a temporary folder."), "info")
        else:
            self.log_line(tr("Agent mode off."), "info")
            if self.job is None:
                self._drop_sandbox()

    def _drop_sandbox(self) -> None:
        if self.sandbox is not None:
            try:
                self.sandbox.cleanup()
            except Exception:                 # a folder Windows still holds: it goes later
                pass
            self.sandbox = None

    def log_line(self, text: str, kind: str = "") -> None:
        """One line (or more) of the sandbox log, colored by `kind`."""
        color = LOG_COLORS.get(kind)
        body = html.escape(str(text).rstrip()).replace("\n", "<br>")
        self.sandbox_log.appendHtml(f"<span style='color:{color}'>{body}</span>" if color
                                    else body)

    def _progress(self, payload: Any) -> None:
        if isinstance(payload, tuple):        # 2.26: the agent's pieces and steps
            kind, value = payload
            if kind == "step":
                for line, tag in value:
                    self.log_line(line, tag)
            else:
                self._piece(str(value))
            return
        self._piece(str(payload))

    def _piece(self, piece: str) -> None:
        self.reply += piece
        self._pieces += 1
        self._dirty = True

    def _flush(self) -> None:
        if self._dirty:
            self._dirty = False
            self.render()
            seconds = max(1e-6, time.time() - self._started)
            self.speed.setText(f"{self._pieces / seconds:.1f} pieces/s")

    def _done(self, _result: Any) -> None:
        self.timer.stop()
        stopped = self.job is not None and self.job.stop_event.is_set()
        self.job = None
        if self.agent is not None:
            self.agent = None
            self.log_line(tr("Agent stopped.") if stopped else tr("Agent done."), "info")
            if not self.agent_mode.isChecked():
                self._drop_sandbox()
        # 2.21.3: a reply cut at the token limit inside a letter's bytes ends in U+FFFD
        reply = self.reply.strip().rstrip("\ufffd").rstrip()
        if reply or not stopped:
            self.messages.append({"role": "assistant",
                                  "content": reply + (" …" if stopped else "")})
        self.reply = ""
        self.studio.end(model_dirty=None)
        self.render()
        seconds = time.time() - self._started
        self.speed.setText(f"{self._pieces} pieces in {seconds:.1f} s")
        self._update_buttons()

    def _failed(self, exc: BaseException) -> None:
        self.timer.stop()
        self.job = None
        self.reply = ""
        if self.agent is not None:
            self.agent = None
            self.log_line(f"ERROR: {exc}", "error")
        if self.messages and self.messages[-1]["role"] == "user":
            self.input.setPlainText(self.messages.pop()["content"])
        self.studio.end(model_dirty=None)
        self.render()
        self._update_buttons()
        self.studio.report_error(exc, "the reply")

    def stop(self) -> None:
        if self.job is not None:
            self.job.stop_event.set()
        if self.agent is not None:
            self.agent.stop()

    def new_chat(self) -> None:
        self.messages = []
        self.reply = ""
        self.speed.setText("")
        self._drop_sandbox()                  # 2.26: a new chat, a new temporary folder
        self.insight_view.clear()
        self.insight_status.setText("")
        self.render()
        self._update_buttons()

    # ---- insights (2.26) -------------------------------------------------------------------
    def show_token_frequencies(self) -> None:
        engine = self.studio.engine
        text = insights.last_reply(self.messages)
        if engine.model is None or not text.strip():
            inform(self, tr("Insights"), tr("There is no reply yet: chat first - the insights "
                                            "read the model's last reply."))
            return
        try:
            pairs = engine.token_frequencies(text)
        except Exception as exc:
            show_error(self, exc, tr("counting the tokens"))
            return
        top = insights.top_tokens(pairs, TOP_TOKENS)
        self.insight_view.show_bars(top, tr("The most frequent tokens of the last reply"))
        self.insight_status.setText(tr("{unique} different tokens in {chars} characters",
                                       unique=f"{len(pairs):,}", chars=f"{len(text):,}"))
        self.insights_section.set_expanded(True)

    def show_attention(self) -> None:
        engine = self.studio.engine
        why = insights.inspectable(engine)
        text = insights.last_exchange(self.messages)
        if why or not text.strip():
            inform(self, tr("Attention map"), why or tr("There is no exchange yet: chat "
                                                        "first."))
            return
        head = int(self.head.value()) - 1 if self.head.value() > 0 else None
        layer = int(self.layer.value())
        try:
            matrix, labels, heads = insights.attention_view(
                engine, text, layer=layer, head=head, max_tokens=int(self.attention_tokens.value()))
        except Exception as exc:
            show_error(self, exc, tr("reading the attention"))
            return
        which = (tr("head {h} of {n}", h=head + 1, n=heads) if head is not None
                 else tr("the average of {n} heads", n=heads))
        self.insight_view.show_heatmap(matrix, labels,
                                       tr("Layer {layer}, {which}: each row is where that "
                                          "token looked", layer=layer, which=which))
        self.insight_status.setText(tr("{n} tokens; brighter is more attention.",
                                       n=len(labels)))
        self.insights_section.set_expanded(True)

    def save_chat(self) -> None:
        path = ask_save(self, "Save the chat", "Dialog (*.jsonl);;Text (*.txt)", "chat.jsonl")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                if path.lower().endswith(".txt"):
                    for m in self.messages:
                        f.write(f"{m['role']}: {m['content']}\n\n")
                else:
                    messages = ([{"role": "system", "content": self.system.text().strip()}]
                                if self.system.text().strip() else []) + self.messages
                    f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        except OSError as err:
            show_error(self, err, "saving the chat")

    # ---- view ----------------------------------------------------------------
    def render(self) -> None:
        palette = self.view.palette()
        text_color = palette.color(palette.ColorRole.Text).name()
        user_bg = "#dbe8ff" if palette.color(palette.ColorRole.Base).lightness() > 128 \
            else "#2d4470"
        bot_bg = "#eef1f5" if palette.color(palette.ColorRole.Base).lightness() > 128 \
            else "#323944"
        parts = [f"<div style='color:{text_color}'>"]
        if not self.messages and not self.job:
            parts.append("<p style='color:gray'>Type the start of a text: the model writes on "
                         "from it.</p>" if self.text_mode() else
                         "<p style='color:gray'>Ask something to see how the model answers. "
                         "Fine-tuned models answer in the style of their dialogs.</p>")
        shown = list(self.messages)
        if self.job is not None:
            shown.append({"role": "assistant", "content": self.reply + " ▌"})
        for m in shown:
            mine = m["role"] == "user"
            body = html.escape(m["content"]).replace("\n", "<br>")
            who = "You" if mine else "Model"
            parts.append(
                f"<table width='100%' cellspacing='0' cellpadding='0'><tr>"
                f"<td align='{'right' if mine else 'left'}'>"
                f"<table cellpadding='8' style='background:{user_bg if mine else bot_bg};'>"
                f"<tr><td><b>{who}</b><br>{body}</td></tr></table></td></tr></table><br>")
        parts.append("</div>")
        self.view.setHtml("".join(parts))
        bar = self.view.verticalScrollBar()
        bar.setValue(bar.maximum())
