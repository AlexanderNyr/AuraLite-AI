"""The Chat tab: talk to the model in memory, reply by reply as it writes (2.19)."""
from __future__ import annotations

import html
import json
import time
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QSpinBox, QTextBrowser, QVBoxLayout,
                               QWidget)

from .common import Job, ask_save, heading, hint, primary, show_error

if TYPE_CHECKING:
    from .window import MainWindow

#: 2.21.3: the Template list's entry that writes on from the message instead of chatting
TEXT_MODE = "text"


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

        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(True)
        self.view.setMinimumHeight(260)
        layout.addWidget(self.view, 1)

        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("Write a message - Enter sends, Shift+Enter starts a new "
                                      "line")
        self.input.setMaximumHeight(90)
        self.input.installEventFilter(_EnterSends(self))
        layout.addWidget(self.input)
        buttons = QHBoxLayout()
        self.send_button = primary(QPushButton("Send"))
        self.stop_button = QPushButton("Stop")
        self.new_button = QPushButton("New chat")
        self.save_button = QPushButton("Save the chat…")
        for button in (self.send_button, self.stop_button, self.new_button, self.save_button):
            buttons.addWidget(button)
        self.speed = hint("")
        buttons.addWidget(self.speed, 1)
        layout.addLayout(buttons)

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
        self._model_changed()
        self.render()

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

    # ---- talking -------------------------------------------------------------
    def send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text or not self.send_button.isEnabled():
            return
        if not self.studio.begin("chat", quiet=True):
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

        self.job = Job("chat", work, self)
        self.job.progress.connect(self._piece)
        self.job.finished.connect(self._done)
        self.job.failed.connect(self._failed)
        self._update_buttons()
        self.render()
        self.timer.start()
        self.job.start()

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
        if self.messages and self.messages[-1]["role"] == "user":
            self.input.setPlainText(self.messages.pop()["content"])
        self.studio.end(model_dirty=None)
        self.render()
        self._update_buttons()
        self.studio.report_error(exc, "the reply")

    def stop(self) -> None:
        if self.job is not None:
            self.job.stop_event.set()

    def new_chat(self) -> None:
        self.messages = []
        self.reply = ""
        self.speed.setText("")
        self.render()
        self._update_buttons()

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
