"""The Model tab: the model in memory, and every other way to load one (2.25).

The classic interface's Model tab: what the model is; open and save it; its LoRA
adapters merged, saved or loaded; a GGUF file opened with llama.cpp's settings
(context, threads, GPU layers, batch, chat format, mmap, mlock); a Hugging Face model
imported into AuraLite's engine, or loaded through transformers - in 4 or 8 bits, with
PEFT LoRA adapters applied, saved and loaded, fine-tuned, pushed to the Hub or loaded
from it.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QPlainTextEdit, QPushButton, QRadioButton,
                               QSpinBox, QVBoxLayout, QWidget)

from ..i18n import N_, tr
from ..modelinfo import model_info_lines
from ..recipes import SettingsError
from .common import (Job, ask_dir, ask_open, ask_text, compact_spins, confirm, heading, hint,
                     inform, package_note, primary, show_error)

if TYPE_CHECKING:
    from .window import MainWindow

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def gguf_note() -> str:
    """"" when llama-cpp-python is installed, else how to get it - prebuilt: PyPI has its
    source only, which needs a compiler."""
    note = package_note("llama_cpp")
    if not note:
        return ""
    from auralite import bundle
    if bundle.packaged():
        return note
    from auralite.engine.gguf_proxy import LLAMA_CPP_INSTALL_HINT
    return tr("Opening GGUF files needs llama-cpp-python, prebuilt: ") + LLAMA_CPP_INSTALL_HINT


def import_text(info: dict) -> str:
    """The line of an import's progress: the file downloading, or the weights converting."""
    if info.get("stage") == "download":
        total = info.get("all_total") or info.get("total") or 0
        done = info.get("all_bytes") or info.get("bytes") or 0
        size = (tr("{done:,.0f} / {total:,.0f} MB", done=done / 1e6, total=total / 1e6)
                if total else tr("{done:,.0f} MB", done=done / 1e6))
        return tr("Downloading {file}: {done}", file=info.get("file"), done=size)
    return tr("Converting the weights {done}/{total}…", done=info.get("done"),
              total=info.get("total"))


def finetune_texts(text: str) -> list[str]:
    """The examples of a text for a transformers fine-tune: its paragraphs (the classic
    tab's rule: separated by blank lines, longer than 30 characters)."""
    return [t.strip() for t in text.split("\n\n") if len(t.strip()) > 30]


class ModelTab(QWidget):
    def __init__(self, window: MainWindow):
        super().__init__()
        self.studio = window
        self.job: Job | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.addWidget(heading(tr("The model")))
        layout.addWidget(hint(tr("The model in memory, and the other ways to load one: a GGUF "
                                 "file with llama.cpp's settings, or a Hugging Face model - "
                                 "imported into AuraLite, or run by transformers with PEFT "
                                 "LoRA adapters and the Hub.")))

        # ---- the model in memory
        memory_box = QGroupBox(tr("The model in memory"))
        memory = QVBoxLayout(memory_box)
        row = QHBoxLayout()
        self.open_button = QPushButton(tr("Open a model…"))
        self.save_button = QPushButton(tr("Save the model as…"))
        row.addWidget(self.open_button)
        row.addWidget(self.save_button)
        row.addSpacing(16)
        self.merge_button = QPushButton(tr("Merge the adapters"))
        self.merge_button.setToolTip(tr("Fold the LoRA adapters into the weights: a plain "
                                        "model again, for GGUF and other formats"))
        self.save_adapter_button = QPushButton(tr("Save the adapters…"))
        self.load_adapter_button = QPushButton(tr("Load adapters…"))
        for button in (self.merge_button, self.save_adapter_button,
                       self.load_adapter_button):
            row.addWidget(button)
        row.addStretch(1)
        memory.addLayout(row)
        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.info.setMinimumHeight(200)
        memory.addWidget(self.info)
        layout.addWidget(memory_box)

        # ---- GGUF
        gguf_box = QGroupBox(tr("GGUF files (llama.cpp)"))
        gguf = QVBoxLayout(gguf_box)
        self.gguf_needs = hint("")
        gguf.addWidget(self.gguf_needs)
        form = QFormLayout()
        self.n_ctx = QSpinBox()
        self.n_ctx.setRange(128, 1 << 20)
        self.n_ctx.setValue(_env_int("AURALITE_GGUF_N_CTX", 4096))
        self.n_ctx.setToolTip(tr("Tokens the model sees at once; more takes more memory"))
        self.gpu_layers = QSpinBox()
        self.gpu_layers.setRange(-1, 1000)
        self.gpu_layers.setValue(_env_int("AURALITE_GGUF_N_GPU_LAYERS", -1))
        self.gpu_layers.setSpecialValueText(tr("as many as fit"))
        self.gpu_layers.setToolTip(tr("Layers on the graphics card (0: none; llama.cpp built "
                                      "for it)"))
        self.threads = QSpinBox()
        self.threads.setRange(1, 1024)
        self.threads.setValue(_env_int("AURALITE_GGUF_N_THREADS",
                                       int(getattr(window.engine, "num_threads", 4) or 4)))
        self.batch = QSpinBox()
        self.batch.setRange(1, 1 << 16)
        self.batch.setValue(_env_int("AURALITE_GGUF_N_BATCH", 512))
        self.batch.setToolTip(tr("Prompt tokens read per step"))
        self.chat_format = QLineEdit(os.environ.get("AURALITE_GGUF_CHAT_FORMAT", ""))
        self.chat_format.setPlaceholderText(tr("auto (chatml, llama-2, mistral-instruct, …)"))
        self.chat_completion = QCheckBox(tr("Use llama.cpp's chat template"))
        self.chat_completion.setChecked(
            os.environ.get("AURALITE_GGUF_USE_CHAT", "0").lower() in _TRUE)
        self.mmap = QCheckBox(tr("Map the file into memory (mmap)"))
        self.mmap.setChecked(os.environ.get("AURALITE_GGUF_USE_MMAP", "1").lower() not in _FALSE)
        self.mlock = QCheckBox(tr("Keep it in memory (mlock)"))
        self.mlock.setChecked(os.environ.get("AURALITE_GGUF_USE_MLOCK", "0").lower() in _TRUE)
        form.addRow(tr("Context (tokens):"), self.n_ctx)
        form.addRow(tr("GPU layers:"), self.gpu_layers)
        form.addRow(tr("Threads:"), self.threads)
        form.addRow(tr("Batch:"), self.batch)
        form.addRow(tr("Chat format:"), self.chat_format)
        checks = QHBoxLayout()
        for box in (self.chat_completion, self.mmap, self.mlock):
            checks.addWidget(box)
        checks.addStretch(1)
        form.addRow(checks)
        gguf.addLayout(form)
        gguf_row = QHBoxLayout()
        self.gguf_button = primary(QPushButton(tr("Open a GGUF file…")))
        gguf_row.addWidget(self.gguf_button)
        gguf_row.addWidget(hint(tr("These settings apply to every GGUF file opened, from the "
                                   "File menu too. GGUF models chat; they do not train.")), 1)
        gguf.addLayout(gguf_row)
        layout.addWidget(gguf_box)

        # ---- Hugging Face
        hf_box = QGroupBox(tr("Hugging Face models"))
        hf = QVBoxLayout(hf_box)
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel(tr("Model:")))
        self.hf_name = QLineEdit()
        self.hf_name.setPlaceholderText(tr("ORG/NAME on the Hub, or a downloaded folder"))
        name_row.addWidget(self.hf_name, 1)
        self.hf_folder_button = QPushButton(tr("Folder…"))
        name_row.addWidget(self.hf_folder_button)
        hf.addLayout(name_row)
        self.local_only = QCheckBox(tr("Local files only (no network)"))
        hf.addWidget(self.local_only)
        precision = QHBoxLayout()
        precision.addWidget(QLabel(tr("Load in:")))
        self.precision_group = QButtonGroup(self)
        self.precision: dict[str, QRadioButton] = {}
        for key, label in (("full", tr("full precision")), ("8bit", tr("8 bits")),
                           ("4bit", tr("4 bits (QLoRA)"))):
            radio = QRadioButton(label)
            self.precision[key] = radio
            self.precision_group.addButton(radio)
            precision.addWidget(radio)
        self.precision["4bit"].setChecked(True)
        precision.addStretch(1)
        hf.addLayout(precision)
        lora_row = QHBoxLayout()
        self.apply_lora = QCheckBox(tr("Add LoRA adapters on load"))
        self.apply_lora.setChecked(True)
        lora_row.addWidget(self.apply_lora)
        lora_row.addWidget(QLabel(tr("LoRA rank:")))
        self.lora_rank = QSpinBox()
        self.lora_rank.setRange(1, 1024)
        self.lora_rank.setValue(16)
        self.lora_rank.setToolTip(tr("8 to 64 is typical; higher learns more and takes more "
                                     "memory"))
        lora_row.addWidget(self.lora_rank)
        self.trust_remote = QCheckBox(tr("Trust remote code (runs code from the repository)"))
        lora_row.addWidget(self.trust_remote)
        lora_row.addStretch(1)
        hf.addLayout(lora_row)

        engine_row = QHBoxLayout()
        self.import_button = primary(QPushButton(tr("Import into AuraLite")))
        self.import_button.setToolTip(tr("Llama, Mistral, Qwen 2/3/3.5 and SmolLM as AuraLite's own "
                                         "model: it trains, chats and exports like one; in 4 "
                                         "bits it is the base of QLoRA (no transformers "
                                         "needed)"))
        engine_row.addWidget(self.import_button)
        engine_row.addWidget(hint(tr("Fine-tune it then on the Fine-tune tab.")), 1)
        hf.addLayout(engine_row)

        self.hf_needs = hint("")
        hf.addWidget(self.hf_needs)
        tf_row = QHBoxLayout()
        self.hf_load_button = QPushButton(tr("Load with transformers"))
        self.hub_load_button = QPushButton(tr("Load from the Hub…"))
        self.hf_lora_button = QPushButton(tr("Add LoRA adapters"))
        self.hf_save_lora_button = QPushButton(tr("Save the adapter…"))
        self.hf_load_lora_button = QPushButton(tr("Load an adapter…"))
        self.push_button = QPushButton(tr("Push to the Hub…"))
        for button in (self.hf_load_button, self.hub_load_button, self.hf_lora_button,
                       self.hf_save_lora_button, self.hf_load_lora_button, self.push_button):
            tf_row.addWidget(button)
        tf_row.addStretch(1)
        hf.addLayout(tf_row)
        tune_row = QHBoxLayout()
        tune_row.addWidget(QLabel(tr("Passes:")))
        self.ft_epochs = _spin(1, 1000, 3)
        tune_row.addWidget(self.ft_epochs)
        tune_row.addWidget(QLabel(tr("Learning rate:")))
        self.ft_lr = QLineEdit("2e-4")
        self.ft_lr.setMaximumWidth(90)
        tune_row.addWidget(self.ft_lr)
        tune_row.addWidget(QLabel(tr("Batch:")))
        self.ft_batch = _spin(1, 1024, 2)
        tune_row.addWidget(self.ft_batch)
        tune_row.addWidget(QLabel(tr("Length:")))
        self.ft_length = _spin(16, 65536, 512)
        tune_row.addWidget(self.ft_length)
        tune_row.addWidget(QLabel(tr("Accumulation:")))
        self.ft_accumulation = _spin(1, 1024, 8)
        tune_row.addWidget(self.ft_accumulation)
        self.ft_button = QPushButton(tr("Fine-tune…"))
        self.ft_stop_button = QPushButton(tr("Stop"))
        self.ft_stop_button.setToolTip(tr("Stop the fine-tune or the import"))
        tune_row.addWidget(self.ft_button)
        tune_row.addWidget(self.ft_stop_button)
        tune_row.addStretch(1)
        hf.addLayout(tune_row)
        self.status = QLabel(tr("Ready"))
        self.status.setWordWrap(True)
        hf.addWidget(self.status)
        layout.addWidget(hf_box)
        layout.addStretch(1)

        self.open_button.clicked.connect(window.open_model_dialog)
        self.save_button.clicked.connect(window.save_model_dialog)
        self.merge_button.clicked.connect(lambda: window.finetune_tab.merge_adapters())
        self.save_adapter_button.clicked.connect(lambda: window.finetune_tab.save_adapters())
        self.load_adapter_button.clicked.connect(lambda: window.finetune_tab.load_adapters())
        self.gguf_button.clicked.connect(self.open_gguf)
        self.hf_folder_button.clicked.connect(self._pick_folder)
        self.import_button.clicked.connect(self.import_native)
        self.hf_load_button.clicked.connect(lambda: self.load_transformers())
        self.hub_load_button.clicked.connect(self.load_from_hub)
        self.hf_lora_button.clicked.connect(self.add_lora)
        self.hf_save_lora_button.clicked.connect(self.save_lora)
        self.hf_load_lora_button.clicked.connect(self.load_lora)
        self.push_button.clicked.connect(self.push_to_hub)
        self.ft_button.clicked.connect(self.finetune)
        self.ft_stop_button.clicked.connect(self.stop)
        window.model_changed.connect(self.refresh)
        window.busy_changed.connect(lambda _busy: self._update_buttons())
        compact_spins(self, 110)
        self.refresh()

    # ---- state ---------------------------------------------------------------------
    def refresh(self) -> None:
        try:
            lines = model_info_lines(self.studio.engine)
        except Exception as err:                      # a model half-way through loading
            lines = [f"({type(err).__name__}: {err})"]
        self.info.setPlainText("\n".join(lines))
        self._update_buttons()

    def _update_buttons(self) -> None:
        engine = self.studio.engine
        idle = self.job is None and not self.studio.busy
        native = engine.model is not None and not engine.is_gguf_model() and \
            not engine.is_hf_model()
        is_hf = engine.model is not None and engine.is_hf_model()
        has_lora = native and bool(getattr(engine, "has_lora", lambda: False)())
        self.open_button.setEnabled(idle)
        self.save_button.setEnabled(idle and native)
        self.merge_button.setEnabled(idle and has_lora)
        self.save_adapter_button.setEnabled(idle and has_lora)
        self.load_adapter_button.setEnabled(idle and native)
        note = gguf_note()
        self.gguf_needs.setText(note)
        self.gguf_needs.setVisible(bool(note))
        self.gguf_button.setEnabled(idle)
        hf_note = package_note("transformers") or package_note("peft")
        self.hf_needs.setText(tr("Loading with transformers: ") + hf_note if hf_note else "")
        self.hf_needs.setVisible(bool(hf_note))
        self.import_button.setEnabled(idle)
        transformers = not hf_note
        self.hf_load_button.setEnabled(idle and transformers)
        self.hub_load_button.setEnabled(idle and transformers)
        peft = is_hf and bool(getattr(engine.hf_proxy, "is_peft", False))
        self.hf_lora_button.setEnabled(idle and is_hf and not peft)      # once
        self.hf_save_lora_button.setEnabled(idle and peft)
        self.hf_load_lora_button.setEnabled(idle and is_hf)
        self.push_button.setEnabled(idle and is_hf)
        self.ft_button.setEnabled(idle and is_hf)
        self.ft_stop_button.setEnabled(self.job is not None and
                                       self.job.name in ("hf-finetune", "hf-import"))

    def gguf_options(self) -> dict[str, Any]:
        """llama.cpp's settings for ``engine.load_gguf_model`` (the window's File → Open
        reads them too)."""
        return {"n_ctx": int(self.n_ctx.value()), "n_gpu_layers": int(self.gpu_layers.value()),
                "n_threads": int(self.threads.value()), "n_batch": int(self.batch.value()),
                "chat_format": self.chat_format.text().strip() or None,
                "use_chat_completion": self.chat_completion.isChecked(),
                "use_mmap": self.mmap.isChecked(), "use_mlock": self.mlock.isChecked()}

    def load_kind(self) -> str:
        return next((k for k, radio in self.precision.items() if radio.isChecked()), "full")

    def _pick_folder(self) -> None:
        path = ask_dir(self, tr("A downloaded Hugging Face model (its folder)"))
        if path:
            self.hf_name.setText(path)
            self.local_only.setChecked(True)

    def open_gguf(self) -> None:
        path = ask_open(self, tr("Open a GGUF file"), tr("GGUF (*.gguf);;All files (*)"),
                        self.studio.settings.last_model_dir)
        if path:
            self.studio.open_model(path)

    # ---- into AuraLite's engine -------------------------------------------------------
    def import_native(self) -> None:
        name = self.hf_name.text().strip()
        if not name:
            show_error(self, SettingsError(tr("enter a Hugging Face model (ORG/NAME) or choose "
                                              "a downloaded folder first")))
            return
        source = name if (os.path.isdir(name) or name.lower().startswith("hf:")) \
            else f"hf:{name}"
        quantize = "nf4" if self.load_kind() == "4bit" else None
        if not self.studio.confirm_replace_model(tr("Importing a model replaces the one in "
                                                    "memory.")):
            return
        if not self.studio.begin(N_("importing a model")):
            return
        engine = self.studio.engine

        def work(job: Job) -> Any:
            engine.unload_model()
            return engine.import_hf_model(source, quantize=quantize,
                                          progress_callback=job.progress.emit,
                                          stop_event=job.stop_event)

        self.status.setText(tr("Importing {name} into AuraLite…", name=name))
        self._run("hf-import", work, lambda info: self._imported(source, info),
                  tr("importing the model"), progress=self._import_progress)

    def _import_progress(self, info: Any) -> None:
        if isinstance(info, dict):
            self.status.setText(import_text(info))

    def _imported(self, source: str, info: Any) -> None:
        from auralite.engine.hf_models import describe_import
        self.job = None
        self.studio.model_loaded(source)
        self.studio.end(model_dirty=False)
        try:
            what = describe_import(dict(info or {}))
        except Exception:
            what = source
        self.status.setText(tr("Imported: {what}. Fine-tune it on the Fine-tune tab.",
                               what=what))
        self.refresh()

    # ---- transformers ----------------------------------------------------------------------
    def _confirm_trust(self, name: str) -> bool:
        if not self.trust_remote.isChecked():
            return True
        return confirm(self, tr("Trust remote code?"),
                       tr("{name} will be loaded with trust_remote_code: Python code from "
                          "that repository runs on this computer with your permissions. "
                          "Continue only if you trust its author.", name=name))

    def load_transformers(self, name: str = "", local_only: bool | None = None) -> None:
        name = name or self.hf_name.text().strip()
        if not name:
            show_error(self, SettingsError(tr("enter a Hugging Face model name or choose a "
                                              "downloaded folder first")))
            return
        note = package_note("transformers") or package_note("peft")
        if note:
            show_error(self, SettingsError(note))
            return
        if not self._confirm_trust(name):
            return
        if not self.studio.confirm_replace_model(tr("Loading a model replaces the one in "
                                                    "memory.")):
            return
        if not self.studio.begin(N_("loading a model")):
            return
        kind = self.load_kind()
        options = {"load_in_4bit": kind == "4bit", "load_in_8bit": kind == "8bit",
                   "apply_lora": self.apply_lora.isChecked(),
                   "lora_rank": int(self.lora_rank.value()),
                   "local_files_only": self.local_only.isChecked() if local_only is None
                   else local_only,
                   "trust_remote_code": self.trust_remote.isChecked(), "verbose": True}
        engine = self.studio.engine

        def work(_job: Job) -> Any:
            engine.unload_model()
            engine.load_hf_model(name, **options)
            return name

        self.status.setText(tr("Loading {name} with transformers…", name=name))
        self._run("hf-load", work, self._loaded, tr("loading the model"))

    def _loaded(self, name: str) -> None:
        self.job = None
        self.studio.model_loaded(name)
        self.studio.end(model_dirty=False)
        engine = self.studio.engine
        try:
            n = engine.model.count_parameters()
            self.status.setText(tr("Loaded {name}: {n} parameters", name=name, n=f"{n:,}"))
        except Exception:
            self.status.setText(tr("Loaded {name}", name=name))
        self.refresh()

    def load_from_hub(self) -> None:
        repo = ask_text(self, tr("Load from the Hub"), tr("The repository (ORG/NAME):"))
        if repo:
            self.load_transformers(repo, local_only=False)

    def add_lora(self) -> None:
        engine = self.studio.engine
        try:
            engine.apply_lora_to_hf(rank=int(self.lora_rank.value()))
        except Exception as err:
            self.studio.report_error(err, tr("adding LoRA adapters"))
            return
        self.status.setText(tr("LoRA adapters added (rank {r}): ready to fine-tune",
                               r=int(self.lora_rank.value())))
        self.studio.model_dirty = True
        self.studio.model_changed.emit()

    def save_lora(self) -> None:
        path = ask_dir(self, tr("A folder for the adapter"))
        if not path:
            return
        try:
            self.studio.engine.save_hf_lora(path)
        except Exception as err:
            self.studio.report_error(err, tr("saving the adapter"))
            return
        self.status.setText(tr("The adapter is saved: {path}", path=path))

    def load_lora(self) -> None:
        path = ask_dir(self, tr("The adapter's folder"))
        if not path:
            return
        try:
            self.studio.engine.load_hf_lora(path)
        except Exception as err:
            self.studio.report_error(err, tr("loading the adapter"))
            return
        self.status.setText(tr("The adapter is loaded: {path}", path=path))
        self.studio.model_changed.emit()

    def push_to_hub(self) -> None:
        engine = self.studio.engine
        if not engine.is_hf_model():
            show_error(self, SettingsError(tr("load a Hugging Face model with transformers "
                                              "first")))
            return
        repo = ask_text(self, tr("Push to the Hub"), tr("The repository (USER/NAME):"))
        if not repo:
            return
        private = confirm(self, tr("Private?"), tr("Make the repository private?"))
        if not self.studio.begin(N_("uploading the model")):
            return

        def work(_job: Job) -> Any:
            engine.push_hf_model_to_hub(repo, commit_message="Upload from AuraLite Studio",
                                        private=private)
            return repo

        self.status.setText(tr("Pushing to {repo}…", repo=repo))
        self._run("hf-push", work, self._pushed, tr("uploading the model"))

    def _pushed(self, repo: str) -> None:
        self.job = None
        self.studio.end(model_dirty=None)
        self.status.setText(tr("Pushed: https://huggingface.co/{repo}", repo=repo))
        self._update_buttons()

    def finetune(self) -> None:
        engine = self.studio.engine
        if not engine.is_hf_model():
            show_error(self, SettingsError(tr("load a Hugging Face model with transformers "
                                              "first (4 bits suits a fine-tune best)")))
            return
        try:
            lr = float(self.ft_lr.text().strip())
        except ValueError:
            show_error(self, SettingsError(tr("the learning rate must be a number (like "
                                              "2e-4)")))
            return
        texts = self._finetune_data()
        if texts is None:
            return
        if not texts:
            show_error(self, SettingsError(tr("not enough text to fine-tune on: paragraphs "
                                              "longer than 30 characters, separated by blank "
                                              "lines")))
            return
        out_dir = ask_dir(self, tr("A folder for the fine-tuned LoRA adapter"))
        if not out_dir:
            return
        if not self.studio.begin(N_("fine-tuning")):
            return
        options = {"epochs": int(self.ft_epochs.value()), "learning_rate": lr,
                   "batch_size": int(self.ft_batch.value()),
                   "max_length": int(self.ft_length.value()),
                   "gradient_accumulation_steps": int(self.ft_accumulation.value())}

        def work(job: Job) -> Any:
            def progress(epoch: Any, total: Any, loss: Any, _val: Any = None) -> None:
                job.progress.emit((epoch, total, loss))
            return engine.finetune_hf(texts, output_dir=out_dir, progress_callback=progress,
                                      stop_event=job.stop_event, **options)

        self.status.setText(tr("Fine-tuning on {n} examples…", n=len(texts)))
        self._run("hf-finetune", work, self._finetuned, tr("the fine-tune"),
                  progress=self._finetune_progress)

    def _finetune_data(self) -> list[str] | None:
        """The examples: the training data chosen on the Train or Fine-tune tab when the
        user wants it, else a text file. None when cancelled."""
        from ..quantize import calibration_text, read_calibration_file
        paths = self.studio.quantize_tab.training_paths()
        if paths and confirm(self, tr("Training data"),
                             tr("Fine-tune on {name}, the data chosen for training?",
                                name=os.path.basename(os.path.normpath(paths[0])))):
            try:
                return finetune_texts(calibration_text(paths, limit=1 << 28))
            except (OSError, ValueError) as err:
                show_error(self, err, tr("reading the training data"))
                return None
        path = ask_open(self, tr("A text to fine-tune on"), tr("Text (*.txt *.md);;All files "
                                                               "(*)"),
                        self.studio.settings.last_data_dir)
        if not path:
            return None
        try:
            return finetune_texts(read_calibration_file(path, limit=1 << 28))
        except OSError as err:
            show_error(self, err, tr("reading the text"))
            return None

    def _finetune_progress(self, event: tuple) -> None:
        epoch, total, loss = event
        try:
            self.status.setText(tr("Fine-tuning… pass {error} of {t}, loss {loss}", error=epoch,
                                   t=total, loss=f"{float(loss):.4f}"))
        except (TypeError, ValueError):
            self.status.setText(tr("Fine-tuning… pass {error} of {t}", error=epoch, t=total))

    def _finetuned(self, saved: Any) -> None:
        stopped = self.job is not None and self.job.stop_event.is_set()
        self.job = None
        self.studio.end(model_dirty=True)
        self.status.setText(tr("The fine-tune stopped; the adapter is in {path}", path=saved)
                            if stopped else
                            tr("Fine-tuned: the adapter is in {path}", path=saved))
        inform(self, tr("Fine-tuning stopped") if stopped else tr("Fine-tuning complete"),
               tr("The LoRA adapter is saved in:\n{path}\n\nLoad it on the base model with "
                  "«Load an adapter…».", path=saved))
        self._update_buttons()

    def stop(self) -> None:
        if self.job is not None:
            self.job.stop_event.set()
            self.status.setText(tr("Stopping after this step…"))

    # ---- jobs ----------------------------------------------------------------------------
    def _run(self, name: str, work: Any, done: Any, what: str, progress: Any = None) -> None:
        self.job = Job(name, work, self)
        if progress is not None:                    # before the start: no event is lost
            self.job.progress.connect(progress)
        self.job.finished.connect(done)
        self.job.failed.connect(lambda exc: self._failed(exc, what))
        self._update_buttons()
        self.job.start()

    def _failed(self, exc: BaseException, what: str) -> None:
        self.job = None
        self.studio.end(model_dirty=None)
        if self.studio.engine.model is None:
            self.studio.model_unloaded()
        self.status.setText(tr("Failed - see the message."))
        self._update_buttons()
        self.studio.report_error(exc, what)


def _spin(low: int, high: int, value: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(low, high)
    spin.setValue(value)
    return spin
