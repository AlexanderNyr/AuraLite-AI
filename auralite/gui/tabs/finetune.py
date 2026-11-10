"""gui/tabs/finetune.py — the Fine-tune tab of the classic interface (2.26, as the Studio's).

A ready model learns from your dialogs, preference pairs or texts: the base (a model of the
catalog, a Hugging Face name or folder, an AuraLite file, the model in memory), the data
with its report, the method (QLoRA, LoRA, every weight) and the advisor's estimate. The run
is recorded in the library with a checkpoint every 10 minutes, so a crash or a power cut
loses at most that much: the Start tab offers to continue it. The form, the texts and the
steps are the Studio's (`studio/finetuneform.py`, `studio/runner.py`); a settings file one
of them saves the other loads. Dialogs written by a teacher model stay on the Train tab.
"""
from __future__ import annotations

import os
import time
from typing import Any

from auralite.studio import runner
from auralite.studio.catalog import CATALOG, find_model
from auralite.studio.finetuneform import (LORA_TARGETS, METHODS, OBJECTIVES, SOURCE_LABELS,
                                          SOURCES, TEMPLATES, base_lines, catalog_label,
                                          estimate_text, hub_spec, settings_from_form)
from auralite.studio.progress import ProgressTracker, ended_by_itself
from auralite.studio.recipes import FineTuneSettings, SettingsError, finetune_params
from auralite.studio.settingsfile import read_settings, write_settings

from .. import common as gui_common
from ..common import filedialog, messagebox, threading, tk, ttk
from ..i18n import N_, tr

#: How often the bar may move (seconds): the engine reports more often than Tk can draw.
_PROGRESS_EVERY = 0.12


def _label_of(pairs: tuple, key: str) -> str:
    """2.28: the label the list shows, in the interface's language."""
    return tr(dict(pairs).get(key, ""))


def _key_of(pairs: tuple, label: str, default: str = "") -> str:
    """The key of a shown label (2.28: translated or English)."""
    return next((k for k, text in pairs if label in (text, tr(text))), default)


class FineTuneTabMixin:
    """The Fine-tune tab (needs `tab_finetune`, `engine`, `root`, `gui_config`)."""

    root: Any
    engine: Any
    gui_config: Any

    # ---- the tab -----------------------------------------------------------------
    def _build_finetune_tab(self) -> None:
        tab = self.tab_finetune                                # type: ignore[attr-defined]
        self.ft_report = None
        self.ft_recommendation = None
        self._ft_thread = None
        self._ft_tracker = None
        self._ft_started = 0.0
        self._ft_last_ui = 0.0
        self._ft_base_chosen = False
        self._ft_method_chosen = False
        self._ft_user_tuned = False
        self._ft_applying = False
        self.ft_stop_event = threading.Event()
        self.ft_finish_event = threading.Event()

        ttk.Label(tab, text=tr("Fine-tune a model"), style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(tab, wraplength=760, justify=tk.LEFT,
                  text=tr("A ready model learns to answer your way from your dialogs - or your "
                       "style and knowledge from texts. Only small adapters train (LoRA), so a "
                       "PC is enough.")).pack(anchor=tk.W, pady=(2, 8))

        # ---- 1. base model
        base = ttk.LabelFrame(tab, text=tr("  1. Base model  "), padding="8")
        base.pack(fill=tk.X, pady=(0, 8))
        self.ft_source_var = tk.StringVar(value="catalog")
        self.ft_source_buttons: dict = {}
        self._ft_catalog_repo = CATALOG[0].repo if CATALOG else ""
        self._ft_catalog_labels: dict[str, str] = {}
        self.ft_catalog_var = tk.StringVar(value="")
        self.ft_hub_var = tk.StringVar(value="")
        self.ft_file_var = tk.StringVar(value="")
        for key in SOURCES:
            row = ttk.Frame(base)
            row.pack(fill=tk.X, pady=1)
            button = ttk.Radiobutton(row, text=tr(SOURCE_LABELS[key]), value=key, width=28,
                                     variable=self.ft_source_var,
                                     command=lambda k=key: self._ft_choose_base(k))
            button.pack(side=tk.LEFT)
            self.ft_source_buttons[key] = button
            if key == "catalog":
                self.ft_catalog_combo = ttk.Combobox(row, textvariable=self.ft_catalog_var,
                                                     state="readonly", width=56)
                self.ft_catalog_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self.ft_catalog_combo.bind("<<ComboboxSelected>>", self._ft_catalog_picked)
            elif key == "hub":
                entry = ttk.Entry(row, textvariable=self.ft_hub_var)
                entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
                entry.bind("<KeyRelease>", lambda _e: self._ft_choose_base("hub"))
                ttk.Button(row, text=tr("Folder…"), command=self._ft_pick_hub_folder).pack(
                    side=tk.LEFT, padx=4)
            elif key == "file":
                entry = ttk.Entry(row, textvariable=self.ft_file_var)
                entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
                entry.bind("<KeyRelease>", lambda _e: self._ft_choose_base("file"))
                ttk.Button(row, text=tr("File…"), command=self._ft_pick_file).pack(side=tk.LEFT,
                                                                                padx=4)
            else:
                self.ft_memory_label = ttk.Label(row, text=tr("none - choose one above"),
                                                 style="Sub.TLabel")
                self.ft_memory_label.pack(side=tk.LEFT)
        self.ft_base_info = ttk.Label(base, text="", wraplength=740, justify=tk.LEFT,
                                      style="Sub.TLabel")
        self.ft_base_info.pack(anchor=tk.W, pady=(4, 0))
        self._ft_fill_catalog()

        # ---- 2. data
        data = ttk.LabelFrame(tab, text=tr("  2. Your data  "), padding="8")
        data.pack(fill=tk.X, pady=(0, 8))
        row = ttk.Frame(data)
        row.pack(fill=tk.X)
        self.ft_data_var = tk.StringVar(value="")
        entry = ttk.Entry(row, textvariable=self.ft_data_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entry.bind("<Return>", lambda _e: self._ft_inspect())
        entry.bind("<FocusOut>", lambda _e: self._ft_inspect_if_changed())
        ttk.Button(row, text=tr("File…"), command=self._ft_pick_data).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text=tr("Folder…"), command=self._ft_pick_data_folder).pack(side=tk.LEFT)
        self.ft_data_report = ttk.Label(data, text=tr("No data chosen yet."), wraplength=740,
                                        justify=tk.LEFT)
        self.ft_data_report.pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(data, wraplength=740, justify=tk.LEFT, style="Sub.TLabel",
                  text=tr('Dialogs: a .jsonl file, one {"messages": [{"role": "user", ...}, '
                       '{"role": "assistant", ...}]} per line (ShareGPT, Alpaca and '
                       'prompt/completion records work too). Preference pairs: prompt, chosen, '
                       'rejected. No dialogs yet? The Train tab\'s "Dialogs from a teacher '
                       'model" writes them.')).pack(anchor=tk.W, pady=(2, 0))
        self._ft_inspected_paths: list[str] = []

        # ---- 3. how to train
        how = ttk.LabelFrame(tab, text=tr("  3. How to train  "), padding="8")
        how.pack(fill=tk.X, pady=(0, 8))
        self.ft_method_var = tk.StringVar(value="qlora")
        for key, label in METHODS:
            ttk.Radiobutton(how, text=tr(label), value=key, variable=self.ft_method_var,
                            command=self._ft_method_picked).pack(anchor=tk.W)
        self.ft_vars: dict = {
            "lora_rank": tk.StringVar(value="16"), "epochs": tk.StringVar(value="2"),
            "seq_length": tk.StringVar(value="512"), "time_limit_min": tk.StringVar(value="0"),
            "lora_lr": tk.StringVar(value="0.001"), "lr": tk.StringVar(value="3e-5"),
            "batch_size": tk.StringVar(value="8"), "prompt_weight": tk.StringVar(value="auto"),
            "val_split": tk.StringVar(value="0.05"), "max_steps": tk.StringVar(value="0"),
            "beta": tk.StringVar(value="auto"), "pref_epochs": tk.StringVar(value="0")}
        # the passes and the window the user changes stay when the base or the data change
        # (the advisor's own changes do not count: _ft_applying)
        for key in ("epochs", "seq_length"):
            self.ft_vars[key].trace_add("write", lambda *_a: self._ft_tuned())
        self.ft_adam8_var = tk.BooleanVar(value=True)
        self.ft_pack_var = tk.BooleanVar(value=True)
        self.ft_memory_plan_var = tk.BooleanVar(value=True)
        self.ft_template_var = tk.StringVar(value=_label_of(TEMPLATES, ""))
        self.ft_targets_var = tk.StringVar(value=_label_of(LORA_TARGETS, "all"))
        self.ft_objective_var = tk.StringVar(value=_label_of(OBJECTIVES, "auto"))
        self.ft_entries: dict = {}

        def field(parent: Any, label: str, key: str, width: int = 8, note: str = "") -> None:
            ttk.Label(parent, text=tr(label)).pack(side=tk.LEFT, padx=(0, 2))
            entry = ttk.Entry(parent, textvariable=self.ft_vars[key], width=width)
            entry.pack(side=tk.LEFT, padx=(0, 10))
            self.ft_entries[key] = entry
            if note:
                ttk.Label(parent, text=tr(note), style="Sub.TLabel").pack(side=tk.LEFT,
                                                                      padx=(0, 10))

        row = ttk.Frame(how)
        row.pack(fill=tk.X, pady=(6, 2))
        field(row, N_("LoRA rank:"), "lora_rank", 5)
        ttk.Checkbutton(row, text=tr("8-bit AdamW (a quarter of the optimizer's memory)"),
                        variable=self.ft_adam8_var, command=self._ft_tuned).pack(side=tk.LEFT)
        row = ttk.Frame(how)
        row.pack(fill=tk.X, pady=2)
        field(row, N_("Passes over the data:"), "epochs", 5)
        field(row, N_("Window (tokens):"), "seq_length", 7)
        field(row, N_("Time limit (min, 0 = none):"), "time_limit_min", 6)
        row = ttk.Frame(how)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=tr("Chat template:")).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Combobox(row, textvariable=self.ft_template_var, state="readonly", width=18,
                     values=[tr(label) for _k, label in TEMPLATES]).pack(side=tk.LEFT)
        self.ft_objective_frame = ttk.Frame(how)
        ttk.Label(self.ft_objective_frame, text=tr("Preference pairs:")).pack(side=tk.LEFT,
                                                                          padx=(0, 2))
        ttk.Combobox(self.ft_objective_frame, textvariable=self.ft_objective_var,
                     state="readonly", width=24,
                     values=[tr(label) for _k, label in OBJECTIVES]).pack(side=tk.LEFT,
                                                                          padx=(0, 10))
        field(self.ft_objective_frame, N_("beta:"), "beta", 6)
        field(self.ft_objective_frame, N_("passes (0 = auto):"), "pref_epochs", 5)
        self._ft_objective_shown = False
        self.ft_estimate_label = ttk.Label(how, text="", style="Sub.TLabel", wraplength=740,
                                           justify=tk.LEFT)
        self.ft_estimate_label.pack(anchor=tk.W, pady=(4, 0))
        self.ft_how_frame = how

        advanced = ttk.LabelFrame(tab, text=tr("  Advanced settings  "), padding="8")
        advanced.pack(fill=tk.X, pady=(0, 8))
        row = ttk.Frame(advanced)
        row.pack(fill=tk.X, pady=2)
        field(row, N_("LoRA learning rate:"), "lora_lr", 8)
        ttk.Label(row, text=tr("LoRA trains:")).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Combobox(row, textvariable=self.ft_targets_var, state="readonly", width=28,
                     values=[tr(label) for _k, label in LORA_TARGETS]).pack(side=tk.LEFT)
        row = ttk.Frame(advanced)
        row.pack(fill=tk.X, pady=2)
        field(row, N_("Learning rate (every weight):"), "lr", 8)
        field(row, N_("Windows per step (batch):"), "batch_size", 5)
        row = ttk.Frame(advanced)
        row.pack(fill=tk.X, pady=2)
        field(row, N_("Prompt weight:"), "prompt_weight", 6,
              N_("0 = only the replies, auto = 0.1 (0 with LoRA)"))
        ttk.Checkbutton(row, text=tr("Whole dialogs per window"),
                        variable=self.ft_pack_var).pack(side=tk.LEFT)
        row = ttk.Frame(advanced)
        row.pack(fill=tk.X, pady=2)
        field(row, N_("Held out for validation:"), "val_split", 6)
        field(row, N_("Steps at most (0 = no limit):"), "max_steps", 8)
        ttk.Checkbutton(row, text=tr("Fit the memory automatically (memory plan)"),
                        variable=self.ft_memory_plan_var).pack(side=tk.LEFT)

        # ---- the run
        run = ttk.LabelFrame(tab, text=tr("  ▶  Fine-tuning  "), padding="8")
        run.pack(fill=tk.X, pady=(0, 8))
        row = ttk.Frame(run)
        row.pack(fill=tk.X)
        self.ft_start_btn = ttk.Button(row, text=tr("▶ Start fine-tuning"),
                                       command=self.start_finetune)
        self.ft_start_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.ft_finish_btn = ttk.Button(row, text=tr("⏹ Finish"), state=tk.DISABLED,
                                        command=self.finish_finetune)
        self.ft_finish_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.ft_stop_btn = ttk.Button(row, text=tr("🛑 Stop"), state=tk.DISABLED,
                                      command=self.stop_finetune)
        self.ft_stop_btn.pack(side=tk.LEFT)
        self.ft_progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(run, variable=self.ft_progress_var, maximum=100.0).pack(fill=tk.X,
                                                                              pady=(6, 2))
        self.ft_status = ttk.Label(run, text=tr("Ready."), wraplength=740, justify=tk.LEFT)
        self.ft_status.pack(anchor=tk.W)
        self.ft_detail = ttk.Label(run, text="", style="Sub.TLabel", wraplength=740,
                                   justify=tk.LEFT)
        self.ft_detail.pack(anchor=tk.W)

        after = ttk.Frame(tab)
        after.pack(fill=tk.X, pady=(0, 4))
        self.ft_after_buttons: dict = {}
        for key, text, command in (
                ("save", N_("💾 Save the model…"), lambda: self.save_model()),   # type: ignore[attr-defined]
                ("chat", N_("💬 Chat with it"), lambda: self._goto_tab("chat")),  # type: ignore[attr-defined]
                ("export", N_("📦 Export…"), lambda: self._goto_tab("export")),  # type: ignore[attr-defined]
                ("merge", N_("🔗 Merge the adapters"),
                 lambda: self.merge_lora_adapters()),                       # type: ignore[attr-defined]
                ("save_adapter", N_("💾 Save the adapters…"),
                 lambda: self.save_lora_adapter()),                         # type: ignore[attr-defined]
                ("load_adapter", N_("📂 Load adapters…"),
                 lambda: self._ft_load_adapters())):
            button = ttk.Button(after, text=tr(text), command=command)
            button.pack(side=tk.LEFT, padx=(0, 4))
            self.ft_after_buttons[key] = button
        files = ttk.Frame(tab)
        files.pack(fill=tk.X)
        ttk.Button(files, text=tr("Load settings…"), command=self.ft_load_settings).pack(
            side=tk.RIGHT)
        ttk.Button(files, text=tr("Save settings…"), command=self.ft_save_settings).pack(
            side=tk.RIGHT, padx=4)

        self._ft_method_picked(chosen=False)
        self._ft_model_changed()
        self._ft_base_changed()

    # ---- the base ----------------------------------------------------------------------
    def _ft_fill_catalog(self) -> None:
        """The catalog's models, marked by the advisor once it has looked at this computer."""
        rec = self.ft_recommendation
        labels = []
        mapping: dict[str, str] = {}
        for i, model in enumerate(CATALOG):
            fit = rec.options[i] if rec is not None and i < len(rec.options) else None
            if fit is not None and fit.model.repo == model.repo:
                text = catalog_label(fit, rec is not None and rec.choice is fit)
            else:
                text = f"{model.display_name} - {model.size_label}"
            labels.append(text)
            mapping[text] = model.repo
        self._ft_catalog_labels = mapping
        try:
            self.ft_catalog_combo.config(values=labels)
        except Exception:                      # a test double
            pass
        current = next((t for t, repo in mapping.items() if repo == self._ft_catalog_repo),
                       labels[0] if labels else "")
        self.ft_catalog_var.set(current)

    def _ft_catalog_picked(self, _event: Any = None) -> None:
        repo = self._ft_catalog_labels.get(str(self.ft_catalog_var.get()))
        if repo:
            self._ft_catalog_repo = repo
        self._ft_choose_base("catalog")

    def _ft_choose_base(self, source: str) -> None:
        self._ft_base_chosen = True
        self.ft_source_var.set(source)
        self._ft_base_changed()

    def ft_source(self) -> str:
        source = str(self.ft_source_var.get() or "catalog")
        return source if source in SOURCES else "catalog"

    def ft_selected_model(self) -> Any:
        source = self.ft_source()
        if source == "catalog":
            return find_model(self._ft_catalog_repo)
        if source == "hub":
            return find_model(str(self.ft_hub_var.get() or ""))
        return None

    def ft_base_spec(self) -> str:
        """What to import or load: ``hf:...``, a folder, a ``.pt`` path, or "" (in memory)."""
        source = self.ft_source()
        if source == "catalog":
            return f"hf:{self._ft_catalog_repo}"
        if source == "hub":
            return hub_spec(str(self.ft_hub_var.get() or ""))
        if source == "file":
            return str(self.ft_file_var.get() or "").strip()
        return ""

    def ft_set_base(self, spec: str) -> None:
        """Choose the base: a model of the catalog, a ``.pt`` file, a Hugging Face name or
        folder, or "" (the model in memory)."""
        model = find_model(spec) if spec else None
        if model is not None:
            self._ft_catalog_repo = model.repo
            self._ft_fill_catalog()
            self._ft_choose_base("catalog")
        elif spec.endswith(".pt"):
            self.ft_file_var.set(spec)
            self._ft_choose_base("file")
        elif spec:
            self.ft_hub_var.set(spec)
            self._ft_choose_base("hub")
        elif self.engine.model is not None:
            self._ft_choose_base("memory")

    def _ft_pick_hub_folder(self) -> None:
        path = filedialog.askdirectory(title=tr("Choose a downloaded Hugging Face model folder"))
        if path:
            self.ft_hub_var.set(path)
            self._ft_choose_base("hub")

    def _ft_pick_file(self) -> None:
        path = filedialog.askopenfilename(title=tr("Choose an AuraLite model"),
                                          filetypes=[(tr("AuraLite models"), "*.pt"),
                                                     (tr("All files"), "*.*")])
        if path:
            self.ft_file_var.set(path)
            self._ft_choose_base("file")

    def _ft_fit_of(self, model: Any) -> Any:
        rec = self.ft_recommendation
        if model is None or rec is None:
            return None
        return next((f for f in rec.options if f.model.repo == model.repo), None)

    def _ft_base_changed(self) -> None:
        lines = base_lines(self._ft_fit_of(self.ft_selected_model()), self.ft_source())
        self.ft_base_info.config(text="\n".join(lines))
        self._ft_update_estimate_line()

    # ---- the method ------------------------------------------------------------------
    def ft_method(self) -> str:
        method = str(self.ft_method_var.get() or "qlora")
        return method if method in dict(METHODS) else "qlora"

    def _ft_method_picked(self, chosen: bool = True) -> None:
        if chosen:
            self._ft_method_chosen = True
        full = self.ft_method() == "full"
        for key, on in (("lora_rank", not full), ("lora_lr", not full), ("lr", full)):
            entry = self.ft_entries.get(key)
            if entry is not None:
                entry.config(state=tk.NORMAL if on else tk.DISABLED)
        self._ft_update_estimate_line()

    def _ft_tuned(self) -> None:
        if not self._ft_applying:
            self._ft_user_tuned = True
        self._ft_update_estimate_line()

    # ---- the data --------------------------------------------------------------------
    def _ft_pick_data(self) -> None:
        path = filedialog.askopenfilename(
            title=tr("Choose the training data"),
            filetypes=[(tr("Data"), "*.jsonl *.json *.txt *.md *.parquet *.zst *.gz"),
                       (tr("All files"), "*.*")])
        if path:
            self.ft_set_data(path)

    def _ft_pick_data_folder(self) -> None:
        path = filedialog.askdirectory(title=tr("Choose a folder of training data"))
        if path:
            self.ft_set_data(path)

    def ft_set_data(self, path: str) -> None:
        self.ft_data_var.set(path)
        self._ft_inspect()

    def ft_data_paths(self) -> list[str]:
        text = str(self.ft_data_var.get() or "").strip()
        return [text] if text else []

    def _ft_inspect_if_changed(self) -> None:
        if self.ft_data_paths() != self._ft_inspected_paths:
            self._ft_inspect()

    def _ft_inspect(self) -> None:
        """Read the data on a worker thread; its report comes back to this tab."""
        paths = self.ft_data_paths()
        self._ft_inspected_paths = list(paths)
        if not paths:
            self.ft_report = None
            self.ft_data_report.config(text=tr("No data chosen yet."))
            self._ft_data_changed()
            return
        self.ft_data_report.config(text=tr("Reading the data…"))

        def work() -> None:
            from auralite.studio.datacheck import DataReport, inspect_data
            try:
                report = inspect_data(paths)
            except Exception as exc:                   # an unreadable file: say why
                report = DataReport(paths=list(paths), problems=[str(exc)])
            try:
                self.root.after(0, self._ft_inspected, report)
            except Exception:                          # the window is closing
                pass

        threading.Thread(target=work, daemon=True).start()

    def _ft_inspected(self, report: Any) -> None:
        if list(report.paths) != self.ft_data_paths():     # the path changed meanwhile
            return
        self.ft_report = report
        self.ft_data_report.config(text="\n".join(report.summary_lines()))
        if report.ok and report.paths and os.path.exists(report.paths[0]):
            try:
                from auralite.studio.library import remember_dataset
                remember_dataset(report.paths[0], report)   # the Studio's library sees it
            except Exception as exc:                       # a read-only home
                gui_common.logger.debug("dataset not remembered: %s", exc)
        self._ft_data_changed()

    def _ft_pairs(self) -> bool:
        report = self.ft_report
        return report is not None and report.kind == "preferences"

    def _ft_data_changed(self) -> None:
        pairs = self._ft_pairs()
        if pairs and not self._ft_objective_shown:
            self.ft_objective_frame.pack(fill=tk.X, pady=2, before=self.ft_estimate_label)
            self._ft_objective_shown = True
        elif not pairs and self._ft_objective_shown:
            self.ft_objective_frame.pack_forget()
            self._ft_objective_shown = False
        self._ft_update_estimates()

    # ---- the advisor -------------------------------------------------------------------
    def _ft_update_estimates(self) -> None:
        """The advisor's look at this computer and the data: the catalog's marks, the
        recommended base and method, its passes, window and optimizer - until the user
        chooses them."""
        hw = getattr(self, "hardware", None)
        if hw is None:
            return
        from auralite.studio.advisor import recommend_finetune
        report = self.ft_report if self.ft_report is not None and self.ft_report.ok else None
        self.ft_recommendation = rec = recommend_finetune(hw, report)
        choice = rec.choice
        if choice is not None:
            if not self._ft_base_chosen:
                self._ft_catalog_repo = choice.model.repo
            self._ft_applying = True
            try:
                if not self._ft_method_chosen:
                    self.ft_method_var.set(choice.method)
                    self._ft_method_picked(chosen=False)
                if not self._ft_user_tuned:
                    s = rec.settings
                    self.ft_vars["epochs"].set(str(int(s.get("epochs", 2))))
                    self.ft_vars["seq_length"].set(str(int(s.get("seq_length", 512))))
                    self.ft_adam8_var.set(s.get("optimizer") == "adamw8bit")
            finally:
                self._ft_applying = False
        self._ft_fill_catalog()
        self._ft_base_changed()

    def _ft_update_estimate_line(self) -> None:
        try:
            seq = int(str(self.ft_vars["seq_length"].get()).strip())
        except (TypeError, ValueError):
            self.ft_estimate_label.config(text="")
            return
        self.ft_estimate_label.config(text=estimate_text(
            self.ft_selected_model(), self.ft_method(), self.ft_recommendation, seq))

    # ---- settings ----------------------------------------------------------------------
    def ft_form(self) -> dict:
        """The form's fields, as `studio.finetuneform.settings_from_form` reads them."""
        form = {key: var.get() for key, var in self.ft_vars.items()}
        form.update(method=self.ft_method(), adam8=bool(self.ft_adam8_var.get()),
                    pack_dialogs=bool(self.ft_pack_var.get()),
                    memory_plan=bool(self.ft_memory_plan_var.get()),
                    chat_template=_key_of(TEMPLATES, str(self.ft_template_var.get())),
                    lora_targets=_key_of(LORA_TARGETS, str(self.ft_targets_var.get()), "all"),
                    objective=_key_of(OBJECTIVES, str(self.ft_objective_var.get()), "auto"))
        return form

    def ft_settings(self) -> FineTuneSettings:
        """The form as settings (SettingsError names a field that cannot be read)."""
        return settings_from_form(self.ft_form(), pairs=self._ft_pairs())

    def ft_apply_settings(self, s: FineTuneSettings) -> None:
        """Fill the whole form from settings (a settings file)."""
        self._ft_applying = True
        try:
            if s.method in dict(METHODS):
                self.ft_method_var.set(s.method)
                self._ft_method_chosen = True
            values = {"lora_rank": s.lora_rank, "epochs": s.epochs, "seq_length": s.seq_length,
                      "time_limit_min": f"{float(s.time_limit_min):g}",
                      "lr": f"{float(s.lr):g}", "batch_size": s.batch_size,
                      "max_steps": s.max_steps, "val_split": f"{float(s.val_split):g}",
                      "pref_epochs": s.preference_epochs,
                      "prompt_weight": "auto" if s.prompt_weight is None
                      else f"{float(s.prompt_weight):g}",
                      "beta": "auto" if s.preference_beta is None
                      else f"{float(s.preference_beta):g}"}
            if s.lora_lr:
                values["lora_lr"] = f"{float(s.lora_lr):g}"
            for key, value in values.items():
                self.ft_vars[key].set(str(value))
            self.ft_adam8_var.set(s.optimizer == "adamw8bit")
            self.ft_pack_var.set(bool(s.pack_dialogs))
            self.ft_memory_plan_var.set(s.memory_plan == "auto")
            if s.chat_template in dict(TEMPLATES):
                self.ft_template_var.set(_label_of(TEMPLATES, s.chat_template))
            if s.lora_targets in dict(LORA_TARGETS):
                self.ft_targets_var.set(_label_of(LORA_TARGETS, s.lora_targets))
            if s.objective in dict(OBJECTIVES):
                self.ft_objective_var.set(_label_of(OBJECTIVES, s.objective))
            self._ft_user_tuned = True            # kept when the base or the data change
        finally:
            self._ft_applying = False
        self._ft_method_picked(chosen=False)

    def ft_save_settings(self) -> None:
        """The form, its base and its data to a settings file (the Studio reads it too)."""
        try:
            settings = self.ft_settings()
        except SettingsError as err:
            messagebox.showerror(tr("Save settings"), str(err))
            return
        path = filedialog.asksaveasfilename(title=tr("Save the settings"), defaultextension=".json",
                                            initialfile="finetune-settings.json",
                                            filetypes=[(tr("Settings"), "*.json")])
        if not path:
            return
        try:
            write_settings(path, "finetune", settings.to_dict(), self.ft_data_paths(),
                           base=self.ft_base_spec() if self.ft_source() != "memory" else "")
        except OSError as err:
            self.report_error(err, tr("saving the settings"), title=tr("Save settings"))  # type: ignore[attr-defined]
            return
        self.ft_status.config(text=tr("Settings saved to {name}.", name=os.path.basename(path)))

    def ft_load_settings(self) -> None:
        """The form from a settings file (the Studio's too), with its base and data."""
        path = filedialog.askopenfilename(title=tr("Load settings"),
                                          filetypes=[(tr("Settings"), "*.json"),
                                                     (tr("All files"), "*.*")])
        if not path:
            return
        try:
            doc = read_settings(path, "finetune")
        except (OSError, ValueError) as err:
            messagebox.showerror(tr("Load settings"), str(err))
            return
        self.ft_apply_settings(FineTuneSettings.from_dict(doc.get("settings") or {}))
        if doc.get("base"):
            self.ft_set_base(str(doc["base"]))
        paths = [p for p in doc.get("data") or [] if isinstance(p, str)]
        if paths and os.path.exists(paths[0]):
            self.ft_set_data(paths[0])
        self.ft_status.config(text=tr("Settings loaded from {name}.", name=os.path.basename(path)))

    # ---- from the Library and the wizard (2.27) ------------------------------------------
    def ft_apply_run(self, run: Any, base: str | None = None) -> None:
        """Fill the form from a run of the Library: *Repeat* (its base) or *Train more*
        (`base`: the run's result). A new model's settings are not a fine-tune's: they
        stay out (its 100 passes would be a very long fine-tune)."""
        from auralite.studio.libraryview import tab_of
        if run.settings and tab_of(run) == "finetune":
            self.ft_apply_settings(FineTuneSettings.from_dict(run.settings))
        self.ft_set_base(str(run.base.get("spec") or "") if base is None else base)
        paths = [str(p) for p in run.data.get("paths") or []]
        if paths:
            self.ft_set_data(paths[0])

    def ft_apply_plan(self, plan: dict, report: Any = None) -> None:
        """Fill the form from the wizard's plan: the base, the method, the passes, the
        window, the rank, the batch, the optimizer and the data (with its report)."""
        settings = dict(plan.get("settings") or {})
        base = str(plan.get("base") or settings.get("base") or "")
        if base:
            self.ft_set_base(base)
        self._ft_applying = True
        try:
            if settings.get("method") in dict(METHODS):
                self.ft_method_var.set(str(settings["method"]))
                self._ft_method_chosen = True
            for key in ("epochs", "seq_length", "lora_rank", "batch_size"):
                if settings.get(key):
                    self.ft_vars[key].set(str(int(settings[key])))
            if "optimizer" in settings:
                self.ft_adam8_var.set(settings["optimizer"] == "adamw8bit")
            self._ft_user_tuned = True            # the plan's numbers stay
        finally:
            self._ft_applying = False
        self._ft_method_picked(chosen=False)
        data = str(plan.get("data") or "")
        if not data:
            return
        self.ft_data_var.set(data)
        if report is not None and list(report.paths) == [data]:
            self._ft_inspected_paths = [data]
            self._ft_inspected(report)            # read by the wizard already
        else:
            self._ft_inspect()

    # ---- the model in memory ------------------------------------------------------------
    def _ft_model_changed(self) -> None:
        engine = self.engine
        has_model = getattr(engine, "model", None) is not None
        busy = self._ft_thread is not None and self._ft_thread.is_alive()
        try:
            native = has_model and not engine.is_gguf_model() and not engine.is_hf_model()
            has_lora = native and bool(engine.has_lora())
        except Exception:
            native = has_lora = False
        self.ft_source_buttons["memory"].config(state=tk.NORMAL if native else tk.DISABLED)
        if not native and self.ft_source() == "memory":
            self.ft_source_var.set("catalog")
        if native:
            try:
                count = int(engine.model.count_parameters())
            except Exception:
                count = 0
            self.ft_memory_label.config(
                text=tr("a native model, {count:,} parameters with LoRA adapters", count=count)
                if has_lora else tr("a native model, {count:,} parameters", count=count))
        else:
            self.ft_memory_label.config(text=tr("none - choose one above"))
        states = {"save": native, "chat": has_model, "export": native, "merge": has_lora,
                  "save_adapter": has_lora, "load_adapter": native}
        for key, on in states.items():
            self.ft_after_buttons[key].config(state=tk.NORMAL if on and not busy
                                              else tk.DISABLED)

    def _ft_load_adapters(self) -> None:
        self.load_lora_adapter()                               # type: ignore[attr-defined]
        self._ft_model_changed()

    # ---- running -------------------------------------------------------------------------
    def start_finetune(self) -> bool:
        """Start fine-tuning with the form (True when it started)."""
        report = self.ft_report
        if report is None or not report.for_finetune:
            problem = (report.problems[0] if report is not None and report.problems else
                       tr("choose dialogs, preference pairs or a text to fine-tune on")
                       if report is None or report.ok else
                       tr("{kind} trains a new model (the Train tab)", kind=report.kind_label))
            messagebox.showerror(tr("Fine-tune"), tr("Cannot start: {problem}.", problem=problem))
            return False
        try:
            settings = self.ft_settings()
            params = finetune_params(settings, report.kind)
        except SettingsError as err:
            messagebox.showerror(tr("Fine-tune"), tr("Cannot start: {error}.", error=err))
            return False
        source, spec = self.ft_source(), self.ft_base_spec()
        engine = self.engine
        if source == "memory" and engine.model is None:
            messagebox.showerror(tr("Fine-tune"), tr("There is no model in memory: choose a base model."))
            return False
        if source != "memory" and not spec:
            messagebox.showerror(tr("Fine-tune"), tr("Choose the base model: a name, a folder or a file."))
            return False
        if source == "file" and not os.path.isfile(spec):
            messagebox.showerror(tr("Fine-tune"), tr("Not found: {spec}", spec=spec))
            return False
        busy = self._busy_with()                               # type: ignore[attr-defined]
        if busy:
            messagebox.showinfo(tr("One run at a time"), tr("Wait until {busy} ends, or stop it.", busy=busy))
            return False
        if source != "memory" and engine.model is not None and not messagebox.askyesno(
                tr("Fine-tune"), tr("The base model replaces the one in memory.\n\nStart?")):
            return False
        record = {"base": runner.base_record(source, spec),
                  "settings": settings.to_dict(), "source": "gui",
                  "hardware": runner.machine_record(engine, getattr(self, "hardware", None)),
                  "data": {"paths": list(report.paths), "kind": report.kind,
                           "records": report.records, "tokens": report.tokens,
                           "bytes": report.bytes}}
        self._ft_begin(tr("Starting…"))
        holder: dict[str, Any] = {}
        started = time.time()

        def on_event(event: str, *payload: Any) -> None:
            if event == "recorder":
                holder["recorder"] = payload[0]
            else:
                self.root.after(0, self._run_event, "finetune", event,  # type: ignore[attr-defined]
                                payload)

        def work() -> None:
            try:
                summary = runner.finetune(engine, source=source, spec=spec, settings=settings,
                                          params=params, paths=list(report.paths),
                                          record=record, progress_callback=self._ft_progress,
                                          stop_event=self.ft_stop_event,
                                          finish_event=self.ft_finish_event, on_event=on_event)
            except BaseException as exc:                   # shown on the window's thread
                self.root.after(0, self._run_failed, "finetune", exc)  # type: ignore[attr-defined]
                return
            self.root.after(0, self._run_done, "finetune", summary,  # type: ignore[attr-defined]
                            holder.get("recorder"), time.time() - started,
                            self.ft_stop_event.is_set())

        self._ft_thread = threading.Thread(target=work, daemon=True)
        self._ft_thread.start()
        return True

    def _ft_begin(self, text: str) -> None:
        """A run begins on this tab: the bar at zero, Start off, Finish and Stop on."""
        self.ft_stop_event.clear()
        self.ft_finish_event.clear()
        self._ft_tracker = ProgressTracker()
        self._ft_started = time.time()
        self._ft_last_ui = 0.0
        self.ft_progress_var.set(0.0)
        self.ft_status.config(text=text)
        self.ft_detail.config(text="")
        self.ft_start_btn.config(state=tk.DISABLED)
        self.ft_finish_btn.config(state=tk.NORMAL)
        self.ft_stop_btn.config(state=tk.NORMAL, text=tr("🛑 Stop"))
        for button in self.ft_after_buttons.values():
            button.config(state=tk.DISABLED)

    def _ft_progress(self, current: Any, total: Any, loss: Any = None, val_loss: Any = None,
                     info: dict | None = None) -> None:
        """The engine's progress, from the training thread (at most ~8 times a second,
        epoch ends and stages always)."""
        info = dict(info or {})
        phase = str(info.get("phase") or "")
        important = (bool(info.get("is_epoch_end")) or current is None or int(current) <= 0
                     or phase in ("setup", "val", "stopped", "epoch_end"))
        now = time.monotonic()
        if not important and now - self._ft_last_ui < _PROGRESS_EVERY:
            return
        self._ft_last_ui = now
        try:
            self.root.after(0, self._ft_show_progress, current, total, loss, val_loss, info)
        except Exception:                                  # the window is closing
            pass

    def _ft_show_progress(self, current: Any, total: Any, loss: Any, val_loss: Any,
                          info: dict) -> None:
        tracker = self._ft_tracker or ProgressTracker()
        self._ft_tracker = tracker
        view = tracker.update(int(current or 0), int(total or 0), loss, val_loss, info)
        self.ft_progress_var.set(float(view.percent))
        self.ft_status.config(text=view.status)
        self.ft_detail.config(text=view.detail)

    def _ft_show_stage(self, text: str, percent: Any = None) -> None:
        self.ft_status.config(text=text)
        if percent is not None:
            self.ft_progress_var.set(float(percent))

    def _ft_show_import(self, info: dict) -> None:
        stage = info.get("stage")
        if stage == "download":
            done, total = int(info.get("all_bytes") or 0), int(info.get("all_total") or 0)
            self._ft_show_stage(tr("Downloading {file}: {done:,.0f} of {total:,.0f} MB",
                                   file=info.get("file", ""), done=done / 1e6,
                                   total=total / 1e6), 100.0 * done / total if total else None)
        elif stage == "convert":
            done, total = int(info.get("done") or 0), int(info.get("total") or 0)
            self._ft_show_stage(tr("Converting the weights: {done}/{total}", done=done,
                                   total=total),
                                100.0 * done / total if total else None)

    def _ft_end(self, text: str, summary: Any = None, stopped: bool = False) -> None:
        """The run has ended (or failed): the bar, the buttons, the line about it."""
        if summary is not None and ended_by_itself(summary, stopped):
            self.ft_progress_var.set(100.0)
        self.ft_status.config(text=text)
        self.ft_detail.config(text="")
        self.ft_start_btn.config(state=tk.NORMAL)
        self.ft_finish_btn.config(state=tk.DISABLED)
        self.ft_stop_btn.config(state=tk.DISABLED, text=tr("🛑 Stop"))
        self._ft_thread = None
        self._ft_model_changed()

    def finish_finetune(self) -> None:
        """Finish: a short learning-rate cooldown, then the run ends with what it learned."""
        self.ft_finish_event.set()
        self.ft_status.config(text=tr("Finishing - cooling down the learning rate…"))

    def stop_finetune(self) -> None:
        """Stop at the next step (the weights learned so far are kept)."""
        self.ft_stop_event.set()
        self.ft_status.config(text=tr("Stopping at the next step…"))
