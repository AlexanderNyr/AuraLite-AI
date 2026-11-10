"""gui/wizard.py - the setup wizard of the classic interface (2.27, as the Studio's).

Goal → this computer → the data → the model → the plan: it fills the Fine-tune or the Train
tab with settings that fit this computer and the data (the advisor) and can start the run.
The steps' texts, tables and plan are the Studio's (`studio/wizardplan.py`).
"""
from __future__ import annotations

import os
from typing import Any

from auralite.studio import wizardplan as steps

from . import common as gui_common
from .common import filedialog, messagebox, threading, tk, ttk
from .i18n import N_, tr

__all__ = ["SetupWizard", "WizardMixin"]

PAGES = ("goal", "computer", "data", "model", "plan")
TITLES = {"goal": N_("What do you want to do?"), "computer": N_("This computer"),
          "data": N_("Your data"), "model": N_("The model"), "plan": N_("The plan")}
GOAL_KEYS = tuple(key for key, _title, _text in steps.GOALS)
PLAN_NOTE = N_("You can stop at any time: Finish ends the run well and keeps the model; the "
               "Console tab shows everything.")


class SetupWizard:
    """The wizard's window. `modal=False` builds it without waiting (tests drive it)."""

    def __init__(self, app: Any, goal: str = "finetune", modal: bool = True):
        self.app = app
        self.report: Any = None
        self.table = steps.ModelTable()
        self.plan: dict[str, Any] | None = None
        self.page = 0
        self._inspected_paths: list[str] = []
        win = self.window = tk.Toplevel(app.root)
        win.title(tr("Set up a training run"))
        win.geometry("820x600")
        win.transient(app.root)
        outer = ttk.Frame(win, padding="14")
        outer.pack(fill=tk.BOTH, expand=True)
        self.title_label = ttk.Label(outer, text="", style="Header.TLabel")
        self.title_label.pack(anchor=tk.W)
        self.subtitle_label = ttk.Label(outer, text="", wraplength=760, justify=tk.LEFT,
                                        style="Sub.TLabel")
        self.subtitle_label.pack(anchor=tk.W, pady=(2, 8))
        body = ttk.Frame(outer)
        body.pack(fill=tk.BOTH, expand=True)
        self.frames = {name: ttk.Frame(body) for name in PAGES}
        self.goal_var = tk.StringVar(value=goal if goal in GOAL_KEYS else "finetune")
        self._build_goal(self.frames["goal"])
        self._build_computer(self.frames["computer"])
        self._build_data(self.frames["data"])
        self._build_model(self.frames["model"])
        self._build_plan(self.frames["plan"])
        buttons = ttk.Frame(outer)
        buttons.pack(fill=tk.X, pady=(10, 0))
        self.cancel_button = ttk.Button(buttons, text=tr("Cancel"), command=self.cancel)
        self.cancel_button.pack(side=tk.LEFT)
        self.finish_button = ttk.Button(buttons, text=tr("Finish"), command=self.finish)
        self.finish_button.pack(side=tk.RIGHT)
        self.next_button = ttk.Button(buttons, text=tr("Next ▸"), command=self.go_next)
        self.next_button.pack(side=tk.RIGHT, padx=4)
        self.back_button = ttk.Button(buttons, text=tr("◂ Back"), command=self.go_back)
        self.back_button.pack(side=tk.RIGHT)
        self.problem_label = ttk.Label(outer, text="", foreground="#d93025", wraplength=760)
        self.problem_label.pack(anchor=tk.W)
        self._show(0)
        if modal:
            try:
                win.grab_set()
                app.root.wait_window(win)
            except (AttributeError, tk.TclError):          # a test double of the root
                pass

    # ---- the pages ---------------------------------------------------------------
    def _build_goal(self, page: Any) -> None:
        for key, title, text in steps.GOALS:
            ttk.Radiobutton(page, text=tr(title), value=key, variable=self.goal_var).pack(
                anchor=tk.W, pady=(6, 0))
            ttk.Label(page, text=tr(text), wraplength=720, justify=tk.LEFT,
                      style="Sub.TLabel").pack(anchor=tk.W, padx=(24, 0))

    def _build_computer(self, page: Any) -> None:
        self.computer_label = ttk.Label(page, text="", justify=tk.LEFT, wraplength=740)
        self.computer_label.pack(anchor=tk.W)
        self.speed_button = ttk.Button(page, text=tr(steps.SPEED_BUTTON),
                                       command=self._speed_test)
        self.speed_button.pack(anchor=tk.W, pady=(10, 0))

    def _build_data(self, page: Any) -> None:
        row = ttk.Frame(page)
        row.pack(fill=tk.X)
        self.data_var = tk.StringVar(value="")
        entry = ttk.Entry(row, textvariable=self.data_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entry.bind("<Return>", lambda _e: self.inspect())
        entry.bind("<FocusOut>", lambda _e: self._inspect_if_changed())
        ttk.Button(row, text=tr("File…"), command=self._pick_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text=tr("Folder…"), command=self._pick_folder).pack(side=tk.LEFT)
        self.data_report = ttk.Label(page, text=tr("No data chosen yet."), wraplength=740,
                                     justify=tk.LEFT)
        self.data_report.pack(anchor=tk.W, pady=(6, 0))
        self.data_explain = ttk.Label(page, text="", wraplength=740, justify=tk.LEFT,
                                      style="Sub.TLabel")
        self.data_explain.pack(anchor=tk.W, pady=(6, 0))

    def _build_model(self, page: Any) -> None:
        self.model_tree = ttk.Treeview(page, columns=("c0",), show="headings", height=8,
                                       selectmode="browse")
        self.model_tree.pack(fill=tk.BOTH, expand=True)
        self.other_label = ttk.Label(page, text=tr(steps.OTHER_LABEL))
        self.other_var = tk.StringVar(value="")
        self.other_entry = ttk.Entry(page, textvariable=self.other_var)
        self.file_row = ttk.Frame(page)
        ttk.Label(self.file_row, text=tr(steps.FILE_PLACEHOLDER)).pack(anchor=tk.W)
        self.file_var = tk.StringVar(value="")
        ttk.Entry(self.file_row, textvariable=self.file_var).pack(side=tk.LEFT, fill=tk.X,
                                                                  expand=True)
        ttk.Button(self.file_row, text=tr("File…"), command=self._pick_model).pack(side=tk.LEFT,
                                                                               padx=4)
        self.model_note = ttk.Label(page, text="", wraplength=740, justify=tk.LEFT,
                                    style="Sub.TLabel")
        self.model_note.pack(anchor=tk.W, pady=(6, 0))

    def _build_plan(self, page: Any) -> None:
        self.plan_label = ttk.Label(page, text="", justify=tk.LEFT, wraplength=740)
        self.plan_label.pack(anchor=tk.W)
        self.start_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(page, text=tr("Start training right away"), variable=self.start_var).pack(
            anchor=tk.W, pady=(10, 0))
        ttk.Label(page, text=tr(PLAN_NOTE), wraplength=740, justify=tk.LEFT,
                  style="Sub.TLabel").pack(anchor=tk.W, pady=(6, 0))

    # ---- moving between pages ---------------------------------------------------------
    def goal(self) -> str:
        goal = str(self.goal_var.get() or "finetune")
        return goal if goal in GOAL_KEYS else "finetune"

    def _show(self, index: int) -> None:
        for frame in self.frames.values():
            frame.pack_forget()
        self.page = max(0, min(index, len(PAGES) - 1))
        name = PAGES[self.page]
        self.title_label.config(text=tr(TITLES[name]))
        subtitle = ""
        if name == "computer":
            subtitle = tr(steps.COMPUTER_SUBTITLE)
            self._refresh_computer()
        elif name == "data":
            subtitle, explanation = steps.data_texts(self.goal())
            self.data_explain.config(text=explanation)
        elif name == "model":
            subtitle = self._fill_model()
        elif name == "plan":
            subtitle = tr(steps.PLAN_SUBTITLE)
            self.plan_label.config(text="\n".join(steps.plan_lines(self.goal(), self.choice(),
                                                                   self.report)))
        self.subtitle_label.config(text=subtitle)
        self.frames[name].pack(fill=tk.BOTH, expand=True)
        self.problem_label.config(text="")
        self.back_button.config(state=tk.NORMAL if self.page > 0 else tk.DISABLED)
        last = self.page == len(PAGES) - 1
        self.next_button.config(state=tk.DISABLED if last else tk.NORMAL)
        self.finish_button.config(state=tk.NORMAL if last else tk.DISABLED)

    def page_complete(self) -> str:
        """"" when the page has what the next one needs, else why not."""
        name = PAGES[self.page]
        goal = self.goal()
        if name == "data" and not steps.data_complete(goal, self.report):
            if self.report is not None and self.report.problems:
                return self.report.problems[0]
            if self.report is not None and self.report.ok:
                return tr("{kind} does not fit this goal: {why}",
                          kind=self.report.kind_label,
                          why=(tr("a new model trains on texts") if goal == "pretrain" else
                               tr("fine-tuning learns from dialogs, preference pairs or "
                                  "texts")))
            return tr("Choose the data first (File… or Folder…).")
        if name == "model" and not steps.model_complete(goal, self.selected_row(),
                                                        self.table.recommendation,
                                                        str(self.other_var.get() or ""),
                                                        str(self.file_var.get() or "")):
            return (tr("Choose your model file (.pt).") if goal == "continue" else
                    tr("Select a model in the list (this computer may still be being looked "
                       "at)."))
        return ""

    def go_next(self) -> bool:
        problem = self.page_complete()
        if problem:
            self.problem_label.config(text=problem)
            return False
        self._show(self.page + 1)
        return True

    def go_back(self) -> None:
        self._show(self.page - 1)

    # ---- this computer -----------------------------------------------------------------
    def _refresh_computer(self) -> None:
        hw = getattr(self.app, "hardware", None)
        self.computer_label.config(text=steps.computer_text(hw))
        self.speed_button.config(state=tk.DISABLED if hw is None or hw.gflops else tk.NORMAL)

    def _speed_test(self) -> None:
        self.app._run_speed_test()
        self.speed_button.config(state=tk.DISABLED)
        try:
            self.app.root.after(2500, self._refresh_computer)
        except Exception:
            pass

    # ---- the data ---------------------------------------------------------------------
    def _pick_file(self) -> None:
        path = filedialog.askopenfilename(
            title=tr("Choose the training data"), parent=self.window,
            filetypes=[(tr("Data"), "*.jsonl *.json *.txt *.md *.parquet *.zst *.gz"),
                       (tr("All files"), "*.*")])
        if path:
            self.set_data(path)

    def _pick_folder(self) -> None:
        path = filedialog.askdirectory(title=tr("Choose a folder of training data"),
                                       parent=self.window)
        if path:
            self.set_data(path)

    def set_data(self, path: str) -> None:
        self.data_var.set(path)
        self.inspect()

    def _paths(self) -> list[str]:
        text = str(self.data_var.get() or "").strip()
        return [text] if text else []

    def _inspect_if_changed(self) -> None:
        if self._paths() != self._inspected_paths:
            self.inspect()

    def inspect(self) -> None:
        """Read the data on a worker thread (the Fine-tune tab's check)."""
        paths = self._paths()
        self._inspected_paths = list(paths)
        if not paths:
            self.report = None
            self.data_report.config(text=tr("No data chosen yet."))
            return
        self.data_report.config(text=tr("Reading the data…"))

        def work() -> None:
            from auralite.studio.datacheck import DataReport, inspect_data
            try:
                report = inspect_data(paths)
            except Exception as exc:
                report = DataReport(paths=list(paths), problems=[str(exc)])
            try:
                self.app.root.after(0, self.inspected, report)
            except Exception:
                pass

        threading.Thread(target=work, daemon=True).start()

    def inspected(self, report: Any) -> None:
        if list(report.paths) != self._paths():
            return
        self.report = report
        self.data_report.config(text="\n".join(report.summary_lines()))
        self.problem_label.config(text="")

    # ---- the model ----------------------------------------------------------------------
    def _fill_model(self) -> str:
        goal = self.goal()
        self.table = table = steps.model_table(goal, getattr(self.app, "hardware", None),
                                               self.report)
        tree = self.model_tree
        try:
            tree.delete(*tree.get_children())
        except Exception:
            pass
        keys = tuple(f"c{i}" for i in range(len(table.headers)))
        if keys:
            tree.config(columns=keys)
            for key, title in zip(keys, table.headers, strict=True):
                tree.heading(key, text=title)
                tree.column(key, width=220 if key == "c0" else gui_common.heading_width(title),
                            stretch=key == "c0")
        tree.tag_configure("unusable", foreground="#888888")
        for i, (cells, _recommended, usable) in enumerate(table.rows):
            tree.insert("", tk.END, iid=str(i), values=cells,
                        tags=() if usable else ("unusable",))
        if table.chosen >= 0:
            try:
                tree.selection_set((str(table.chosen),))
                tree.see(str(table.chosen))
            except Exception:
                pass
        continuing = goal == "continue"
        for widget in (self.model_tree, self.other_label, self.other_entry, self.file_row):
            widget.pack_forget()
        note = self.model_note                     # the widgets go above the note, in order
        if not continuing and table.rows:
            self.model_tree.pack(fill=tk.BOTH, expand=True, before=note)
        if goal == "finetune":
            self.other_label.pack(anchor=tk.W, pady=(6, 0), before=note)
            self.other_entry.pack(fill=tk.X, before=note)
        if continuing:
            self.file_row.pack(fill=tk.X, pady=(4, 0), before=note)
        self.model_note.config(text=table.note)
        return table.subtitle

    def selected_row(self) -> int:
        try:
            selection = [str(i) for i in self.model_tree.selection()]
        except Exception:
            selection = []
        if selection and selection[0].isdigit():
            return int(selection[0])
        return self.table.chosen if self.table.rows else -1

    def _pick_model(self) -> None:
        path = filedialog.askopenfilename(title=tr("Choose your model"), parent=self.window,
                                          filetypes=[(tr("AuraLite models"), "*.pt"),
                                                     (tr("All files"), "*.*")])
        if path:
            self.file_var.set(path)

    def choice(self) -> dict[str, Any]:
        return steps.choose(self.goal(), self.table.recommendation, self.selected_row(),
                            str(self.other_var.get() or ""), str(self.file_var.get() or ""))

    # ---- the end ------------------------------------------------------------------------
    def finish(self) -> dict[str, Any] | None:
        """The plan: the tab opens filled in (and starts, when asked)."""
        self.plan = steps.make_plan(self.goal(), self.choice(), self.report,
                                    bool(self.start_var.get()))
        self._close()
        self.app.apply_plan(self.plan, self.report)
        return self.plan

    def cancel(self) -> None:
        self._close()

    def _close(self) -> None:
        try:
            self.window.grab_release()
            self.window.destroy()
        except Exception:
            pass


class WizardMixin:
    """`open_wizard` and `apply_plan` of `AIApp`."""

    root: Any

    def open_wizard(self, goal: str = "finetune") -> SetupWizard | None:
        """Tools -> Setup wizard… (Ctrl+N), the Start tab's way in."""
        busy = self._busy_with()                               # type: ignore[attr-defined]
        if busy:
            messagebox.showinfo(tr("AuraLite is busy"), tr("Wait until {busy} ends, or stop it.", busy=busy))
            return None
        try:
            return SetupWizard(self, goal)
        except tk.TclError as err:                     # pragma: no cover - a closing window
            gui_common.logger.warning("the wizard did not open: %s", err)
            return None

    def apply_plan(self, plan: dict[str, Any], report: Any = None) -> None:
        """Open the tab of the wizard's plan, filled in, and start it if asked."""
        from auralite.engine.sizing import MODEL_PRESETS
        goal = str(plan.get("goal") or "finetune")
        settings = dict(plan.get("settings") or {})
        data = str(plan.get("data") or "")
        if goal == "pretrain":
            form = self.train_form()                           # type: ignore[attr-defined]
            shape = MODEL_PRESETS.get(str(settings.get("preset") or ""))
            if shape:
                form.update({key: str(int(shape[key])) for key in
                             ("d_model", "n_layers", "n_heads", "d_ff", "seq_length")})
                form["n_kv_heads"] = str(int(shape.get("n_kv_heads") or 0))
            for key, name in (("time_limit_min", "time_limit"), ("bpe_vocab_size",
                                                                  "bpe_vocab_size"),
                              ("epochs", "epochs")):
                if settings.get(key) is not None:
                    form[name] = str(int(settings[key]))
            form["continue"] = False                   # a new model, not the one in memory
            self.apply_train_form(form)                        # type: ignore[attr-defined]
            if data and os.path.exists(data):
                self._select_data(data)                        # type: ignore[attr-defined]
            self._goto_tab("train")                            # type: ignore[attr-defined]
            if plan.get("start"):
                self.root.after(100, self.start_training)      # type: ignore[attr-defined]
        else:
            self.ft_apply_plan(plan, report)                   # type: ignore[attr-defined]
            self._goto_tab("finetune")                         # type: ignore[attr-defined]
            if plan.get("start"):
                self.root.after(100, self.start_finetune)      # type: ignore[attr-defined]
