"""gui/tabs/library.py — the Library tab of the classic interface (2.27, as the Studio's).

* **Runs** - every training run of the two interfaces and of ``auralite train``
  (:mod:`auralite.runs`), newest first: the selected run's curve, settings, results and
  files; *Continue* (a stopped, failed or interrupted run goes on exactly where it stood),
  *Open the result*, *Evaluate* (a chat model on the dialogs its run held out), *Train
  more*, *Repeat*, *Compare* (two runs or more), a name and notes, *Delete*.
* **Models** and **Datasets** - what is on the disk (:mod:`auralite.studio.library`), the
  room it takes, and what to do with it.

The rows, texts and button states are the Studio's (`studio/libraryview.py`).
"""
from __future__ import annotations

import os
import time
from typing import Any

from auralite import runs as runlib
from auralite.studio import library
from auralite.studio import libraryview as view

from .. import common as gui_common
from ..common import messagebox, threading, tk, ttk
from ..menu import open_folder

_RUN_KEYS = ("name", "status", "kind", "best", "time", "started")
_MODEL_KEYS = ("name", "kind", "size", "changed", "about")
_DATA_KEYS = ("name", "kind", "size", "changed", "runs", "about")


def _tree(parent: Any, keys: tuple[str, ...], titles: tuple[str, ...], height: int = 8,
          selectmode: Any = "browse") -> Any:
    tree = ttk.Treeview(parent, columns=keys, show="headings", height=height,
                        selectmode=selectmode)
    for key, title in zip(keys, titles, strict=True):
        tree.heading(key, text=title)
        wide = key in ("name", "about")
        tree.column(key, width=260 if wide else max(90, 8 * len(title) + 16), stretch=wide,
                    anchor=tk.E if key in ("best", "time", "size", "runs") else tk.W)
    return tree


class LibraryTabMixin:
    """The Library tab (needs `tab_library`, `engine`, `root`, `gui_config`)."""

    root: Any
    engine: Any
    gui_config: Any

    # ---- the tab -----------------------------------------------------------------
    def _build_library_tab(self) -> None:
        tab = self.tab_library                                 # type: ignore[attr-defined]
        self.lib_runs: list[runlib.Run] = []
        self.lib_shown_runs: list[runlib.Run] = []
        self.lib_models: list[library.LibraryItem] = []
        self.lib_datasets: list[library.LibraryItem] = []
        self._lib_shown_id = ""
        self._lib_eval_thread: Any = None
        self.lib_eval_stop = threading.Event()
        self._lib_after: Any = None
        self._lib_ticking = False
        self._lib_after_import = ""
        self.lib_compare_window: Any = None

        top = ttk.Frame(tab)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Library", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Button(top, text="Refresh", command=self._lib_refresh).pack(side=tk.RIGHT)
        self.lib_disk_label = ttk.Label(top, text="", style="Sub.TLabel")
        self.lib_disk_label.pack(side=tk.RIGHT, padx=8)
        self.lib_pages = ttk.Notebook(tab)
        self.lib_pages.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        pages = []
        for title, build in (("Runs", self._lib_build_runs), ("Models", self._lib_build_models),
                             ("Datasets", self._lib_build_datasets)):
            page = ttk.Frame(self.lib_pages, padding="6")
            build(page)
            self.lib_pages.add(page, text=title)
            pages.append(page)
        self.lib_page_frames = pages
        self._lib_refresh()
        self._lib_tick()

    # ==== runs =========================================================================
    def _lib_build_runs(self, page: Any) -> None:
        tools = ttk.Frame(page)
        tools.pack(fill=tk.X)
        ttk.Label(tools, text="Find:").pack(side=tk.LEFT)
        self.lib_search_var = tk.StringVar(value="")
        search = ttk.Entry(tools, textvariable=self.lib_search_var, width=48)
        search.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        search.bind("<KeyRelease>", lambda _e: self._lib_fill_runs())
        ttk.Button(tools, text="Runs folder",
                   command=lambda: open_folder(str(runlib.runs_root()))).pack(side=tk.RIGHT)
        self.lib_compare_btn = ttk.Button(tools, text="Compare",
                                          command=self.lib_compare_selected)
        self.lib_compare_btn.pack(side=tk.RIGHT, padx=4)
        ttk.Label(page, text=f"{view.SEARCH_PLACEHOLDER}. {view.COMPARE_TIP}.",
                  style="Sub.TLabel").pack(anchor=tk.W)
        self.lib_runs_tree = _tree(page, _RUN_KEYS, view.RUN_COLUMNS, height=8,
                                   selectmode="extended")
        self.lib_runs_tree.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        for status, color in view.STATUS_COLORS.items():
            self.lib_runs_tree.tag_configure(status, foreground=color)
        self.lib_runs_tree.bind("<<TreeviewSelect>>", lambda _e: self._lib_run_selected())
        self.lib_runs_tree.bind("<Double-1>", lambda _e: self.lib_open_result())
        self.lib_empty_label = ttk.Label(page, text=view.EMPTY_RUNS, wraplength=740,
                                         justify=tk.LEFT, style="Sub.TLabel")
        self._lib_empty_shown = False

        details = ttk.LabelFrame(page, text="  The selected run  ", padding="8")
        details.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.lib_details_frame = details
        row = ttk.Frame(details)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Name:").pack(side=tk.LEFT)
        self.lib_name_var = tk.StringVar(value="")
        self.lib_name_entry = ttk.Entry(row, textvariable=self.lib_name_var)
        self.lib_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.lib_save_btn = ttk.Button(row, text="Save the name and notes",
                                       command=self.lib_save_notes)
        self.lib_save_btn.pack(side=tk.LEFT)
        body = ttk.Frame(details)
        body.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.lib_facts = ttk.Label(left, text="Select a run.", justify=tk.LEFT, wraplength=420)
        self.lib_facts.pack(anchor=tk.W, fill=tk.X)
        ttk.Label(left, text="Notes: what you tried, what you saw",
                  style="Sub.TLabel").pack(anchor=tk.W, pady=(6, 0))
        self.lib_notes = tk.Text(left, height=3, width=48, wrap=tk.WORD)
        self.lib_notes.pack(fill=tk.X)
        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        if gui_common.HAS_MATPLOTLIB:
            self.lib_fig = gui_common.Figure(figsize=(4.6, 2.4), dpi=80)
            self.lib_ax = self.lib_fig.add_subplot(111)
            self.lib_canvas = gui_common.FigureCanvasTkAgg(self.lib_fig, master=right)
            self.lib_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.lib_curve_label = None
        else:
            self.lib_fig = self.lib_ax = self.lib_canvas = None
            self.lib_curve_label = ttk.Label(right, text="", justify=tk.LEFT, wraplength=320,
                                             style="Sub.TLabel")
            self.lib_curve_label.pack(anchor=tk.W)
        buttons = ttk.Frame(details)
        buttons.pack(fill=tk.X, pady=(8, 0))
        self.lib_run_buttons: dict[str, Any] = {}
        for key, text, command in (
                ("continue", "▶ Continue", self.lib_continue_selected),
                ("open", "Open the result", self.lib_open_result),
                ("evaluate", "Evaluate", self.lib_evaluate_selected),
                ("more", "Train more", self.lib_train_more),
                ("repeat", "Repeat", self.lib_repeat),
                ("files", "Show files", self.lib_show_files),
                ("delete", "Delete…", self.lib_delete_menu)):
            button = ttk.Button(buttons, text=text, command=command)
            button.pack(side=tk.LEFT, padx=(0, 4))
            self.lib_run_buttons[key] = button
        self.lib_hint = ttk.Label(details, text="", style="Sub.TLabel", wraplength=740,
                                  justify=tk.LEFT)
        self.lib_hint.pack(anchor=tk.W, pady=(4, 0))
        task = ttk.Frame(details)
        task.pack(fill=tk.X)
        self.lib_task_label = ttk.Label(task, text="", style="Sub.TLabel", wraplength=640,
                                        justify=tk.LEFT)
        self.lib_task_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lib_task_stop = ttk.Button(task, text="🛑 Stop", command=self.lib_stop_task,
                                        state=tk.DISABLED)
        self.lib_task_stop.pack(side=tk.RIGHT)

    # ---- filling -------------------------------------------------------------------
    def _lib_refresh(self) -> None:
        """Read the runs, models and datasets again."""
        try:
            self.lib_runs = runlib.list_runs()
        except Exception as exc:                       # an unreadable runs folder
            gui_common.logger.warning("could not read the runs: %s", exc)
            self.lib_runs = []
        self._lib_fill_runs()
        self._lib_refresh_models()
        self._lib_refresh_datasets()
        try:
            self.lib_disk_label.config(text=view.disk_line(library.disk_summary()))
        except Exception as exc:                       # cosmetic
            gui_common.logger.debug("disk summary not shown: %s", exc)

    def _lib_runs_changed(self) -> None:
        """A run started or ended: its record is new."""
        if getattr(self, "lib_runs_tree", None) is None:
            return
        if getattr(self, "_active_tab_name", "") == "library":
            self._lib_refresh()
            return
        try:
            self.lib_runs = runlib.list_runs()
        except Exception:
            return
        self._lib_fill_runs()

    def _lib_tick(self) -> None:
        """While a run trains and the Library shows, its row follows it (every 3 s)."""
        if self._lib_ticking:                  # a test's root runs after() at once
            return
        self._lib_ticking = True
        try:
            self._lib_after = None
            if getattr(self, "_active_tab_name", "") == "library" and any(
                    run.status == "running" for run in self.lib_runs):
                self.lib_runs = runlib.list_runs()
                self._lib_fill_runs()
            self._lib_after = self.root.after(view.REFRESH_MS, self._lib_tick)
        except Exception:                      # the window is closing
            pass
        finally:
            self._lib_ticking = False

    def _stop_library_ticks(self) -> None:
        after = getattr(self, "_lib_after", None)
        if after is not None:
            try:
                self.root.after_cancel(after)
            except Exception:
                pass
            self._lib_after = None

    def _lib_fill_runs(self) -> None:
        words = str(self.lib_search_var.get() or "").lower().split()
        shown = [run for run in self.lib_runs if view.run_matches(run, words)]
        selected = set(self._lib_selected_ids()) or {self._lib_shown_id}
        tree = self.lib_runs_tree
        try:
            tree.delete(*tree.get_children())
        except Exception:                      # a test double
            pass
        for run in shown:
            tree.insert("", tk.END, iid=run.id, values=view.run_row(run), tags=(run.status,))
        self.lib_shown_runs = shown
        keep = [run.id for run in shown if run.id in selected]
        try:
            tree.selection_set(keep)
        except Exception:
            pass
        if not self.lib_runs and not self._lib_empty_shown:
            self.lib_empty_label.pack(anchor=tk.W, pady=(4, 0), before=self.lib_details_frame)
            self._lib_empty_shown = True
        elif self.lib_runs and self._lib_empty_shown:
            self.lib_empty_label.pack_forget()
            self._lib_empty_shown = False
        self._lib_run_selected()

    def _lib_selected_ids(self) -> list[str]:
        try:
            return [str(iid) for iid in self.lib_runs_tree.selection()]
        except Exception:
            return []

    def lib_selected_runs(self) -> list[runlib.Run]:
        """The selected runs, in the table's order."""
        ids = set(self._lib_selected_ids())
        return [run for run in self.lib_shown_runs if run.id in ids]

    def lib_current_run(self) -> runlib.Run | None:
        runs = self.lib_selected_runs()
        return runs[0] if runs else None

    def lib_select_run(self, run_id: str) -> bool:
        """Select a run by its id (the Start tab's *Show in the library*)."""
        self.lib_search_var.set("")
        self._lib_fill_runs()
        if not any(run.id == run_id for run in self.lib_shown_runs):
            return False
        try:
            self.lib_runs_tree.selection_set((run_id,))
            self.lib_runs_tree.see(run_id)
            self.lib_pages.select(0)
        except Exception:                      # a test double
            pass
        self._lib_shown_id = ""
        self._lib_run_selected(run_id)
        return True

    def _lib_busy(self) -> bool:
        return bool(self._busy_with())                         # type: ignore[attr-defined]

    def _lib_run_selected(self, run_id: str = "") -> None:
        runs = self.lib_selected_runs()
        run = runs[0] if runs else next((r for r in self.lib_shown_runs if r.id == run_id), None)
        self.lib_compare_btn.config(state=tk.NORMAL if len(runs) >= 2 else tk.DISABLED)
        if run is None:
            self._lib_shown_id = ""
            self.lib_name_var.set("")
            self.lib_notes.delete("1.0", tk.END)
            self.lib_facts.config(text="Select a run." if self.lib_runs else "")
            self._lib_draw_curve(None)
        else:
            if run.id != self._lib_shown_id:
                self.lib_name_var.set(run.name)
                self.lib_notes.delete("1.0", tk.END)
                self.lib_notes.insert("1.0", run.notes)
            self._lib_shown_id = run.id
            self.lib_facts.config(text="\n".join(f"{label}: {text}" for label, text in
                                                 runlib.describe_run(run)))
            self._lib_draw_curve(run)
        self._lib_update_run_buttons(run)

    def _lib_draw_curve(self, run: runlib.Run | None) -> None:
        train, val = run.curve() if run is not None else ([], [])
        if self.lib_ax is not None:
            ax = self.lib_ax
            ax.clear()
            if train:                          # a short run's few points stay visible
                ax.plot([s for s, _ in train], [v for _, v in train], label="training",
                        marker="." if len(train) < 30 else None)
            if val:
                ax.plot([s for s, _ in val], [v for _, v in val], "o", label="validation")
            if train or val:
                ax.set_xlabel("Step")
                ax.set_ylabel("Loss")
                ax.legend()
            try:
                self.lib_fig.tight_layout()                    # type: ignore[union-attr]
                self.lib_canvas.draw_idle()                    # type: ignore[union-attr]
            except Exception:
                pass
        elif self.lib_curve_label is not None:
            if not train and not val:
                text = "" if run is None else "No curve recorded."
            else:
                text = f"Curve: {len(train)} points" + (
                    f", the last training loss {train[-1][1]:.4f}" if train else "") + (
                    f", the best validation loss {min(v for _, v in val):.4f}" if val else "")
            self.lib_curve_label.config(text=text)

    def _lib_update_run_buttons(self, run: runlib.Run | None = None) -> None:
        busy = self._lib_busy()
        can = view.run_actions(run, busy)
        for key, button in self.lib_run_buttons.items():
            button.config(state=tk.NORMAL if can.get(key) else tk.DISABLED)
        for widget in (self.lib_save_btn, self.lib_name_entry):
            widget.config(state=tk.NORMAL if can["notes"] else tk.DISABLED)
        self.lib_hint.config(text=view.continue_tip(run) if run is not None and not busy
                             else "")

    # ---- run actions -----------------------------------------------------------------
    def lib_save_notes(self) -> None:
        run = self.lib_current_run()
        if run is None:
            return
        name = str(self.lib_name_var.get() or "").strip() or run.name
        notes = self.lib_notes.get("1.0", tk.END).strip()
        try:
            runlib.update_run(run, name=name, notes=notes)
        except (OSError, ValueError) as err:
            self.report_error(err, "saving the run's notes")    # type: ignore[attr-defined]
            return
        self._lib_shown_id = ""
        self._lib_refresh()

    def lib_continue_selected(self) -> None:
        run = self.lib_current_run()
        if run is not None:
            self.continue_recorded_run(run)                    # type: ignore[attr-defined]

    def _lib_check_idle(self) -> bool:
        busy = self._busy_with()                               # type: ignore[attr-defined]
        if busy:
            messagebox.showinfo("One thing at a time", f"Wait until {busy} ends, or stop it.")
            return False
        return True

    def lib_open_result(self) -> None:
        run = self.lib_current_run()
        if run is None or run.result_path is None or not self._lib_check_idle():
            return
        self.load_model(str(run.result_path))                  # type: ignore[attr-defined]
        if self.engine.model is not None:
            self._goto_tab("chat")                             # type: ignore[attr-defined]

    def lib_train_more(self) -> None:
        run = self.lib_current_run()
        if run is None or run.result_path is None:
            return
        self.ft_apply_run(run, base=str(run.result_path))      # type: ignore[attr-defined]
        self._goto_tab("finetune")                             # type: ignore[attr-defined]

    def lib_repeat(self) -> None:
        run = self.lib_current_run()
        if run is None:
            return
        tab = view.tab_of(run)
        if tab == "train":
            self.train_apply_run(run)                          # type: ignore[attr-defined]
        else:
            self.ft_apply_run(run)                             # type: ignore[attr-defined]
        self._goto_tab(tab)                                    # type: ignore[attr-defined]

    def lib_show_files(self) -> None:
        run = self.lib_current_run()
        if run is not None:
            open_folder(str(run.folder))

    def lib_delete_menu(self) -> None:
        """*Delete…*: only the model files (the record and the curve stay), or the run."""
        menu = getattr(self, "lib_delete_popup", None)
        if menu is None:                       # made once, shown under the button
            menu = self.lib_delete_popup = tk.Menu(self.root, tearoff=0)
            menu.add_command(label="Delete its model files (keep the record and curve)",
                             command=lambda: self.lib_delete_selected(files_only=True))
            menu.add_command(label="Delete the run",
                             command=lambda: self.lib_delete_selected(files_only=False))
        button = self.lib_run_buttons["delete"]
        try:
            menu.tk_popup(button.winfo_rootx(), button.winfo_rooty() + button.winfo_height())
        except Exception:                      # a test double
            pass

    def lib_delete_selected(self, files_only: bool) -> None:
        run = self.lib_current_run()
        if run is None or run.status == "running":
            return
        if not messagebox.askyesno("Delete", view.delete_run_question(run, files_only)):
            return
        try:
            runlib.delete_run(run, files_only=files_only)
        except (OSError, ValueError) as err:
            self.report_error(err, "deleting the run")          # type: ignore[attr-defined]
        self._lib_shown_id = ""
        self._lib_refresh()
        refresh = getattr(self, "_start_refresh_interrupted", None)
        if callable(refresh):
            refresh()

    def lib_compare_selected(self) -> None:
        runs = self.lib_selected_runs()
        if len(runs) < 2:
            messagebox.showinfo("Compare runs", view.SELECT_TWO)
            return
        if self.lib_compare_window is not None:
            try:
                self.lib_compare_window.destroy()
            except Exception:
                pass
        self.lib_compare_window = self._lib_compare_window(runs[:6])

    def _lib_compare_window(self, runs: list[runlib.Run]) -> Any:
        """Two runs or more side by side: their curves and what differs (bold)."""
        comparison = runlib.compare_runs(runs)
        win = tk.Toplevel(self.root)
        win.title(f"Compare {len(runs)} runs")
        win.geometry("940x640")
        outer = ttk.Frame(win, padding="10")
        outer.pack(fill=tk.BOTH, expand=True)
        if gui_common.HAS_MATPLOTLIB:
            fig = gui_common.Figure(figsize=(8.6, 3.0), dpi=80)
            ax = fig.add_subplot(111)
            for name, train, val in comparison["series"]:
                if train:
                    line = ax.plot([s for s, _ in train], [v for _, v in train], label=name,
                                   marker="." if len(train) < 30 else None)
                    color = line[0].get_color() if line else None
                else:
                    color = None
                if val:
                    ax.plot([s for s, _ in val], [v for _, v in val], "o", color=color)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.legend()
            try:
                fig.tight_layout()
            except Exception:                  # cosmetic
                pass
            canvas = gui_common.FigureCanvasTkAgg(fig, master=outer)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            canvas.draw_idle()
        ttk.Label(outer, text=view.COMPARE_HINT, style="Sub.TLabel").pack(anchor=tk.W,
                                                                       pady=(4, 0))
        keys = tuple(f"run{i}" for i in range(len(runs)))
        table = ttk.Treeview(outer, columns=keys, show="tree headings",
                             height=min(14, len(comparison["rows"])))
        table.heading("#0", text="")
        table.column("#0", width=150, stretch=False)
        for key, run in zip(keys, runs, strict=True):
            table.heading(key, text=run.name)
            table.column(key, width=180, stretch=True)
        table.tag_configure("differs", font=("Segoe UI", 10, "bold"))
        for label, values, differs in comparison["rows"]:
            table.insert("", tk.END, text=label, values=[str(v) for v in values],
                         tags=("differs",) if differs else ())
        table.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        ttk.Button(outer, text="Close", command=win.destroy).pack(anchor=tk.E, pady=(8, 0))
        self.lib_compare_table = table
        self.lib_comparison = comparison
        return win

    # ---- evaluation --------------------------------------------------------------------
    def lib_evaluate_selected(self) -> bool:
        """A chat model answers the dialogs its run held out (True when it started)."""
        run = self.lib_current_run()
        if run is None or run.result_path is None:
            return False
        if str(run.data.get("kind")) not in ("dialogs", "preferences"):
            messagebox.showinfo("Evaluate", view.EVAL_TEXT_ONLY)
            return False
        if not self._lib_check_idle():
            return False
        if self.engine.model is not None and not messagebox.askyesno(
                "Evaluate", "The evaluation loads the run's result in place of the model in "
                            "memory.\n\nEvaluate?"):
            return False
        engine = self.engine
        path = str(run.result_path)
        self.lib_eval_stop.clear()
        self.lib_task_label.config(text=f"Loading {run.name}…")
        self.lib_task_stop.config(state=tk.NORMAL)

        def progress(done: int, total: int) -> None:
            self.root.after(0, lambda: self.lib_task_label.config(
                text=f"Evaluating: {done} of {total} dialogs"))

        def work() -> None:
            try:
                engine.unload_model()
                engine.load_model(path)
                self.root.after(0, self._model_in_memory_changed,   # type: ignore[attr-defined]
                                f"Loaded: {run.name}")
                dialogs = runlib.held_out_dialogs(run)
                total = min(view.EVAL_DIALOGS, len(dialogs))
                self.root.after(0, lambda: self.lib_task_label.config(
                    text=f"Replying to {total} held-out dialogs…"))
                started = time.time()
                result = engine.evaluate_chat(dialogs, limit=view.EVAL_DIALOGS,
                                              max_new_tokens=view.EVAL_TOKENS,
                                              on_progress=progress, stop_event=self.lib_eval_stop)
                result["seconds"] = time.time() - started
            except BaseException as exc:                   # shown on the window's thread
                self.root.after(0, self._lib_eval_failed, exc)
                return
            self.root.after(0, self._lib_evaluated, run, result)

        self._lib_eval_thread = threading.Thread(target=work, daemon=True)
        self._lib_update_run_buttons(run)
        self._lib_eval_thread.start()
        return True

    def _lib_evaluated(self, run: runlib.Run, result: dict) -> None:
        self._lib_eval_thread = None
        self.lib_task_stop.config(state=tk.DISABLED)
        summary = runlib.evaluation_summary(result)
        try:
            runlib.update_run(run, evaluation=summary)
        except (OSError, ValueError) as err:
            gui_common.logger.warning("the evaluation was not saved with the run: %s", err)
        self.lib_task_label.config(text=view.evaluation_line(summary))
        self._lib_shown_id = ""
        self._lib_refresh()

    def _lib_eval_failed(self, exc: BaseException) -> None:
        self._lib_eval_thread = None
        self.lib_task_stop.config(state=tk.DISABLED)
        self.lib_task_label.config(text="The evaluation failed - see the message.")
        self._model_in_memory_changed()                        # type: ignore[attr-defined]
        self._lib_update_run_buttons(self.lib_current_run())
        self.report_error(exc, "the evaluation", title="Evaluate")  # type: ignore[attr-defined]

    def lib_stop_task(self) -> None:
        if self._lib_eval_thread is not None:
            self.lib_eval_stop.set()
            self.lib_task_label.config(text="Stopping after this reply…")

    # ==== models ==========================================================================
    def _lib_build_models(self, page: Any) -> None:
        ttk.Label(page, text=view.MODELS_HINT, wraplength=740, justify=tk.LEFT,
                  style="Sub.TLabel").pack(anchor=tk.W)
        self.lib_models_tree = _tree(page, _MODEL_KEYS, view.MODEL_COLUMNS, height=12)
        self.lib_models_tree.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.lib_models_tree.tag_configure("gone", foreground="#888888")
        self.lib_models_tree.bind("<<TreeviewSelect>>", lambda _e: self._lib_update_model_buttons())
        self.lib_models_tree.bind("<Double-1>", lambda _e: self.lib_chat_with_model())
        buttons = ttk.Frame(page)
        buttons.pack(fill=tk.X, pady=(6, 0))
        self.lib_model_buttons: dict[str, Any] = {}
        for key, text, command in (
                ("chat", "💬 Chat with it", self.lib_chat_with_model),
                ("finetune", "🎯 Fine-tune it", self.lib_finetune_model),
                ("export", "📦 Export…", self.lib_export_model),
                ("show", "Show in folder", lambda: self._lib_show(self.lib_selected_model())),
                ("forget", "Remove from the list",
                 lambda: self._lib_forget(self.lib_selected_model())),
                ("delete", "Delete…", lambda: self._lib_delete(self.lib_selected_model()))):
            button = ttk.Button(buttons, text=text, command=command)
            button.pack(side=tk.LEFT, padx=(0, 4))
            self.lib_model_buttons[key] = button

    def _lib_fill_items(self, tree: Any, items: list[library.LibraryItem],
                        columns: tuple[str, ...]) -> None:
        try:
            tree.delete(*tree.get_children())
        except Exception:                      # a test double
            pass
        for i, item in enumerate(items):
            tree.insert("", tk.END, iid=str(i), values=[item.name, *view.item_cells(item, columns)],
                        tags=() if item.exists else ("gone",))

    def _lib_selected_item(self, tree: Any, items: list[library.LibraryItem]
                           ) -> library.LibraryItem | None:
        try:
            selection = [str(i) for i in tree.selection()]
        except Exception:
            return None
        if not selection or not selection[0].isdigit():
            return None
        index = int(selection[0])
        return items[index] if 0 <= index < len(items) else None

    def _lib_refresh_models(self) -> None:
        recent = [p for p in getattr(self.gui_config, "recent_models", []) if isinstance(p, str)]
        try:
            self.lib_models = library.list_models(runs=self.lib_runs, recent=recent)
        except Exception as exc:
            gui_common.logger.warning("could not list the models: %s", exc)
            self.lib_models = []
        self._lib_fill_items(self.lib_models_tree, self.lib_models,
                             ("kind_label", "size", "changed", "detail"))
        self._lib_update_model_buttons()

    def lib_selected_model(self) -> library.LibraryItem | None:
        return self._lib_selected_item(self.lib_models_tree, self.lib_models)

    def _lib_update_model_buttons(self) -> None:
        can = view.model_actions(self.lib_selected_model(), self._lib_busy())
        for key, button in self.lib_model_buttons.items():
            button.config(state=tk.NORMAL if can[key] else tk.DISABLED)

    def lib_chat_with_model(self) -> None:
        item = self.lib_selected_model()
        if item is None or not item.exists or not self._lib_check_idle():
            return
        if item.kind == "download":
            # imported into AuraLite's engine (the Model tab's way); the Chat tab opens after
            spec = str(item.extra.get("spec") or item.path)
            self.hf_model_var.set(spec)                        # type: ignore[attr-defined]
            self._lib_after_import = "chat"
            self.lib_task_label.config(text=f"Importing {item.name}: the Chat tab opens when it "
                                            f"is in memory (the Train tab's status shows how far "
                                            f"it is).")
            self._import_hf_native()                           # type: ignore[attr-defined]
            return
        self.load_model(item.path)                             # type: ignore[attr-defined]
        if self.engine.model is not None:
            self._goto_tab("chat")                             # type: ignore[attr-defined]

    def lib_finetune_model(self) -> None:
        item = self.lib_selected_model()
        if item is None:
            return
        self.ft_set_base(str(item.extra.get("spec") or item.path))  # type: ignore[attr-defined]
        self._goto_tab("finetune")                             # type: ignore[attr-defined]

    def lib_export_model(self) -> None:
        item = self.lib_selected_model()
        if item is None or not self._lib_check_idle():
            return
        self.load_model(item.path)                             # type: ignore[attr-defined]
        if self.engine.model is not None:
            self._goto_tab("export")                           # type: ignore[attr-defined]

    def _lib_show(self, item: library.LibraryItem | None) -> None:
        if item is not None and item.exists:
            open_folder(item.path)

    def _lib_forget(self, item: library.LibraryItem | None) -> None:
        if item is None:
            return
        library.forget(item.path)
        recent = [p for p in getattr(self.gui_config, "recent_models", []) if isinstance(p, str)]
        kept = [p for p in recent if os.path.normcase(os.path.abspath(p)) !=
                os.path.normcase(os.path.abspath(item.path))]
        if len(kept) != len(recent):
            self.gui_config = self.gui_config.update(recent_models=kept)
            try:
                self.gui_config.save()
            except Exception:                  # a read-only home: forgotten for this run
                pass
            refresh = getattr(self, "_start_refresh_recent", None)
            if callable(refresh):
                refresh()
        self._lib_refresh()

    def _lib_delete(self, item: library.LibraryItem | None) -> None:
        if item is None or not item.exists or not self._lib_check_idle():
            return
        if not messagebox.askyesno("Delete", view.delete_item_question(item)):
            return
        try:
            library.delete_item(item)
        except (OSError, ValueError) as err:
            self.report_error(err, "deleting it")               # type: ignore[attr-defined]
        self._lib_refresh()

    # ==== datasets ========================================================================
    def _lib_build_datasets(self, page: Any) -> None:
        ttk.Label(page, text=view.DATASETS_HINT, wraplength=740, justify=tk.LEFT,
                  style="Sub.TLabel").pack(anchor=tk.W)
        self.lib_data_tree = _tree(page, _DATA_KEYS, view.DATA_COLUMNS, height=12)
        self.lib_data_tree.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.lib_data_tree.tag_configure("gone", foreground="#888888")
        self.lib_data_tree.bind("<<TreeviewSelect>>", lambda _e: self._lib_update_data_buttons())
        buttons = ttk.Frame(page)
        buttons.pack(fill=tk.X, pady=(6, 0))
        self.lib_data_buttons: dict[str, Any] = {}
        for key, text, command in (
                ("finetune", "🎯 Fine-tune on it", lambda: self.lib_use_data("finetune")),
                ("train", "🧠 Train on it", lambda: self.lib_use_data("train")),
                ("show", "Show in folder", lambda: self._lib_show(self.lib_selected_dataset())),
                ("forget", "Remove from the list",
                 lambda: self._lib_forget(self.lib_selected_dataset())),
                ("delete", "Delete…", lambda: self._lib_delete(self.lib_selected_dataset()))):
            button = ttk.Button(buttons, text=text, command=command)
            button.pack(side=tk.LEFT, padx=(0, 4))
            self.lib_data_buttons[key] = button

    def _lib_refresh_datasets(self) -> None:
        try:
            self.lib_datasets = library.list_datasets(runs=self.lib_runs)
        except Exception as exc:
            gui_common.logger.warning("could not list the datasets: %s", exc)
            self.lib_datasets = []
        self._lib_fill_items(self.lib_data_tree, self.lib_datasets,
                             ("kind_label", "size", "changed", "runs", "detail"))
        self._lib_update_data_buttons()

    def lib_selected_dataset(self) -> library.LibraryItem | None:
        return self._lib_selected_item(self.lib_data_tree, self.lib_datasets)

    def _lib_update_data_buttons(self) -> None:
        can = view.data_actions(self.lib_selected_dataset(), self._lib_busy())
        for key, button in self.lib_data_buttons.items():
            button.config(state=tk.NORMAL if can[key] else tk.DISABLED)

    def lib_use_data(self, tab: str) -> None:
        item = self.lib_selected_dataset()
        if item is None:
            return
        if tab == "finetune":
            self.ft_set_data(item.path)                        # type: ignore[attr-defined]
        else:
            self._select_data(item.path)                       # type: ignore[attr-defined]
        self._goto_tab(tab)                                    # type: ignore[attr-defined]
