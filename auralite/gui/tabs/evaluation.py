"""gui/tabs/evaluation.py — EvaluationTabMixin for the AuraLite GUI (v2.6.5).

Extracted verbatim from `gui_app.py`; behaviour is unchanged.
"""
from ..common import (filedialog, json, messagebox, threading, tk, ttk)
from ..i18n import tr

class EvaluationTabMixin:
    def _build_evaluation_tab(self):
        tab = self.tab_eval
        self._build_chat_eval_frame(tab)

        try:
            from auralite.evaluation import HAS_LM_EVAL
        except ImportError:
            HAS_LM_EVAL = False

        if not HAS_LM_EVAL:
            ttk.Label(tab,
                      text=tr("⚠ lm-evaluation-harness not installed.\n"
                           "Install with: pip install lm-eval"),
                      font=("Segoe UI", 12, "bold")).pack(pady=40)
            return

        # ---- Task Selection ----
        task_frame = ttk.LabelFrame(tab, text=tr("  📋  Tasks  "), padding="10")
        task_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(task_frame, text=tr("Tasks (comma separated):")).pack(anchor=tk.W)
        self.eval_tasks_var = tk.StringVar(value="arc_easy,hellaswag,winogrande")
        ttk.Entry(task_frame, textvariable=self.eval_tasks_var, width=60).pack(fill=tk.X, pady=4)

        # Common tasks
        common_frame = ttk.Frame(task_frame)
        common_frame.pack(fill=tk.X, pady=4)
        for task in ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "gsm8k", "mmlu"]:
            ttk.Button(common_frame, text=task,
                       command=lambda t=task: self._add_eval_task(t)).pack(side=tk.LEFT, padx=2)

        # ---- Options ----
        opt_frame = ttk.LabelFrame(tab, text=tr("  ⚙️  Options  "), padding="8")
        opt_frame.pack(fill=tk.X, pady=(0, 8))

        row = ttk.Frame(opt_frame)
        row.pack(fill=tk.X)

        ttk.Label(row, text=tr("Few-shot:")).pack(side=tk.LEFT, padx=4)
        self.eval_fewshot_var = tk.StringVar(value="0")
        ttk.Entry(row, textvariable=self.eval_fewshot_var, width=5).pack(side=tk.LEFT, padx=4)

        ttk.Label(row, text=tr("Batch size:")).pack(side=tk.LEFT, padx=(16, 4))
        self.eval_batch_var = tk.StringVar(value="1")
        ttk.Entry(row, textvariable=self.eval_batch_var, width=5).pack(side=tk.LEFT, padx=4)

        ttk.Label(row, text=tr("Limit (optional):")).pack(side=tk.LEFT, padx=(16, 4))
        self.eval_limit_var = tk.StringVar(value="")
        ttk.Entry(row, textvariable=self.eval_limit_var, width=8).pack(side=tk.LEFT, padx=4)

        # ---- Run Button ----
        run_frame = ttk.Frame(tab)
        run_frame.pack(fill=tk.X, pady=8)

        self.eval_run_btn = ttk.Button(run_frame, text=tr("🚀 Run Evaluation"),
                                       command=self._run_evaluation, state=tk.DISABLED)
        self.eval_run_btn.pack(side=tk.LEFT, padx=4)

        self.eval_save_btn = ttk.Button(run_frame, text=tr("💾 Save Results"),
                                        command=self._save_eval_results, state=tk.DISABLED)
        self.eval_save_btn.pack(side=tk.LEFT, padx=4)

        # ---- Results ----
        res_frame = ttk.LabelFrame(tab, text=tr("  📊  Results  "), padding="8")
        res_frame.pack(fill=tk.BOTH, expand=True)

        ysb = ttk.Scrollbar(res_frame, orient=tk.VERTICAL)
        self.eval_result_text = tk.Text(res_frame, font=("Consolas", 10),
                                        yscrollcommand=ysb.set, wrap=tk.WORD)
        ysb.config(command=self.eval_result_text.yview)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        self.eval_result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.eval_status = ttk.Label(tab, text=tr("Ready"), style="Sub.TLabel")
        self.eval_status.pack(anchor=tk.W, pady=4)


    def _add_eval_task(self, task: str):
        current = self.eval_tasks_var.get().strip()
        if current:
            self.eval_tasks_var.set(current + "," + task)
        else:
            self.eval_tasks_var.set(task)


    def _run_evaluation(self):
        if not self.engine.model:
            messagebox.showwarning(tr("No Model"), tr("Load or train a model first."))
            return

        tasks = [t.strip() for t in self.eval_tasks_var.get().split(",") if t.strip()]
        if not tasks:
            messagebox.showwarning(tr("No Tasks"), tr("Please specify at least one task."))
            return

        try:
            fewshot = int(self.eval_fewshot_var.get())
            batch_size = int(self.eval_batch_var.get())
            limit = int(self.eval_limit_var.get()) if self.eval_limit_var.get().strip() else None
        except ValueError:
            messagebox.showerror(tr("Invalid Input"), tr("Few-shot, batch size and limit must be integers."))
            return

        self.eval_run_btn.config(state=tk.DISABLED)
        self.eval_result_text.delete("1.0", tk.END)
        self.eval_result_text.insert(tk.END, tr("Evaluating on {tasks}...", tasks=tasks) + "\n\n")
        self.eval_status.config(text=tr("Status: Running evaluation..."))

        def run():
            try:
                results = self.engine.evaluate_model(
                    tasks=tasks,
                    num_fewshot=fewshot,
                    batch_size=batch_size,
                    limit=limit,
                )
                self.last_eval_results = results

                def update_ui():
                    self.eval_result_text.delete("1.0", tk.END)
                    if "results" in results:
                        for task_name, metrics in results["results"].items():
                            self.eval_result_text.insert(tk.END, f"=== {task_name} ===\n")
                            for k, v in metrics.items():
                                if isinstance(v, (int, float)):
                                    self.eval_result_text.insert(tk.END, f"  {k}: {v:.4f}\n")
                                else:
                                    self.eval_result_text.insert(tk.END, f"  {k}: {v}\n")
                            self.eval_result_text.insert(tk.END, "\n")
                    else:
                        self.eval_result_text.insert(tk.END, str(results))

                    self.eval_save_btn.config(state=tk.NORMAL)
                    if hasattr(self, "eval_status"):
                        self.eval_status.config(text=tr("Status: Evaluation complete ✅"))

                self.root.after(0, update_ui)

            except Exception as e:
                self.root.after(0, lambda exc=e: self.report_error(
                    exc, tr("the evaluation"), title=tr("Evaluation Error")))
            finally:
                self.root.after(0, lambda: self.eval_run_btn.config(state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()


    def _save_eval_results(self):
        path = filedialog.asksaveasfilename(
            title=tr("Save Evaluation Results"),
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), (tr("All files"), "*.*")]
        )
        if not path:
            return
        try:
            if getattr(self, "last_eval_results", None) is not None:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.last_eval_results, f, indent=2, ensure_ascii=False)
            else:
                text = self.eval_result_text.get("1.0", "end-1c")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
            messagebox.showinfo(tr("Saved"), tr("Results saved to {path}", path=path))
        except Exception as e:
            self.report_error(e, tr("saving the results"), title=tr("Save Error"))

    # ---- 2.16: a chat model on held-out dialogs (no lm-eval needed) -------
    def _build_chat_eval_frame(self, tab):
        frame = ttk.LabelFrame(tab, text=tr("  💬  Chat model on dialogs  "), padding="10")
        frame.pack(fill=tk.X, pady=(0, 8))
        row = ttk.Frame(frame)
        row.pack(fill=tk.X)
        self.chat_eval_btn = ttk.Button(row, text=tr("💬 Evaluate on dialogs…"),
                                        command=self._run_chat_eval)
        self.chat_eval_btn.pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text=tr("Dialogs (max):")).pack(side=tk.LEFT, padx=(12, 4))
        self.chat_eval_limit_var = tk.StringVar(value="300")
        ttk.Entry(row, textvariable=self.chat_eval_limit_var, width=6).pack(side=tk.LEFT)
        self.chat_eval_status = ttk.Label(
            row, text=tr("Replies to the last user turn of held-out dialogs (.jsonl), compared "
                      "with the dialog's own reply"), style="Sub.TLabel")
        self.chat_eval_status.pack(side=tk.LEFT, padx=12)
        self.chat_eval_table = ttk.Treeview(frame, columns=("metric", "value"),
                                            show="headings", height=7)
        for column, width in (("metric", 260), ("value", 520)):
            self.chat_eval_table.heading(column, text=column.title())
            self.chat_eval_table.column(column, width=width, anchor=tk.W)
        self.chat_eval_table.pack(fill=tk.X, pady=(6, 0))

    def _run_chat_eval(self):
        if not self.engine.model:
            messagebox.showwarning(tr("No Model"), tr("Load or train a model first."))
            return
        try:
            limit = int(self.chat_eval_limit_var.get()) if self.chat_eval_limit_var.get().strip() \
                else None
        except ValueError:
            messagebox.showerror(tr("Invalid Input"), tr("The number of dialogs must be an integer."))
            return
        path = filedialog.askopenfilename(
            title=tr("Dialogs to evaluate on (held out from training)"),
            filetypes=[(tr("Dialogs"), "*.jsonl *.json"), (tr("All files"), "*.*")])
        if not path:
            return
        self.chat_eval_btn.config(state=tk.DISABLED)
        for item in self.chat_eval_table.get_children():
            self.chat_eval_table.delete(item)
        self.chat_eval_status.config(text=tr("Status: replying…"))

        def progress(done: int, total: int) -> None:
            self.root.after(0, lambda: self.chat_eval_status.config(
                text=tr("Status: replying… {done}/{total}", done=done, total=total)))

        def run():
            try:
                result = self.engine.evaluate_chat(path, limit=limit, on_progress=progress)
                self.last_chat_eval = result
                self.root.after(0, lambda: self._show_chat_eval(result))
            except Exception as e:  # noqa: BLE001 - shown to the user
                self.root.after(0, lambda exc=e: self.report_error(
                    exc, tr("the chat evaluation"), title=tr("Chat evaluation")))
                self.root.after(0, lambda: self.chat_eval_status.config(text=tr("Status: failed")))
            finally:
                self.root.after(0, lambda: self.chat_eval_btn.config(state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()

    def _show_chat_eval(self, result: dict):
        # 2.25: the rows are shared with the Studio's Evaluate tab (studio/evaluate.py)
        from auralite.studio.evaluate import chat_eval_rows
        rows = chat_eval_rows(result)
        for metric, value in rows:
            self.chat_eval_table.insert("", tk.END, values=(metric, value))
        self.chat_eval_status.config(
            text=tr("Status: {dialogs} dialogs, {template} template, "
                 "{seconds:.0f} s", dialogs=result['dialogs'], template=result['template'], seconds=result['seconds']))
