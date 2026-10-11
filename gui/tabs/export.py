"""gui/tabs/export.py — ExportTabMixin for the AuraLite GUI (v2.6.5).

Extracted verbatim from `gui_app.py`; behaviour is unchanged.
"""
from ..common import (ModelExporter, filedialog, messagebox, threading, tk, ttk)

class ExportTabMixin:
    def _build_export_tab(self):
        tab = self.tab_export

        if ModelExporter is None:
            ttk.Label(tab,
                      text="⚠ Export module not available (export.py missing).",
                      font=("Segoe UI", 12, "bold")).pack(pady=40)
            return

        # ---- TorchScript Settings ----
        ts_frame = ttk.LabelFrame(tab, text="  🔥  TorchScript Settings  ", padding="10")
        ts_frame.pack(fill=tk.X, pady=(0, 8))

        row1 = ttk.Frame(ts_frame)
        row1.pack(fill=tk.X, pady=2)

        ttk.Label(row1, text="Method:").pack(side=tk.LEFT, padx=4)
        self.exp_ts_method_var = tk.StringVar(value="trace")
        ttk.Radiobutton(row1, text="Trace (recommended)", value="trace",
                        variable=self.exp_ts_method_var).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(row1, text="Script", value="script",
                        variable=self.exp_ts_method_var).pack(side=tk.LEFT, padx=4)

        self.exp_ts_opt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, text="Optimize for inference",
                        variable=self.exp_ts_opt_var).pack(side=tk.LEFT, padx=(20, 4))

        # ---- ONNX Settings ----
        onnx_frame = ttk.LabelFrame(tab, text="  🌐  ONNX Settings  ", padding="10")
        onnx_frame.pack(fill=tk.X, pady=(0, 8))

        row2 = ttk.Frame(onnx_frame)
        row2.pack(fill=tk.X, pady=2)

        ttk.Label(row2, text="Opset Version:").pack(side=tk.LEFT, padx=4)
        self.exp_onnx_opset_var = tk.StringVar(value="17")
        ttk.Combobox(row2, textvariable=self.exp_onnx_opset_var,
                     values=["14", "15", "16", "17"], state="readonly", width=6).pack(side=tk.LEFT, padx=4)

        self.exp_onnx_dyn_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text="Dynamic axes (batch & seq_len)",
                        variable=self.exp_onnx_dyn_var).pack(side=tk.LEFT, padx=(20, 4))

        # ---- Actions ----
        action_frame = ttk.Frame(tab)
        action_frame.pack(fill=tk.X, pady=10)

        self.exp_ts_btn = ttk.Button(action_frame, text="📦 Export TorchScript",
                                     command=self._export_torchscript)
        self.exp_ts_btn.pack(side=tk.LEFT, padx=4)

        self.exp_onnx_btn = ttk.Button(action_frame, text="📦 Export ONNX",
                                       command=self._export_onnx)
        self.exp_onnx_btn.pack(side=tk.LEFT, padx=4)

        self.exp_all_btn = ttk.Button(action_frame, text="🚀 Export All (Folder)",
                                      command=self._export_all)
        self.exp_all_btn.pack(side=tk.LEFT, padx=4)

        # ---- Results / Output ----
        res_frame = ttk.LabelFrame(tab, text="  📄  Export Log  ", padding="8")
        res_frame.pack(fill=tk.BOTH, expand=True)

        ysb = ttk.Scrollbar(res_frame, orient=tk.VERTICAL)
        self.exp_log_text = tk.Text(res_frame, font=("Consolas", 10),
                                    yscrollcommand=ysb.set, wrap=tk.WORD)
        ysb.config(command=self.exp_log_text.yview)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        self.exp_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.exp_status = ttk.Label(tab, text="Status: Ready to export loaded model", style="Sub.TLabel")
        self.exp_status.pack(anchor=tk.W, pady=4)


    def _check_can_export(self) -> bool:
        if not self.engine.model:
            messagebox.showwarning("No Model", "Please load or train a PyTorch model first.")
            return False
        if self.engine.is_gguf_model() or self.engine.is_hf_model():
            messagebox.showwarning("Unsupported Model", "Only native PyTorch AuraLite models can be exported.")
            return False
        return True


    def _log_export(self, msg: str):
        self.exp_log_text.insert(tk.END, msg + "\n")
        self.exp_log_text.see(tk.END)


    def _export_torchscript(self):
        if not self._check_can_export():
            return
        path = filedialog.asksaveasfilename(title="Save TorchScript Model", defaultextension=".pt",
                                          filetypes=[("TorchScript", "*.pt"), ("All files", "*.*")])
        if not path:
            return

        method = self.exp_ts_method_var.get()
        opt = self.exp_ts_opt_var.get()
        self.exp_status.config(text="Status: Exporting to TorchScript...")

        def run():
            try:
                exporter = ModelExporter(self.engine.model, self.engine.tokenizer, self.engine.device)
                saved_path = exporter.export_torchscript(path, method=method, optimize=opt)
                self.root.after(0, lambda: (
                    self._log_export(f"[TorchScript] Successfully saved to {saved_path}"),
                    self.exp_status.config(text="Status: TorchScript export complete ✅")
                ))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: (
                    messagebox.showerror("Export Error", err),
                    self.exp_status.config(text="Status: Export failed ✗")
                ))

        threading.Thread(target=run, daemon=True).start()


    def _export_onnx(self):
        if not self._check_can_export():
            return
        path = filedialog.asksaveasfilename(title="Save ONNX Model", defaultextension=".onnx",
                                          filetypes=[("ONNX", "*.onnx"), ("All files", "*.*")])
        if not path:
            return

        try:
            opset = int(self.exp_onnx_opset_var.get())
        except ValueError:
            opset = 17
        dyn = self.exp_onnx_dyn_var.get()
        self.exp_status.config(text="Status: Exporting to ONNX...")

        def run():
            try:
                exporter = ModelExporter(self.engine.model, self.engine.tokenizer, self.engine.device)
                saved_path = exporter.export_onnx(path, opset_version=opset, dynamic_axes=dyn)
                self.root.after(0, lambda: (
                    self._log_export(f"[ONNX] Successfully saved and verified at {saved_path}"),
                    self.exp_status.config(text="Status: ONNX export complete ✅")
                ))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: (
                    messagebox.showerror("ONNX Export Error", err),
                    self.exp_status.config(text="Status: Export failed ✗")
                ))

        threading.Thread(target=run, daemon=True).start()


    def _export_all(self):
        if not self._check_can_export():
            return
        dir_path = filedialog.askdirectory(title="Choose folder to export all models")
        if not dir_path:
            return

        self.exp_status.config(text="Status: Exporting all formats...")

        def run():
            try:
                exporter = ModelExporter(self.engine.model, self.engine.tokenizer, self.engine.device)
                ts_path, onnx_path = exporter.export_all(dir_path)
                self.root.after(0, lambda: (
                    self._log_export(f"[Export All] Saved TorchScript: {ts_path}"),
                    self._log_export(f"[Export All] Saved ONNX: {onnx_path}"),
                    self.exp_status.config(text="Status: Export all complete ✅")
                ))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: (
                    messagebox.showerror("Export All Error", err),
                    self.exp_status.config(text="Status: Export failed ✗")
                ))

        threading.Thread(target=run, daemon=True).start()

