"""gui/tabs/quantization.py — QuantizationTabMixin for the AuraLite GUI (v2.6.5).

Extracted verbatim from `gui_app.py`; behaviour is unchanged.
"""
from .. import common as gui_common
from ..common import (BitWidth, METHOD_DESCRIPTIONS, METHOD_SUPPORTED_BITS, QuantConfig, QuantMethod, QuantizationEngine, compare_quantizations, filedialog, format_comparison_table, messagebox, os, threading, tk, ttk)

class QuantizationTabMixin:
    def _build_quantization_tab(self):
        tab = self.tab_quant

        if not gui_common.HAS_QUANTIZATION:
            ttk.Label(tab,
                      text="⚠ Quantization module not found (quantization.py missing).",
                      font=("Segoe UI", 12, "bold")).pack(pady=40)
            return

        # ---- Method & Bits selection ---------------------------------
        method_frame = ttk.LabelFrame(tab, text="  ⚡  Method & Precision  ",
                                      padding="10")
        method_frame.pack(fill=tk.X, pady=(0, 6))

        row1 = ttk.Frame(method_frame)
        row1.pack(fill=tk.X, pady=2)

        ttk.Label(row1, text="Method:").pack(side=tk.LEFT, padx=4)
        self.q_method_var = tk.StringVar(value="dynamic")
        method_names = [
            ("Dynamic INT8", "dynamic"),
            ("Static INT8", "static"),
            ("QAT", "qat"),
            ("GPTQ", "gptq"),
            ("AWQ", "awq"),
            ("Half (FP16/BF16)", "half"),
        ]
        self.q_method_combo = ttk.Combobox(
            row1, textvariable=self.q_method_var,
            values=[n[1] for n in method_names],
            state="readonly", width=14)
        self.q_method_combo.pack(side=tk.LEFT, padx=4)
        self.q_method_combo.bind("<<ComboboxSelected>>",
                                  self._on_quant_method_changed)

        ttk.Label(row1, text="Bits:").pack(side=tk.LEFT, padx=(16, 4))
        self.q_bits_var = tk.StringVar(value="int8")
        self.q_bits_combo = ttk.Combobox(
            row1, textvariable=self.q_bits_var,
            values=["int2", "int3", "int4", "int8", "fp16", "bf16"],
            state="readonly", width=8)
        self.q_bits_combo.pack(side=tk.LEFT, padx=4)

        # Method description
        self.q_desc_label = ttk.Label(method_frame, text="", wraplength=750,
                                      style="Sub.TLabel")
        self.q_desc_label.pack(fill=tk.X, pady=(4, 0))
        self._update_quant_description()

        # ---- Advanced Options ----------------------------------------
        opts_frame = ttk.LabelFrame(tab, text="  🔧  Advanced Options  ",
                                    padding="8")
        opts_frame.pack(fill=tk.X, pady=(0, 6))

        opts_grid = ttk.Frame(opts_frame)
        opts_grid.pack(fill=tk.X)

        # Row 1: GPTQ options
        ttk.Label(opts_grid, text="Group size:").grid(
            row=0, column=0, sticky=tk.W, padx=4, pady=2)
        self.q_group_var = tk.StringVar(value="128")
        ttk.Entry(opts_grid, textvariable=self.q_group_var,
                  width=6).grid(row=0, column=1, sticky=tk.W, padx=4, pady=2)

        ttk.Label(opts_grid, text="Block size:").grid(
            row=0, column=2, sticky=tk.W, padx=(12, 4), pady=2)
        self.q_block_var = tk.StringVar(value="128")
        ttk.Entry(opts_grid, textvariable=self.q_block_var,
                  width=6).grid(row=0, column=3, sticky=tk.W, padx=4, pady=2)

        ttk.Label(opts_grid, text="Percdamp:").grid(
            row=0, column=4, sticky=tk.W, padx=(12, 4), pady=2)
        self.q_percdamp_var = tk.StringVar(value="0.01")
        ttk.Entry(opts_grid, textvariable=self.q_percdamp_var,
                  width=6).grid(row=0, column=5, sticky=tk.W, padx=4, pady=2)

        # Row 2: AWQ / QAT / calibration options
        ttk.Label(opts_grid, text="Calib. samples:").grid(
            row=1, column=0, sticky=tk.W, padx=4, pady=2)
        self.q_calib_n_var = tk.StringVar(value="128")
        ttk.Entry(opts_grid, textvariable=self.q_calib_n_var,
                  width=6).grid(row=1, column=1, sticky=tk.W, padx=4, pady=2)

        ttk.Label(opts_grid, text="Calib. seq_len:").grid(
            row=1, column=2, sticky=tk.W, padx=(12, 4), pady=2)
        self.q_calib_seq_var = tk.StringVar(value="64")
        ttk.Entry(opts_grid, textvariable=self.q_calib_seq_var,
                  width=6).grid(row=1, column=3, sticky=tk.W, padx=4, pady=2)

        ttk.Label(opts_grid, text="AWQ alpha:").grid(
            row=1, column=4, sticky=tk.W, padx=(12, 4), pady=2)
        self.q_awq_alpha_var = tk.StringVar(value="0.5")
        ttk.Entry(opts_grid, textvariable=self.q_awq_alpha_var,
                  width=6).grid(row=1, column=5, sticky=tk.W, padx=4, pady=2)

        # Row 3: QAT options
        ttk.Label(opts_grid, text="QAT epochs:").grid(
            row=2, column=0, sticky=tk.W, padx=4, pady=2)
        self.q_qat_epochs_var = tk.StringVar(value="5")
        ttk.Entry(opts_grid, textvariable=self.q_qat_epochs_var,
                  width=6).grid(row=2, column=1, sticky=tk.W, padx=4, pady=2)

        ttk.Label(opts_grid, text="QAT LR:").grid(
            row=2, column=2, sticky=tk.W, padx=(12, 4), pady=2)
        self.q_qat_lr_var = tk.StringVar(value="1e-5")
        ttk.Entry(opts_grid, textvariable=self.q_qat_lr_var,
                  width=8).grid(row=2, column=3, sticky=tk.W, padx=4, pady=2)

        self.q_symmetric_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts_grid, text="Symmetric",
                        variable=self.q_symmetric_var).grid(
            row=2, column=4, sticky=tk.W, padx=(12, 4), pady=2)

        self.q_perchannel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts_grid, text="Per-channel",
                        variable=self.q_perchannel_var).grid(
            row=2, column=5, sticky=tk.W, padx=4, pady=2)

        # ---- Calibration Data ----------------------------------------
        calib_frame = ttk.LabelFrame(tab, text="  📋  Calibration Data  ",
                                     padding="8")
        calib_frame.pack(fill=tk.X, pady=(0, 6))

        calib_btn_row = ttk.Frame(calib_frame)
        calib_btn_row.pack(fill=tk.X, pady=2)

        self.q_calib_btn = ttk.Button(
            calib_btn_row, text="📂 Select Calibration File",
            command=self._select_calib_file)
        self.q_calib_btn.pack(side=tk.LEFT, padx=4)

        ttk.Button(calib_btn_row, text="📝 Use Training File",
                   command=self._use_training_file_for_calib).pack(
            side=tk.LEFT, padx=4)

        self.q_calib_label = ttk.Label(calib_frame,
                                       text="No calibration data loaded",
                                       foreground="gray")
        self.q_calib_label.pack(anchor=tk.W, pady=2)

        self.q_calib_text = ""  # will hold calibration text

        # ---- Action Buttons ------------------------------------------
        action_frame = ttk.Frame(tab)
        action_frame.pack(fill=tk.X, pady=(0, 6))

        self.q_quantize_btn = ttk.Button(
            action_frame, text="⚡ Quantize Model",
            command=self._quantize_model, state=tk.DISABLED)
        self.q_quantize_btn.pack(side=tk.LEFT, padx=4)

        self.q_benchmark_btn = ttk.Button(
            action_frame, text="📊 Benchmark",
            command=self._benchmark_quantized, state=tk.DISABLED)
        self.q_benchmark_btn.pack(side=tk.LEFT, padx=4)

        self.q_compare_btn = ttk.Button(
            action_frame, text="🔄 Compare All Methods",
            command=self._compare_all_methods, state=tk.DISABLED)
        self.q_compare_btn.pack(side=tk.LEFT, padx=4)

        self.q_save_btn = ttk.Button(
            action_frame, text="💾 Save Quantized",
            command=self._save_quantized, state=tk.DISABLED)
        self.q_save_btn.pack(side=tk.LEFT, padx=4)

        self.q_revert_btn = ttk.Button(
            action_frame, text="↩ Revert to Original",
            command=self._revert_model, state=tk.DISABLED)
        self.q_revert_btn.pack(side=tk.LEFT, padx=4)

        # Progress
        self.q_progress_var = tk.DoubleVar()
        self.q_progress_bar = ttk.Progressbar(tab,
                                               variable=self.q_progress_var,
                                               maximum=100)
        self.q_progress_bar.pack(fill=tk.X, pady=(0, 4))

        self.q_status_label = ttk.Label(tab, text="Status: Load a model to begin")
        self.q_status_label.pack(anchor=tk.W, pady=(0, 4))

        # ---- Results Display -----------------------------------------
        result_frame = ttk.LabelFrame(tab, text="  📊  Results  ", padding="6")
        result_frame.pack(fill=tk.BOTH, expand=True)

        result_toolbar = ttk.Frame(result_frame)
        result_toolbar.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(result_toolbar, text="📋 Copy",
                   command=self._copy_quant_result).pack(side=tk.RIGHT, padx=2)
        ttk.Button(result_toolbar, text="🗑 Clear",
                   command=self._clear_quant_result).pack(side=tk.RIGHT, padx=2)

        q_ysb = ttk.Scrollbar(result_frame, orient=tk.VERTICAL)
        self.q_result_text = tk.Text(
            result_frame, font=("Consolas", 10), wrap=tk.NONE,
            yscrollcommand=q_ysb.set, height=10)
        q_ysb.config(command=self.q_result_text.yview)
        q_ysb.pack(side=tk.RIGHT, fill=tk.Y)
        self.q_result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Color tags for results
        self.q_result_text.tag_configure("header", font=("Consolas", 10, "bold"),
                                          foreground="#0066cc")
        self.q_result_text.tag_configure("good", foreground="#228B22")
        self.q_result_text.tag_configure("warn", foreground="#CC8800")
        self.q_result_text.tag_configure("error", foreground="#CC0000",
                                         font=("Consolas", 10, "bold"))

        # State for revert
        self._q_original_model = None
        self._q_last_config = None
        self._q_last_result = None


    def _on_quant_method_changed(self, *_):
        """Update available bits when method changes."""
        method = self.q_method_var.get()
        try:
            m = QuantMethod(method)
            supported = METHOD_SUPPORTED_BITS.get(m, [])
            bit_values = [b.value for b in supported]
            self.q_bits_combo.config(values=bit_values)
            if self.q_bits_var.get() not in bit_values and bit_values:
                self.q_bits_var.set(bit_values[0])
        except Exception:
            pass
        self._update_quant_description()


    def _update_quant_description(self):
        method = self.q_method_var.get()
        try:
            m = QuantMethod(method)
            desc = METHOD_DESCRIPTIONS.get(m, "")
            self.q_desc_label.config(text=desc)
        except Exception:
            self.q_desc_label.config(text="")


    def _select_calib_file(self):
        path = filedialog.askopenfilename(
            title="Select Calibration File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                self.q_calib_text = f.read()
            n = len(self.q_calib_text)
            self.q_calib_label.config(
                text=f"Loaded: {os.path.basename(path)} ({n:,} chars)",
                foreground="black")
            self._update_quant_buttons()
        except Exception as e:
            messagebox.showerror("File Error", str(e))


    def _use_training_file_for_calib(self):
        if self.selected_file_path:
            try:
                with open(self.selected_file_path, "r",
                          encoding="utf-8", errors="ignore") as f:
                    self.q_calib_text = f.read()
                n = len(self.q_calib_text)
                self.q_calib_label.config(
                    text=f"Using training file: "
                         f"{os.path.basename(self.selected_file_path)} "
                         f"({n:,} chars)",
                    foreground="black")
                self._update_quant_buttons()
            except Exception as e:
                messagebox.showerror("File Error", str(e))
        else:
            messagebox.showinfo("No File",
                                "Select a training file first on the Training tab.")


    def _update_quant_buttons(self):
        """Enable/disable quantization buttons based on current state."""
        has_model = (self.engine.model is not None and
                     not self.engine.is_gguf_model() and
                     not self.engine.is_hf_model())
        self.q_quantize_btn.config(
            state=tk.NORMAL if has_model else tk.DISABLED)
        self.q_compare_btn.config(
            state=tk.NORMAL if (has_model and self.q_calib_text)
            else tk.DISABLED)
        self.q_benchmark_btn.config(
            state=tk.NORMAL if (has_model and self._q_original_model
                                and self.q_calib_text) else tk.DISABLED)
        self.q_save_btn.config(
            state=tk.NORMAL if (has_model and self._q_last_result
                                and not self._q_last_result.errors)
            else tk.DISABLED)
        self.q_revert_btn.config(
            state=tk.NORMAL if self._q_original_model is not None
            else tk.DISABLED)


    def _build_quant_config(self) -> "QuantConfig":
        """Build QuantConfig from GUI fields."""
        return QuantConfig(
            method=QuantMethod(self.q_method_var.get()),
            bits=BitWidth(self.q_bits_var.get()),
            calibration_text=self.q_calib_text,
            calibration_samples=int(self.q_calib_n_var.get() or "128"),
            calibration_seq_length=int(self.q_calib_seq_var.get() or "64"),
            gptq_block_size=int(self.q_block_var.get() or "128"),
            gptq_percdamp=float(self.q_percdamp_var.get() or "0.01"),
            gptq_group_size=int(self.q_group_var.get() or "128"),
            awq_alpha=float(self.q_awq_alpha_var.get() or "0.5"),
            qat_epochs=int(self.q_qat_epochs_var.get() or "5"),
            qat_lr=float(self.q_qat_lr_var.get() or "1e-5"),
            symmetric=bool(self.q_symmetric_var.get()),
            per_channel=bool(self.q_perchannel_var.get()),
        )


    def _quant_progress(self, step, total, message):
        """Progress callback for quantization (called from worker thread)."""
        if total > 0:
            pct = (step / total) * 100
        else:
            pct = 0
        self.root.after(0, lambda: (
            self.q_progress_var.set(pct),
            self.q_status_label.config(text=f"Status: {message}")))


    def _quantize_model(self):
        if not gui_common.HAS_QUANTIZATION:
            messagebox.showerror("Error", "Quantization module not available.")
            return
        if self.engine.model is None or self.engine.is_gguf_model():
            messagebox.showwarning("No Model",
                                   "Load a native AuraLite .pt model first.")
            return

        try:
            config = self._build_quant_config()
        except (ValueError, KeyError) as e:
            messagebox.showerror("Config Error", str(e))
            return

        errors = config.validate()
        if errors:
            messagebox.showerror("Validation Error",
                                 "\n".join(f"• {e}" for e in errors))
            return

        # Needs calibration?
        needs_calib = config.method in (
            QuantMethod.STATIC, QuantMethod.GPTQ,
            QuantMethod.AWQ, QuantMethod.QAT)
        if needs_calib and not self.q_calib_text:
            messagebox.showwarning(
                "Calibration Needed",
                f"{config.method.value} requires calibration data.\n"
                "Load a calibration file or use the training file.")
            return

        import copy
        self._q_original_model = copy.deepcopy(self.engine.model)
        self._q_last_config = config

        self.q_quantize_btn.config(state=tk.DISABLED)
        self.q_compare_btn.config(state=tk.DISABLED)
        self.q_status_label.config(text="Status: Quantizing…")

        def run():
            try:
                q_engine = QuantizationEngine()
                q_model, result = q_engine.quantize(
                    self.engine.model, config,
                    tokenizer=self.engine.tokenizer,
                    device=self.engine.device,
                    progress_callback=self._quant_progress)

                self._q_last_result = result

                if not result.errors:
                    self.engine.model = q_model

                def update_ui():
                    self.q_result_text.delete("1.0", tk.END)
                    self.q_result_text.insert(tk.END, result.summary())
                    if result.errors:
                        self.q_status_label.config(
                            text="Status: Quantization FAILED ✗")
                        for e in result.errors:
                            self.q_result_text.insert(tk.END, f"\n✗ {e}",
                                                       "error")
                    else:
                        self.q_status_label.config(
                            text=f"Status: Quantized ✅ "
                                 f"({result.compression_ratio:.2f}× compression)")
                        self.param_label.config(
                            text=f"Parameters: {result.quantized_params:,} "
                                 f"(quantized)")
                    self._refresh_model_info()
                    self._update_quant_buttons()

                self.root.after(0, update_ui)

            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror(
                    "Quantization Error", err))
            finally:
                self.root.after(0, lambda: (
                    self.q_quantize_btn.config(state=tk.NORMAL),
                    self.q_compare_btn.config(state=tk.NORMAL),
                    self._update_quant_buttons()))

        threading.Thread(target=run, daemon=True).start()


    def _benchmark_quantized(self):
        if (self._q_original_model is None or self.engine.model is None
                or not self.q_calib_text):
            messagebox.showwarning(
                "Cannot Benchmark",
                "Need both original and quantized model, plus calibration data.")
            return

        self.q_benchmark_btn.config(state=tk.DISABLED)
        self.q_status_label.config(text="Status: Benchmarking…")

        def run():
            try:
                q_engine = QuantizationEngine()
                result = q_engine.benchmark(
                    self._q_original_model, self.engine.model,
                    self.engine.tokenizer, self.q_calib_text,
                    self.engine.device,
                    seq_length=int(self.q_calib_seq_var.get() or "64"),
                    progress_callback=self._quant_progress)

                self._q_last_result = result

                def update_ui():
                    self.q_result_text.delete("1.0", tk.END)
                    self.q_result_text.insert(tk.END,
                                              "📊 BENCHMARK RESULTS\n",
                                              "header")
                    self.q_result_text.insert(tk.END, "─" * 50 + "\n")
                    self.q_result_text.insert(tk.END, result.summary())
                    self.q_status_label.config(
                        text=f"Status: Benchmark complete ✅  "
                             f"(PPL Δ{result.perplexity_delta:+.2f}, "
                             f"speed {result.speedup:.2f}×)")

                self.root.after(0, update_ui)
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror(
                    "Benchmark Error", err))
            finally:
                self.root.after(0, lambda: (
                    self.q_benchmark_btn.config(state=tk.NORMAL),
                    self._update_quant_buttons()))

        threading.Thread(target=run, daemon=True).start()


    def _compare_all_methods(self):
        if self.engine.model is None or not self.q_calib_text:
            messagebox.showwarning("Cannot Compare",
                                   "Need a loaded model and calibration data.")
            return

        self.q_compare_btn.config(state=tk.DISABLED)
        self.q_status_label.config(text="Status: Comparing all methods…")

        def run():
            try:
                import copy
                model_copy = copy.deepcopy(self.engine.model)

                results = compare_quantizations(
                    model_copy, self.engine.tokenizer,
                    self.q_calib_text, self.engine.device,
                    seq_length=int(self.q_calib_seq_var.get() or "64"),
                    progress_callback=self._quant_progress)

                table = format_comparison_table(results)

                def update_ui():
                    self.q_result_text.delete("1.0", tk.END)
                    self.q_result_text.insert(tk.END,
                                              "🔄 COMPARISON OF ALL METHODS\n",
                                              "header")
                    self.q_result_text.insert(tk.END, table)
                    self.q_status_label.config(
                        text=f"Status: Comparison complete ✅ "
                             f"({len(results)} methods tested)")

                self.root.after(0, update_ui)
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror(
                    "Compare Error", err))
            finally:
                self.root.after(0, lambda: (
                    self.q_compare_btn.config(state=tk.NORMAL),
                    self._update_quant_buttons()))

        threading.Thread(target=run, daemon=True).start()


    def _save_quantized(self):
        if self.engine.model is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save Quantized Model",
            defaultextension=".pt",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
        if not path:
            return
        try:
            config = self._q_last_config or QuantConfig()
            QuantizationEngine.save_quantized(
                self.engine.model, path, config,
                tokenizer=self.engine.tokenizer,
                params_used=self.engine.params_used,
                result=self._q_last_result)
            messagebox.showinfo("Saved",
                                f"Quantized model saved to:\n{path}")
            self.q_status_label.config(
                text=f"Status: Saved ✅ ({os.path.basename(path)})")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))


    def _revert_model(self):
        if self._q_original_model is None:
            return
        self.engine.model = self._q_original_model
        self._q_original_model = None
        self._q_last_result = None
        self._q_last_config = None
        self.q_status_label.config(text="Status: Reverted to original model ↩")
        self._refresh_model_info()
        self._update_quant_buttons()
        if self.engine.model is not None:
            n = self.engine.model.count_parameters()
            self.param_label.config(text=f"Parameters: {n:,}")


    def _copy_quant_result(self):
        text = self.q_result_text.get("1.0", "end-1c")
        if text.strip():
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.q_status_label.config(text="Status: Results copied ✅")


    def _clear_quant_result(self):
        self.q_result_text.delete("1.0", tk.END)

