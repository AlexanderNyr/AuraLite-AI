"""gui/tabs/training.py — TrainTabMixin for the AuraLite GUI (v2.6.5).

Extracted verbatim from `gui_app.py`; behaviour is unchanged.
"""
from .. import common as gui_common
from ..common import (CONFIG_PRESETS, DEFAULT_RECIPE, cluster_lan_address, run_cluster_worker, Figure, FigureCanvasTkAgg, ParamValidationError, _fmt_duration, estimate_n_params, filedialog, messagebox, os, parse_dropout, recommend_epochs, recommend_gen_length, threading, time, tk, ttk, validate_params)

R = DEFAULT_RECIPE      # 2.12: one recipe for the GUI, the CLI and train()

class TrainTabMixin:
    def _build_training_tab(self):
        tab = self.tab_train

        # ---- Configuration Presets -------------------------------------
        preset_frame = ttk.LabelFrame(tab, text="  📋  Configuration Presets  ",
                                      padding="8")
        preset_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(preset_frame, text="Choose a preset:").pack(side=tk.LEFT, padx=4)
        self.preset_var = tk.StringVar(value="Small (default)")
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                     values=list(CONFIG_PRESETS.keys()),
                                     state="readonly", width=30)
        preset_combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(preset_frame, text="📥 Apply Preset",
                   command=self._apply_preset).pack(side=tk.LEFT, padx=4)

        # ---- Architecture & Hyperparameters ---------------------------
        hp_frame = ttk.LabelFrame(tab, text="  ⚙️  Architecture & Hyperparameters  ",
                                  padding="10")
        hp_frame.pack(fill=tk.X, pady=(0, 8))

        grid = ttk.Frame(hp_frame)
        grid.pack(fill=tk.X)

        self.params = {
            "lr":         tk.StringVar(value=str(R["lr"])),
            "epochs":     tk.StringVar(value="100"),
            "d_model":    tk.StringVar(value="128"),
            "d_ff":       tk.StringVar(value="256"),
            "n_heads":    tk.StringVar(value="4"),
            "n_layers":   tk.StringVar(value="4"),
            "seq_length": tk.StringVar(value="64"),
            "batch_size": tk.StringVar(value=str(R["batch_size"])),
            "dropout":    tk.StringVar(value=str(R["dropout"])),
            "grad_clip":  tk.StringVar(value=str(R["grad_clip"])),
        }

        labels = [
            ("Learning Rate:",       "lr"),
            ("Model Dim (D_Model):", "d_model"),
            ("Epochs:",              "epochs"),
            ("FF Dim (D_FF):",       "d_ff"),
            ("Heads (N_Heads):",     "n_heads"),
            ("Layers (N_Layers):",   "n_layers"),
            ("Context Window (Seq):","seq_length"),
            ("Batch Size:",          "batch_size"),
            ("Dropout:",             "dropout"),
            ("Grad Clip:",           "grad_clip"),
        ]

        for i, (text, key) in enumerate(labels):
            row = i // 2
            col = (i % 2) * 2
            ttk.Label(grid, text=text).grid(row=row, column=col,
                                            sticky=tk.W, padx=5, pady=3)
            ttk.Entry(grid, textvariable=self.params[key],
                      width=10).grid(row=row, column=col + 1,
                                      sticky=tk.W, padx=5, pady=3)

        # ---- Auto-recommend epochs --------------------------------------
        auto_row = ttk.Frame(hp_frame)
        auto_row.pack(fill=tk.X, pady=(6, 0))

        ttk.Button(auto_row, text="🎯 Auto-recommend Epochs",
                   command=self._auto_epochs).pack(side=tk.LEFT, padx=4)
        ttk.Label(auto_row,
                  text="(needs a selected file — uses ~20 tokens/param heuristic)",
                  style="Sub.TLabel").pack(side=tk.LEFT, padx=4)
        self.auto_epochs_hint = ttk.Label(auto_row, text="", style="Sub.TLabel",
                                          foreground="#0a6")
        self.auto_epochs_hint.pack(side=tk.LEFT, padx=8)

        # ---- Tokenizer & options ----------------------------------------
        tok_frame = ttk.LabelFrame(tab, text="  🔤  Tokenizer & Options  ",
                                   padding="10")
        tok_frame.pack(fill=tk.X, pady=(0, 8))

        row1 = ttk.Frame(tok_frame)
        row1.pack(fill=tk.X, pady=2)

        ttk.Label(row1, text="Tokenizer:").pack(side=tk.LEFT, padx=4)
        self.tok_var = tk.StringVar(value=R["tokenizer"])
        ttk.Radiobutton(row1, text="BPE (recommended)", value="bpe",
                        variable=self.tok_var).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(row1, text="Char-level", value="char",
                        variable=self.tok_var).pack(side=tk.LEFT, padx=4)

        ttk.Label(row1, text="BPE Vocab:").pack(side=tk.LEFT, padx=(16, 4))
        self.bpe_vocab_var = tk.StringVar(value=str(R["bpe_vocab_size"]))
        ttk.Entry(row1, textvariable=self.bpe_vocab_var,
                  width=7).pack(side=tk.LEFT, padx=4)

        ttk.Label(row1, text="Val Split:").pack(side=tk.LEFT, padx=(16, 4))
        self.val_split_var = tk.StringVar(value=str(R["val_split"]))
        ttk.Entry(row1, textvariable=self.val_split_var,
                  width=6).pack(side=tk.LEFT, padx=4)

        # 2.9: stop when val loss has not improved for N epochs and keep the
        # best epoch's weights (needs a validation split). 2.12: on by default.
        ttk.Label(row1, text="Early stop:").pack(side=tk.LEFT, padx=(12, 4))
        self.early_stop_var = tk.StringVar(value=str(R["early_stopping_patience"]))
        ttk.Entry(row1, textvariable=self.early_stop_var,
                  width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(row1, text="(epochs, 0=off)", style="Sub.TLabel").pack(
            side=tk.LEFT, padx=2)

        # 2.12: a time limit; the learning-rate decay is sized to end at it.
        ttk.Label(row1, text="Time limit:").pack(side=tk.LEFT, padx=(12, 4))
        self.time_limit_var = tk.StringVar(value="0")
        ttk.Entry(row1, textvariable=self.time_limit_var,
                  width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(row1, text="(min, 0=off)", style="Sub.TLabel").pack(
            side=tk.LEFT, padx=2)

        # NEW: GQA, Accumulation, ALiBi, LoRA
        row1b = ttk.Frame(tok_frame)
        row1b.pack(fill=tk.X, pady=2)

        ttk.Label(row1b, text="KV Heads (GQA):").pack(side=tk.LEFT, padx=4)
        self.n_kv_heads_var = tk.StringVar(value="0")
        ttk.Entry(row1b, textvariable=self.n_kv_heads_var,
                  width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(row1b, text="(0=MHA)", style="Sub.TLabel").pack(
            side=tk.LEFT, padx=2)

        ttk.Label(row1b, text="Accumulation:").pack(side=tk.LEFT, padx=(12, 4))
        self.accum_var = tk.StringVar(value="1")
        ttk.Entry(row1b, textvariable=self.accum_var,
                  width=4).pack(side=tk.LEFT, padx=2)

        self.alibi_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1b, text="ALiBi",
                        variable=self.alibi_var).pack(side=tk.LEFT, padx=8)

        ttk.Label(row1b, text="LoRA rank:").pack(side=tk.LEFT, padx=(8, 4))
        self.lora_var = tk.StringVar(value="0")
        ttk.Entry(row1b, textvariable=self.lora_var,
                  width=4).pack(side=tk.LEFT, padx=2)

        self.checkpoint_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1b, text="Gradient Checkpointing (VRAM saving)",
                        variable=self.checkpoint_var).pack(side=tk.LEFT, padx=12)

        # NEW: Multi-GPU (DDP) toggle — v2.3
        self.ddp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1b, text="Multi-GPU (DDP)",
                        variable=self.ddp_var).pack(side=tk.LEFT, padx=12)

        # 2.14: training on several computers of the local network
        # (auralite/engine/cluster.py). This computer either waits for N
        # workers when training starts, or joins another computer's runs.
        row_cluster = ttk.Frame(tok_frame)
        row_cluster.pack(fill=tk.X, pady=2)
        ttk.Label(row_cluster, text="Cluster workers:").pack(side=tk.LEFT, padx=4)
        self.cluster_workers_var = tk.StringVar(value="0")
        ttk.Entry(row_cluster, textvariable=self.cluster_workers_var,
                  width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(row_cluster, text="Port:").pack(side=tk.LEFT, padx=(8, 4))
        self.cluster_port_var = tk.StringVar(value="29500")
        ttk.Entry(row_cluster, textvariable=self.cluster_port_var,
                  width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(row_cluster, text=f"(this computer: {cluster_lan_address()}; 0=off)",
                  style="Sub.TLabel").pack(side=tk.LEFT, padx=4)
        ttk.Label(row_cluster, text="Join as worker:").pack(side=tk.LEFT, padx=(16, 4))
        self.cluster_join_var = tk.StringVar(value="")
        ttk.Entry(row_cluster, textvariable=self.cluster_join_var,
                  width=18).pack(side=tk.LEFT, padx=2)
        self.cluster_join_btn = ttk.Button(row_cluster, text="Join",
                                           command=self.toggle_cluster_worker)
        self.cluster_join_btn.pack(side=tk.LEFT, padx=4)
        # its own line: a worker that cannot connect says why, at some length
        self.cluster_status_label = ttk.Label(tok_frame, text="", style="Sub.TLabel",
                                              wraplength=900, justify=tk.LEFT)
        self.cluster_status_label.pack(fill=tk.X, padx=4)
        self._cluster_worker_stop = None

        # NEW: RoPE Scaling (v2.4)
        rope_row = ttk.Frame(tok_frame)
        rope_row.pack(fill=tk.X, pady=2)

        ttk.Label(rope_row, text="RoPE Scaling:").pack(side=tk.LEFT, padx=4)
        self.rope_type_var = tk.StringVar(value="none")
        rope_combo = ttk.Combobox(rope_row, textvariable=self.rope_type_var,
                                  values=["none", "linear", "ntk", "yarn"],
                                  state="readonly", width=10)
        rope_combo.pack(side=tk.LEFT, padx=2)

        ttk.Label(rope_row, text="Factor:").pack(side=tk.LEFT, padx=(12, 4))
        self.rope_factor_var = tk.StringVar(value="1.0")
        ttk.Entry(rope_row, textvariable=self.rope_factor_var, width=6).pack(side=tk.LEFT, padx=2)
        # 2.11: measured — scaling alone does not let a model read past the
        # length it was trained on (docs/engine.md, "Context length").
        ttk.Label(rope_row, style="Sub.TLabel",
                  text="not a free extension: train on longer windows to read further"
                  ).pack(side=tk.LEFT, padx=(8, 2))

        # NEW: Modern training stack (v2.6) — QK-norm + Muon + WSD
        modern_row = ttk.Frame(tok_frame)
        modern_row.pack(fill=tk.X, pady=2)

        self.qk_norm_var = tk.BooleanVar(value=bool(R["use_qk_norm"]))
        ttk.Checkbutton(modern_row, text="QK-norm (modern, stable attention)",
                        variable=self.qk_norm_var).pack(side=tk.LEFT, padx=4)

        ttk.Label(modern_row, text="Optimizer:").pack(side=tk.LEFT, padx=(12, 4))
        # Muon + QK-norm: 5 % fewer bits per byte than AdamW on a 100 KB text
        # and a tie on a megabyte at equal time (docs/training.md § 8).
        self.opt_var = tk.StringVar(value=R["optimizer"])
        ttk.Combobox(modern_row, textvariable=self.opt_var,
                     values=["muon", "adamw"],
                     state="readonly", width=7).pack(side=tk.LEFT, padx=2)

        ttk.Label(modern_row, text="Muon LR:").pack(side=tk.LEFT, padx=(8, 2))
        self.muon_lr_var = tk.StringVar(value=str(R["muon_lr"]))
        ttk.Entry(modern_row, textvariable=self.muon_lr_var,
                  width=7).pack(side=tk.LEFT, padx=2)

        ttk.Label(modern_row, text="LR Schedule:").pack(side=tk.LEFT, padx=(12, 4))
        self.sched_var = tk.StringVar(value=R["lr_schedule"])
        ttk.Combobox(modern_row, textvariable=self.sched_var,
                     values=["wsd", "cosine"],
                     state="readonly", width=7).pack(side=tk.LEFT, padx=2)

        row2 = ttk.Frame(tok_frame)
        row2.pack(fill=tk.X, pady=2)

        self.compile_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="torch.compile (faster, slow first epoch)",
                        variable=self.compile_var).pack(side=tk.LEFT, padx=4)

        self.continue_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="Continue training current model",
                        variable=self.continue_var).pack(side=tk.LEFT, padx=12)

        # Architecture: input/output embedding tying (untied by default).
        self.tie_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="Tie in/out embeddings (legacy)",
                        variable=self.tie_var).pack(side=tk.LEFT, padx=12)

        ttk.Label(row2, text="Autosave every N epochs (0=off):").pack(
            side=tk.LEFT, padx=(8, 4))
        self.autosave_var = tk.StringVar(value="0")
        ttk.Entry(row2, textvariable=self.autosave_var,
                  width=5).pack(side=tk.LEFT, padx=4)

        # ---- File + start/stop -----------------------------------------
        run_frame = ttk.LabelFrame(tab, text="  🚀  Run  ", padding="10")
        run_frame.pack(fill=tk.X, pady=(0, 8))

        top_row = ttk.Frame(run_frame)
        top_row.pack(fill=tk.X, pady=2)

        self.file_btn = ttk.Button(top_row, text="📂 Select .txt File",
                                   command=self.select_file)
        self.file_btn.pack(side=tk.LEFT, padx=4)

        self.train_btn = ttk.Button(top_row, text="🚀 Start Training",
                                    command=self.start_training,
                                    state=tk.DISABLED)
        self.train_btn.pack(side=tk.LEFT, padx=4)

        self.stop_btn = ttk.Button(top_row, text="🛑 Stop",
                                   command=self.stop_training,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        # NEW: Save/Load config
        self.save_cfg_btn = ttk.Button(top_row, text="💾 Save Config",
                                       command=self.save_config)
        self.save_cfg_btn.pack(side=tk.LEFT, padx=4)

        self.load_cfg_btn = ttk.Button(top_row, text="📂 Load Config",
                                       command=self.load_config)
        self.load_cfg_btn.pack(side=tk.LEFT, padx=4)

        self.file_label = ttk.Label(run_frame, text="No file selected",
                                    foreground="gray")
        self.file_label.pack(pady=2, anchor=tk.W)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(run_frame,
                                            variable=self.progress_var,
                                            maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=4)

        # Live detail line under the bar (batch / phase / speed / ETA)
        self.progress_detail_var = tk.StringVar(value="")
        self.progress_detail_label = ttk.Label(
            run_frame, textvariable=self.progress_detail_var, style="Sub.TLabel")
        self.progress_detail_label.pack(fill=tk.X, pady=(0, 2), anchor=tk.W)

        self.status_label = ttk.Label(run_frame,
                                      text="Status: Waiting for file…")
        self.status_label.pack(pady=(0, 2), anchor=tk.W)

        # ---- Loss history ------------------------------------------------
        hist_frame = ttk.LabelFrame(tab, text="  📉  Loss History  ", padding="6")
        hist_frame.pack(fill=tk.BOTH, expand=True)

        if gui_common.HAS_MATPLOTLIB:
            # Matplotlib plot for loss visualization
            self.fig = Figure(figsize=(6, 3), dpi=80)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel("Loss")
            self.ax.set_title("Training Loss")
            self.canvas = FigureCanvasTkAgg(self.fig, master=hist_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            # Text fallback
            self.loss_text = tk.Text(hist_frame, height=6,
                                     font=("Consolas", 9),
                                     state=tk.DISABLED, wrap=tk.NONE)
            self.loss_text.pack(fill=tk.BOTH, expand=True)
            ttk.Label(hist_frame, text="(Install matplotlib for live plots)",
                      style="Sub.TLabel").pack()


    def _apply_preset(self):
        """Apply a configuration preset to all fields."""
        preset_name = self.preset_var.get()
        preset = CONFIG_PRESETS.get(preset_name)
        if not preset:
            return
        for key, value in preset.items():
            if key == "n_kv_heads":
                self.n_kv_heads_var.set(str(value))
            elif key == "use_gradient_checkpointing":
                self.checkpoint_var.set(bool(value))
            elif key == "rope_scaling":
                if value and isinstance(value, dict):
                    self.rope_type_var.set(value.get("type", "none") or "none")
                    self.rope_factor_var.set(str(value.get("factor", 1.0)))
                else:
                    self.rope_type_var.set("none")
                    self.rope_factor_var.set("1.0")
            else:
                if key in self.params:
                    self.params[key].set(str(value))
        self.status_label.config(
            text=f"Status: Preset '{preset_name}' applied ✅")


    def _auto_epochs(self):
        """Pick a reasonable number of epochs based on dataset & model size.

        Reads current architecture fields, estimates total params, peeks at
        the selected training file to get an approximate token count, and
        plugs both into `recommend_epochs`. The user can always tweak the
        result manually afterwards.
        """
        if not self.selected_file_path:
            messagebox.showinfo(
                "No file",
                "Select a training .txt file first — auto-recommendation needs "
                "to know how big your dataset is.")
            return

        try:
            d_model    = int(self.params["d_model"].get())
            d_ff       = int(self.params["d_ff"].get())
            n_heads    = int(self.params["n_heads"].get())
            n_layers   = int(self.params["n_layers"].get())
            seq_length = int(self.params["seq_length"].get())
            batch_size = int(self.params["batch_size"].get())
            n_kv_heads = int(self.n_kv_heads_var.get()) or None
            bpe_vocab  = int(self.bpe_vocab_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid params",
                "Some hyperparameter fields contain non-numeric values.")
            return

        tok_kind = self.tok_var.get()

        # Approximate vocab + token count without doing a full BPE pass:
        # for char-level we count unique chars; for BPE the trained vocab
        # caps at bpe_vocab; raw token count ≈ chars / 3 (typical English).
        try:
            with open(self.selected_file_path, "r", encoding="utf-8",
                      errors="ignore") as f:
                text_sample = f.read()
        except Exception as e:
            messagebox.showerror("File Error", f"Could not read file:\n{e}")
            return

        n_chars = len(text_sample)
        if tok_kind == "bpe":
            vocab = min(bpe_vocab, max(2, len(set(text_sample))))
            n_tokens = max(seq_length + 2, int(n_chars / 3.0))   # rough BPE estimate
        else:
            vocab = max(2, len(set(text_sample)))
            n_tokens = n_chars

        n_params = estimate_n_params(vocab, d_model, n_layers, d_ff,
                                     n_heads, n_kv_heads)
        epochs = recommend_epochs(n_tokens, n_params, batch_size, seq_length)

        self.params["epochs"].set(str(epochs))
        hint = (f"~{n_params/1e6:.2f}M params · ~{n_tokens:,} tokens "
                f"· vocab≈{vocab} → {epochs} epochs")
        self.auto_epochs_hint.config(text=hint)
        self.status_label.config(text=f"Status: Auto-set Epochs = {epochs} ✅")
        print(f"[AuraLite] Auto-recommend: {hint}")


    def _auto_gen_length(self):
        """Pick a generation length based on the seed and the model's context."""
        seed = self.seed_entry.get()
        tokenizer = self.engine.tokenizer
        max_seq_len = (self.engine.model.max_seq_len
                       if self.engine.model is not None else 4096)

        full = bool(getattr(self.engine, "full_context", False))
        length = recommend_gen_length(seed, tokenizer, max_seq_len=max_seq_len,
                                      full_context=full)
        # Keep the slider in its own range; bump the upper bound if needed.
        try:
            slider_max = float(self.len_scale.cget("to"))
            if length > slider_max:
                self.len_scale.configure(to=max(slider_max, length))
        except tk.TclError:
            pass
        self.len_scale.set(length)
        self._update_len_display(length)
        print(f"[AuraLite] Auto-recommend gen length: {length} tokens "
              f"(seed≈{len(seed)} chars, max_seq_len={max_seq_len}"
              f"{', full context' if full else ''})")


    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Training File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if file_path:
            self.selected_file_path = file_path
            self.file_label.config(
                text=f"Selected: {os.path.basename(file_path)}",
                foreground="black",
            )
            self.train_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Ready to train")


    # ---- cluster (2.14) ----------------------------------------------
    def _cluster_train_params(self) -> dict:
        """`cluster_workers` / `cluster_port` when this computer leads a cluster."""
        workers = int(str(self.cluster_workers_var.get() or 0).strip() or 0)
        if workers <= 0:
            return {}
        return {"cluster_workers": workers,
                "cluster_port": int(str(self.cluster_port_var.get() or 29500).strip())}

    def toggle_cluster_worker(self):
        """Join another computer's training runs, or leave them."""
        if self._cluster_worker_stop is not None:
            self._cluster_worker_stop.set()
            self.cluster_join_btn.config(text="Join")
            self.cluster_status_label.config(text="Leaving after this step…")
            return
        address = str(self.cluster_join_var.get() or "").strip()
        if not address:
            messagebox.showerror("Cluster", "Enter the address of the main computer "
                                            "(for example 192.168.1.10 or 192.168.1.10:29500).")
            return
        stop = threading.Event()
        self._cluster_worker_stop = stop
        self.cluster_join_btn.config(text="Leave")

        def status(message: str) -> None:
            self.root.after(0, lambda m=message: self.cluster_status_label.config(text=m))
            print(f"[cluster] {message}")

        def run():
            try:
                run_cluster_worker(address, on_status=status, stop_event=stop)
            except Exception as err:   # noqa: BLE001 - shown to the user
                status(f"Cluster error: {err}")
            finally:
                def done():
                    self._cluster_worker_stop = None
                    self.cluster_join_btn.config(text="Join")
                self.root.after(0, done)

        threading.Thread(target=run, daemon=True).start()

    def update_progress(self, current, total, loss, val_loss=None, info=None):
        """Called from the training thread — marshal the actual UI update
        onto the Tk main loop via root.after (tkinter is not thread-safe).

        `info` (optional dict from AuraLiteEngine) carries live batch/ETA fields
        so the progress bar moves *between* epochs, not only after each one.
        Mid-epoch updates are throttled (~8 Hz) so Tk stays responsive.
        """
        if info is None:
            info = {}
        # Copy dict so the worker thread can mutate its next payload safely.
        info = dict(info)
        phase = str(info.get("phase") or "")
        is_epoch_end = bool(info.get("is_epoch_end")) or phase == "epoch_end"
        important = (is_epoch_end or phase in ("setup", "val", "stopped", "cluster")
                     or current <= 0)
        now = time.time()
        # Always schedule important events; throttle the rest (~8 UI fps).
        if (not important) and (now - getattr(self, "_progress_ui_last_ts", 0.0) < 0.12):
            self._progress_ui_pending = (current, total, loss, val_loss, info)
            return
        self._progress_ui_last_ts = now
        self._progress_ui_pending = None
        self.root.after(0, self._apply_progress, current, total, loss, val_loss, info)


    def _apply_progress(self, current, total, loss, val_loss, info=None):
        if info is None:
            info = {}
        phase = str(info.get("phase") or ("epoch_end" if info.get("is_epoch_end") else "train"))
        is_epoch_end = bool(info.get("is_epoch_end")) or phase == "epoch_end"

        # Prefer engine-provided overall percent (includes intra-epoch fraction).
        if "percent" in info and info["percent"] is not None:
            try:
                percent = float(info["percent"])
            except (TypeError, ValueError):
                percent = (current / max(1, total)) * 100.0
        else:
            percent = (current / max(1, total)) * 100.0
        percent = max(0.0, min(100.0, percent))
        self.progress_var.set(percent)

        # LR: prefer info, fall back to live scheduler
        lr = info.get("lr")
        if lr is None:
            lr = self.engine.scheduler.get_lr() if self.engine.scheduler else 0
        try:
            lr_f = float(lr)
        except (TypeError, ValueError):
            lr_f = 0.0

        now = time.time()
        elapsed = info.get("elapsed")
        if elapsed is None:
            elapsed = (now - self.train_start_time) if self.train_start_time else 0.0
        try:
            elapsed = float(elapsed)
        except (TypeError, ValueError):
            elapsed = 0.0

        # ETA from engine (batch-accurate) when present
        eta_seconds = info.get("eta_seconds")
        eta_str = "—"
        if eta_seconds is not None:
            try:
                eta_str = _fmt_duration(float(eta_seconds))
            except (TypeError, ValueError):
                eta_str = "—"

        # On epoch end keep rolling epoch-duration stats (for plot / fallback ETA)
        if is_epoch_end and current > 0:
            if self._last_epoch_ts is not None:
                self.epoch_times.append(now - self._last_epoch_ts)
                if len(self.epoch_times) > 20:
                    self.epoch_times = self.epoch_times[-20:]
            self._last_epoch_ts = now
            if eta_seconds is None and self.epoch_times:
                remaining_epochs = max(0, total - current)
                eta_str = _fmt_duration(
                    (sum(self.epoch_times) / len(self.epoch_times)) * remaining_epochs)

        # Speed line
        speed_bits = []
        tps = info.get("tokens_per_sec")
        bps = info.get("batches_per_sec")
        if tps is not None:
            try:
                speed_bits.append(f"{float(tps):.0f} tok/s")
            except (TypeError, ValueError):
                pass
        if bps is not None:
            try:
                speed_bits.append(f"{float(bps):.2f} batch/s")
            except (TypeError, ValueError):
                pass
        if is_epoch_end and self.epoch_times:
            speed_bits.append(f"{self.epoch_times[-1]:.1f}s/ep")
        speed_str = ("  |  " + " · ".join(speed_bits)) if speed_bits else ""

        batch = info.get("batch")
        batches = info.get("batches")
        batch_part = ""
        if batch is not None and batches is not None:
            try:
                batch_part = f"  |  Batch {int(batch)}/{int(batches)}"
            except (TypeError, ValueError):
                batch_part = ""

        epoch_pct = info.get("epoch_percent")
        epoch_pct_part = ""
        if epoch_pct is not None and not is_epoch_end and phase == "train":
            try:
                epoch_pct_part = f" ({float(epoch_pct):.0f}% of epoch)"
            except (TypeError, ValueError):
                pass

        # `info`/progress values come from a worker thread (and from the engine's
        # own payloads), so they are treated as untrusted input: a string here
        # used to raise ValueError and freeze the progress strip mid-training.
        val_part = ""
        if val_loss is not None:
            try:
                val_part = f"  |  Val: {float(val_loss):.4f}"
            except (TypeError, ValueError):
                val_part = f"  |  Val: {val_loss}"
            # 2.11: bits per byte compares char and BPE models
            bpb = (info or {}).get("val_bpb") if isinstance(info, dict) else None
            try:
                if bpb is not None:
                    val_part += f" ({float(bpb):.3f} bits/byte)"
            except (TypeError, ValueError):
                pass
        loss_part = ""
        try:
            if loss is not None and float(loss) > 0:
                loss_part = f"  |  Loss: {float(loss):.4f}"
        except (TypeError, ValueError):
            pass

        if current <= 0 and phase in ("setup", "cluster"):
            head = info.get("message") or "Preparing…"
            self.status_label.config(
                text=f"Status: {head}  |  Elapsed: {_fmt_duration(elapsed)}"
            )
        elif phase == "val":
            head = info.get("message") or f"Epoch {current}/{total} — validation…"
            self.status_label.config(
                text=f"Status: {head}{loss_part}"
                     f"  |  LR: {lr_f:.6f}"
                     f"  |  Elapsed: {_fmt_duration(elapsed)}  |  ETA: {eta_str}"
            )
        elif phase == "stopped":
            head = info.get("message") or "Stopping…"
            self.status_label.config(text=f"Status: {head} 🛑")
        else:
            self.status_label.config(
                text=f"Status: Epoch {current}/{total}{epoch_pct_part}"
                     f"{batch_part}{loss_part}{val_part}"
                     f"  |  LR: {lr_f:.6f}{speed_str}"
                     f"  |  Elapsed: {_fmt_duration(elapsed)}  |  ETA: {eta_str}"
            )

        # Detail strip under the bar — always shows "what is happening now"
        detail_bits = []
        msg = info.get("message")
        if msg:
            detail_bits.append(str(msg))
        gstep = info.get("global_step")
        tstep = info.get("total_steps")
        if gstep is not None and tstep is not None:
            try:
                detail_bits.append(f"step {int(gstep)}/{int(tstep)}")
            except (TypeError, ValueError):
                pass
        detail_bits.append(f"total {percent:.1f}%")
        if eta_str and eta_str != "—":
            detail_bits.append(f"~{eta_str} left")
        try:
            self.progress_detail_var.set(" · ".join(detail_bits))
        except Exception:
            pass

        # Loss history / plot — only once per finished epoch (avoid spam)
        if is_epoch_end and current > 0:
            vtxt = f"{val_loss:.4f}" if val_loss is not None else None
            # Avoid duplicate epoch rows if engine somehow double-fires
            if not self.loss_history or self.loss_history[-1][0] != current:
                self.loss_history.append((current, float(loss) if loss is not None else 0.0, val_loss))
                try:
                    train_txt = f"{float(loss):.4f}"
                except (TypeError, ValueError):
                    train_txt = str(loss)
                self._append_loss_line(
                    f"epoch {current:>4}/{total}   train {train_txt}   "
                    f"val {vtxt or '  —  '}   eta {eta_str}")
                self._update_loss_plot()


    def stop_training(self):
        # 2.12: the first press finishes the run with a short learning-rate
        # decay (a run that ends at the peak LR is measurably worse); the
        # second press stops at once.
        if not self.finish_event.is_set():
            self.finish_event.set()
            self.stop_btn.config(text="🛑 Stop now")
            self.status_label.config(
                text="Status: Finishing — cooling down the learning rate… "
                     "(press Stop again to stop now) ⏳")
            return
        self.stop_event.set()
        self.status_label.config(text="Status: Stopping… 🛑")


    def start_training(self):
        if not self.selected_file_path:
            return

        try:
            params = {
                "lr":          float(self.params["lr"].get()),
                "epochs":      int(self.params["epochs"].get()),
                "d_model":     int(self.params["d_model"].get()),
                "d_ff":        int(self.params["d_ff"].get()),
                "n_heads":     int(self.params["n_heads"].get()),
                "n_layers":    int(self.params["n_layers"].get()),
                "seq_length":  int(self.params["seq_length"].get()),
                "batch_size":  int(self.params["batch_size"].get()),
                "dropout":     parse_dropout(self.params["dropout"].get()),
                "grad_clip":   float(self.params["grad_clip"].get()),
                "tokenizer":       self.tok_var.get(),
                "bpe_vocab_size":  int(self.bpe_vocab_var.get()),
                "val_split":       float(self.val_split_var.get()),
                "early_stopping_patience": int(self.early_stop_var.get() or 0),
                "time_budget_s": 60.0 * float(str(self.time_limit_var.get() or 0)),
                # Live progress bar: ~1% of epoch + time-based ticks
                "progress_every_seconds": 0.5,
                "use_compile":     bool(self.compile_var.get()),
                "continue_training": bool(self.continue_var.get()),
                "autosave_every":  int(self.autosave_var.get()),
                "n_kv_heads":      int(self.n_kv_heads_var.get()) or None,
                "accumulation_steps": int(self.accum_var.get()) or 1,
                "use_alibi":         bool(self.alibi_var.get()),
                "lora_rank":         int(self.lora_var.get()) or 0,
                "use_gradient_checkpointing": bool(self.checkpoint_var.get()),
                # NEW: RoPE scaling
                "rope_scaling": {
                    "type": self.rope_type_var.get() if self.rope_type_var.get() != "none" else None,
                    "factor": float(self.rope_factor_var.get()) if self.rope_type_var.get() != "none" else 1.0,
                } if self.rope_type_var.get() != "none" else None,
                # NEW: Multi-GPU (DDP) — v2.3
                "use_ddp": bool(self.ddp_var.get()),
                # 2.14: other computers that train with this one
                **self._cluster_train_params(),
                # NEW: modern training stack — v2.6
                "use_qk_norm": bool(self.qk_norm_var.get()),
                "optimizer": self.opt_var.get(),
                "muon_lr": float(self.muon_lr_var.get()),
                "lr_schedule": self.sched_var.get(),
                # Architecture: tie the input embedding with the output head
                # (legacy; off by default to avoid gradient cancellation).
                "tie_word_embeddings": bool(self.tie_var.get()),
            }
        except ValueError:
            messagebox.showerror("Params Error",
                                 "Please enter valid numbers in all fields!")
            return

        # NEW: Validate parameters before training
        errors = validate_params(params)
        if errors:
            err_msg = "Parameter validation errors:\n\n" + "\n".join(
                f"• {e}" for e in errors
            )
            messagebox.showerror("Validation Error", err_msg)
            return

        if params["continue_training"] and self.engine.model is None:
            messagebox.showwarning(
                "No model",
                "«Continue training» is checked, but there is no model "
                "in memory — a new one will be created.")
            params["continue_training"] = False

        if params["continue_training"] and self.engine.is_gguf_model():
            messagebox.showwarning(
                "GGUF is inference-only",
                ".gguf models cannot be continued/trained in AuraLite. "
                "Uncheck «Continue training current model» to train a new native .pt model."
            )
            return

        if params["continue_training"] and self.engine.is_hf_model():
            messagebox.showwarning(
                "Hugging Face models use LoRA / QLoRA fine-tuning",
                "Loaded Hugging Face models should be fine-tuned from the Model tab. "
                "Uncheck «Continue training current model» to train a new native .pt model."
            )
            return

        if params["autosave_every"] > 0:
            base = os.path.splitext(self.selected_file_path)[0]
            params["autosave_path"] = base + "_autosave.pt"

        try:
            with open(self.selected_file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            messagebox.showerror("File Error", f"Could not read file:\n{e}")
            return

        self.stop_event.clear()
        self.finish_event.clear()
        self.stop_btn.config(text="🛑 Stop")
        self.loss_history = []
        # Reset ETA tracking
        self.train_start_time = time.time()
        self.epoch_times = []
        self._last_epoch_ts = self.train_start_time
        self.progress_var.set(0)
        self._progress_ui_last_ts = 0.0
        self._progress_ui_pending = None
        try:
            self.progress_detail_var.set("Preparing for training…")
        except Exception:
            pass
        self.status_label.config(text="Status: Starting… ⏳")
        self.train_btn.config(state=tk.DISABLED)
        self.file_btn.config(state=tk.DISABLED)
        self.gen_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        if gui_common.HAS_MATPLOTLIB:
            self.ax.clear()
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel("Loss")
            self.ax.set_title("Training Loss")
            self.canvas.draw()
        else:
            self.loss_text.config(state=tk.NORMAL)
            self.loss_text.delete("1.0", tk.END)
            self.loss_text.config(state=tk.DISABLED)

        def run():
            try:
                summary = self.engine.train(
                    text, params,
                    progress_callback=self.update_progress,
                    stop_event=self.stop_event,
                    finish_event=self.finish_event,
                )
                summary = summary if isinstance(summary, dict) else {}
                total_time = (time.time() - self.train_start_time
                              if self.train_start_time else 0.0)
                total_str = _fmt_duration(total_time)
                if summary.get("stop_reason") == "early_stopping":
                    kept = ("restored" if summary.get("restored_best")
                            else "of the last epoch kept")
                    msg = (f"Status: Stopped early after epoch "
                           f"{summary.get('epochs_completed')} — weights {kept}, best "
                           f"val loss {summary.get('best_val_loss', 0.0):.4f} at epoch "
                           f"{summary.get('best_epoch')} ✅  |  Total: {total_str}")
                    print(f"[AuraLite] Training stopped early after {total_str}.")
                elif summary.get("stop_reason") in ("finished", "time_budget"):
                    why = ("time limit reached" if summary["stop_reason"] == "time_budget"
                           else "finished early")
                    msg = (f"Status: Training {why} after a learning-rate cooldown ✅  |  "
                           f"Total: {total_str}")
                    print(f"[AuraLite] Training {why} after {total_str}.")
                elif self.stop_event.is_set():
                    msg = (f"Status: Stopped. 🛑 Weights preserved — "
                           f"you can generate or save.  |  Total: {total_str}")
                    print(f"[AuraLite] Training stopped after {total_str}.")
                else:
                    msg = f"Status: Training complete! ✅  |  Total: {total_str}"
                    print(f"[AuraLite] Training finished in {total_str}.")
                self.root.after(0, lambda m=msg: self.status_label.config(text=m))
                # Whether finished or stopped mid-way, the model holds
                # learned weights — enable generation and saving.
                if self.engine.model is not None:
                    self.is_trained = True
                    self.root.after(0, lambda: self.gen_btn.config(
                        state=tk.NORMAL))
                    self.root.after(0, lambda: self.save_btn.config(
                        state=tk.NORMAL))
                    n = self.engine.model.count_parameters()
                    self.root.after(0, lambda c=n: self.param_label.config(
                        text=f"Parameters: {c:,}"))
                    self.root.after(0, self._refresh_model_info)
                    if gui_common.HAS_QUANTIZATION:
                        self.root.after(0, self._update_quant_buttons)
            except ParamValidationError as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror(
                    "Validation Error", err))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror(
                    "Train Error", f"Error during training:\n{err}"))
            finally:
                self.root.after(0, self._reset_train_buttons)

        threading.Thread(target=run, daemon=True).start()


    def _reset_train_buttons(self):
        self.train_btn.config(state=tk.NORMAL)
        self.file_btn.config(state=tk.NORMAL)
        self.load_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED, text="🛑 Stop")


    def _append_loss_line(self, line):
        if not gui_common.HAS_MATPLOTLIB:
            self.loss_text.config(state=tk.NORMAL)
            self.loss_text.insert(tk.END, line + "\n")
            self.loss_text.see(tk.END)
            self.loss_text.config(state=tk.DISABLED)


    def _update_loss_plot(self):
        """Update matplotlib loss plot with new data."""
        if not gui_common.HAS_MATPLOTLIB or not self.loss_history:
            return
        epochs       = [x[0] for x in self.loss_history]
        train_losses = [x[1] for x in self.loss_history]
        val_pairs    = [(x[0], x[2]) for x in self.loss_history if x[2] is not None]
        val_epochs   = [p[0] for p in val_pairs]
        val_losses   = [p[1] for p in val_pairs]

        self.ax.clear()
        self.ax.plot(epochs, train_losses, 'b-o', markersize=3,
                     label="Train Loss", linewidth=1.2)
        if val_losses:
            # A single val point would be invisible as a line — force markers.
            self.ax.plot(val_epochs, val_losses, 'r-s', markersize=5,
                         label="Val Loss", linewidth=1.2)
            title = "Training & Validation Loss"
        else:
            title = "Training Loss (validation disabled — text too short or val_split=0)"
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title(title, fontsize=10)
        self.ax.legend(loc="best")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()


    def _update_temp_display(self, *_):
        try:
            self.temp_display.config(text=f"{self.temp_var.get():.2f}")
        except tk.TclError:
            pass


    def _update_len_display(self, val):
        try:
            self.len_display.config(text=str(int(float(val))))
        except (ValueError, tk.TclError):
            pass

