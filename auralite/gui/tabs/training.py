"""gui/tabs/training.py — TrainTabMixin for the AuraLite GUI (v2.6.5).

Extracted verbatim from `gui_app.py`; behaviour is unchanged.
"""
from .. import common as gui_common
from ..common import (CONFIG_PRESETS, DEFAULT_RECIPE, cluster_lan_address, run_cluster_worker, Figure, FigureCanvasTkAgg, ParamValidationError, _fmt_duration, describe_tokenizer, estimate_n_params, filedialog, messagebox, os, parse_dropout, read_tokenizer_file, recommend_epochs, recommend_gen_length, threading, time, tk, ttk, validate_params)
from ..common import SFT_TEMPLATES, DialogSet, data_kind, dialog_plain_text, load_training_data
from ..common import EngineTeacher, PreferenceSet, distill_dialogs, load_teacher, read_prompts

R = DEFAULT_RECIPE      # 2.12: one recipe for the GUI, the CLI and train()

#: 2.14.1: tokenizer kinds the "Save the model's tokenizer" button can write.
_SAVEABLE_TOKENIZERS = ("bpe", "char", "hf_bpe")
_NO_TOKENIZER_FILE = "Own tokenizer: none — BPE is trained with the model"


def _stage_text(info: dict) -> str:
    """The line under the progress bar for a stage (tokenizer, tokenizing, epoch, validation)."""
    if info.get("phase") != "train" and info.get("message"):
        return str(info["message"])
    parts = [str(info.get("stage_label") or info.get("stage") or "")]
    try:
        done, total = int(info.get("stage_done") or 0), int(info.get("stage_total") or 0)
        unit = str(info.get("stage_unit") or "")
        if total:
            parts.append(f"{done:,}/{total:,} {unit} ({100.0 * done / total:.0f}%)")
        rate = info.get("stage_rate")
        if rate:
            parts.append(f"{float(rate):,.1f} {unit}/s")
        eta = info.get("stage_eta_seconds")
        if eta is not None and done < total:
            parts.append(f"~{_fmt_duration(float(eta))} left")
    except (TypeError, ValueError):
        pass
    return " · ".join(p for p in parts if p)


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
        # 2.14.1: a tokenizer file (your own, or one trained with the button below)
        ttk.Radiobutton(row1, text="Own file", value="file",
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
        # 2.16: "auto" = the recipe's patience on text, off on dialogs.
        ttk.Label(row1, text="Early stop:").pack(side=tk.LEFT, padx=(12, 4))
        self.early_stop_var = tk.StringVar(value="auto")
        ttk.Entry(row1, textvariable=self.early_stop_var,
                  width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(row1, text=f"(epochs, 0=off; auto: {R['early_stopping_patience']}, "
                             f"dialogs off)", style="Sub.TLabel").pack(side=tk.LEFT, padx=2)

        # 2.12: a time limit; the learning-rate decay is sized to end at it.
        ttk.Label(row1, text="Time limit:").pack(side=tk.LEFT, padx=(12, 4))
        self.time_limit_var = tk.StringVar(value="0")
        ttk.Entry(row1, textvariable=self.time_limit_var,
                  width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(row1, text="(min, 0=off)", style="Sub.TLabel").pack(
            side=tk.LEFT, padx=2)

        # 2.14.1: plug in your own tokenizer, or train only the tokenizer
        row_tok = ttk.Frame(tok_frame)
        row_tok.pack(fill=tk.X, pady=2)
        self.tokenizer_path_var = tk.StringVar(value="")
        self._tokenizer_vocab = 0
        self.tok_load_btn = ttk.Button(row_tok, text="📂 Load tokenizer…",
                                       command=self.load_tokenizer_file)
        self.tok_load_btn.pack(side=tk.LEFT, padx=4)
        self.tok_train_btn = ttk.Button(row_tok, text="🔤 Train tokenizer only",
                                        command=self.train_tokenizer_only)
        self.tok_train_btn.pack(side=tk.LEFT, padx=4)
        self.tok_save_btn = ttk.Button(row_tok, text="💾 Save model's tokenizer…",
                                       command=self.save_tokenizer_file)
        self.tok_save_btn.pack(side=tk.LEFT, padx=4)
        self.tokenizer_label = ttk.Label(row_tok, text=_NO_TOKENIZER_FILE, style="Sub.TLabel")
        self.tokenizer_label.pack(side=tk.LEFT, padx=8)

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
        # 2.16: the layers with adapters, and LoRA's learning rate
        self.lora_targets_var = tk.StringVar(value="all")
        ttk.Combobox(row1b, textvariable=self.lora_targets_var, values=["all", "ffn", "attn"],
                     state="readonly", width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(row1b, text="LoRA LR:").pack(side=tk.LEFT, padx=(6, 2))
        self.lora_lr_var = tk.StringVar(value="auto")
        ttk.Entry(row1b, textvariable=self.lora_lr_var,
                  width=7).pack(side=tk.LEFT, padx=2)

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

        self.file_btn = ttk.Button(top_row, text="📂 Select File",
                                   command=self.select_file)
        self.file_btn.pack(side=tk.LEFT, padx=4)
        # 2.15: a folder of documents (its .txt / .md / .jsonl files)
        self.folder_btn = ttk.Button(top_row, text="📁 Select Folder",
                                     command=self.select_folder)
        self.folder_btn.pack(side=tk.LEFT, padx=4)

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

        data_row = ttk.Frame(run_frame)
        data_row.pack(fill=tk.X, pady=2)
        self.file_label = ttk.Label(data_row, text="No file selected",
                                    foreground="gray")
        self.file_label.pack(side=tk.LEFT, anchor=tk.W)
        # 2.15: a .jsonl of dialogs fine-tunes a chat model in this template
        self.dialog_template_var = tk.StringVar(value="chatml")
        ttk.Combobox(data_row, textvariable=self.dialog_template_var,
                     values=list(SFT_TEMPLATES), state="readonly",
                     width=8).pack(side=tk.RIGHT, padx=4)
        ttk.Label(data_row, text="Chat template (dialogs):").pack(side=tk.RIGHT, padx=4)
        # 2.16: the prompt's weight in the loss, and whole dialogs per window
        self.dialog_pack_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(data_row, text="Pack whole dialogs",
                        variable=self.dialog_pack_var).pack(side=tk.RIGHT, padx=8)
        self.prompt_weight_var = tk.StringVar(value="auto")
        ttk.Entry(data_row, textvariable=self.prompt_weight_var,
                  width=5).pack(side=tk.RIGHT, padx=2)
        ttk.Label(data_row, text="Prompt weight:").pack(side=tk.RIGHT, padx=(8, 2))
        # 2.16: a .jsonl of preference pairs (prompt, chosen, rejected) trains the
        # chat model in memory with DPO or SimPO ("sft": on the chosen replies)
        pref_row = ttk.Frame(run_frame)
        pref_row.pack(fill=tk.X, pady=2)
        ttk.Label(pref_row, text="Preference pairs:").pack(side=tk.LEFT, padx=4)
        self.pref_method_var = tk.StringVar(value="dpo")
        ttk.Combobox(pref_row, textvariable=self.pref_method_var,
                     values=["dpo", "simpo", "sft"], state="readonly",
                     width=6).pack(side=tk.LEFT, padx=2)
        self.pref_beta_var = tk.StringVar(value="auto")
        self.pref_lr_var = tk.StringVar(value="auto")
        self.pref_epochs_var = tk.StringVar(value="auto")
        for label, var in (("Beta:", self.pref_beta_var), ("LR:", self.pref_lr_var),
                           ("Epochs:", self.pref_epochs_var)):
            ttk.Label(pref_row, text=label).pack(side=tk.LEFT, padx=(8, 2))
            ttk.Entry(pref_row, textvariable=var, width=7).pack(side=tk.LEFT, padx=2)
        ttk.Label(pref_row, style="Sub.TLabel",
                  text="(auto: DPO beta 0.1, LR 3e-4; SimPO beta 2.0, LR 1e-4; 3 epochs; "
                       "trains the chat model in memory)").pack(side=tk.LEFT, padx=6)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(run_frame,
                                            variable=self.progress_var,
                                            maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=4)

        # Live detail line under the bar (batch / phase / speed / ETA). 2.14.2: the
        # one bar also shows the stages before training — training the tokenizer,
        # tokenizing the corpus — with done/total, speed and time left on this line
        # (2.14.1 had a second bar for them).
        self.progress_detail_var = tk.StringVar(value="")
        self.progress_detail_label = ttk.Label(
            run_frame, textvariable=self.progress_detail_var, style="Sub.TLabel")
        self.progress_detail_label.pack(fill=tk.X, pady=(0, 2), anchor=tk.W)
        self._train_phase = ""
        self._bar_busy = False

        self.status_label = ttk.Label(run_frame,
                                      text="Status: Waiting for file…")
        self.status_label.pack(pady=(0, 2), anchor=tk.W)

        # ---- Dialogs from a teacher model (2.16) --------------------------
        self._build_distill_frame(tab)

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
            text_sample = self._selected_text()       # 2.15: files, folders, dialogs
        except Exception as e:
            messagebox.showerror("File Error", f"Could not read file:\n{e}")
            return

        n_chars = len(text_sample)
        if tok_kind == "file" and getattr(self, "_tokenizer_vocab", 0):
            vocab = int(self._tokenizer_vocab)                 # 2.14.1: the loaded file's
            n_tokens = max(seq_length + 2, int(n_chars / 3.0))
        elif tok_kind in ("bpe", "file"):
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
            filetypes=[("Text, dialogs or pairs", "*.txt *.md *.jsonl *.json"),
                       ("Text files", "*.txt"), ("Dialogs or preference pairs (JSONL)", "*.jsonl"),
                       ("All files", "*.*")],
        )
        if file_path:
            self._select_data(file_path)

    def select_folder(self):
        """2.15: train on every text file of a folder (one document each)."""
        folder = filedialog.askdirectory(title="Select a Folder of Training Files")
        if folder:
            self._select_data(folder)

    def _select_data(self, path: str) -> None:
        self.selected_file_path = path
        name = os.path.basename(os.path.normpath(path)) or path
        kind = data_kind([path])
        if os.path.isdir(path):
            text = f"Selected folder: {name}"
        else:
            text = f"Selected: {name}"
        if kind == "dialogs":
            text += " — dialogs: fine-tunes a chat model (check «Continue training» to " \
                    "fine-tune the model in memory)"
        elif kind == "preferences":
            text += " — preference pairs: trains the chat model in memory (see «Preference " \
                    "pairs»)"
        self.file_label.config(text=text, foreground="black")
        self.train_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Ready to train")

    def early_stop_param(self) -> int | str:
        """The «Early stop» field: a patience in epochs, or "auto" (2.16)."""
        text = str(self.early_stop_var.get() or "").strip().lower()
        return "auto" if text in ("", "auto") else int(text)

    def prompt_weight_param(self) -> float | str:
        """The «Prompt weight» field: a weight from 0 to 1, or "auto" (2.16:
        0.1, or 0 with LoRA)."""
        text = str(self.prompt_weight_var.get() or "").strip().lower()
        return "auto" if text in ("", "auto") else float(text)

    def preference_params(self) -> dict:
        """The «Preference pairs» row (2.16): the method, and beta, the learning
        rate and the epochs unless "auto" (preference.py's defaults)."""
        params: dict = {"method": self.pref_method_var.get() or "dpo"}
        for var, key, kind in ((self.pref_beta_var, "preference_beta", float),
                               (self.pref_lr_var, "lr", float),
                               (self.pref_epochs_var, "epochs", int)):
            text = str(var.get() or "").strip().lower()
            if text not in ("", "auto"):
                params[key] = kind(text)
        return params

    def lora_params(self) -> dict:
        """LoRA's layers and learning rate from the form (2.16; "auto": 10 x LR)."""
        params: dict = {"lora_targets": self.lora_targets_var.get() or "all"}
        text = str(self.lora_lr_var.get() or "").strip().lower()
        if text not in ("", "auto"):
            params["lora_lr"] = float(text)
        return params

    def _selected_text(self) -> str:
        """The text of the selected file or folder (dialogs: without the markers), 2.15."""
        data = load_training_data([self.selected_file_path])
        if isinstance(data, DialogSet):
            template = self.dialog_template_var.get() or "chatml"
            return "\n".join(dialog_plain_text(d, template) for d in data.dialogs)
        return data.text


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
        if getattr(self, "_train_phase", ""):
            self._train_phase = phase          # 2.14.1: Stop acts at once before training
        if current <= 0 and phase in ("setup", "cluster"):
            # 2.14.2: before the first step the one bar follows the current stage
            self._apply_stage(info)
            try:
                elapsed = float(info.get("elapsed") or 0.0)
            except (TypeError, ValueError):
                elapsed = 0.0
            if not elapsed and self.train_start_time:
                elapsed = time.time() - self.train_start_time
            head = info.get("message") or "Preparing…"
            self.status_label.config(
                text=f"Status: {head}  |  Elapsed: {_fmt_duration(elapsed)}")
            return

        # Prefer engine-provided overall percent (includes intra-epoch fraction).
        if "percent" in info and info["percent"] is not None:
            try:
                percent = float(info["percent"])
            except (TypeError, ValueError):
                percent = (current / max(1, total)) * 100.0
        else:
            percent = (current / max(1, total)) * 100.0
        percent = max(0.0, min(100.0, percent))
        self._set_progress_bar(percent)

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

        if phase == "val":
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
        msg = _stage_text(info) if phase == "val" and info.get("stage") else info.get("message")
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


    def _set_progress_bar(self, percent) -> None:
        """2.14.2: the one progress bar — a percentage, or (``None``) a busy
        animation for a step without a count (building the model, compiling)."""
        bar = getattr(self, "progress_bar", None)
        if percent is None:
            if not getattr(self, "_bar_busy", False):
                self._bar_busy = True
                try:
                    if bar is not None:
                        bar.config(mode="indeterminate")
                        bar.start(15)
                except Exception:
                    pass
            return
        if getattr(self, "_bar_busy", False):
            self._bar_busy = False
            try:
                if bar is not None:
                    bar.stop()
                    bar.config(mode="determinate")
            except Exception:
                pass
        try:
            self.progress_var.set(max(0.0, min(100.0, float(percent))))
        except (TypeError, ValueError):
            pass

    def _apply_stage(self, info) -> None:
        """A stage on the one bar (2.14.2; 2.14.1 drew a second bar): its percent
        (a busy animation without a count) and its line — done/total, speed,
        time left — under the bar."""
        if not isinstance(info, dict):
            return
        percent = None
        try:
            percent = info.get("stage_percent")
            if percent is None:
                done = float(info.get("stage_done") or 0)
                total = float(info.get("stage_total") or 0)
                percent = 100.0 * done / total if total else None
        except (TypeError, ValueError):
            percent = None
        self._set_progress_bar(percent)
        try:
            text = _stage_text(info) if info.get("stage") else str(info.get("message") or "")
            self.progress_detail_var.set(text or "Preparing…")
        except Exception:
            pass

    def stop_training(self):
        # 2.14.1: before the first training step (tokenizer, tokenizing) and
        # while only the tokenizer trains there is nothing to cool down.
        if getattr(self, "_train_phase", "") in ("setup", "cluster", "tokenizer"):
            self.finish_event.set()
            self.stop_event.set()
            self.status_label.config(text="Status: Stopping… 🛑")
            return
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

        # 2.14.1: "Own file" trains with the loaded / trained tokenizer file
        tokenizer_path = ""
        if self.tok_var.get() == "file":
            tokenizer_path = str(self.tokenizer_path_var.get() or "").strip()
            if not tokenizer_path:
                messagebox.showinfo(
                    "No tokenizer file",
                    "«Own file» is selected: load a tokenizer (📂 Load tokenizer…) or "
                    "train one (🔤 Train tokenizer only) first.")
                return
            if not os.path.exists(tokenizer_path):
                messagebox.showerror("Tokenizer file",
                                     f"The tokenizer file is missing:\n{tokenizer_path}")
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
                "tokenizer":       "bpe" if tokenizer_path else self.tok_var.get(),
                "bpe_vocab_size":  int(self.bpe_vocab_var.get()),
                "val_split":       float(self.val_split_var.get()),
                "early_stopping_patience": self.early_stop_param(),
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
                **self.lora_params(),
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
        if tokenizer_path:
            params["tokenizer_path"] = tokenizer_path

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
            base = os.path.splitext(os.path.normpath(self.selected_file_path))[0]
            params["autosave_path"] = base + "_autosave.pt"

        try:
            # 2.15: a file, a folder of documents, or dialogs (.jsonl)
            data = load_training_data([self.selected_file_path])
        except Exception as e:
            messagebox.showerror("File Error", f"Could not read file:\n{e}")
            return
        if isinstance(data, PreferenceSet):
            # 2.16: DPO or SimPO on the chat model in memory, with the row's own
            # learning rate and epochs (the form's are for training a model)
            try:
                preference = self.preference_params()
            except ValueError:
                messagebox.showerror("Params Error", "Beta and LR must be numbers and Epochs a "
                                                     "whole number (or auto).")
                return
            if preference["method"] == "sft":
                data = data.as_dialogs()
            else:
                if self.engine.model is None or self.engine.is_gguf_model() \
                        or self.engine.is_hf_model():
                    messagebox.showwarning(
                        "No chat model",
                        "Preference pairs train the AuraLite chat model in memory: load one "
                        "(or fine-tune one on dialogs) first.")
                    return
                for key in ("lr", "epochs"):
                    params.pop(key, None)
                params.update(preference, continue_training=True)
                errors = validate_params(params)
                if errors:
                    messagebox.showerror("Validation Error", "\n".join(f"• {e}" for e in errors))
                    return
        if isinstance(data, DialogSet):
            params["chat_template"] = self.dialog_template_var.get() or "chatml"
            # 2.16: the prompt's weight in the loss, whole dialogs per window
            try:
                params["prompt_loss_weight"] = self.prompt_weight_param()
            except ValueError:
                messagebox.showerror("Params Error", "Prompt weight must be a number from 0 "
                                                     "to 1.")
                return
            params["dialog_packing"] = bool(self.dialog_pack_var.get())
            errors = validate_params(params)
            if errors:
                messagebox.showerror("Validation Error", "\n".join(f"• {e}" for e in errors))
                return
            if params.get("cluster_workers"):
                messagebox.showwarning(
                    "Dialogs train on one computer",
                    "Chat fine-tuning on dialogs runs on this computer only: set the "
                    "number of other computers to 0.")
                return

        self.stop_event.clear()
        self.finish_event.clear()
        self.stop_btn.config(text="🛑 Stop")
        self.loss_history = []
        # Reset ETA tracking
        self.train_start_time = time.time()
        self.epoch_times = []
        self._last_epoch_ts = self.train_start_time
        self._set_progress_bar(0.0)
        self._progress_ui_last_ts = 0.0
        self._progress_ui_pending = None
        try:
            self.progress_detail_var.set("Preparing for training…")
        except Exception:
            pass
        self._train_phase = "setup"
        self.status_label.config(text="Status: Starting… ⏳")
        self._set_tokenizer_buttons(tk.DISABLED)
        self.train_btn.config(state=tk.DISABLED)
        self.file_btn.config(state=tk.DISABLED)
        self.folder_btn.config(state=tk.DISABLED)
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
                    data, params,
                    progress_callback=self.update_progress,
                    stop_event=self.stop_event,
                    finish_event=self.finish_event,
                )
                summary = summary if isinstance(summary, dict) else {}
                total_time = (time.time() - self.train_start_time
                              if self.train_start_time else 0.0)
                total_str = _fmt_duration(total_time)
                if summary.get("stopped_during"):
                    # 2.14.1: stopped while the tokenizer trained / the corpus was
                    # tokenized: train() put the engine back as it was.
                    msg = (f"Status: Stopped before training began "
                           f"({summary['stopped_during']}) — the model in memory is "
                           f"unchanged 🛑  |  Total: {total_str}")
                    print(f"[AuraLite] Training stopped before it began ({total_str}).")
                elif summary.get("stop_reason") == "early_stopping":
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


    # ---- dialogs from a teacher model (2.16) ----------------------------
    def _build_distill_frame(self, tab) -> None:
        """«Generate dialogs from a model»: a teacher answers a prompts file (auralite distill)."""
        frame = ttk.LabelFrame(tab, text="  🎓  Generate dialogs from a model  ", padding="8")
        frame.pack(fill=tk.X, pady=(0, 8))
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Teacher:").pack(side=tk.LEFT, padx=4)
        self.distill_teacher_var = tk.StringVar(value="")
        ttk.Entry(row, textvariable=self.distill_teacher_var, width=36).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="📂", width=3, command=self.select_teacher).pack(side=tk.LEFT)
        self.distill_memory_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="the model in memory",
                        variable=self.distill_memory_var).pack(side=tk.LEFT, padx=6)
        ttk.Label(row, text="Prompts:").pack(side=tk.LEFT, padx=(10, 4))
        self.distill_prompts_var = tk.StringVar(value="")
        ttk.Entry(row, textvariable=self.distill_prompts_var, width=30).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="📂", width=3, command=self.select_prompts).pack(side=tk.LEFT)
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        self.distill_samples_var = tk.StringVar(value="1")
        self.distill_max_tokens_var = tk.StringVar(value="256")
        self.distill_temperature_var = tk.StringVar(value="auto")
        for label, var, width in (("Samples:", self.distill_samples_var, 4),
                                  ("Max tokens:", self.distill_max_tokens_var, 6),
                                  ("Temperature:", self.distill_temperature_var, 6)):
            ttk.Label(row, text=label).pack(side=tk.LEFT, padx=(8, 2))
            ttk.Entry(row, textvariable=var, width=width).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="Judge:").pack(side=tk.LEFT, padx=(8, 2))
        self.distill_judge_var = tk.StringVar(value="none")
        ttk.Combobox(row, textvariable=self.distill_judge_var,
                     values=["none", "exact", "contains", "number", "regex:", "teacher"],
                     width=14).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="System:").pack(side=tk.LEFT, padx=(8, 2))
        self.distill_system_var = tk.StringVar(value="")
        ttk.Entry(row, textvariable=self.distill_system_var, width=30).pack(side=tk.LEFT, padx=2)
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        self.distill_btn = ttk.Button(row, text="🧪 Generate dialogs…", command=self.start_distill)
        self.distill_btn.pack(side=tk.LEFT, padx=4)
        self.distill_stop_btn = ttk.Button(row, text="🛑 Stop", command=self.stop_distill,
                                           state=tk.DISABLED)
        self.distill_stop_btn.pack(side=tk.LEFT, padx=4)
        self.distill_status = ttk.Label(row, style="Sub.TLabel",
                                        text="A judge keeps only right replies and also writes "
                                             "preference pairs; a stopped run resumes.")
        self.distill_status.pack(side=tk.LEFT, padx=8)
        self._distill_stop = threading.Event()
        self.last_distill: dict | None = None

    def select_teacher(self):
        path = filedialog.askopenfilename(
            title="Select a Teacher Model",
            filetypes=[("Models", "*.gguf *.pt *.safetensors"), ("All files", "*.*")])
        if path:
            self.distill_teacher_var.set(path)
            self.distill_memory_var.set(False)

    def select_prompts(self):
        path = filedialog.askopenfilename(
            title="Select a Prompts File",
            filetypes=[("Prompts", "*.txt *.jsonl *.json"), ("All files", "*.*")])
        if path:
            self.distill_prompts_var.set(path)

    def distill_settings(self) -> dict:
        """The form of «Generate dialogs from a model» as distill_dialogs() arguments."""
        temperature = str(self.distill_temperature_var.get() or "").strip().lower()
        judge = str(self.distill_judge_var.get() or "").strip()
        return {"samples": int(str(self.distill_samples_var.get() or 1).strip()),
                "max_tokens": int(str(self.distill_max_tokens_var.get() or 256).strip()),
                "temperature": None if temperature in ("", "auto") else float(temperature),
                "judge": None if judge.lower() in ("", "none") else judge,
                "system": str(self.distill_system_var.get() or "").strip() or None}

    def start_distill(self):
        prompts_path = str(self.distill_prompts_var.get() or "").strip()
        teacher_path = str(self.distill_teacher_var.get() or "").strip()
        in_memory = bool(self.distill_memory_var.get())
        if not prompts_path or not os.path.isfile(prompts_path):
            messagebox.showinfo("Prompts", "Choose a prompts file: one question per line "
                                           "(.txt), or a .jsonl of prompts or dialogs.")
            return
        if in_memory and self.engine.model is None:
            messagebox.showinfo("Teacher", "There is no model in memory to be the teacher.")
            return
        if not in_memory and not teacher_path:
            messagebox.showinfo("Teacher", "Choose a teacher model (.gguf, .pt, a Hugging Face "
                                           "folder or an http:// llama.cpp server), or use the "
                                           "model in memory.")
            return
        try:
            settings = self.distill_settings()
            prompts, _skipped = read_prompts(prompts_path)
        except (OSError, ValueError) as err:
            messagebox.showerror("Generate dialogs", str(err))
            return
        if not prompts:
            messagebox.showerror("Generate dialogs", f"No prompts in {prompts_path}.")
            return
        stem = os.path.splitext(prompts_path)[0]
        out = filedialog.asksaveasfilename(title="Save the Dialogs As",
                                           initialfile=os.path.basename(stem) + "_dialogs.jsonl",
                                           defaultextension=".jsonl",
                                           filetypes=[("Dialogs (JSONL)", "*.jsonl")])
        if not out:
            return
        pairs = os.path.splitext(out)[0] + "_pairs.jsonl" if settings["judge"] else None
        self._distill_stop.clear()
        self.distill_btn.config(state=tk.DISABLED)
        self.distill_stop_btn.config(state=tk.NORMAL)
        if in_memory:
            self.train_btn.config(state=tk.DISABLED)      # the model in memory is busy
        self.distill_status.config(text="Loading the teacher…")

        def progress(info: dict) -> None:
            eta = info.get("eta_seconds")
            text = (f"{info['done']:,}/{info['total']:,} prompts | {info['kept']:,} dialogs"
                    + (f" | {info['pairs']:,} pairs" if pairs else "")
                    + (f" | {_fmt_duration(float(eta))} left" if eta else ""))
            self.root.after(0, lambda t=text: self.distill_status.config(text=t))

        def run():
            try:
                teacher = EngineTeacher(self.engine, name="model in memory") if in_memory \
                    else load_teacher(teacher_path)
                meta = distill_dialogs(teacher, prompts, out, pairs_path=pairs,
                                       prompts_file=prompts_path, on_progress=progress,
                                       stop_event=self._distill_stop, **settings)
                self.last_distill = meta
                done = meta["progress"]
                text = (f"Saved {meta['files']['dialogs']['records']:,} dialogs to "
                        f"{os.path.basename(out)}"
                        + (f" and {meta['files']['pairs']['records']:,} pairs to "
                           f"{os.path.basename(pairs)}" if pairs else ""))
                if meta.get("stopped"):
                    text += (f" — stopped at {done['done']:,}/{done['total']:,}; Generate again "
                             f"with the same file to continue")
                self.root.after(0, lambda t=text: self.distill_status.config(text=t))
                if meta["files"]["dialogs"]["records"]:
                    self.root.after(0, lambda: self._select_data(out))
            except Exception as err:
                message = str(err)
                self.root.after(0, lambda m=message: messagebox.showerror("Generate dialogs", m))
                self.root.after(0, lambda: self.distill_status.config(text="Failed"))
            finally:
                self.root.after(0, lambda: self.distill_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.distill_stop_btn.config(state=tk.DISABLED))
                if in_memory and self.selected_file_path:
                    self.root.after(0, lambda: self.train_btn.config(state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()

    def stop_distill(self):
        """Stop after the current prompt; the same output file resumes later."""
        self._distill_stop.set()
        self.distill_status.config(text="Stopping after this prompt…")

    def _reset_train_buttons(self):
        self._train_phase = ""
        if getattr(self, "_bar_busy", False):        # a run that ended during the setup
            self._set_progress_bar(0.0)
        self.train_btn.config(state=tk.NORMAL)
        self.file_btn.config(state=tk.NORMAL)
        self.folder_btn.config(state=tk.NORMAL)
        self.load_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED, text="🛑 Stop")
        self._set_tokenizer_buttons(tk.NORMAL)

    # ---- Own tokenizer / training only the tokenizer (2.14.1) -------------
    def _set_tokenizer_buttons(self, state) -> None:
        for name in ("tok_load_btn", "tok_train_btn", "tok_save_btn"):
            button = getattr(self, name, None)
            if button is not None:
                button.config(state=state)

    def _use_tokenizer_file(self, path: str, tokenizer) -> None:
        """Select "Own file" with this tokenizer for the next training."""
        self.tokenizer_path_var.set(path)
        self.tok_var.set("file")
        self._tokenizer_vocab = int(getattr(tokenizer, "vocab_size", 0) or 0)
        self.tokenizer_label.config(
            text=f"Own tokenizer: {os.path.basename(path)} — {describe_tokenizer(tokenizer)}")

    def load_tokenizer_file(self):
        """Plug in a tokenizer: AuraLite file, tokenizer.json, vocab.json, checkpoint."""
        path = filedialog.askopenfilename(
            title="Load a tokenizer",
            filetypes=[("Tokenizers (JSON)", "*.json"),
                       ("AuraLite checkpoints", "*.pt *.safetensors"),
                       ("GPT-2 merges.txt", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            tokenizer, info = read_tokenizer_file(path)
        except Exception as e:
            messagebox.showerror("Tokenizer Error", f"Could not load the tokenizer:\n{e}")
            return
        self._use_tokenizer_file(path, tokenizer)
        self.status_label.config(
            text=f"Status: Tokenizer loaded ✅ ({describe_tokenizer(tokenizer)}; "
                 f"read as {info.get('format')}) — the next training uses it")

    def save_tokenizer_file(self):
        """Save the tokenizer of the model in memory (to train other models with it)."""
        tokenizer = self.engine.tokenizer
        if tokenizer is None or getattr(tokenizer, "kind", None) not in _SAVEABLE_TOKENIZERS:
            messagebox.showinfo("No tokenizer",
                                "Train or load a native AuraLite model first — its tokenizer "
                                "is what gets saved.")
            return
        path = filedialog.asksaveasfilename(
            title="Save the model's tokenizer", defaultextension=".json",
            initialfile="tokenizer.json",
            filetypes=[("Tokenizer (JSON)", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.engine.save_tokenizer(path)
        except Exception as e:
            messagebox.showerror("Tokenizer Error", f"Could not save the tokenizer:\n{e}")
            return
        self.status_label.config(text=f"Status: Tokenizer saved ✅ ({os.path.basename(path)}: "
                                      f"{describe_tokenizer(tokenizer)})")

    def train_tokenizer_only(self):
        """Train only the tokenizer on the selected text, save it and select it."""
        if not self.selected_file_path:
            messagebox.showinfo("No file", "Select a training file or folder first "
                                           "(📂 Select File / 📁 Select Folder).")
            return
        kind = "char" if self.tok_var.get() == "char" else "bpe"
        try:
            vocab = int(self.bpe_vocab_var.get())
            val_split = float(self.val_split_var.get())
        except ValueError:
            messagebox.showerror("Params Error", "BPE Vocab and Val Split must be numbers.")
            return
        stem = os.path.splitext(os.path.basename(os.path.normpath(self.selected_file_path)))[0]
        size = f"{vocab}" if kind == "bpe" else ""
        path = filedialog.asksaveasfilename(
            title="Save the tokenizer as", defaultextension=".json",
            initialfile=f"{stem}_{kind}{size}_tokenizer.json",
            filetypes=[("Tokenizer (JSON)", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            # 2.15: a corpus is cut like train() cuts it (held-out documents)
            data = load_training_data([self.selected_file_path])
            text = data if not isinstance(data, DialogSet) else self._selected_text()
        except Exception as e:
            messagebox.showerror("File Error", f"Could not read file:\n{e}")
            return
        self.stop_event.clear()
        self.finish_event.clear()
        self._train_phase = "tokenizer"
        self._set_progress_bar(0.0)
        self.progress_detail_var.set("Training the tokenizer…")
        self.status_label.config(text="Status: Training the tokenizer… ⏳ (Stop keeps the "
                                      "merges learned so far)")
        self._set_tokenizer_buttons(tk.DISABLED)
        self.train_btn.config(state=tk.DISABLED)
        self.file_btn.config(state=tk.DISABLED)
        self.folder_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        started = time.time()

        def progress(info):
            self.root.after(0, self._apply_stage, dict(info))

        def run():
            try:
                tokenizer = self.engine.train_tokenizer(
                    text, vocab, kind=kind, val_split=val_split, progress_callback=progress,
                    stop_event=self.stop_event, path=path)
                info = dict(getattr(self.engine, "last_tokenizer_info", None) or {})
                self.root.after(0, self._tokenizer_trained, path, tokenizer, info,
                                time.time() - started)
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror(
                    "Tokenizer Error", f"Could not train the tokenizer:\n{err}"))
            finally:
                self.root.after(0, self._reset_train_buttons)

        threading.Thread(target=run, daemon=True).start()

    def _tokenizer_trained(self, path: str, tokenizer, info: dict, seconds: float) -> None:
        self._use_tokenizer_file(path, tokenizer)
        details = [describe_tokenizer(tokenizer)]
        if info.get("chars_per_token"):
            details.append(f"{float(info['chars_per_token']):.2f} chars/token")
        if info.get("stopped"):
            details.append("stopped early with the merges learned so far")
        elif info.get("exhausted"):
            details.append("the text had no more pairs to merge")
        self._set_progress_bar(100.0)
        self.status_label.config(
            text=f"Status: Tokenizer trained and saved ✅ ({os.path.basename(path)}: "
                 f"{'; '.join(details)})  |  {_fmt_duration(seconds)} — the next training "
                 f"uses it")


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

