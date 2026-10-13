"""gui/tabs/model.py — ModelTabMixin for the AuraLite GUI (v2.6.5).

Extracted verbatim from `gui_app.py`; behaviour is unchanged.
"""
from .. import common as gui_common
from ..common import (GGUFNotAvailableError, HFNotAvailableError, filedialog, messagebox, os, threading, tk, ttk)

class ModelTabMixin:
    def _build_model_tab(self):
        tab = self.tab_model

        io_frame = ttk.LabelFrame(tab, text="  💾  Save / Load  ", padding="12")
        io_frame.pack(fill=tk.X, pady=(0, 8))

        btn_row = ttk.Frame(io_frame)
        btn_row.pack(pady=4)

        self.save_btn = ttk.Button(btn_row, text="💾 Save Model",
                                   command=self.save_model, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=6)

        self.load_btn = ttk.Button(btn_row, text="📂 Load Model",
                                   command=self.load_model)
        self.load_btn.pack(side=tk.LEFT, padx=6)

        self.model_file_label = ttk.Label(io_frame, text="No model loaded",
                                          foreground="gray")
        self.model_file_label.pack(pady=2)

        # ==================================================================
        #  NEW: Hugging Face + LoRA / QLoRA section (any model)
        # ==================================================================
        hf_frame = ttk.LabelFrame(tab, text="  🤗  Hugging Face — Any Model + LoRA / QLoRA  ",
                                  padding="12")
        hf_frame.pack(fill=tk.X, pady=(12, 8))

        ttk.Label(hf_frame, text="Model name (HF Hub) or path to local folder:").pack(anchor=tk.W)
        self.hf_model_var = tk.StringVar(value="Qwen/Qwen2-0.5B-Instruct")
        hf_entry = ttk.Entry(hf_frame, textvariable=self.hf_model_var, font=("Segoe UI", 10))
        hf_entry.pack(fill=tk.X, pady=2)

        # NEW: Browse button for already downloaded local models
        browse_row = ttk.Frame(hf_frame)
        browse_row.pack(fill=tk.X, pady=(0, 4))

        ttk.Button(browse_row, text="📁 Browse local folder (already downloaded model)",
                   command=self._browse_local_hf_model).pack(side=tk.LEFT)

        self.hf_local_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(browse_row, text="Use local files only (offline, no internet)",
                        variable=self.hf_local_only_var).pack(side=tk.LEFT, padx=12)

        hf_opts = ttk.Frame(hf_frame)
        hf_opts.pack(fill=tk.X, pady=4)

        self.hf_4bit_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(hf_opts, text="Load in 4-bit (QLoRA — best for fine-tuning)",
                        variable=self.hf_4bit_var).pack(side=tk.LEFT)

        self.hf_8bit_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(hf_opts, text="8-bit", variable=self.hf_8bit_var).pack(side=tk.LEFT, padx=12)

        self.hf_apply_lora_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(hf_opts, text="Apply LoRA on load", variable=self.hf_apply_lora_var).pack(side=tk.LEFT)

        lora_row = ttk.Frame(hf_frame)
        lora_row.pack(fill=tk.X, pady=2)
        ttk.Label(lora_row, text="LoRA rank:").pack(side=tk.LEFT)
        self.hf_lora_rank_var = tk.StringVar(value="16")
        ttk.Entry(lora_row, textvariable=self.hf_lora_rank_var, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Label(lora_row, text="(8–64 typical; higher = more capacity)").pack(side=tk.LEFT, padx=6)

        hf_btn_row = ttk.Frame(hf_frame)
        hf_btn_row.pack(fill=tk.X, pady=6)

        self.hf_load_btn = ttk.Button(hf_btn_row, text="📥 Load HF Model",
                                      command=self._load_hf_model)
        self.hf_load_btn.pack(side=tk.LEFT, padx=4)

        self.hf_apply_lora_btn = ttk.Button(hf_btn_row, text="🔧 Apply LoRA (for fine-tune)",
                                            command=self._apply_lora_to_hf, state=tk.DISABLED)
        self.hf_apply_lora_btn.pack(side=tk.LEFT, padx=4)

        self.hf_save_lora_btn = ttk.Button(hf_btn_row, text="💾 Save LoRA Adapter",
                                           command=self._save_hf_lora, state=tk.DISABLED)
        self.hf_save_lora_btn.pack(side=tk.LEFT, padx=4)

        self.hf_load_lora_btn = ttk.Button(hf_btn_row, text="📂 Load LoRA Adapter",
                                           command=self._load_hf_lora, state=tk.DISABLED)
        self.hf_load_lora_btn.pack(side=tk.LEFT, padx=4)

        # Fine-tune button (works on loaded HF model)
        self.hf_finetune_btn = ttk.Button(hf_btn_row, text="🚀 Fine-tune (LoRA/QLoRA)",
                                          command=self._finetune_hf_from_gui, state=tk.DISABLED)
        self.hf_finetune_btn.pack(side=tk.LEFT, padx=4)

        # NEW: Hub push/pull buttons
        hub_row = ttk.Frame(hf_btn_row)
        hub_row.pack(fill=tk.X, pady=(8, 0))

        self.hf_push_btn = ttk.Button(hub_row, text="☁️ Push to Hub",
                                      command=self._push_hf_to_hub, state=tk.DISABLED)
        self.hf_push_btn.pack(side=tk.LEFT, padx=4)

        self.hf_pull_btn = ttk.Button(hub_row, text="📥 Load from Hub",
                                      command=self._load_hf_from_hub)
        self.hf_pull_btn.pack(side=tk.LEFT, padx=4)

        ttk.Label(hf_frame,
                  text="Tip: Use small models first (0.5B–3B). 4-bit + LoRA lets you fine-tune on consumer GPUs.",
                  style="Sub.TLabel", foreground="#666").pack(anchor=tk.W, pady=(4, 0))

        # --- Model info ---
        info_frame = ttk.LabelFrame(tab, text="  ℹ️  Model Info  ", padding="6")
        info_frame.pack(fill=tk.BOTH, expand=True)

        self.model_info = tk.Text(info_frame, font=("Consolas", 10),
                                  state=tk.DISABLED, wrap=tk.WORD)
        self.model_info.pack(fill=tk.BOTH, expand=True)


    def save_model(self):
        if self.engine.is_gguf_model():
            messagebox.showinfo(
                "GGUF",
                ".gguf is an external inference-only model file. It cannot be "
                "saved as an AuraLite .pt checkpoint; keep/copy the original .gguf file."
            )
            return
        if self.engine.is_hf_model():
            messagebox.showinfo(
                "Hugging Face",
                "Hugging Face models are managed separately. Save adapters via 'Save LoRA Adapter' "
                "or use the Hub / save_pretrained workflow instead of AuraLite .pt saving."
            )
            return
        path = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pt",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.engine.save_model(path)
            self.model_file_label.config(
                text=f"Saved: {os.path.basename(path)}", foreground="black")
            messagebox.showinfo("Saved",
                                f"Model saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))


    def _ask_gguf_options(self, path: str) -> dict | None:
        """Modal dialog for llama.cpp / GGUF loading options."""
        dlg = tk.Toplevel(self.root)
        dlg.title("GGUF / llama.cpp options")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)

        outer = ttk.Frame(dlg, padding="12")
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(outer, text=f"Model: {os.path.basename(path)}",
                  font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))

        n_ctx_var = tk.StringVar(value=os.environ.get("AURALITE_GGUF_N_CTX", "4096"))
        gpu_var = tk.StringVar(value=os.environ.get("AURALITE_GGUF_N_GPU_LAYERS", "-1"))
        threads_var = tk.StringVar(value=os.environ.get("AURALITE_GGUF_N_THREADS", str(self.engine.num_threads)))
        batch_var = tk.StringVar(value=os.environ.get("AURALITE_GGUF_N_BATCH", "512"))
        chat_format_var = tk.StringVar(value=os.environ.get("AURALITE_GGUF_CHAT_FORMAT", ""))
        use_chat_var = tk.BooleanVar(
            value=os.environ.get("AURALITE_GGUF_USE_CHAT", "0").lower() in {"1", "true", "yes", "on"})
        mmap_var = tk.BooleanVar(
            value=os.environ.get("AURALITE_GGUF_USE_MMAP", "1").lower() not in {"0", "false", "no", "off"})
        mlock_var = tk.BooleanVar(
            value=os.environ.get("AURALITE_GGUF_USE_MLOCK", "0").lower() in {"1", "true", "yes", "on"})

        rows = [
            ("Context (n_ctx):", n_ctx_var, "Max tokens in context; larger = more RAM/VRAM"),
            ("GPU layers:", gpu_var, "-1 = offload as much as llama.cpp can; 0 = CPU only"),
            ("CPU threads:", threads_var, "Usually number of physical/logical CPU cores"),
            ("Batch (n_batch):", batch_var, "Prompt processing batch size"),
            ("Chat format:", chat_format_var, "Optional: llama-2, chatml, mistral-instruct, ..."),
        ]
        for i, (label, var, hint) in enumerate(rows, start=1):
            ttk.Label(outer, text=label).grid(row=i, column=0, sticky=tk.W, padx=(0, 8), pady=3)
            ttk.Entry(outer, textvariable=var, width=18).grid(row=i, column=1, sticky=tk.W, pady=3)
            ttk.Label(outer, text=hint, style="Sub.TLabel").grid(row=i, column=2, sticky=tk.W, padx=(8, 0), pady=3)

        opts_row = ttk.Frame(outer)
        opts_row.grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=(8, 2))
        ttk.Checkbutton(opts_row, text="Use chat completion / template",
                        variable=use_chat_var).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Checkbutton(opts_row, text="mmap", variable=mmap_var).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Checkbutton(opts_row, text="mlock", variable=mlock_var).pack(side=tk.LEFT)

        ttk.Label(outer,
                  text="Note: GGUF is inference-only here. Training/fine-tuning uses AuraLite .pt models.",
                  style="Sub.TLabel", foreground="#a60").grid(
            row=7, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))

        result = {"value": None}

        def on_ok():
            try:
                result["value"] = {
                    "n_ctx": int(n_ctx_var.get()),
                    "n_gpu_layers": int(gpu_var.get()),
                    "n_threads": int(threads_var.get()) if threads_var.get().strip() else None,
                    "n_batch": int(batch_var.get()),
                    "chat_format": chat_format_var.get().strip() or None,
                    "use_chat_completion": bool(use_chat_var.get()),
                    "use_mmap": bool(mmap_var.get()),
                    "use_mlock": bool(mlock_var.get()),
                }
            except ValueError:
                messagebox.showerror("GGUF options", "n_ctx, GPU layers, threads and batch must be integers.", parent=dlg)
                return
            dlg.destroy()

        def on_cancel():
            result["value"] = None
            dlg.destroy()

        btns = ttk.Frame(outer)
        btns.grid(row=8, column=0, columnspan=3, sticky=tk.E, pady=(12, 0))
        ttk.Button(btns, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Load GGUF", command=on_ok).pack(side=tk.RIGHT, padx=4)

        dlg.protocol("WM_DELETE_WINDOW", on_cancel)
        dlg.update_idletasks()
        x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - dlg.winfo_width()) // 2)
        y = self.root.winfo_rooty() + max(0, (self.root.winfo_height() - dlg.winfo_height()) // 2)
        dlg.geometry(f"+{x}+{y}")
        self.root.wait_window(dlg)
        return result["value"]


    def load_model(self):
        path = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[
                ("Supported models", "*.pt *.gguf"),
                ("AuraLite / PyTorch", "*.pt"),
                ("GGUF / llama.cpp", "*.gguf"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            if path.lower().endswith(".gguf"):
                opts = self._ask_gguf_options(path)
                if opts is None:
                    return
                self.status_label.config(text=f"Status: Loading GGUF… ({os.path.basename(path)})")
                self.root.update_idletasks()
                self.engine.load_gguf_model(path, **opts)
            else:
                self.engine.load_model(path)
            self.is_trained = True
            self.gen_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.DISABLED if self.engine.is_gguf_model() else tk.NORMAL)
            n = self.engine.model.count_parameters()
            self.param_label.config(text=f"Parameters: {n:,}" if n else "Parameters: —")
            self.model_file_label.config(
                text=f"Loaded: {os.path.basename(path)}", foreground="black")
            if self.engine.is_gguf_model():
                self.status_label.config(
                    text=f"Status: GGUF model loaded ✅  ({os.path.basename(path)})")
            else:
                self.status_label.config(
                    text=f"Status: Model loaded ✅  ({os.path.basename(path)})")
            if gui_common.HAS_QUANTIZATION:
                self._update_quant_buttons()
            # Fill GUI fields from stored params
            p = self.engine.params_used
            for key in ("lr", "epochs", "d_model", "d_ff", "n_heads",
                        "n_layers", "seq_length", "batch_size", "dropout",
                        "grad_clip"):
                if key in p:
                    self.params[key].set(str(p[key]))
            if "tokenizer" in p:
                self.tok_var.set(str(p["tokenizer"]))
            if "bpe_vocab_size" in p:
                self.bpe_vocab_var.set(str(p["bpe_vocab_size"]))
            if "n_kv_heads" in p and p["n_kv_heads"] is not None:
                self.n_kv_heads_var.set(str(p["n_kv_heads"]))
            self._refresh_model_info()
        except GGUFNotAvailableError as e:
            messagebox.showerror(
                "GGUF support is not installed",
                f"{e}\n\nInstall it with:\n  pip install llama-cpp-python"
            )
        except Exception as e:
            messagebox.showerror("Load Error", str(e))


    def _browse_local_hf_model(self):
        """Open folder dialog to select an already downloaded HF model directory."""
        folder = filedialog.askdirectory(
            title="Choose the folder with the downloaded Hugging Face model (must contain config.json)"
        )
        if folder:
            self.hf_model_var.set(folder)
            self.hf_local_only_var.set(True)   # automatically enable offline mode
            self.status_label.config(text=f"Status: Selected local model: {os.path.basename(folder)}")


    def _load_hf_model(self):
        model_name = self.hf_model_var.get().strip()
        if not model_name:
            messagebox.showwarning("No model", "Please enter a Hugging Face model name or select a local folder")
            return

        load_4bit = self.hf_4bit_var.get()
        load_8bit = self.hf_8bit_var.get()
        apply_lora = self.hf_apply_lora_var.get()
        local_only = self.hf_local_only_var.get()

        if load_4bit and load_8bit:
            messagebox.showwarning("Invalid options", "Choose either 4-bit or 8-bit loading, not both.")
            return

        try:
            lora_rank = int(self.hf_lora_rank_var.get())
        except ValueError:
            lora_rank = 16

        self.hf_load_btn.config(state=tk.DISABLED)
        self.status_label.config(text=f"Status: Loading HF model {model_name}…")
        self.root.update_idletasks()

        def run():
            try:
                self.engine.load_hf_model(
                    model_name,
                    load_in_4bit=load_4bit,
                    load_in_8bit=load_8bit,
                    apply_lora=apply_lora,
                    lora_rank=lora_rank,
                    local_files_only=local_only,
                    verbose=True,
                )

                def update_ui():
                    self.gen_btn.config(state=tk.NORMAL)
                    self.save_btn.config(state=tk.DISABLED)
                    self.hf_apply_lora_btn.config(state=tk.NORMAL)
                    self.hf_save_lora_btn.config(state=tk.NORMAL if self.engine.is_hf_model() and getattr(self.engine.hf_proxy, 'is_peft', False) else tk.DISABLED)
                    self.hf_load_lora_btn.config(state=tk.NORMAL)
                    self.hf_finetune_btn.config(state=tk.NORMAL)
                    self.hf_push_btn.config(state=tk.NORMAL)

                    n = self.engine.model.count_parameters()
                    self.param_label.config(text=f"Parameters: {n:,}")
                    self.model_file_label.config(text=f"HF: {model_name}", foreground="black")
                    self.status_label.config(text=f"Status: HF model loaded ✅ ({model_name})")
                    self._refresh_model_info()

                    # Disable native training / saving buttons for HF models
                    self.train_btn.config(state=tk.DISABLED)
                    if gui_common.HAS_QUANTIZATION:
                        self._update_quant_buttons()

                self.root.after(0, update_ui)

            except HFNotAvailableError as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror(
                    "HF support missing",
                    f"{err}\n\nRun: pip install -r requirements.txt"
                ))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror("HF Load Error", err))
            finally:
                self.root.after(0, lambda: self.hf_load_btn.config(state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()


    def _apply_lora_to_hf(self):
        if not self.engine.is_hf_model():
            messagebox.showwarning("No HF model", "Load a Hugging Face model first.")
            return

        try:
            rank = int(self.hf_lora_rank_var.get())
        except ValueError:
            rank = 16

        try:
            self.engine.apply_lora_to_hf(rank=rank)
            self.hf_save_lora_btn.config(state=tk.NORMAL)
            self.hf_finetune_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"Status: LoRA (rank {rank}) applied ✅")
            self._refresh_model_info()
        except Exception as e:
            messagebox.showerror("LoRA Error", str(e))


    def _save_hf_lora(self):
        if not self.engine.is_hf_model():
            return
        # tkinter has no `asksaveasdirectory` (that call raised AttributeError
        # before v2.6.5, so the button silently did nothing): ask for a folder
        # and allow a new one, since LoRA adapters are written to a fresh dir.
        path = filedialog.askdirectory(
            title="Choose folder to save LoRA adapter", mustexist=False)
        if not path:
            return
        try:
            self.engine.save_hf_lora(path)
            messagebox.showinfo("Saved", f"LoRA adapter saved to:\n{path}")
            self.status_label.config(text="Status: LoRA adapter saved ✅")
        except Exception as e:
            messagebox.showerror("Save LoRA Error", str(e))


    def _load_hf_lora(self):
        if not self.engine.is_hf_model():
            messagebox.showwarning("No base model", "Load the base HF model first, then load adapter.")
            return
        path = filedialog.askdirectory(title="Select folder with saved LoRA adapter")
        if not path:
            return
        try:
            self.engine.load_hf_lora(path)
            self.hf_save_lora_btn.config(state=tk.NORMAL)
            self.hf_finetune_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Status: LoRA adapter loaded ✅")
            self._refresh_model_info()
        except Exception as e:
            messagebox.showerror("Load LoRA Error", str(e))


    def _push_hf_to_hub(self):
        if not self.engine.is_hf_model():
            messagebox.showwarning("No HF model", "Load a Hugging Face model first.")
            return

        repo_id = tk.simpledialog.askstring(
            "Push to Hub",
            "Enter repository ID (e.g. username/model-name):",
            parent=self.root
        )
        if not repo_id:
            return

        private = messagebox.askyesno("Private?", "Make repository private?", parent=self.root)

        self.hf_push_btn.config(state=tk.DISABLED)
        self.status_label.config(text=f"Status: Pushing to {repo_id}...")

        def run():
            try:
                self.engine.push_hf_model_to_hub(
                    repo_id=repo_id,
                    commit_message="Uploaded from AuraLite AI",
                    private=private
                )
                self.root.after(0, lambda: messagebox.showinfo(
                    "Success", f"Model successfully pushed to:\nhttps://huggingface.co/{repo_id}"
                ))
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Status: Pushed to {repo_id} ✅"
                ))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror("Push Error", err))
            finally:
                self.root.after(0, lambda: self.hf_push_btn.config(state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()


    def _load_hf_from_hub(self):
        repo_id = tk.simpledialog.askstring(
            "Load from Hub",
            "Enter repository ID or model name:",
            parent=self.root
        )
        if not repo_id:
            return

        load_4bit = self.hf_4bit_var.get()
        load_8bit = self.hf_8bit_var.get()
        apply_lora = self.hf_apply_lora_var.get()

        if load_4bit and load_8bit:
            messagebox.showwarning("Invalid options", "Choose either 4-bit or 8-bit loading, not both.")
            return

        try:
            lora_rank = int(self.hf_lora_rank_var.get())
        except ValueError:
            lora_rank = 16

        self.status_label.config(text=f"Status: Loading {repo_id} from Hub...")
        self.hf_load_btn.config(state=tk.DISABLED)

        def run():
            try:
                self.engine.load_hf_model(
                    repo_id,
                    load_in_4bit=load_4bit,
                    load_in_8bit=load_8bit,
                    apply_lora=apply_lora,
                    lora_rank=lora_rank,
                    local_files_only=False,
                    verbose=True,
                )

                def update_ui():
                    self.gen_btn.config(state=tk.NORMAL)
                    self.save_btn.config(state=tk.DISABLED)
                    self.hf_apply_lora_btn.config(state=tk.NORMAL)
                    self.hf_save_lora_btn.config(state=tk.NORMAL)
                    self.hf_load_lora_btn.config(state=tk.NORMAL)
                    self.hf_finetune_btn.config(state=tk.NORMAL)
                    self.hf_push_btn.config(state=tk.NORMAL)

                    n = self.engine.model.count_parameters()
                    self.param_label.config(text=f"Parameters: {n:,}")
                    self.model_file_label.config(text=f"HF: {repo_id}", foreground="black")
                    self.status_label.config(text=f"Status: Loaded from Hub ✅ ({repo_id})")
                    self._refresh_model_info()
                    self.train_btn.config(state=tk.DISABLED)
                    if gui_common.HAS_QUANTIZATION:
                        self._update_quant_buttons()

                self.root.after(0, update_ui)

            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror("HF Hub Load Error", err))
            finally:
                self.root.after(0, lambda: self.hf_load_btn.config(state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()


    def _finetune_hf_from_gui(self):
        if not self.engine.is_hf_model():
            messagebox.showwarning("No HF model", "Load a Hugging Face model first (preferably in 4-bit).")
            return

        # Ask for training text
        if self.selected_file_path:
            use_training = messagebox.askyesno(
                "Training data",
                "Use the currently selected .txt file for fine-tuning?"
            )
            if use_training:
                try:
                    with open(self.selected_file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    texts = [t.strip() for t in text.split("\n\n") if len(t.strip()) > 30]
                except Exception as e:
                    messagebox.showerror("File Error", str(e))
                    return
            else:
                texts = None
        else:
            texts = None

        if not texts:
            # Ask user to select a file or use a small example
            path = filedialog.askopenfilename(title="Select .txt file for fine-tuning",
                                              filetypes=[("Text", "*.txt"), ("All", "*.*")])
            if not path:
                messagebox.showinfo("Info", "Fine-tuning cancelled.")
                return
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                texts = [t.strip() for t in text.split("\n\n") if len(t.strip()) > 30]
            except Exception as e:
                messagebox.showerror("File Error", str(e))
                return

        if not texts:
            messagebox.showwarning("No data", "Not enough text for fine-tuning.")
            return

        self.hf_finetune_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Fine-tuning with LoRA/QLoRA… (see console)")

        def run():
            try:
                out_dir = self.engine.finetune_hf(
                    texts,
                    output_dir="hf_lora_finetuned",
                    epochs=3,
                    learning_rate=2e-4,
                    batch_size=2,                    # small for consumer GPUs
                    max_length=512,
                    gradient_accumulation_steps=8,
                )
                self.root.after(0, lambda: messagebox.showinfo(
                    "Fine-tuning complete",
                    f"LoRA adapter saved to:\n{out_dir}\n\nYou can now load it with 'Load LoRA Adapter'."
                ))
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Status: Fine-tuning complete ✅ Adapter in {out_dir}"))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror("Fine-tune Error", err))
            finally:
                self.root.after(0, lambda: self.hf_finetune_btn.config(state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()

