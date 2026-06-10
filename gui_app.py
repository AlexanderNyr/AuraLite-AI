import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from model_engine import AuraLiteEngine
import threading
import multiprocessing
import os


class AIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AuraLite AI v2.1 — Modern Transformer Edition")
        self.root.geometry("820x760")
        self.root.minsize(760, 680)
        self.root.configure(bg="#f5f6f7")

        self.engine = AuraLiteEngine()
        self.is_trained = False
        self.selected_file_path = None
        self.stop_event = threading.Event()

        # ---- Styles ----------------------------------------------------
        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TLabel", font=("Segoe UI", 10), background="#f5f6f7")
        style.configure("Header.TLabel",
                         font=("Segoe UI", 16, "bold"), background="#f5f6f7")
        style.configure("Sub.TLabel",
                         font=("Segoe UI", 9, "italic"), background="#f5f6f7")
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"),
                        padding=(14, 6))

        # ---- Header ------------------------------------------------------
        main_frame = ttk.Frame(root, padding="12")
        main_frame.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(main_frame,
                           text="🌟 AuraLite AI v2.1 — Modern Edition",
                           style="Header.TLabel")
        header.pack(pady=(0, 2))

        if self.engine.device.type == "cuda":
            dev = "GPU: CUDA 🟢"
        else:
            dev = f"CPU: {self.engine.num_threads} threads"
        info_row = ttk.Frame(main_frame)
        info_row.pack(pady=(0, 6))
        self.device_label = ttk.Label(info_row, text=f"Hardware: {dev}",
                                      style="Sub.TLabel")
        self.device_label.pack(side=tk.LEFT, padx=8)
        self.param_label = ttk.Label(info_row, text="Parameters: —",
                                     style="Sub.TLabel")
        self.param_label.pack(side=tk.LEFT, padx=8)

        # ==================================================================
        #  Notebook — 3 tabs
        # ==================================================================
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_train = ttk.Frame(self.notebook, padding="12")
        self.tab_gen   = ttk.Frame(self.notebook, padding="12")
        self.tab_model = ttk.Frame(self.notebook, padding="12")

        self.notebook.add(self.tab_train, text=" 🏋️  Training ")
        self.notebook.add(self.tab_gen,   text=" ✨  Generation ")
        self.notebook.add(self.tab_model, text=" 💾  Model ")

        self._build_training_tab()
        self._build_generation_tab()
        self._build_model_tab()

    # ==================================================================
    #  TAB 1 — Training
    # ==================================================================
    def _build_training_tab(self):
        tab = self.tab_train

        # ---- Architecture & Hyperparameters ---------------------------
        hp_frame = ttk.LabelFrame(tab, text="  ⚙️  Architecture & Hyperparameters  ",
                                  padding="10")
        hp_frame.pack(fill=tk.X, pady=(0, 8))

        grid = ttk.Frame(hp_frame)
        grid.pack(fill=tk.X)

        self.params = {
            "lr":         tk.StringVar(value="0.0003"),
            "epochs":     tk.StringVar(value="100"),
            "d_model":    tk.StringVar(value="128"),
            "d_ff":       tk.StringVar(value="256"),
            "n_heads":    tk.StringVar(value="4"),
            "n_layers":   tk.StringVar(value="4"),
            "seq_length": tk.StringVar(value="64"),
            "batch_size": tk.StringVar(value="32"),
            "dropout":    tk.StringVar(value="0.1"),
            "grad_clip":  tk.StringVar(value="1.0"),
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

        # ---- Tokenizer & extras ----------------------------------------
        tok_frame = ttk.LabelFrame(tab, text="  🔤  Tokenizer & Options  ",
                                   padding="10")
        tok_frame.pack(fill=tk.X, pady=(0, 8))

        row1 = ttk.Frame(tok_frame)
        row1.pack(fill=tk.X, pady=2)

        ttk.Label(row1, text="Tokenizer:").pack(side=tk.LEFT, padx=4)
        self.tok_var = tk.StringVar(value="bpe")
        ttk.Radiobutton(row1, text="BPE (recommended)", value="bpe",
                        variable=self.tok_var).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(row1, text="Char-level", value="char",
                        variable=self.tok_var).pack(side=tk.LEFT, padx=4)

        ttk.Label(row1, text="BPE Vocab:").pack(side=tk.LEFT, padx=(16, 4))
        self.bpe_vocab_var = tk.StringVar(value="512")
        ttk.Entry(row1, textvariable=self.bpe_vocab_var,
                  width=7).pack(side=tk.LEFT, padx=4)

        ttk.Label(row1, text="Val Split:").pack(side=tk.LEFT, padx=(16, 4))
        self.val_split_var = tk.StringVar(value="0.1")
        ttk.Entry(row1, textvariable=self.val_split_var,
                  width=6).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(tok_frame)
        row2.pack(fill=tk.X, pady=2)

        self.compile_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="torch.compile (faster, slow first epoch)",
                        variable=self.compile_var).pack(side=tk.LEFT, padx=4)

        self.continue_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="Continue training current model",
                        variable=self.continue_var).pack(side=tk.LEFT, padx=12)

        ttk.Label(row2, text="Autosave every N epochs (0 = off):").pack(
            side=tk.LEFT, padx=(16, 4))
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

        self.file_label = ttk.Label(run_frame, text="No file selected",
                                    foreground="gray")
        self.file_label.pack(pady=2, anchor=tk.W)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(run_frame,
                                            variable=self.progress_var,
                                            maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=4)

        self.status_label = ttk.Label(run_frame,
                                      text="Status: Waiting for file…")
        self.status_label.pack(pady=(0, 2), anchor=tk.W)

        # ---- Loss history ------------------------------------------------
        hist_frame = ttk.LabelFrame(tab, text="  📉  Loss History  ", padding="6")
        hist_frame.pack(fill=tk.BOTH, expand=True)

        self.loss_text = tk.Text(hist_frame, height=6, font=("Consolas", 9),
                                 state=tk.DISABLED, wrap=tk.NONE)
        self.loss_text.pack(fill=tk.BOTH, expand=True)

    # ==================================================================
    #  TAB 2 — Generation
    # ==================================================================
    def _build_generation_tab(self):
        tab = self.tab_gen

        # --- Sampling settings ---
        gen_settings = ttk.LabelFrame(tab, text="  🎛️  Sampling  ", padding="10")
        gen_settings.pack(fill=tk.X, pady=(0, 8))

        srow = ttk.Frame(gen_settings)
        srow.pack(fill=tk.X)

        ttk.Label(srow, text="🌡️ Temperature:").grid(row=0, column=0,
                                                     sticky=tk.W, padx=4)
        self.temp_var = tk.DoubleVar(value=0.8)
        self.temp_scale = ttk.Scale(srow, from_=0.1, to=2.0,
                                    variable=self.temp_var,
                                    orient=tk.HORIZONTAL, length=150)
        self.temp_scale.grid(row=0, column=1, padx=4)
        self.temp_display = ttk.Label(srow, text="0.80")
        self.temp_display.grid(row=0, column=2, padx=2)
        self.temp_var.trace_add("write", self._update_temp_display)

        ttk.Label(srow, text="Top-K:").grid(row=0, column=3,
                                            sticky=tk.W, padx=(14, 4))
        self.topk_var = tk.StringVar(value="50")
        ttk.Entry(srow, textvariable=self.topk_var,
                  width=6).grid(row=0, column=4, padx=4)

        ttk.Label(srow, text="Top-P:").grid(row=0, column=5,
                                            sticky=tk.W, padx=(14, 4))
        self.topp_var = tk.StringVar(value="0.9")
        ttk.Entry(srow, textvariable=self.topp_var,
                  width=6).grid(row=0, column=6, padx=4)

        ttk.Label(srow, text="Rep. Penalty:").grid(row=1, column=0,
                                                   sticky=tk.W, padx=4,
                                                   pady=(6, 0))
        self.rep_var = tk.StringVar(value="1.1")
        ttk.Entry(srow, textvariable=self.rep_var,
                  width=6).grid(row=1, column=1, sticky=tk.W, padx=4,
                                pady=(6, 0))
        ttk.Label(srow, text="(1.0 = off, 1.1–1.3 fights loops)",
                  style="Sub.TLabel").grid(row=1, column=2, columnspan=4,
                                           sticky=tk.W, pady=(6, 0))

        # --- Seed + length ---
        seed_frame = ttk.LabelFrame(tab, text="  🌱  Prompt  ", padding="10")
        seed_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(seed_frame, text="Seed phrase:").pack(anchor=tk.W)
        self.seed_entry = ttk.Entry(seed_frame, font=("Segoe UI", 11))
        self.seed_entry.pack(fill=tk.X, pady=4)
        self.seed_entry.insert(0, "The quick")

        len_row = ttk.Frame(seed_frame)
        len_row.pack(fill=tk.X, pady=2)
        ttk.Label(len_row, text="Length (tokens):").pack(side=tk.LEFT, padx=4)
        self.len_scale = ttk.Scale(len_row, from_=10, to=1000,
                                   orient=tk.HORIZONTAL)
        self.len_scale.set(100)
        self.len_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.len_display = ttk.Label(len_row, text="100")
        self.len_display.pack(side=tk.LEFT, padx=4)
        self.len_scale.configure(command=self._update_len_display)

        self.gen_btn = ttk.Button(seed_frame, text="📝 Generate Text",
                                  command=self.generate_text,
                                  state=tk.DISABLED)
        self.gen_btn.pack(pady=6)

        # --- Output ---
        out_frame = ttk.LabelFrame(tab, text="  📄  Output  ", padding="6")
        out_frame.pack(fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(out_frame, height=10,
                                   font=("Consolas", 11), wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)

    # ==================================================================
    #  TAB 3 — Model
    # ==================================================================
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

        # --- Model info ---
        info_frame = ttk.LabelFrame(tab, text="  ℹ️  Model Info  ", padding="6")
        info_frame.pack(fill=tk.BOTH, expand=True)

        self.model_info = tk.Text(info_frame, font=("Consolas", 10),
                                  state=tk.DISABLED, wrap=tk.WORD)
        self.model_info.pack(fill=tk.BOTH, expand=True)

    def _refresh_model_info(self):
        m = self.engine.model
        lines = []
        if m is None:
            lines.append("No model in memory.")
        else:
            tok = self.engine.tokenizer
            lines.append(f"Parameters      : {m.count_parameters():,}")
            lines.append(f"Vocab size      : {self.engine.vocab_size}")
            lines.append(f"Tokenizer       : {tok.kind if tok else '—'}")
            lines.append(f"d_model         : {m.d_model}")
            lines.append(f"d_ff            : {m.d_ff}")
            lines.append(f"n_heads         : {m.n_heads}")
            lines.append(f"n_layers        : {m.n_layers}")
            lines.append(f"n_kv_heads      : {m.n_kv_heads or m.n_heads} "
                         f"({'GQA' if m.n_kv_heads else 'MHA'})")
            lines.append(f"max_seq_len     : {m.max_seq_len}")
            lines.append(f"dropout         : {m.dropout}")
            lines.append(f"device          : {self.engine.device}")
            if self.engine.last_val_loss is not None:
                lines.append(f"last val loss   : {self.engine.last_val_loss:.4f}")
            if self.engine.params_used:
                lines.append("")
                lines.append("Last training params:")
                for k, v in self.engine.params_used.items():
                    lines.append(f"  {k} = {v}")
        self.model_info.config(state=tk.NORMAL)
        self.model_info.delete("1.0", tk.END)
        self.model_info.insert(tk.END, "\n".join(lines))
        self.model_info.config(state=tk.DISABLED)

    def _append_loss_line(self, line):
        self.loss_text.config(state=tk.NORMAL)
        self.loss_text.insert(tk.END, line + "\n")
        self.loss_text.see(tk.END)
        self.loss_text.config(state=tk.DISABLED)

    # ==================================================================
    #  Callbacks
    # ==================================================================

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

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def update_progress(self, current, total, loss, val_loss=None):
        """Called from the training thread — marshal the actual UI update
        onto the Tk main loop via root.after (tkinter is not thread-safe)."""
        self.root.after(0, self._apply_progress, current, total, loss, val_loss)

    def _apply_progress(self, current, total, loss, val_loss):
        percent = (current / total) * 100
        self.progress_var.set(percent)
        lr = self.engine.scheduler.get_lr() if self.engine.scheduler else 0
        val_part = f"  |  Val: {val_loss:.4f}" if val_loss is not None else ""
        self.status_label.config(
            text=f"Epoch {current}/{total}  |  Loss: {loss:.4f}{val_part}"
                 f"  |  LR: {lr:.6f}"
        )
        vtxt = f"{val_loss:.4f}" if val_loss is not None else "  —  "
        self._append_loss_line(
            f"epoch {current:>4}/{total}   train {loss:.4f}   val {vtxt}")

    # ------------------------------------------------------------------
    def stop_training(self):
        self.stop_event.set()
        self.status_label.config(text="Status: Stopping… 🛑")

    # ------------------------------------------------------------------
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
                "dropout":     float(self.params["dropout"].get()),
                "grad_clip":   float(self.params["grad_clip"].get()),
                "tokenizer":       self.tok_var.get(),
                "bpe_vocab_size":  int(self.bpe_vocab_var.get()),
                "val_split":       float(self.val_split_var.get()),
                "use_compile":     bool(self.compile_var.get()),
                "continue_training": bool(self.continue_var.get()),
                "autosave_every":  int(self.autosave_var.get()),
            }
        except ValueError:
            messagebox.showerror("Params Error",
                                 "Please enter valid numbers in all fields!")
            return

        if params["continue_training"] and self.engine.model is None:
            messagebox.showwarning(
                "No model",
                "«Continue training» is checked, but there is no model "
                "in memory — a new one will be created.")
            params["continue_training"] = False

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
        self.train_btn.config(state=tk.DISABLED)
        self.file_btn.config(state=tk.DISABLED)
        self.gen_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        self.loss_text.config(state=tk.NORMAL)
        self.loss_text.delete("1.0", tk.END)
        self.loss_text.config(state=tk.DISABLED)

        def run():
            try:
                self.engine.train(
                    text, params,
                    progress_callback=self.update_progress,
                    stop_event=self.stop_event,
                )
                if self.stop_event.is_set():
                    self.root.after(0, lambda: self.status_label.config(
                        text="Status: Stopped. 🛑 Weights preserved — "
                             "you can generate or save."))
                else:
                    self.root.after(0, lambda: self.status_label.config(
                        text="Status: Training complete! ✅"))
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
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Train Error", f"Error during training:\n{e}"))
            finally:
                self.root.after(0, self._reset_train_buttons)

        threading.Thread(target=run, daemon=True).start()

    def _reset_train_buttons(self):
        self.train_btn.config(state=tk.NORMAL)
        self.file_btn.config(state=tk.NORMAL)
        self.load_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    def generate_text(self):
        seed = self.seed_entry.get()
        length = int(self.len_scale.get())
        try:
            temperature = float(self.temp_var.get())
            top_k = int(self.topk_var.get())
            top_p = float(self.topp_var.get())
            rep_pen = float(self.rep_var.get())
        except ValueError:
            messagebox.showwarning("Warning",
                                   "Invalid generation settings!")
            return

        if not seed:
            messagebox.showwarning("Warning",
                                   "Please enter a seed phrase.")
            return

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "Generating… please wait…\n")
        self.gen_btn.config(state=tk.DISABLED)

        def run():
            try:
                res = self.engine.generate(seed, length,
                                           temperature, top_k, top_p,
                                           repetition_penalty=rep_pen)
                self.root.after(0, self._display_result, res)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Gen Error", f"Error during generation:\n{e}"))
            finally:
                self.root.after(0, lambda: self.gen_btn.config(
                    state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()

    def _display_result(self, text):
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)

    # ------------------------------------------------------------------
    def save_model(self):
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

    def load_model(self):
        path = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.engine.load_model(path)
            self.is_trained = True
            self.gen_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            n = self.engine.model.count_parameters()
            self.param_label.config(text=f"Parameters: {n:,}")
            self.model_file_label.config(
                text=f"Loaded: {os.path.basename(path)}", foreground="black")
            self.status_label.config(
                text=f"Status: Model loaded ✅  ({os.path.basename(path)})")
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
            self._refresh_model_info()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))


# ======================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = AIApp(root)
    root.mainloop()
