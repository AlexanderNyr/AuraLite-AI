import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from model_engine import (
    AuraLiteEngine, validate_params, ParamValidationError,
    estimate_n_params, recommend_epochs, recommend_gen_length,
)
from web_tools import build_web_context
import threading
import multiprocessing
import os
import sys
import io
import time
import json

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ======================================================================
#  Configuration Presets
# ======================================================================

CONFIG_PRESETS = {
    "Tiny (CPU-friendly)": {
        "d_model": 64, "d_ff": 128, "n_heads": 2, "n_layers": 2,
        "seq_length": 32, "batch_size": 16, "lr": 0.001, "epochs": 200,
        "dropout": 0.1, "grad_clip": 1.0, "n_kv_heads": 0,
    },
    "Small (default)": {
        "d_model": 128, "d_ff": 256, "n_heads": 4, "n_layers": 4,
        "seq_length": 64, "batch_size": 32, "lr": 0.0003, "epochs": 100,
        "dropout": 0.1, "grad_clip": 1.0, "n_kv_heads": 0,
    },
    "Medium (GPU recommended)": {
        "d_model": 256, "d_ff": 512, "n_heads": 8, "n_layers": 6,
        "seq_length": 128, "batch_size": 64, "lr": 0.0003, "epochs": 50,
        "dropout": 0.1, "grad_clip": 1.0, "n_kv_heads": 0,
    },
    "Large (powerful GPU)": {
        "d_model": 512, "d_ff": 1024, "n_heads": 8, "n_layers": 8,
        "seq_length": 256, "batch_size": 32, "lr": 0.0001, "epochs": 30,
        "dropout": 0.1, "grad_clip": 1.0, "n_kv_heads": 0,
    },
    "GQA-efficient (Medium)": {
        "d_model": 256, "d_ff": 512, "n_heads": 8, "n_layers": 6,
        "seq_length": 128, "batch_size": 64, "lr": 0.0003, "epochs": 50,
        "dropout": 0.1, "grad_clip": 1.0, "n_kv_heads": 2,
    },
}


class ConsoleRedirector(io.TextIOBase):
    """Thread-safe redirector that forwards writes to a Tk Text widget.

    Also keeps writing to the original stream (so logs are still visible
    in the real terminal if launched from one), and buffers lines so that
    they are flushed to the widget via root.after (Tk is not thread-safe).

    Lines are colourised based on a simple keyword heuristic
    (ERROR / WARNING / NOTE / OK / etc).
    """

    # (keyword substring, tag name) — first match wins, case-insensitive.
    LEVEL_RULES = [
        ("traceback",     "error"),
        ("error",         "error"),
        ("exception",     "error"),
        ("critical",      "error"),
        ("fail",          "error"),
        ("warning",       "warn"),
        ("warn:",         "warn"),
        ("deprecat",      "warn"),
        ("note:",         "info"),
        ("info:",         "info"),
        ("[auralite]",    "engine"),
        ("epoch",         "epoch"),
        ("✅",             "ok"),
        ("🛑",             "warn"),
        ("complete",      "ok"),
        ("finished",      "ok"),
        ("disabled",      "warn"),
    ]

    def __init__(self, widget: tk.Text, root: tk.Tk, original):
        super().__init__()
        self.widget   = widget
        self.root     = root
        self.original = original
        self._lock    = threading.Lock()
        self._buffer  = ""           # line-level buffer for accurate tagging

    def write(self, s: str) -> int:
        if not s:
            return 0
        # Mirror to the original stream so terminal users still see output.
        try:
            if self.original is not None:
                self.original.write(s)
        except Exception:
            pass
        # Buffer until newline so we can colour each whole line.
        self._buffer += s
        if "\n" in self._buffer:
            lines = self._buffer.split("\n")
            # Last chunk may be a partial line — keep it buffered.
            self._buffer = lines.pop()
            for line in lines:
                self._schedule(line + "\n")
        return len(s)

    def _schedule(self, line: str):
        tag = self._classify(line)
        try:
            self.root.after(0, self._append, line, tag)
        except Exception:
            pass  # window may be closing

    @classmethod
    def _classify(cls, line: str) -> str | None:
        low = line.lower()
        for needle, tag in cls.LEVEL_RULES:
            if needle in low:
                return tag
        return None

    def _append(self, s: str, tag: str | None):
        with self._lock:
            try:
                self.widget.config(state=tk.NORMAL)
                if tag:
                    self.widget.insert(tk.END, s, tag)
                else:
                    self.widget.insert(tk.END, s)
                # Cap at ~5000 lines so the widget stays responsive.
                line_count = int(self.widget.index("end-1c").split(".")[0])
                if line_count > 5000:
                    self.widget.delete("1.0", f"{line_count - 5000}.0")
                self.widget.see(tk.END)
                self.widget.config(state=tk.DISABLED)
            except tk.TclError:
                pass  # widget destroyed

    def flush(self):
        # Flush any trailing partial line.
        if self._buffer:
            tail, self._buffer = self._buffer, ""
            self._schedule(tail)
        try:
            if self.original is not None:
                self.original.flush()
        except Exception:
            pass


def _fmt_duration(seconds: float) -> str:
    """Format seconds as a compact human-readable string."""
    if seconds is None or seconds < 0 or seconds != seconds:  # NaN check
        return "—"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}h {m:02d}m {s:02d}s"


class AIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AuraLite AI v2.1 — Modern Transformer Edition")
        self.root.geometry("920x840")
        self.root.minsize(820, 700)
        self.root.configure(bg="#f5f6f7")

        self.engine = AuraLiteEngine()
        self.is_trained = False
        self.selected_file_path = None
        self.stop_event = threading.Event()
        self.loss_history = []  # [(epoch, train_loss, val_loss), ...]

        # ETA tracking
        self.train_start_time: float | None = None
        self.epoch_times: list[float] = []   # seconds per completed epoch
        self._last_epoch_ts: float | None = None

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

        self.tab_train   = ttk.Frame(self.notebook, padding="12")
        self.tab_gen     = ttk.Frame(self.notebook, padding="12")
        self.tab_model   = ttk.Frame(self.notebook, padding="12")
        self.tab_console = ttk.Frame(self.notebook, padding="12")

        self.notebook.add(self.tab_train,   text=" 🏋️  Training ")
        self.notebook.add(self.tab_gen,     text=" ✨  Generation ")
        self.notebook.add(self.tab_model,   text=" 💾  Model ")
        self.notebook.add(self.tab_console, text=" 🖥️  Console ")

        self._build_training_tab()
        self._build_generation_tab()
        self._build_model_tab()
        self._build_console_tab()

        # Hook stdout/stderr into the console tab. Do it AFTER the widget
        # exists. Keep references so we can restore them on shutdown.
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = ConsoleRedirector(self.console_text, self.root, self._orig_stdout)
        sys.stderr = ConsoleRedirector(self.console_text, self.root, self._orig_stderr)
        print(f"[AuraLite] Console attached. Device: {self.engine.device}, "
              f"threads: {self.engine.num_threads}")

        # Restore original streams on window close.
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ==================================================================
    #  TAB 1 — Training
    # ==================================================================
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

        row2 = ttk.Frame(tok_frame)
        row2.pack(fill=tk.X, pady=2)

        self.compile_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="torch.compile (faster, slow first epoch)",
                        variable=self.compile_var).pack(side=tk.LEFT, padx=4)

        self.continue_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="Continue training current model",
                        variable=self.continue_var).pack(side=tk.LEFT, padx=12)

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

        self.status_label = ttk.Label(run_frame,
                                      text="Status: Waiting for file…")
        self.status_label.pack(pady=(0, 2), anchor=tk.W)

        # ---- Loss history ------------------------------------------------
        hist_frame = ttk.LabelFrame(tab, text="  📉  Loss History  ", padding="6")
        hist_frame.pack(fill=tk.BOTH, expand=True)

        if HAS_MATPLOTLIB:
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
        self.len_display = ttk.Label(len_row, text="100", width=5)
        self.len_display.pack(side=tk.LEFT, padx=4)
        self.len_scale.configure(command=self._update_len_display)

        ttk.Button(len_row, text="🎯 Auto",
                   command=self._auto_gen_length, width=8).pack(side=tk.LEFT, padx=4)

        # NEW: Batch generation
        self.batch_row = ttk.Frame(seed_frame)
        self.batch_row.pack(fill=tk.X, pady=2)
        self.batch_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.batch_row, text="Batch mode (multiple prompts)",
                        variable=self.batch_var).pack(side=tk.LEFT, padx=4)
        self.batch_entry = ttk.Entry(self.batch_row, font=("Segoe UI", 10))
        self.batch_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.batch_entry.insert(0, "Prompt 1 | Prompt 2 | Prompt 3")
        self.batch_entry.pack_forget()  # Hidden by default

        self.gen_btn = ttk.Button(seed_frame, text="📝 Generate Text",
                                  command=self.generate_text,
                                  state=tk.DISABLED)
        self.gen_btn.pack(pady=6)

        self.stream_var = tk.BooleanVar(value=False)
        self.stream_cb = ttk.Checkbutton(seed_frame,
            text="Streaming output (token-by-token)",
            variable=self.stream_var)
        self.stream_cb.pack(pady=2)

        # NEW: Thinking mode + Web search
        smart_row = ttk.Frame(seed_frame)
        smart_row.pack(fill=tk.X, pady=2)

        self.thinking_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(smart_row,
            text="🧠 Thinking mode (two-pass draft → answer)",
            variable=self.thinking_var).pack(side=tk.LEFT, padx=4)

        self.websearch_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(smart_row,
            text="🌐 Web search (DuckDuckGo)",
            variable=self.websearch_var,
            command=self._toggle_web_query).pack(side=tk.LEFT, padx=12)

        self.web_query_row = ttk.Frame(seed_frame)
        ttk.Label(self.web_query_row,
                  text="Search query (empty = use seed):").pack(
            side=tk.LEFT, padx=4)
        self.web_query_entry = ttk.Entry(self.web_query_row,
                                         font=("Segoe UI", 10))
        self.web_query_entry.pack(side=tk.LEFT, fill=tk.X,
                                  expand=True, padx=4)

        # --- Output ---
        out_frame = ttk.LabelFrame(tab, text="  📄  Output  ", padding="6")
        out_frame.pack(fill=tk.BOTH, expand=True)

        # Toolbar above the result widget
        out_toolbar = ttk.Frame(out_frame)
        out_toolbar.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(out_toolbar, text="Generated text",
                  style="Sub.TLabel").pack(side=tk.LEFT, padx=2)

        ttk.Button(out_toolbar, text="🗑 Clear",
                   command=self._clear_result).pack(side=tk.RIGHT, padx=2)
        ttk.Button(out_toolbar, text="💾 Save…",
                   command=self._save_result).pack(side=tk.RIGHT, padx=2)
        ttk.Button(out_toolbar, text="➕ Append to file…",
                   command=self._append_result).pack(side=tk.RIGHT, padx=2)
        ttk.Button(out_toolbar, text="📋 Copy",
                   command=self._copy_result).pack(side=tk.RIGHT, padx=2)

        # Text + scrollbar
        text_wrap = ttk.Frame(out_frame)
        text_wrap.pack(fill=tk.BOTH, expand=True)

        res_ysb = ttk.Scrollbar(text_wrap, orient=tk.VERTICAL)
        self.result_text = tk.Text(text_wrap, height=10,
                                   font=("Consolas", 11), wrap=tk.WORD,
                                   yscrollcommand=res_ysb.set,
                                   undo=True)
        res_ysb.config(command=self.result_text.yview)
        res_ysb.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right-click context menu on the result widget
        self._result_menu = tk.Menu(self.root, tearoff=0)
        self._result_menu.add_command(label="Copy selection",
                                      command=self._copy_selection_result)
        self._result_menu.add_command(label="Copy all",
                                      command=self._copy_result)
        self._result_menu.add_separator()
        self._result_menu.add_command(label="Save to file…",
                                      command=self._save_result)
        self._result_menu.add_command(label="Append to file…",
                                      command=self._append_result)
        self._result_menu.add_separator()
        self._result_menu.add_command(label="Clear",
                                      command=self._clear_result)
        # bind right click (Button-3 on Linux/Win, Button-2 on macOS)
        self.result_text.bind("<Button-3>", self._show_result_menu)
        self.result_text.bind("<Button-2>", self._show_result_menu)
        # Ctrl/Cmd+S to save
        self.result_text.bind("<Control-s>", lambda e: (self._save_result(), "break"))
        self.result_text.bind("<Command-s>", lambda e: (self._save_result(), "break"))

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

    # ==================================================================
    #  TAB 4 — Console
    # ==================================================================
    def _build_console_tab(self):
        tab = self.tab_console

        # Toolbar: clear + copy + autoscroll toggle
        toolbar = ttk.Frame(tab)
        toolbar.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(toolbar, text="🖥️ Live stdout / stderr from the engine",
                  style="Sub.TLabel").pack(side=tk.LEFT, padx=4)

        ttk.Button(toolbar, text="🗑 Clear",
                   command=self._clear_console).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="📋 Copy All",
                   command=self._copy_console).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="💾 Save to file…",
                   command=self._save_console).pack(side=tk.RIGHT, padx=2)

        # Output area with scrollbar
        out_frame = ttk.LabelFrame(tab, text="  📜  Output  ", padding="4")
        out_frame.pack(fill=tk.BOTH, expand=True)

        ysb = ttk.Scrollbar(out_frame, orient=tk.VERTICAL)
        self.console_text = tk.Text(
            out_frame,
            font=("Consolas", 10),
            bg="#1e1e1e", fg="#dcdcdc",
            insertbackground="#dcdcdc",
            wrap=tk.NONE,
            state=tk.DISABLED,
            yscrollcommand=ysb.set,
        )
        ysb.config(command=self.console_text.yview)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        self.console_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ---- Colour tags ----------------------------------------------
        # VS-Code-ish palette on a dark background.
        self.console_text.tag_configure("error",  foreground="#f48771",
                                        font=("Consolas", 10, "bold"))
        self.console_text.tag_configure("warn",   foreground="#dcdcaa")
        self.console_text.tag_configure("info",   foreground="#9cdcfe")
        self.console_text.tag_configure("ok",     foreground="#6a9955",
                                        font=("Consolas", 10, "bold"))
        self.console_text.tag_configure("engine", foreground="#c586c0")
        self.console_text.tag_configure("epoch",  foreground="#4ec9b0")

        # Legend strip
        legend = ttk.Frame(tab)
        legend.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(legend, text="Legend:", style="Sub.TLabel").pack(side=tk.LEFT, padx=4)
        for txt, color in [
            ("[AuraLite]", "#c586c0"),
            ("epoch",      "#4ec9b0"),
            ("INFO",       "#9cdcfe"),
            ("WARNING",    "#dcdcaa"),
            ("ERROR",      "#f48771"),
            ("✅ done",     "#6a9955"),
        ]:
            tk.Label(legend, text=txt, fg=color, bg="#f5f6f7",
                     font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=6)

    def _clear_console(self):
        self.console_text.config(state=tk.NORMAL)
        self.console_text.delete("1.0", tk.END)
        self.console_text.config(state=tk.DISABLED)

    def _copy_console(self):
        try:
            text = self.console_text.get("1.0", tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_label.config(text="Status: Console copied to clipboard ✅")
        except tk.TclError:
            pass

    def _save_console(self):
        path = filedialog.asksaveasfilename(
            title="Save console log",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.console_text.get("1.0", tk.END))
            messagebox.showinfo("Saved", f"Console log saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _on_close(self):
        """Restore stdout/stderr and close the window cleanly."""
        try:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
        except Exception:
            pass
        try:
            self.stop_event.set()
        except Exception:
            pass
        self.root.destroy()

    def _refresh_model_info(self):
        m = self.engine.model
        lines = []
        if m is None:
            lines.append("No model in memory.")
        else:
            tok = self.engine.tokenizer
            lines.append(f"Parameters      : {m.count_parameters():,}")
            if hasattr(m, 'lora_rank') and m.lora_rank > 0:
                lines.append(f"Trainable (LoRA): {m.count_trainable_parameters():,} "
                             f"(rank {m.lora_rank})")
            lines.append(f"Vocab size      : {self.engine.vocab_size}")
            lines.append(f"Tokenizer       : {tok.kind if tok else '—'}")
            lines.append(f"d_model         : {m.d_model}")
            lines.append(f"d_ff            : {m.d_ff}")
            lines.append(f"n_heads         : {m.n_heads}")
            lines.append(f"n_layers        : {m.n_layers}")
            n_kv = m.n_kv_heads or m.n_heads
            lines.append(f"n_kv_heads      : {n_kv} "
                         f"({'GQA' if m.n_kv_heads and m.n_kv_heads < m.n_heads else 'MHA'})")
            lines.append(f"max_seq_len     : {m.max_seq_len}")
            lines.append(f"dropout         : {m.dropout}")
            lines.append(f"ALiBi           : {'Yes ✅' if m.use_alibi else 'No'}")
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
        if not HAS_MATPLOTLIB:
            self.loss_text.config(state=tk.NORMAL)
            self.loss_text.insert(tk.END, line + "\n")
            self.loss_text.see(tk.END)
            self.loss_text.config(state=tk.DISABLED)

    def _update_loss_plot(self):
        """Update matplotlib loss plot with new data."""
        if not HAS_MATPLOTLIB or not self.loss_history:
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

    def _apply_preset(self):
        """Apply a configuration preset to all fields."""
        preset_name = self.preset_var.get()
        preset = CONFIG_PRESETS.get(preset_name)
        if not preset:
            return
        for key, value in preset.items():
            if key == "n_kv_heads":
                self.n_kv_heads_var.set(str(value))
            else:
                if key in self.params:
                    self.params[key].set(str(value))
        self.status_label.config(
            text=f"Status: Preset '{preset_name}' applied ✅")

    # ---- Auto-recommend Epochs ------------------------------------------
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

    # ---- Auto-recommend Generation Length -------------------------------
    def _auto_gen_length(self):
        """Pick a generation length based on the seed and the model's context."""
        seed = self.seed_entry.get()
        tokenizer = self.engine.tokenizer
        max_seq_len = (self.engine.model.max_seq_len
                       if self.engine.model is not None else 4096)

        length = recommend_gen_length(seed, tokenizer, max_seq_len=max_seq_len)
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
              f"(seed≈{len(seed)} chars, max_seq_len={max_seq_len})")

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

        # ---- ETA computation ----------------------------------------------
        now = time.time()
        if self._last_epoch_ts is not None:
            self.epoch_times.append(now - self._last_epoch_ts)
            # Keep a rolling window of the most recent 20 epochs for stability.
            if len(self.epoch_times) > 20:
                self.epoch_times = self.epoch_times[-20:]
        self._last_epoch_ts = now

        elapsed = (now - self.train_start_time) if self.train_start_time else 0.0
        remaining_epochs = max(0, total - current)
        eta_str = "—"
        speed_str = ""
        if self.epoch_times:
            avg_epoch = sum(self.epoch_times) / len(self.epoch_times)
            eta_seconds = avg_epoch * remaining_epochs
            eta_str = _fmt_duration(eta_seconds)
            speed_str = f"  |  {avg_epoch:.2f}s/epoch"

        val_part = f"  |  Val: {val_loss:.4f}" if val_loss is not None else ""
        self.status_label.config(
            text=f"Epoch {current}/{total}  |  Loss: {loss:.4f}{val_part}"
                 f"  |  LR: {lr:.6f}{speed_str}"
                 f"  |  Elapsed: {_fmt_duration(elapsed)}  |  ETA: {eta_str}"
        )

        vtxt = f"{val_loss:.4f}" if val_loss is not None else None
        self.loss_history.append((current, loss, val_loss))
        self._append_loss_line(
            f"epoch {current:>4}/{total}   train {loss:.4f}   val {vtxt or '  —  '}"
            f"   eta {eta_str}")
        self._update_loss_plot()

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
                "n_kv_heads":      int(self.n_kv_heads_var.get()) or None,
                "accumulation_steps": int(self.accum_var.get()) or 1,
                "use_alibi":         bool(self.alibi_var.get()),
                "lora_rank":         int(self.lora_var.get()) or 0,
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
        self.loss_history = []
        # Reset ETA tracking
        self.train_start_time = time.time()
        self.epoch_times = []
        self._last_epoch_ts = self.train_start_time
        self.train_btn.config(state=tk.DISABLED)
        self.file_btn.config(state=tk.DISABLED)
        self.gen_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        if HAS_MATPLOTLIB:
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
                self.engine.train(
                    text, params,
                    progress_callback=self.update_progress,
                    stop_event=self.stop_event,
                )
                total_time = (time.time() - self.train_start_time
                              if self.train_start_time else 0.0)
                total_str = _fmt_duration(total_time)
                if self.stop_event.is_set():
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
            except ParamValidationError as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Validation Error", str(e)))
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
    def _toggle_web_query(self, *_):
        """Show / hide the web-search query row."""
        if self.websearch_var.get():
            self.web_query_row.pack(fill=tk.X, pady=2)
        else:
            self.web_query_row.pack_forget()

    def _fetch_web_context(self, seed: str) -> str:
        """Run a web search (blocking — call from a worker thread).

        Returns formatted snippet context, or '' on failure / no results.
        """
        query = self.web_query_entry.get().strip() or seed
        try:
            ctx = build_web_context(query, max_results=4)
            if ctx:
                print(f"🌐 Web search OK: {len(ctx)} chars of context "
                      f"for query '{query}'")
            else:
                print(f"🌐 Web search: no results for '{query}'")
            return ctx
        except Exception as e:
            print(f"⚠️ Web search failed ({e}) — generating without it.")
            return ""

    def generate_text(self):
        # Handle batch mode
        if self.batch_var.get():
            self._generate_batch()
            return

        # Thinking / web-search mode has its own pipeline
        if self.thinking_var.get() or self.websearch_var.get():
            self._generate_thinking()
            return

        # Handle streaming mode
        if self.stream_var.get():
            self._generate_streaming()
            return

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

    # NEW: Thinking mode (+ optional web search) generation
    def _generate_thinking(self):
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

        use_web = self.websearch_var.get()
        use_thinking = self.thinking_var.get()

        self.result_text.delete("1.0", tk.END)
        steps = []
        if use_web:
            steps.append("🌐 searching the web")
        if use_thinking:
            steps.append("🧠 thinking")
        steps.append("✍️ generating")
        self.result_text.insert(tk.END, " → ".join(steps) + " …\n")
        self.gen_btn.config(state=tk.DISABLED)

        def run():
            try:
                web_ctx = self._fetch_web_context(seed) if use_web else ""

                if use_thinking:
                    thoughts, final = self.engine.generate_with_thinking(
                        seed, length, temperature, top_k, top_p,
                        repetition_penalty=rep_pen,
                        web_context=web_ctx or None)
                else:
                    # Web search only: prepend context, generate once
                    prompt = f"{web_ctx}\n{seed}" if web_ctx else seed
                    full = self.engine.generate(
                        prompt, length, temperature, top_k, top_p,
                        repetition_penalty=rep_pen)
                    thoughts = ""
                    final = seed + full[len(prompt):]

                parts = []
                if web_ctx:
                    parts.append("🌐 WEB CONTEXT\n──────────────\n"
                                 + web_ctx)
                if thoughts:
                    parts.append("🧠 THINKING (draft pass)\n"
                                 "──────────────\n" + thoughts)
                parts.append("✅ ANSWER\n──────────────\n" + final)
                self.root.after(0, self._display_result,
                                "\n\n".join(parts))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Gen Error", f"Error during generation:\n{e}"))
            finally:
                self.root.after(0, lambda: self.gen_btn.config(
                    state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()

    # NEW: Streaming generation
    def _generate_streaming(self):
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
        self.result_text.insert(tk.END, "Generating…\n")
        self.gen_btn.config(state=tk.DISABLED)

        def run():
            try:
                for token_text in self.engine.generate_streaming(
                    seed, length, temperature, top_k, top_p,
                    repetition_penalty=rep_pen
                ):
                    # tkinter is NOT thread-safe: marshal the widget update
                    # onto the main loop via root.after instead of calling
                    # root.update() directly from this worker thread.
                    self.root.after(0, lambda t=token_text:
                        (self.result_text.insert(tk.END, t),
                         self.result_text.see(tk.END)))
                self.root.after(0, lambda: self.result_text.insert(
                    tk.END, "\n\n✅ Generation complete."))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Gen Error", f"Error during generation:\n{e}"))
            finally:
                self.root.after(0, lambda: self.gen_btn.config(
                    state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()

    # NEW: Batch generation
    def _generate_batch(self):
        batch_text = self.batch_entry.get()
        if not batch_text.strip():
            messagebox.showwarning("Warning", "Please enter prompts separated by '|'")
            return

        prompts = [p.strip() for p in batch_text.split("|") if p.strip()]
        if not prompts:
            messagebox.showwarning("Warning", "No valid prompts found")
            return

        length = int(self.len_scale.get())
        try:
            temperature = float(self.temp_var.get())
            top_k = int(self.topk_var.get())
            top_p = float(self.topp_var.get())
            rep_pen = float(self.rep_var.get())
        except ValueError:
            messagebox.showwarning("Warning", "Invalid generation settings!")
            return

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"Generating {len(prompts)} prompts in batch…\n")
        self.gen_btn.config(state=tk.DISABLED)

        def run():
            try:
                results = self.engine.generate_batch(
                    prompts, length, temperature, top_k, top_p,
                    repetition_penalty=rep_pen
                )
                for i, (prompt, result) in enumerate(zip(prompts, results)):
                    self.root.after(0, lambda p=prompt, r=result, idx=i:
                        self.result_text.insert(tk.END,
                            f"\n{'='*40}\nPrompt {idx+1}: {p}\n{'='*40}\n{r}\n"))
                self.root.after(0, lambda: self.result_text.insert(
                    tk.END, f"\n✅ Batch generation complete ({len(prompts)} prompts)"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Gen Error", f"Error during batch generation:\n{e}"))
            finally:
                self.root.after(0, lambda: self.gen_btn.config(
                    state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()

    def _display_result(self, text):
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)

    # ---- Output toolbar handlers --------------------------------------
    def _get_result_text(self) -> str:
        # strip the trailing newline tkinter always appends
        return self.result_text.get("1.0", "end-1c")

    def _copy_result(self):
        text = self._get_result_text()
        if not text.strip():
            self.status_label.config(text="Status: Output is empty — nothing to copy.")
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update()  # ensure clipboard persists after focus change
            n = len(text)
            self.status_label.config(
                text=f"Status: Copied {n:,} characters to clipboard ✅")
        except tk.TclError as e:
            messagebox.showerror("Copy Error", str(e))

    def _copy_selection_result(self):
        try:
            sel = self.result_text.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            return self._copy_result()  # nothing selected → copy all
        if not sel:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(sel)
        self.root.update()
        self.status_label.config(
            text=f"Status: Copied selection ({len(sel):,} chars) ✅")

    def _clear_result(self):
        self.result_text.delete("1.0", tk.END)

    def _save_result(self):
        text = self._get_result_text()
        if not text.strip():
            messagebox.showinfo("Empty", "Nothing to save — output is empty.")
            return
        path = filedialog.asksaveasfilename(
            title="Save generated text",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("Markdown", "*.md"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self.status_label.config(
                text=f"Status: Saved output ✅ ({os.path.basename(path)})")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _append_result(self):
        text = self._get_result_text()
        if not text.strip():
            messagebox.showinfo("Empty", "Nothing to append — output is empty.")
            return
        path = filedialog.asksaveasfilename(
            title="Append generated text to file (existing file will be appended)",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("Markdown", "*.md"),
                       ("All files", "*.*")],
            confirmoverwrite=False,
        )
        if not path:
            return
        try:
            sep = "\n\n" + "=" * 50 + "\n"
            with open(path, "a", encoding="utf-8") as f:
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    f.write(sep)
                f.write(text)
            self.status_label.config(
                text=f"Status: Appended output ✅ ({os.path.basename(path)})")
        except Exception as e:
            messagebox.showerror("Append Error", str(e))

    def _show_result_menu(self, event):
        try:
            self._result_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._result_menu.grab_release()

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
            if "n_kv_heads" in p and p["n_kv_heads"] is not None:
                self.n_kv_heads_var.set(str(p["n_kv_heads"]))
            self._refresh_model_info()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    # NEW: Save / Load config
    def save_config(self):
        path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            config = {
                "lr": float(self.params["lr"].get()),
                "epochs": int(self.params["epochs"].get()),
                "d_model": int(self.params["d_model"].get()),
                "d_ff": int(self.params["d_ff"].get()),
                "n_heads": int(self.params["n_heads"].get()),
                "n_layers": int(self.params["n_layers"].get()),
                "seq_length": int(self.params["seq_length"].get()),
                "batch_size": int(self.params["batch_size"].get()),
                "dropout": float(self.params["dropout"].get()),
                "grad_clip": float(self.params["grad_clip"].get()),
                "tokenizer": self.tok_var.get(),
                "bpe_vocab_size": int(self.bpe_vocab_var.get()),
                "val_split": float(self.val_split_var.get()),
                "use_compile": bool(self.compile_var.get()),
                "continue_training": bool(self.continue_var.get()),
                "autosave_every": int(self.autosave_var.get()),
                "n_kv_heads": int(self.n_kv_heads_var.get()) or None,
                "accumulation_steps": int(self.accum_var.get()) or 1,
                "use_alibi": bool(self.alibi_var.get()),
                "lora_rank": int(self.lora_var.get()) or 0,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            self.status_label.config(
                text=f"Status: Config saved ✅ ({os.path.basename(path)})")
        except Exception as e:
            messagebox.showerror("Save Config Error", str(e))

    def load_config(self):
        path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
            # Apply config to GUI
            for key in ("lr", "epochs", "d_model", "d_ff", "n_heads",
                        "n_layers", "seq_length", "batch_size", "dropout",
                        "grad_clip"):
                if key in config:
                    self.params[key].set(str(config[key]))
            if "tokenizer" in config:
                self.tok_var.set(config["tokenizer"])
            if "bpe_vocab_size" in config:
                self.bpe_vocab_var.set(str(config["bpe_vocab_size"]))
            if "val_split" in config:
                self.val_split_var.set(str(config["val_split"]))
            if "n_kv_heads" in config:
                self.n_kv_heads_var.set(str(config["n_kv_heads"] or 0))
            if "accumulation_steps" in config:
                self.accum_var.set(str(config["accumulation_steps"]))
            if "lora_rank" in config:
                self.lora_var.set(str(config["lora_rank"]))
            if "use_compile" in config:
                self.compile_var.set(config["use_compile"])
            if "use_alibi" in config:
                self.alibi_var.set(config["use_alibi"])
            self.status_label.config(
                text=f"Status: Config loaded ✅ ({os.path.basename(path)})")
            messagebox.showinfo("Config Loaded",
                                f"Configuration loaded from:\n{path}")
        except Exception as e:
            messagebox.showerror("Load Config Error", str(e))


# ======================================================================
#  Show/hide batch entry based on checkbox
# ======================================================================

def _toggle_batch_entry(self, *_):
    # NOTE: batch_entry's master is self.batch_row, so it must be re-packed
    # inside that same frame. Using `before=self.gen_btn` (a child of a
    # different frame) raised a TclError — fixed by packing within batch_row.
    if self.batch_var.get():
        self.batch_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
    else:
        self.batch_entry.pack_forget()


# Monkey-patch the method into the class
AIApp._toggle_batch_entry = _toggle_batch_entry

# Bind after creation — need to modify __init__ slightly
# Actually, let's just do it in the main block

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = AIApp(root)
    # Bind batch mode checkbox
    app.batch_var.trace_add("write", app._toggle_batch_entry)
    root.mainloop()
