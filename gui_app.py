import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from model_engine import (
    AuraLiteEngine, validate_params, ParamValidationError,
    estimate_n_params, recommend_epochs, recommend_gen_length,
    GGUFNotAvailableError,
    HFNotAvailableError,
    HAS_CHAT_SUPPORT,
    CHAT_TEMPLATES,
)
from web_tools import build_web_context
try:
    from quantization import (
        QuantizationEngine, QuantConfig, QuantResult,
        QuantMethod, BitWidth,
        METHOD_SUPPORTED_BITS, METHOD_DESCRIPTIONS,
        compare_quantizations, format_comparison_table,
    )
    HAS_QUANTIZATION = True
except ImportError:
    HAS_QUANTIZATION = False
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
        "use_gradient_checkpointing": False,
        "rope_scaling": None,
    },
    "Small (default)": {
        "d_model": 128, "d_ff": 256, "n_heads": 4, "n_layers": 4,
        "seq_length": 64, "batch_size": 32, "lr": 0.0003, "epochs": 100,
        "dropout": 0.1, "grad_clip": 1.0, "n_kv_heads": 0,
        "use_gradient_checkpointing": False,
        "rope_scaling": None,
    },
    "Medium (GPU recommended)": {
        "d_model": 256, "d_ff": 512, "n_heads": 8, "n_layers": 6,
        "seq_length": 128, "batch_size": 64, "lr": 0.0003, "epochs": 50,
        "dropout": 0.1, "grad_clip": 1.0, "n_kv_heads": 0,
        "use_gradient_checkpointing": True,
        "rope_scaling": {"type": "yarn", "factor": 4.0},  # 4x context
    },
    "Large (powerful GPU)": {
        "d_model": 512, "d_ff": 1024, "n_heads": 8, "n_layers": 8,
        "seq_length": 256, "batch_size": 32, "lr": 0.0001, "epochs": 30,
        "dropout": 0.1, "grad_clip": 1.0, "n_kv_heads": 0,
        "use_gradient_checkpointing": True,
        "rope_scaling": {"type": "yarn", "factor": 8.0},
    },
    "GQA-efficient (Medium)": {
        "d_model": 256, "d_ff": 512, "n_heads": 8, "n_layers": 6,
        "seq_length": 128, "batch_size": 64, "lr": 0.0003, "epochs": 50,
        "dropout": 0.1, "grad_clip": 1.0, "n_kv_heads": 2,
        "use_gradient_checkpointing": True,
        "rope_scaling": {"type": "ntk", "factor": 4.0},
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
        self.root.title("AuraLite AI v2.3 — Modern Transformer Edition")
        self.root.geometry("920x840")
        self.root.minsize(820, 700)
        self.root.configure(bg="#f5f6f7")

        # Dark theme toggle (NEW v2.5)
        self.dark_mode = tk.BooleanVar(value=False)

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
        self._apply_theme(light=True)

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
        self.tab_chat    = ttk.Frame(self.notebook, padding="12")
        self.tab_model   = ttk.Frame(self.notebook, padding="12")
        self.tab_quant   = ttk.Frame(self.notebook, padding="12")
        self.tab_eval    = ttk.Frame(self.notebook, padding="12")
        self.tab_export  = ttk.Frame(self.notebook, padding="12")
        self.tab_console = ttk.Frame(self.notebook, padding="12")

        self.notebook.add(self.tab_train,   text=" 🏋️  Training ")
        self.notebook.add(self.tab_gen,     text=" ✨  Generation ")
        self.notebook.add(self.tab_chat,    text=" 💬  Chat ")
        self.notebook.add(self.tab_model,   text=" 💾  Model ")
        self.notebook.add(self.tab_quant,   text=" ⚡  Quantization ")
        self.notebook.add(self.tab_eval,    text=" 📊  Evaluation ")
        self.notebook.add(self.tab_export,  text=" 📦  Export ")
        self.notebook.add(self.tab_console, text=" 🖥️  Console ")

        self._build_training_tab()
        self._build_generation_tab()
        self._build_chat_tab()
        self._build_model_tab()
        self._build_quantization_tab()
        self._build_evaluation_tab()
        self._build_export_tab()
        self._build_console_tab()

        # Hook stdout/stderr into the console tab. Do it AFTER the widget
        # exists. Keep references so we can restore them on shutdown.
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = ConsoleRedirector(self.console_text, self.root, self._orig_stdout)
        sys.stderr = ConsoleRedirector(self.console_text, self.root, self._orig_stderr)
        print(f"[AuraLite] Console attached. Device: {self.engine.device}, "
              f"threads: {self.engine.num_threads}")

        # Dark mode toggle in header
        theme_btn = ttk.Checkbutton(info_row, text="🌙 Dark", variable=self.dark_mode,
                                    command=self._toggle_dark_mode)
        theme_btn.pack(side=tk.RIGHT, padx=8)

        # Restore original streams on window close.
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ==================================================================
    #  Theme handling (NEW v2.5)
    # ==================================================================
    def _apply_theme(self, light: bool = True):
        """Apply light or dark theme to the entire application."""
        style = ttk.Style()

        if light:
            # Light theme (original)
            bg = "#f5f6f7"
            fg = "#000000"
            entry_bg = "#ffffff"
            text_bg = "#ffffff"
            console_bg = "#1e1e1e"
            console_fg = "#dcdcdc"
        else:
            # Dark theme
            bg = "#2b2b2b"
            fg = "#eeeeee"
            entry_bg = "#3c3f41"
            text_bg = "#2b2b2b"
            console_bg = "#1e1e1e"
            console_fg = "#dcdcdc"

        self.root.configure(bg=bg)

        # ttk styles
        style.configure("TFrame", background=bg)
        style.configure("TLabel", font=("Segoe UI", 10), background=bg, foreground=fg)
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), background=bg, foreground=fg)
        style.configure("Sub.TLabel", font=("Segoe UI", 9, "italic"), background=bg, foreground=fg)
        style.configure("TLabelframe", background=bg)
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"), background=bg, foreground=fg)
        style.configure("TNotebook", background=bg)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(14, 6))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TEntry", fieldbackground=entry_bg, foreground=fg)
        style.configure("TCombobox", fieldbackground=entry_bg, foreground=fg)

        # Text widgets that need manual styling
        for widget_name in ["result_text", "model_info", "q_result_text", "console_text", "chat_text", "loss_text"]:
            if hasattr(self, widget_name):
                w = getattr(self, widget_name)
                try:
                    if "console" in widget_name:
                        w.configure(bg=console_bg, fg=console_fg, insertbackground=console_fg)
                    else:
                        w.configure(bg=text_bg, fg=fg, insertbackground=fg)
                except tk.TclError:
                    pass

    def _toggle_dark_mode(self):
        """Toggle between light and dark theme."""
        is_dark = self.dark_mode.get()
        self._apply_theme(light=not is_dark)

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

        self.checkpoint_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1b, text="Gradient Checkpointing (VRAM saving)",
                        variable=self.checkpoint_var).pack(side=tk.LEFT, padx=12)

        # NEW: Multi-GPU (DDP) toggle — v2.3
        self.ddp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1b, text="Multi-GPU (DDP)",
                        variable=self.ddp_var).pack(side=tk.LEFT, padx=12)

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
        ttk.Label(srow, text="Min-P:").grid(row=1, column=2,
                                            sticky=tk.W, padx=(14, 4),
                                            pady=(6, 0))
        self.minp_var = tk.StringVar(value="0.0")
        ttk.Entry(srow, textvariable=self.minp_var,
                  width=6).grid(row=1, column=3, sticky=tk.W, padx=4,
                                pady=(6, 0))
        ttk.Label(srow, text="(1.0 = off, 1.1–1.3 fights loops)",
                  style="Sub.TLabel").grid(row=1, column=4, columnspan=3,
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

        # Chat streaming toggle (in generation tab for reference)
        self.chat_stream_var = tk.BooleanVar(value=True)

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

    # ==================================================================
    #  TAB 4 — Quantization
    # ==================================================================
    def _build_quantization_tab(self):
        tab = self.tab_quant

        if not HAS_QUANTIZATION:
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

    # ---- Quantization tab helpers ------------------------------------

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
        if not HAS_QUANTIZATION:
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
                            text=f"Status: Quantization FAILED ✗")
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
                self.root.after(0, lambda: messagebox.showerror(
                    "Quantization Error", str(e)))
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
                self.root.after(0, lambda: messagebox.showerror(
                    "Benchmark Error", str(e)))
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
                self.root.after(0, lambda: messagebox.showerror(
                    "Compare Error", str(e)))
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

    # ==================================================================
    #  TAB — Evaluation (NEW v2.7)
    # ==================================================================
    def _build_evaluation_tab(self):
        tab = self.tab_eval

        try:
            from evaluation import HAS_LM_EVAL
        except ImportError:
            HAS_LM_EVAL = False

        if not HAS_LM_EVAL:
            ttk.Label(tab,
                      text="⚠ lm-evaluation-harness not installed.\n"
                           "Install with: pip install lm-eval",
                      font=("Segoe UI", 12, "bold")).pack(pady=40)
            return

        # ---- Task Selection ----
        task_frame = ttk.LabelFrame(tab, text="  📋  Tasks  ", padding="10")
        task_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(task_frame, text="Tasks (comma separated):").pack(anchor=tk.W)
        self.eval_tasks_var = tk.StringVar(value="arc_easy,hellaswag,winogrande")
        ttk.Entry(task_frame, textvariable=self.eval_tasks_var, width=60).pack(fill=tk.X, pady=4)

        # Common tasks
        common_frame = ttk.Frame(task_frame)
        common_frame.pack(fill=tk.X, pady=4)
        for task in ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "gsm8k", "mmlu"]:
            ttk.Button(common_frame, text=task,
                       command=lambda t=task: self._add_eval_task(t)).pack(side=tk.LEFT, padx=2)

        # ---- Options ----
        opt_frame = ttk.LabelFrame(tab, text="  ⚙️  Options  ", padding="8")
        opt_frame.pack(fill=tk.X, pady=(0, 8))

        row = ttk.Frame(opt_frame)
        row.pack(fill=tk.X)

        ttk.Label(row, text="Few-shot:").pack(side=tk.LEFT, padx=4)
        self.eval_fewshot_var = tk.StringVar(value="0")
        ttk.Entry(row, textvariable=self.eval_fewshot_var, width=5).pack(side=tk.LEFT, padx=4)

        ttk.Label(row, text="Batch size:").pack(side=tk.LEFT, padx=(16, 4))
        self.eval_batch_var = tk.StringVar(value="1")
        ttk.Entry(row, textvariable=self.eval_batch_var, width=5).pack(side=tk.LEFT, padx=4)

        ttk.Label(row, text="Limit (optional):").pack(side=tk.LEFT, padx=(16, 4))
        self.eval_limit_var = tk.StringVar(value="")
        ttk.Entry(row, textvariable=self.eval_limit_var, width=8).pack(side=tk.LEFT, padx=4)

        # ---- Run Button ----
        run_frame = ttk.Frame(tab)
        run_frame.pack(fill=tk.X, pady=8)

        self.eval_run_btn = ttk.Button(run_frame, text="🚀 Run Evaluation",
                                       command=self._run_evaluation, state=tk.DISABLED)
        self.eval_run_btn.pack(side=tk.LEFT, padx=4)

        self.eval_save_btn = ttk.Button(run_frame, text="💾 Save Results",
                                        command=self._save_eval_results, state=tk.DISABLED)
        self.eval_save_btn.pack(side=tk.LEFT, padx=4)

        # ---- Results ----
        res_frame = ttk.LabelFrame(tab, text="  📊  Results  ", padding="8")
        res_frame.pack(fill=tk.BOTH, expand=True)

        ysb = ttk.Scrollbar(res_frame, orient=tk.VERTICAL)
        self.eval_result_text = tk.Text(res_frame, font=("Consolas", 10),
                                        yscrollcommand=ysb.set, wrap=tk.WORD)
        ysb.config(command=self.eval_result_text.yview)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        self.eval_result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.eval_status = ttk.Label(tab, text="Ready", style="Sub.TLabel")
        self.eval_status.pack(anchor=tk.W, pady=4)

    def _add_eval_task(self, task: str):
        current = self.eval_tasks_var.get().strip()
        if current:
            self.eval_tasks_var.set(current + "," + task)
        else:
            self.eval_tasks_var.set(task)

    def _run_evaluation(self):
        if not self.engine.model:
            messagebox.showwarning("No Model", "Load or train a model first.")
            return

        tasks = [t.strip() for t in self.eval_tasks_var.get().split(",") if t.strip()]
        if not tasks:
            messagebox.showwarning("No Tasks", "Please specify at least one task.")
            return

        try:
            fewshot = int(self.eval_fewshot_var.get())
            batch_size = int(self.eval_batch_var.get())
            limit = int(self.eval_limit_var.get()) if self.eval_limit_var.get().strip() else None
        except ValueError:
            messagebox.showerror("Invalid Input", "Few-shot, batch size and limit must be integers.")
            return

        self.eval_run_btn.config(state=tk.DISABLED)
        self.eval_result_text.delete("1.0", tk.END)
        self.eval_result_text.insert(tk.END, f"Evaluating on {tasks}...\n\n")
        self.eval_status.config(text="Status: Running evaluation...")

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
                        self.eval_status.config(text="Status: Evaluation complete ✅")

                self.root.after(0, update_ui)

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Evaluation Error", str(e)))
            finally:
                self.root.after(0, lambda: self.eval_run_btn.config(state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()

    def _save_eval_results(self):
        path = filedialog.asksaveasfilename(
            title="Save Evaluation Results",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
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
            messagebox.showinfo("Saved", f"Results saved to {path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    # ==================================================================
    #  TAB 5 — Console
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

    def _toggle_batch_entry(self, *_):
        """Show / hide the batch entry based on the batch mode checkbox."""
        if self.batch_var.get():
            self.batch_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        else:
            self.batch_entry.pack_forget()

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
        elif self.engine.is_gguf_model():
            tok = self.engine.tokenizer
            n_params = m.count_parameters() if hasattr(m, "count_parameters") else 0
            lines.append("Backend         : GGUF / llama.cpp")
            lines.append(f"File            : {getattr(m, 'path', '—')}")
            lines.append(f"Parameters      : {n_params:,}" if n_params else "Parameters      : —")
            lines.append(f"Vocab size      : {self.engine.vocab_size or '—'}")
            lines.append(f"Tokenizer       : {tok.kind if tok else 'gguf'}")
            lines.append(f"max_seq_len     : {getattr(m, 'max_seq_len', '—')}")
            lines.append(f"threads         : {getattr(m, 'n_threads', '—')}")
            lines.append(f"GPU layers      : {getattr(m, 'n_gpu_layers', '—')}")
            lines.append(f"Batch           : {getattr(m, 'n_batch', '—')}")
            lines.append(f"mmap / mlock    : {getattr(m, 'use_mmap', '—')} / {getattr(m, 'use_mlock', '—')}")
            lines.append(f"Chat completion : {'Yes' if getattr(m, 'use_chat_completion', False) else 'No'}")
            lines.append(f"Chat format     : {getattr(m, 'chat_format', None) or 'auto/default'}")
            lines.append("Training        : not supported for .gguf (inference-only)")
            meta = getattr(m, "metadata", {}) or {}
            if meta:
                lines.append("")
                lines.append("GGUF metadata:")
                for k in sorted(meta)[:40]:
                    lines.append(f"  {k} = {meta[k]}")
        elif self.engine.is_hf_model():
            tok = self.engine.tokenizer
            info = self.engine.hf_proxy.get_info() if self.engine.hf_proxy else {}
            lines.append("Backend         : Hugging Face / transformers")
            lines.append(f"Model           : {info.get('model', self.engine.hf_path or '—')}")
            lines.append(f"Parameters      : {m.count_parameters():,}")
            lines.append(f"Trainable       : {m.count_trainable_parameters():,}")
            lines.append(f"Vocab size      : {self.engine.vocab_size or '—'}")
            lines.append(f"Tokenizer       : {type(tok).__name__ if tok else '—'}")
            lines.append(f"max_seq_len     : {info.get('max_seq_len', getattr(m, 'max_seq_len', '—'))}")
            lines.append(f"device          : {info.get('device', self.engine.device)}")
            lines.append(f"Quantized       : {'Yes' if info.get('is_quantized') else 'No'}")
            lines.append(f"LoRA / PEFT     : {'Yes' if info.get('is_peft') else 'No'}")
            if info.get('lora_config'):
                lines.append(f"LoRA config     : {info['lora_config']}")
            lines.append("Training        : use the HF / LoRA fine-tuning tools in the Model tab")
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
                "use_gradient_checkpointing": bool(self.checkpoint_var.get()),
                # NEW: RoPE scaling
                "rope_scaling": {
                    "type": self.rope_type_var.get() if self.rope_type_var.get() != "none" else None,
                    "factor": float(self.rope_factor_var.get()) if self.rope_type_var.get() != "none" else 1.0,
                } if self.rope_type_var.get() != "none" else None,
                # NEW: Multi-GPU (DDP) — v2.3
                "use_ddp": bool(self.ddp_var.get()),
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
                    if HAS_QUANTIZATION:
                        self.root.after(0, self._update_quant_buttons)
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
            min_p = float(self.minp_var.get())
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
                                           repetition_penalty=rep_pen,
                                           min_p=min_p)
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
            min_p = float(self.minp_var.get())
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
                        web_context=web_ctx or None, min_p=min_p)
                else:
                    # Web search only: prepend context, generate once
                    prompt = f"{web_ctx}\n{seed}" if web_ctx else seed
                    full = self.engine.generate(
                        prompt, length, temperature, top_k, top_p,
                        repetition_penalty=rep_pen, min_p=min_p)
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
            min_p = float(self.minp_var.get())
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
                    repetition_penalty=rep_pen, min_p=min_p
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
            min_p = float(self.minp_var.get())
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
                    repetition_penalty=rep_pen, min_p=min_p
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

    # ==================================================================
    #  TAB — Chat (NEW v2.3)
    # ==================================================================
    def _build_chat_tab(self):
        tab = self.tab_chat

        if not HAS_CHAT_SUPPORT:
            ttk.Label(tab,
                      text="⚠ Chat interface not available (chat_interface.py missing).",
                      font=("Segoe UI", 12, "bold")).pack(pady=40)
            return

        # ---- Chat Template Selection ----
        template_frame = ttk.LabelFrame(tab, text="  💬  Chat Template  ", padding="8")
        template_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(template_frame, text="Template:").pack(side=tk.LEFT, padx=4)
        self.chat_template_var = tk.StringVar(value="chatml")
        templates = list(CHAT_TEMPLATES.keys()) if 'CHAT_TEMPLATES' in globals() else ["chatml", "llama2", "mistral", "simple"]
        self.chat_template_combo = ttk.Combobox(
            template_frame, textvariable=self.chat_template_var,
            values=templates, state="readonly", width=18
        )
        self.chat_template_combo.pack(side=tk.LEFT, padx=4)

        self.chat_system_var = tk.StringVar(value="You are a helpful assistant.")
        ttk.Label(template_frame, text="System prompt:").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Entry(template_frame, textvariable=self.chat_system_var,
                  width=40).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # ---- Chat History ----
        history_frame = ttk.LabelFrame(tab, text="  📜  Conversation  ", padding="8")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # Chat display
        chat_display_frame = ttk.Frame(history_frame)
        chat_display_frame.pack(fill=tk.BOTH, expand=True)

        ysb = ttk.Scrollbar(chat_display_frame, orient=tk.VERTICAL)
        self.chat_text = tk.Text(
            chat_display_frame,
            font=("Segoe UI", 11),
            wrap=tk.WORD,
            yscrollcommand=ysb.set,
            state=tk.DISABLED,
            bg="#f8f9fa",
        )
        ysb.config(command=self.chat_text.yview)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Color tags
        self.chat_text.tag_configure("user", foreground="#0066cc", font=("Segoe UI", 11, "bold"))
        self.chat_text.tag_configure("assistant", foreground="#228B22")
        self.chat_text.tag_configure("system", foreground="#666666", font=("Segoe UI", 10, "italic"))

        # Input area
        input_frame = ttk.Frame(history_frame)
        input_frame.pack(fill=tk.X, pady=(8, 0))

        self.chat_input = ttk.Entry(input_frame, font=("Segoe UI", 11))
        self.chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.chat_input.bind("<Return>", lambda e: self._send_chat_message())

        self.chat_send_btn = ttk.Button(input_frame, text="Send", command=self._send_chat_message)
        self.chat_send_btn.pack(side=tk.LEFT, padx=2)

        self.chat_stream_var = tk.BooleanVar(value=True)
        self.chat_stream_cb = ttk.Checkbutton(input_frame, text="Stream",
                                              variable=self.chat_stream_var)
        self.chat_stream_cb.pack(side=tk.LEFT, padx=4)

        self.chat_clear_btn = ttk.Button(input_frame, text="Clear History", command=self._clear_chat_history)
        self.chat_clear_btn.pack(side=tk.LEFT, padx=2)

        # Status
        self.chat_status = ttk.Label(tab, text="Chat ready. Load a model to start.", style="Sub.TLabel")
        self.chat_status.pack(anchor=tk.W, pady=4)

        # Initialize chat history
        self.chat_history = []

    def _send_chat_message(self):
        if not self.engine.model:
            messagebox.showwarning("No Model", "Please load or train a model first.")
            return

        user_msg = self.chat_input.get().strip()
        if not user_msg:
            return

        self.chat_input.delete(0, tk.END)

        # Add user message to history
        self.chat_history.append({"role": "user", "content": user_msg})

        # Update display
        self._append_chat_message("user", user_msg)

        # Generate response
        self.chat_send_btn.config(state=tk.DISABLED)
        self.chat_status.config(text="Generating response...")

        use_stream = bool(getattr(self, 'chat_stream_var', None) and self.chat_stream_var.get())

        def run():
            try:
                template = self.chat_template_var.get()
                system_prompt = self.chat_system_var.get().strip() or None

                if use_stream:
                    # Streaming mode
                    response_parts = []
                    self.root.after(0, lambda: self._begin_chat_stream())
                    for token in self.engine.generate_chat_streaming(
                        self.chat_history,
                        max_new_tokens=256,
                        temperature=0.7,
                        top_k=40,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        chat_template=template,
                        system_prompt=system_prompt,
                    ):
                        response_parts.append(token)
                        # Update UI live
                        self.root.after(0, lambda t=token: self._append_chat_token(t))

                    full_response = "".join(response_parts).strip()
                    if full_response:
                        # Replace the streaming tokens with final message
                        self.root.after(0, lambda: self._finalize_chat_response(full_response))
                    else:
                        self.root.after(0, lambda: self._append_chat_message("assistant", "[No response]"))
                else:
                    # Non-streaming
                    response = self.engine.generate_chat(
                        self.chat_history,
                        max_new_tokens=256,
                        temperature=0.7,
                        top_k=40,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        chat_template=template,
                        system_prompt=system_prompt,
                    )

                    if response:
                        self.chat_history.append({"role": "assistant", "content": response})
                        self.root.after(0, lambda: self._append_chat_message("assistant", response))
                    else:
                        self.root.after(0, lambda: self._append_chat_message("assistant", "[No response generated]"))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Chat Error", str(e)))
            finally:
                self.root.after(0, lambda: (
                    self.chat_send_btn.config(state=tk.NORMAL),
                    self.chat_status.config(text="Ready")
                ))

        threading.Thread(target=run, daemon=True).start()

    def _append_chat_token(self, token: str):
        """Append a single token during streaming."""
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, token)
        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)

    def _begin_chat_stream(self):
        """Insert the assistant prefix before streaming tokens arrive."""
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, "🤖 Assistant: ", "assistant")
        self.chat_text.config(state=tk.DISABLED)

    def _finalize_chat_response(self, full_response: str):
        """Clean up after streaming and store the final response."""
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, "\n\n")
        self.chat_text.config(state=tk.DISABLED)

        # Make sure we have the response in history
        if self.chat_history and self.chat_history[-1]["role"] == "assistant":
            self.chat_history[-1]["content"] = full_response
        else:
            self.chat_history.append({"role": "assistant", "content": full_response})

    def _append_chat_message(self, role: str, content: str):
        self.chat_text.config(state=tk.NORMAL)
        if role == "user":
            self.chat_text.insert(tk.END, "👤 You: ", "user")
            self.chat_text.insert(tk.END, content + "\n\n")
        elif role == "assistant":
            self.chat_text.insert(tk.END, "🤖 Assistant: ", "assistant")
            self.chat_text.insert(tk.END, content + "\n\n")
        else:
            self.chat_text.insert(tk.END, f"[{role}] {content}\n\n", role)
        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)

    def _clear_chat_history(self):
        self.chat_history.clear()
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.delete("1.0", tk.END)
        self.chat_text.config(state=tk.DISABLED)
        self.chat_status.config(text="Chat history cleared.")

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
            if HAS_QUANTIZATION:
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

    # ==================================================================
    #  NEW: Hugging Face + LoRA/QLoRA GUI methods
    # ==================================================================

    def _browse_local_hf_model(self):
        """Open folder dialog to select an already downloaded HF model directory."""
        folder = filedialog.askdirectory(
            title="Выбери папку со скачанной моделью Hugging Face (должна содержать config.json)"
        )
        if folder:
            self.hf_model_var.set(folder)
            self.hf_local_only_var.set(True)   # автоматически включаем оффлайн-режим
            self.status_label.config(text=f"Status: Выбрана локальная модель: {os.path.basename(folder)}")

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
                    if HAS_QUANTIZATION:
                        self._update_quant_buttons()

                self.root.after(0, update_ui)

            except HFNotAvailableError as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "HF support missing",
                    f"{e}\n\nRun: pip install -r requirements.txt"
                ))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("HF Load Error", str(e)))
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
        path = filedialog.asksaveasdirectory(title="Choose folder to save LoRA adapter")
        if not path:
            return
        try:
            self.engine.save_hf_lora(path)
            messagebox.showinfo("Saved", f"LoRA adapter saved to:\n{path}")
            self.status_label.config(text=f"Status: LoRA adapter saved ✅")
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

    # ==================================================================
    #  NEW: Hugging Face Hub push/pull GUI handlers (v2.6)
    # ==================================================================

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
                self.root.after(0, lambda: messagebox.showerror("Push Error", str(e)))
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
                    if HAS_QUANTIZATION:
                        self._update_quant_buttons()

                self.root.after(0, update_ui)

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("HF Hub Load Error", str(e)))
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
                self.root.after(0, lambda: messagebox.showerror("Fine-tune Error", str(e)))
            finally:
                self.root.after(0, lambda: self.hf_finetune_btn.config(state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()

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
                "use_gradient_checkpointing": bool(self.checkpoint_var.get()),
                "use_ddp": bool(self.ddp_var.get()),
                "rope_scaling": {
                    "type": self.rope_type_var.get() if self.rope_type_var.get() != "none" else None,
                    "factor": float(self.rope_factor_var.get()) if self.rope_type_var.get() != "none" else 1.0,
                } if self.rope_type_var.get() != "none" else None,
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
            if "use_gradient_checkpointing" in config:
                self.checkpoint_var.set(bool(config["use_gradient_checkpointing"]))
            if "use_ddp" in config:
                self.ddp_var.set(bool(config["use_ddp"]))
            if "rope_scaling" in config and config["rope_scaling"]:
                self.rope_type_var.set(config["rope_scaling"].get("type", "none") or "none")
                self.rope_factor_var.set(str(config["rope_scaling"].get("factor", 1.0)))
            elif "rope_scaling" in config:
                self.rope_type_var.set("none")
                self.rope_factor_var.set("1.0")
            self.status_label.config(
                text=f"Status: Config loaded ✅ ({os.path.basename(path)})")
            messagebox.showinfo("Config Loaded",
                                f"Configuration loaded from:\n{path}")
        except Exception as e:
            messagebox.showerror("Load Config Error", str(e))


# ======================================================================
#  Show/hide batch entry based on checkbox
# ======================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = AIApp(root)
    # Bind batch mode checkbox
    app.batch_var.trace_add("write", app._toggle_batch_entry)
    root.mainloop()
