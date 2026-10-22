"""gui/common.py — shared imports, constants and helpers for the GUI (v2.6.5).

Split out of the historical 4.5k-line `gui_app.py` monolith. This module is
the import hub for the tab mixins: it keeps the original module-level
definitions (optional-dependency probes, presets, `ConsoleRedirector`,
`_fmt_duration`) so the mixins stay tiny and consistent.
"""
# ruff: noqa: F401  — this module intentionally re-hosts the shared import hub
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from auralite.engine import (
    AuraLiteEngine, validate_params, ParamValidationError,
    estimate_n_params, recommend_epochs, recommend_gen_length,
    GGUFNotAvailableError,
    HFNotAvailableError,
    HAS_CHAT_SUPPORT,
    CHAT_TEMPLATES,
)
# 2.12: the training tab's defaults are the engine's recipe.
from auralite.engine.recipe import DEFAULT_RECIPE, parse_dropout
# 2.14: training on several computers (the training tab's Cluster row).
from auralite.engine.cluster import lan_address as cluster_lan_address
from auralite.engine.cluster import run_worker as run_cluster_worker
# 2.14.1: your own tokenizer / training only the tokenizer (the training tab).
from auralite.engine.tokenizer_io import describe_tokenizer, read_tokenizer_file
from auralite.web_tools import build_web_context
# Agent framework (optional)
try:
    from auralite.agent import AuraLiteAgent, Sandbox
    from auralite.agent.tools import TOOL_REGISTRY, build_system_prompt
    HAS_AGENT = True
except ImportError:
    HAS_AGENT = False
    AuraLiteAgent = None
    Sandbox = None
    TOOL_REGISTRY = {}
    build_system_prompt = lambda *a, **k: ""

try:
    from auralite.export import ModelExporter
except ImportError:
    ModelExporter = None
try:
    from auralite.quantization import (
        QuantizationEngine, QuantConfig, QuantResult,
        QuantMethod, BitWidth,
        METHOD_SUPPORTED_BITS, METHOD_DESCRIPTIONS,
        compare_quantizations, format_comparison_table,
    )
    HAS_QUANTIZATION = True
except ImportError:
    HAS_QUANTIZATION = False
# Image generator (from-scratch diffusion) — optional, needs Pillow.
try:
    from auralite.image_engine import (
        ImageGenEngine, validate_image_params, tensor_to_uint8, HAS_PIL,
        IMAGE_PRESETS, PREDICTION_TYPES, NOISE_SCHEDULES, LR_SCHEDULES, AMP_DTYPES,
        MODEL_TYPES, SAMPLERS_BY_TYPE, GAN_LOSSES, GAN_AUGMENTS, FLOW_TIME_SAMPLING,
    )
    HAS_IMAGEGEN = True
except Exception:
    HAS_IMAGEGEN = False
    HAS_PIL = False
    ImageGenEngine = None
    validate_image_params = lambda *a, **k: []
    IMAGE_PRESETS = {}
    PREDICTION_TYPES = ("eps", "x0", "v")
    NOISE_SCHEDULES = ("linear", "cosine", "sigmoid")
    LR_SCHEDULES = ("constant", "cosine", "warmup_cosine")
    AMP_DTYPES = ("fp16", "bf16", "none")
    # 2.14.2: image model types and their settings
    MODEL_TYPES = ("diffusion", "flow", "gan")
    SAMPLERS_BY_TYPE = {"diffusion": ("ddim", "ddpm", "dpm++2m"),
                        "flow": ("euler", "heun", "dpm++2m"), "gan": ()}
    GAN_LOSSES = ("rpgan", "hinge", "nonsaturating")
    GAN_AUGMENTS = ("ada", "diffaug", "off")
    FLOW_TIME_SAMPLING = ("logit_normal", "uniform")
import threading
import multiprocessing
import os
import sys
import io
import time
import json
import logging

#: Logger used by the GUI modules (`auralite.gui`, inherits the CLI/library setup).
logger = logging.getLogger("auralite.gui")

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
        "dropout": "auto", "grad_clip": 1.0, "n_kv_heads": 0,
        "use_gradient_checkpointing": False,
        "rope_scaling": None,
    },
    "Small (default)": {
        "d_model": 128, "d_ff": 256, "n_heads": 4, "n_layers": 4,
        "seq_length": 64, "batch_size": 32, "lr": 0.001, "epochs": 100,
        "dropout": "auto", "grad_clip": 1.0, "n_kv_heads": 0,
        "use_gradient_checkpointing": False,
        "rope_scaling": None,
    },
    "Medium (GPU recommended)": {
        "d_model": 256, "d_ff": 512, "n_heads": 8, "n_layers": 6,
        "seq_length": 128, "batch_size": 64, "lr": 0.0003, "epochs": 50,
        "dropout": "auto", "grad_clip": 1.0, "n_kv_heads": 0,
        # 2.9: off — recomputing activations made each step ~36 % slower
        # for ~0.9 GB less memory at this size (docs/training.md § 4).
        "use_gradient_checkpointing": False,
        # 2.11: no RoPE scaling — it did not extend the context in any
        # measurement (docs/engine.md § 9); generation keeps the trained length.
        "rope_scaling": None,
    },
    "Large (powerful GPU)": {
        "d_model": 512, "d_ff": 1024, "n_heads": 8, "n_layers": 8,
        "seq_length": 256, "batch_size": 32, "lr": 0.0001, "epochs": 30,
        "dropout": "auto", "grad_clip": 1.0, "n_kv_heads": 0,
        "use_gradient_checkpointing": True,
        "rope_scaling": None,
    },
    "GQA-efficient (Medium)": {
        "d_model": 256, "d_ff": 512, "n_heads": 8, "n_layers": 6,
        "seq_length": 128, "batch_size": 64, "lr": 0.0003, "epochs": 50,
        "dropout": "auto", "grad_clip": 1.0, "n_kv_heads": 2,
        "use_gradient_checkpointing": False,
        "rope_scaling": None,
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


