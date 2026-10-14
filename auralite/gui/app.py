"""gui/app.py — the AuraLite tkinter application (v2.6.5).

The GUI class is assembled from tab mixins (see `gui/tabs/`); the historical
entry point `python gui_app.py` re-exports `AIApp`/`main` from here.
"""
import multiprocessing

from .tabs.training import TrainTabMixin
from .tabs.image import ImageTabMixin
from .tabs.generation import GenerationTabMixin
from .tabs.model import ModelTabMixin
from .tabs.quantization import QuantizationTabMixin
from .tabs.evaluation import EvaluationTabMixin
from .tabs.export import ExportTabMixin
from .tabs.console import ConsoleTabMixin
from .tabs.chat import ChatTabMixin
from .tabs.result_view import ResultViewMixin
from . import common as gui_common
from . import session as gui_session
from .async_ops import TkAsyncBridge
from .config import load_config
from .i18n import LANGUAGES, available_languages, language_name, set_language, tr
from .common import (AuraLiteEngine, ConsoleRedirector, ImageGenEngine, filedialog, json, messagebox, os, sys, threading, tk, ttk)


def _sanitize_geometry(geometry: str, avail_w: int, avail_h: int) -> str:
    """Return ``"WxH"`` if the stored geometry is usable, else an empty string."""
    if not geometry or not isinstance(geometry, str):
        return ""
    width, _, height = geometry.partition("x")
    try:
        win_w, win_h = int(width), int(height.split("+")[0])
    except (TypeError, ValueError):
        return ""
    if win_w < 400 or win_h < 300:
        return ""
    if avail_w and win_w > avail_w:      # saved on a bigger monitor
        return ""
    if avail_h and win_h > avail_h:
        return ""
    return f"{win_w}x{win_h}"


class AIApp(TrainTabMixin, ImageTabMixin, GenerationTabMixin, ModelTabMixin,
            QuantizationTabMixin, EvaluationTabMixin, ExportTabMixin, ConsoleTabMixin,
            ChatTabMixin, ResultViewMixin):
    """AuraLite desktop app: tabs live in `gui/tabs/*`, shared state here."""

    #: Notebook tab names, in build order — used for titles and for the
    #: session snapshot (`gui/session.py` stores the active tab by name).
    _TAB_NAMES = ("train", "image", "gen", "chat", "model", "quant", "eval",
                  "export", "console")

    def __init__(self, root):
        self.root = root

        # ---- persisted settings (v2.6.5) --------------------------------
        self.gui_config = load_config()
        self.language = set_language(self.gui_config.language)
        self.last_prompt = ""
        self._tab_names = list(self._TAB_NAMES)

        self.root.title("AuraLite AI v2.6.5 — Modern Transformer Edition")
        # Default 800x600, clamped to the physical screen so the window (and its
        # content) never exceeds what the monitor can show. The scrollable tabs
        # handle the rest, so nothing is clipped on 1280x720 / 1280x800 / 1280x1024.
        win_w, win_h = 800, 600
        # Guard the screen-probe calls: tests pass a DummyTk mock without them.
        try:
            avail_w = self.root.winfo_screenwidth()
            avail_h = self.root.winfo_screenheight()
        except AttributeError:
            avail_w = avail_h = 0
        if avail_h and avail_h < win_h + 40:      # taskbar / scale margin
            win_h = max(400, avail_h - 72)
        if avail_w and avail_w < win_w + 40:
            win_w = max(560, avail_w - 72)
        # A geometry saved by a previous run wins, as long as it is a plausible
        # WxH string and fits the current screen.
        saved_geometry = _sanitize_geometry(self.gui_config.geometry, avail_w, avail_h)
        self.root.geometry(saved_geometry or f"{win_w}x{win_h}")
        self.root.minsize(min(560, win_w), min(420, win_h))
        self.root.configure(bg="#f5f6f7")

        # Dark theme toggle (NEW v2.5)
        self.dark_mode = tk.BooleanVar(value=False)

        self.engine = AuraLiteEngine()
        self.is_trained = False
        self.selected_file_path = None
        self.stop_event = threading.Event()
        self.loss_history = []  # [(epoch, train_loss, val_loss), ...]

        # ---- Image generator (from-scratch diffusion) state ----
        self.image_engine = ImageGenEngine() if gui_common.HAS_IMAGEGEN else None
        self.image_dir = None
        self.image_is_trained = False
        self.image_stop_event = threading.Event()
        self.image_loss_history: list[float] = []
        self._img_progress_ui_last_ts: float = 0.0
        self._last_generated_images: list = []

        # ETA tracking
        self.train_start_time: float | None = None
        self.epoch_times: list[float] = []   # seconds per completed epoch
        self._progress_ui_last_ts: float = 0.0  # throttle Tk updates from train thread
        self._progress_ui_pending = None        # latest skipped payload
        self._last_epoch_ts: float | None = None

        # ---- Styles ----------------------------------------------------
        self._apply_theme(light=True)

        # ---- Main container ---------------------------------------------
        main_frame = ttk.Frame(root, padding="12")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # The old big top banner was removed; its info now lives in a footer
        # at the bottom of the window. Packed FIRST with side=BOTTOM so it
        # stays pinned below the (expanding) notebook.
        if self.engine.device.type == "cuda":
            dev = "GPU: CUDA 🟢"
        else:
            dev = f"CPU: {self.engine.num_threads} threads"
        info_row = ttk.Frame(main_frame)
        info_row.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))
        ttk.Label(info_row, text="🌟 AuraLite AI — Modern Edition",
                  style="Sub.TLabel").pack(side=tk.LEFT, padx=8)
        self.device_label = ttk.Label(info_row, text=f"Hardware: {dev}",
                                      style="Sub.TLabel")
        self.device_label.pack(side=tk.LEFT, padx=8)
        self.param_label = ttk.Label(info_row, text="Parameters: —",
                                     style="Sub.TLabel")
        self.param_label.pack(side=tk.LEFT, padx=8)
        # Dark theme toggle (moved down together with the rest of the header).
        theme_btn = ttk.Checkbutton(info_row, text="🌙 Dark", variable=self.dark_mode,
                                    command=self._toggle_dark_mode)
        theme_btn.pack(side=tk.RIGHT, padx=8)

        # ==================================================================
        #  Notebook — tabs
        # ==================================================================
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Each tab is wrapped in a scrollable canvas so its content is never
        # clipped on small screens. self.tab_* stays the inner frame, so the
        # existing _build_*_tab() methods keep working unchanged.
        self._scroll_canvases = []  # keep refs so we can theme their background
        self._tab_widgets: dict[str, object] = {}
        for _name in self._tab_names:
            container, canvas, inner = self._make_scrollable_tab()
            setattr(self, f"tab_{_name}", inner)
            self._scroll_canvases.append((container, canvas, inner))
            self.notebook.add(container, text=tr(f"tab.{_name}"))
            self._tab_widgets[_name] = container
        self._active_tab_name = self._tab_names[0]

        self._build_training_tab()
        self._build_image_tab()
        self._build_generation_tab()
        self._build_chat_tab()
        self._build_model_tab()
        self._build_quantization_tab()
        self._build_evaluation_tab()
        self._build_export_tab()
        self._build_console_tab()

        # Language selector (v2.6.5): tabs are re-labelled in place, the choice
        # is remembered in gui_config.json. Shown only when the catalogue has
        # more than one language.
        try:
            self.language_var = tk.StringVar(value=language_name(self.language))
            if len(available_languages()) < 2:
                raise LookupError("single-language catalogue")
            lang_menu = ttk.OptionMenu(info_row, self.language_var,
                                       language_name(self.language),
                                       *[language_name(code) for code in available_languages()],
                                       command=self._on_language_change)
            lang_menu.pack(side=tk.RIGHT, padx=8)
            ttk.Label(info_row, text=tr("btn.language"), style="Sub.TLabel").pack(side=tk.RIGHT)
        except Exception as exc:      # one language, or an exotic Tk build
            gui_common.logger.debug("language selector not shown: %s", exc)

        # Hook stdout/stderr into the console tab. Do it AFTER the widget
        # exists. Keep references so we can restore them on shutdown.
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = ConsoleRedirector(self.console_text, self.root, self._orig_stdout)
        sys.stderr = ConsoleRedirector(self.console_text, self.root, self._orig_stderr)
        print(f"[AuraLite] Console attached. Device: {self.engine.device}, "
              f"threads: {self.engine.num_threads}")

        # Background-work bridge (training/generation callbacks on the Tk thread).
        try:
            self.bridge = TkAsyncBridge(self.root)
        except Exception as exc:      # pragma: no cover - defensive
            gui_common.logger.debug("async bridge unavailable: %s", exc)
            self.bridge = None

        # Restore the previous session (last corpus, active tab, …).
        self._restore_session()

        # Restore original streams on window close.
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)


    def _make_scrollable_tab(self):
        """Build a (container, canvas, inner) triple.

        The container is what the notebook holds; the canvas + vertical
        scrollbar provide scrolling so tall content is never clipped on
        720p-style screens; the inner frame is where the tab's widgets live
        (exposed as self.tab_<name> for the existing _build_*_tab methods).
        """
        container = ttk.Frame(self.notebook)
        canvas = tk.Canvas(container, highlightthickness=0, bd=0, bg="#f5f6f7")
        vsb = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)

        inner = ttk.Frame(canvas, padding="12")
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        canvas.configure(yscrollcommand=vsb.set)

        def _on_canvas_configure(event):
            # Inner frame always matches the canvas width.
            canvas.itemconfigure(inner_id, width=event.width)
            # If content is shorter than the viewport, stretch inner to fill.
            req_h = inner.winfo_reqheight()
            if req_h < event.height:
                canvas.itemconfigure(inner_id, height=event.height)
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_inner_configure(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        canvas.bind("<Configure>", _on_canvas_configure)
        inner.bind("<Configure>", _on_inner_configure)

        def _on_mousewheel(event):
            if event.num == 4:            # Linux/Windows wheel up
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:          # Linux/Windows wheel down
                canvas.yview_scroll(1, "units")
            else:                          # Windows <MouseWheel>
                canvas.yview_scroll(-1 * int(event.delta / 120), "units")

        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Button-4>", _on_mousewheel)
        canvas.bind("<Button-5>", _on_mousewheel)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        return container, canvas, inner


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

        # Scrollable tab canvases follow the theme background.
        for _container, _canvas, _inner in getattr(self, "_scroll_canvases", []):
            try:
                _canvas.configure(bg=bg)
            except tk.TclError:
                pass

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
        for widget_name in ["result_text", "model_info", "q_result_text", "console_text", "chat_text", "loss_text", "exp_log_text", "eval_result_text"]:
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


    # ---- language / session (v2.6.5) ---------------------------------
    def _on_language_change(self, display_name):
        """Switch the UI language and re-label the notebook tabs in place."""
        code = next((c for c, name in LANGUAGES.items() if name == display_name), "en")
        self.language = set_language(code)
        self.gui_config = self.gui_config.update(language=self.language)
        for name, widget in getattr(self, "_tab_widgets", {}).items():
            try:
                self.notebook.tab(widget, text=tr(f"tab.{name}"))
            except Exception:      # tab already gone / mocked notebook
                pass
        self._save_session(quiet=True)

    def _restore_session(self):
        """Apply the last session snapshot, if the user left one behind."""
        if not getattr(self.gui_config, "restore_session", True):
            return []
        try:
            restored = gui_session.restore_to_app(self)
        except Exception as exc:      # pragma: no cover - defensive
            gui_common.logger.warning("session restore failed: %s", exc)
            return []
        if restored:
            print(tr("msg.session_restored", what=", ".join(restored)))
        return restored

    def _save_session(self, quiet: bool = False):
        """Persist paths/choices so the next start feels continuous."""
        if not getattr(self.gui_config, "autosave_session", True):
            return None
        try:
            state = gui_session.snapshot_from_app(self)
            path = gui_session.save_session(state)
        except Exception as exc:
            gui_common.logger.warning("session save failed: %s", exc)
            return None
        if not quiet:
            print(tr("msg.session_saved", path=path))
        return path

    def _on_close(self):
        """Persist the session, restore stdout/stderr and close the window."""
        # Remember the window size for the next run before Tk goes away.
        try:
            geometry = self.root.geometry()
            if isinstance(geometry, str) and "x" in geometry:
                self.gui_config = self.gui_config.update(geometry=geometry.split("+")[0])
        except Exception:
            pass
        self._save_session()
        try:
            self.gui_config.save()
        except Exception as exc:      # pragma: no cover - read-only home
            gui_common.logger.warning("could not save GUI config: %s", exc)
        bridge = getattr(self, "bridge", None)
        if bridge is not None:
            try:
                bridge.shutdown()
            except Exception:         # pragma: no cover - defensive
                pass
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
                "use_qk_norm": bool(self.qk_norm_var.get()),
                "optimizer": self.opt_var.get(),
                "muon_lr": float(self.muon_lr_var.get()),
                "lr_schedule": self.sched_var.get(),
                "tie_word_embeddings": bool(self.tie_var.get()),
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
            if "use_qk_norm" in config:
                self.qk_norm_var.set(bool(config["use_qk_norm"]))
            if "optimizer" in config:
                self.opt_var.set(config["optimizer"] if config["optimizer"] in ("muon", "adamw") else "muon")
            if "muon_lr" in config:
                self.muon_lr_var.set(str(config["muon_lr"]))
            if "lr_schedule" in config:
                self.sched_var.set(config["lr_schedule"] if config["lr_schedule"] in ("wsd", "cosine") else "wsd")
            if "tie_word_embeddings" in config:
                self.tie_var.set(bool(config["tie_word_embeddings"]))
            self.status_label.config(
                text=f"Status: Config loaded ✅ ({os.path.basename(path)})")
            messagebox.showinfo("Config Loaded",
                                f"Configuration loaded from:\n{path}")
        except Exception as e:
            messagebox.showerror("Load Config Error", str(e))


def main():
    """Launch the tkinter application (python gui_app.py)."""
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = AIApp(root)
    # Bind batch mode checkbox
    app.batch_var.trace_add("write", app._toggle_batch_entry)
    root.mainloop()


if __name__ == "__main__":
    main()
