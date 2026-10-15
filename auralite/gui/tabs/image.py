"""gui/tabs/image.py — ImageTabMixin for the AuraLite GUI.

Extracted from `gui_app.py` in v2.6.5. 2.10 adds the validation split and
early-stopping fields, the val-loss curve, the DPM-Solver++ sampler and
`.safetensors` save/load, and class-conditional training (one sub-folder per
class) with a class picker and classifier-free guidance in the Generate row.
"""
from .. import common as gui_common
from ..common import (AMP_DTYPES, Figure, FigureCanvasTkAgg, IMAGE_PRESETS, LR_SCHEDULES, NOISE_SCHEDULES, PREDICTION_TYPES, _fmt_duration, filedialog, messagebox, os, threading, time, tk, ttk, validate_image_params)

#: Sampler choices offered in the Generate row (engine names).
IMAGE_SAMPLERS = ("dpm++2m", "ddim", "ddpm")
#: Class-picker entries besides the class names of a class-conditional model.
_ANY_CLASS, _ALL_CLASSES = "(any)", "(one of each)"
#: Checkpoint file types for the save/load dialogs.
_IMAGE_CKPT_TYPES = [("PyTorch checkpoint", "*.pt"), ("safetensors (no pickle)", "*.safetensors"),
                     ("All files", "*.*")]

class ImageTabMixin:
    def _build_image_tab(self):
        tab = self.tab_image

        if not gui_common.HAS_IMAGEGEN or not gui_common.HAS_PIL:
            msg = ("The image generator needs Pillow.\n\n"
                   "Install it with:\n    pip install Pillow\n\n"
                   "Then restart AuraLite.")
            ttk.Label(tab, text="🖼️  Image Generator (unavailable)",
                      style="Header.TLabel").pack(anchor=tk.W, padx=10, pady=(12, 4))
            ttk.Label(tab, text=msg, style="Sub.TLabel",
                      justify=tk.LEFT).pack(anchor=tk.W, padx=10)
            return

        ttk.Label(
            tab,
            text=("Train your OWN image generator from scratch (DDPM diffusion, self-attention "
                  "U-Net) — no pretrained weights. Full control over architecture, the modern "
                  "diffusion stack and sampling. Small resolutions train even on CPU."),
            style="Sub.TLabel", wraplength=780, justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=10, pady=(10, 6))

        # ---- Presets ----
        pf = ttk.LabelFrame(tab, text="  📋  Configuration Presets  ", padding="8")
        pf.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(pf, text="Choose a preset:").pack(side=tk.LEFT, padx=4)
        self.img_preset_var = tk.StringVar(value="Small (default)")
        ttk.Combobox(pf, textvariable=self.img_preset_var,
                     values=list(IMAGE_PRESETS.keys()), state="readonly",
                     width=28).pack(side=tk.LEFT, padx=4)
        ttk.Button(pf, text="📥 Apply Preset",
                   command=self._apply_image_preset).pack(side=tk.LEFT, padx=4)

        self.img_vars = {
            # architecture
            "img_size":        tk.StringVar(value="32"),
            "color":           tk.StringVar(value="RGB"),
            "base_channels":   tk.StringVar(value="48"),
            "channel_mults":   tk.StringVar(value="auto"),
            "num_res_blocks":  tk.StringVar(value="2"),
            "attn_resolutions": tk.StringVar(value="auto"),
            "attn_heads":      tk.StringVar(value="0"),
            "time_emb_dim":    tk.StringVar(value="0"),
            "dropout":         tk.StringVar(value="0.0"),
            # diffusion
            "timesteps":       tk.StringVar(value="400"),
            "schedule":        tk.StringVar(value="cosine"),
            "prediction_type": tk.StringVar(value="v"),
            "min_snr_gamma":   tk.StringVar(value="5.0"),
            # optimization
            "epochs":          tk.StringVar(value="80"),
            "batch_size":      tk.StringVar(value="32"),
            "lr":              tk.StringVar(value="0.0006"),
            "weight_decay":    tk.StringVar(value="0.0"),
            "grad_clip":       tk.StringVar(value="1.0"),
            "accumulation_steps": tk.StringVar(value="1"),
            "lr_schedule":     tk.StringVar(value="warmup_cosine"),
            "warmup_steps":    tk.StringVar(value="200"),
            "amp_dtype":       tk.StringVar(value="none"),
            "ema_decay":       tk.StringVar(value="0.999"),
            "seed":            tk.StringVar(value="0"),
            "autosave_every":  tk.StringVar(value="0"),
            # validation (2.10)
            "val_split":       tk.StringVar(value="0.05"),
            "early_stopping_patience": tk.StringVar(value="0"),
        }
        self.img_use_ema_var = tk.BooleanVar(value=True)
        self.img_ckpt_var = tk.BooleanVar(value=False)
        self.img_compile_var = tk.BooleanVar(value=False)
        self.img_augment_var = tk.BooleanVar(value=False)
        self.img_continue_var = tk.BooleanVar(value=False)
        self.img_classes_var = tk.BooleanVar(value=False)

        def _grid_field(parent, r, c, label, var, width=10, combo=None):
            ttk.Label(parent, text=label).grid(row=r, column=c * 2, sticky=tk.W, padx=6, pady=3)
            if combo is not None:
                w = ttk.Combobox(parent, textvariable=var, values=list(combo),
                                 state="readonly", width=width)
            else:
                w = ttk.Entry(parent, textvariable=var, width=width)
            w.grid(row=r, column=c * 2 + 1, sticky=tk.W, padx=6, pady=3)
            return w

        # ---- Architecture ----
        arch = ttk.LabelFrame(tab, text="  🧱  Architecture (U-Net)  ", padding="10")
        arch.pack(fill=tk.X, pady=(0, 8))
        g = ttk.Frame(arch); g.pack(fill=tk.X)
        _grid_field(g, 0, 0, "Image size (px)", self.img_vars["img_size"], 8,
                    combo=("16", "24", "32", "48", "64", "96", "128"))
        _grid_field(g, 0, 1, "Color", self.img_vars["color"], 10, combo=("RGB", "Grayscale"))
        _grid_field(g, 1, 0, "Base channels", self.img_vars["base_channels"])
        _grid_field(g, 1, 1, "Res-blocks / level", self.img_vars["num_res_blocks"])
        _grid_field(g, 2, 0, "Channel mults", self.img_vars["channel_mults"])
        _grid_field(g, 2, 1, "Attn resolutions", self.img_vars["attn_resolutions"])
        _grid_field(g, 3, 0, "Attention heads (0=auto)", self.img_vars["attn_heads"])
        _grid_field(g, 3, 1, "Time-emb dim (0=auto)", self.img_vars["time_emb_dim"])
        _grid_field(g, 4, 0, "Dropout", self.img_vars["dropout"])
        ttk.Label(arch, text="channel mults / attn resolutions accept 'auto' or a comma list "
                             "(e.g. 1,2,4  ·  16,8)", style="Sub.TLabel").pack(anchor=tk.W, pady=(4, 0))

        # ---- Diffusion & modern stack ----
        diff = ttk.LabelFrame(tab, text="  🌫️  Diffusion & Modern Stack  ", padding="10")
        diff.pack(fill=tk.X, pady=(0, 8))
        g2 = ttk.Frame(diff); g2.pack(fill=tk.X)
        _grid_field(g2, 0, 0, "Diffusion steps (T)", self.img_vars["timesteps"])
        _grid_field(g2, 0, 1, "Noise schedule", self.img_vars["schedule"], 12, combo=NOISE_SCHEDULES)
        _grid_field(g2, 1, 0, "Prediction type", self.img_vars["prediction_type"], 12, combo=PREDICTION_TYPES)
        _grid_field(g2, 1, 1, "Min-SNR γ (0=off)", self.img_vars["min_snr_gamma"])
        _grid_field(g2, 2, 0, "AMP precision", self.img_vars["amp_dtype"], 12, combo=AMP_DTYPES)
        _grid_field(g2, 2, 1, "EMA decay", self.img_vars["ema_decay"])
        row = ttk.Frame(diff); row.pack(fill=tk.X, pady=(6, 0))
        ttk.Checkbutton(row, text="EMA weights", variable=self.img_use_ema_var).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(row, text="Gradient checkpointing", variable=self.img_ckpt_var).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(row, text="torch.compile", variable=self.img_compile_var).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(row, text="Augment (h-flip)", variable=self.img_augment_var).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(row, text="Classes from sub-folders",
                        variable=self.img_classes_var).pack(side=tk.LEFT, padx=8)

        # ---- Optimization ----
        opt = ttk.LabelFrame(tab, text="  ⚙️  Optimization  ", padding="10")
        opt.pack(fill=tk.X, pady=(0, 8))
        g3 = ttk.Frame(opt); g3.pack(fill=tk.X)
        _grid_field(g3, 0, 0, "Epochs", self.img_vars["epochs"])
        _grid_field(g3, 0, 1, "Batch size", self.img_vars["batch_size"])
        _grid_field(g3, 1, 0, "Learning rate", self.img_vars["lr"])
        _grid_field(g3, 1, 1, "LR schedule", self.img_vars["lr_schedule"], 14, combo=LR_SCHEDULES)
        _grid_field(g3, 2, 0, "Warmup steps", self.img_vars["warmup_steps"])
        _grid_field(g3, 2, 1, "Grad accumulation", self.img_vars["accumulation_steps"])
        _grid_field(g3, 3, 0, "Weight decay", self.img_vars["weight_decay"])
        _grid_field(g3, 3, 1, "Grad clip", self.img_vars["grad_clip"])
        _grid_field(g3, 4, 0, "Seed", self.img_vars["seed"])
        _grid_field(g3, 4, 1, "Autosave every N epochs", self.img_vars["autosave_every"])
        _grid_field(g3, 5, 0, "Validation split", self.img_vars["val_split"])
        _grid_field(g3, 5, 1, "Early stop (epochs, 0=off)",
                    self.img_vars["early_stopping_patience"])
        row2 = ttk.Frame(opt); row2.pack(fill=tk.X, pady=(6, 0))
        ttk.Checkbutton(row2, text="Continue training current model",
                        variable=self.img_continue_var).pack(side=tk.LEFT, padx=4)

        # ---- Run ----
        run = ttk.LabelFrame(tab, text="  🚀  Train  ", padding="10")
        run.pack(fill=tk.X, pady=(0, 8))
        rr = ttk.Frame(run); rr.pack(fill=tk.X, pady=2)
        self.img_folder_btn = ttk.Button(rr, text="📂 Select Image Folder",
                                         command=self.select_image_folder)
        self.img_folder_btn.pack(side=tk.LEFT, padx=4)
        self.img_train_btn = ttk.Button(rr, text="🚀 Start Training",
                                        command=self.start_image_training, state=tk.DISABLED)
        self.img_train_btn.pack(side=tk.LEFT, padx=4)
        self.img_stop_btn = ttk.Button(rr, text="🛑 Stop",
                                       command=self.stop_image_training, state=tk.DISABLED)
        self.img_stop_btn.pack(side=tk.LEFT, padx=4)
        self.img_folder_label = ttk.Label(run, text="No folder selected", foreground="gray")
        self.img_folder_label.pack(pady=2, anchor=tk.W)
        self.img_progress_var = tk.DoubleVar()
        ttk.Progressbar(run, variable=self.img_progress_var, maximum=100).pack(fill=tk.X, pady=4)
        self.img_progress_detail = tk.StringVar(value="")
        ttk.Label(run, textvariable=self.img_progress_detail, style="Sub.TLabel").pack(fill=tk.X, anchor=tk.W)
        self.img_status_label = ttk.Label(run, text="Status: waiting for a folder…")
        self.img_status_label.pack(pady=(0, 2), anchor=tk.W)
        if gui_common.HAS_MATPLOTLIB:
            self.img_fig = Figure(figsize=(6, 2.2), dpi=80)
            self.img_ax = self.img_fig.add_subplot(111)
            self.img_ax.set_xlabel("Epoch"); self.img_ax.set_ylabel("Loss")
            self.img_ax.set_title("Diffusion training loss")
            self.img_canvas = FigureCanvasTkAgg(self.img_fig, master=run)
            self.img_canvas.get_tk_widget().pack(fill=tk.X, pady=(6, 0))

        # ---- Generate ----
        gen = ttk.LabelFrame(tab, text="  ✨  Generate  ", padding="10")
        gen.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        grow = ttk.Frame(gen); grow.pack(fill=tk.X, pady=2)
        ttk.Label(grow, text="Samples").pack(side=tk.LEFT, padx=(4, 2))
        self.img_n_var = tk.StringVar(value="4")
        ttk.Entry(grow, textvariable=self.img_n_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(grow, text="Sampler").pack(side=tk.LEFT, padx=(10, 2))
        self.img_sampler_var = tk.StringVar(value="dpm++2m")
        ttk.Combobox(grow, textvariable=self.img_sampler_var, values=IMAGE_SAMPLERS,
                     state="readonly", width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(grow, text="Steps").pack(side=tk.LEFT, padx=(10, 2))
        self.img_steps_var = tk.StringVar(value="20")
        ttk.Entry(grow, textvariable=self.img_steps_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(grow, text="η (stochastic)").pack(side=tk.LEFT, padx=(10, 2))
        self.img_eta_var = tk.StringVar(value="0.0")
        ttk.Entry(grow, textvariable=self.img_eta_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(grow, text="Seed").pack(side=tk.LEFT, padx=(10, 2))
        self.img_seed_gen_var = tk.StringVar(value="")
        ttk.Entry(grow, textvariable=self.img_seed_gen_var, width=7).pack(side=tk.LEFT, padx=2)
        self.img_gen_ema_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grow, text="EMA weights", variable=self.img_gen_ema_var).pack(side=tk.LEFT, padx=(10, 2))
        crow = ttk.Frame(gen); crow.pack(fill=tk.X, pady=2)
        ttk.Label(crow, text="Class").pack(side=tk.LEFT, padx=(4, 2))
        self.img_class_var = tk.StringVar(value=_ANY_CLASS)
        self.img_class_combo = ttk.Combobox(crow, textvariable=self.img_class_var,
                                            values=(_ANY_CLASS,), state="readonly", width=16)
        self.img_class_combo.pack(side=tk.LEFT, padx=2)
        ttk.Label(crow, text="Guidance").pack(side=tk.LEFT, padx=(10, 2))
        self.img_guidance_var = tk.StringVar(value="2.0")
        ttk.Entry(crow, textvariable=self.img_guidance_var, width=5).pack(side=tk.LEFT, padx=2)

        brow = ttk.Frame(gen); brow.pack(fill=tk.X, pady=4)
        self.img_gen_btn = ttk.Button(brow, text="✨ Generate", state=tk.DISABLED,
                                      command=self.generate_images)
        self.img_gen_btn.pack(side=tk.LEFT, padx=4)
        self.img_save_samples_btn = ttk.Button(brow, text="💾 Save Samples", state=tk.DISABLED,
                                               command=self.save_generated_images)
        self.img_save_samples_btn.pack(side=tk.LEFT, padx=4)
        self.img_save_model_btn = ttk.Button(brow, text="💾 Save Model", state=tk.DISABLED,
                                             command=self.save_image_model)
        self.img_save_model_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(brow, text="📂 Load Model", command=self.load_image_model).pack(side=tk.LEFT, padx=4)
        self.img_param_label = ttk.Label(gen, text="Parameters: —", style="Sub.TLabel")
        self.img_param_label.pack(anchor=tk.W, pady=(0, 4))
        if gui_common.HAS_MATPLOTLIB:
            self.img_preview_fig = Figure(figsize=(6, 2.6), dpi=80)
            self.img_preview_canvas = FigureCanvasTkAgg(self.img_preview_fig, master=gen)
            self.img_preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        else:
            ttk.Label(gen, text="(Install matplotlib to preview generated images)",
                      style="Sub.TLabel").pack()


    def _apply_image_preset(self):
        preset = IMAGE_PRESETS.get(self.img_preset_var.get())
        if not preset:
            return
        for key, value in preset.items():
            if key in self.img_vars:
                self.img_vars[key].set(str(value))
            elif key == "use_ema":
                self.img_use_ema_var.set(bool(value))
            elif key == "use_gradient_checkpointing":
                self.img_ckpt_var.set(bool(value))
            elif key == "augment_hflip":
                self.img_augment_var.set(bool(value))
        self.img_status_label.config(
            text=f"Status: preset '{self.img_preset_var.get()}' applied ✅")


    @staticmethod
    def _parse_int_list(text):
        """Parse "16,32,64" / "16;32" into a list; None for auto/none/empty.

        v2.6.5: this lived without `self` *and* without `@staticmethod`, so every
        call site (`self._parse_int_list(...)`) passed the instance as the first
        argument and raised TypeError — the image tab could never start training.
        """
        text = (text or "").strip().lower()
        if text in ("", "auto", "none"):
            return None
        return [int(x) for x in text.replace(";", ",").split(",") if x.strip()]


    def select_image_folder(self):
        d = filedialog.askdirectory(title="Select a folder of training images")
        if not d:
            return
        try:
            from auralite.image_engine import ImageFolderDataset
            n = len(ImageFolderDataset._scan(d))
        except Exception:
            n = 0
        if n == 0:
            messagebox.showwarning("No images",
                                   "No supported image files found in that folder.")
            return
        self.image_dir = d
        self.img_folder_label.config(
            text=f"{n} images · {os.path.basename(d.rstrip(os.sep)) or d}", foreground="black")
        self.img_train_btn.config(state=tk.NORMAL)
        self.img_status_label.config(text=f"Status: ready — {n} images found.")


    def stop_image_training(self):
        self.image_stop_event.set()
        self.img_status_label.config(text="Status: stopping… 🛑")


    def start_image_training(self):
        if not self.image_dir or self.image_engine is None:
            return
        try:
            params = {
                "img_size":        int(self.img_vars["img_size"].get()),
                "channels":        1 if self.img_vars["color"].get() == "Grayscale" else 3,
                "base_channels":   int(self.img_vars["base_channels"].get()),
                "channel_mults":   self._parse_int_list(self.img_vars["channel_mults"].get()),
                "num_res_blocks":  int(self.img_vars["num_res_blocks"].get()),
                "attn_resolutions": self._parse_int_list(self.img_vars["attn_resolutions"].get()),
                "attn_heads":      int(self.img_vars["attn_heads"].get()),
                "time_emb_dim":    int(self.img_vars["time_emb_dim"].get()),
                "dropout":         float(self.img_vars["dropout"].get()),
                "timesteps":       int(self.img_vars["timesteps"].get()),
                "schedule":        self.img_vars["schedule"].get(),
                "prediction_type": self.img_vars["prediction_type"].get(),
                "min_snr_gamma":   float(self.img_vars["min_snr_gamma"].get()),
                "epochs":          int(self.img_vars["epochs"].get()),
                "batch_size":      int(self.img_vars["batch_size"].get()),
                "lr":              float(self.img_vars["lr"].get()),
                "weight_decay":    float(self.img_vars["weight_decay"].get()),
                "grad_clip":       float(self.img_vars["grad_clip"].get()),
                "accumulation_steps": int(self.img_vars["accumulation_steps"].get()),
                "lr_schedule":     self.img_vars["lr_schedule"].get(),
                "warmup_steps":    int(self.img_vars["warmup_steps"].get()),
                "amp_dtype":       self.img_vars["amp_dtype"].get(),
                "use_ema":         bool(self.img_use_ema_var.get()),
                "ema_decay":       float(self.img_vars["ema_decay"].get()),
                "use_gradient_checkpointing": bool(self.img_ckpt_var.get()),
                "use_compile":     bool(self.img_compile_var.get()),
                "augment_hflip":   bool(self.img_augment_var.get()),
                "seed":            int(self.img_vars["seed"].get()),
                "continue_training": bool(self.img_continue_var.get()),
                "autosave_every":  int(self.img_vars["autosave_every"].get() or 0),
                "val_split":       float(self.img_vars["val_split"].get() or 0),
                "class_conditional": bool(self.img_classes_var.get()),
                "early_stopping_patience": int(
                    self.img_vars["early_stopping_patience"].get() or 0),
            }
        except ValueError:
            messagebox.showerror("Params Error", "Please enter valid numbers.")
            return

        errors = validate_image_params(params)
        if errors:
            messagebox.showerror("Validation Error", "\n".join(f"• {e}" for e in errors))
            return

        if params["autosave_every"] > 0:
            params["autosave_path"] = os.path.join(self.image_dir, "image_autosave.pt")

        self.image_stop_event.clear()
        self.image_loss_history = []
        self.image_val_history = []
        self.img_progress_var.set(0)
        self._img_progress_ui_last_ts = 0.0
        self.img_status_label.config(text="Status: starting… ⏳")
        self.img_train_btn.config(state=tk.DISABLED)
        self.img_folder_btn.config(state=tk.DISABLED)
        self.img_stop_btn.config(state=tk.NORMAL)
        self.img_gen_btn.config(state=tk.DISABLED)
        if gui_common.HAS_MATPLOTLIB:
            self.img_ax.clear()
            self.img_ax.set_xlabel("Epoch"); self.img_ax.set_ylabel("Loss")
            self.img_ax.set_title("Diffusion training loss")
            self.img_canvas.draw()

        def run():
            try:
                self.image_engine.train(
                    self.image_dir, params,
                    progress_callback=self.update_image_progress,
                    stop_event=self.image_stop_event,
                )
                stopped = self.image_stop_event.is_set()
                msg = ("Status: stopped. 🛑 Weights kept — you can generate or save."
                       if stopped else "Status: training complete! ✅")
                summary = getattr(self.image_engine, "last_train_summary", None)
                if not stopped and isinstance(summary, dict) and summary:
                    msg += self._image_summary_text(summary)
                self.root.after(0, lambda m=msg: self.img_status_label.config(text=m))
                if self.image_engine.model is not None:
                    self.image_is_trained = True
                    n = self.image_engine.count_parameters()
                    self.root.after(0, lambda: self.img_gen_btn.config(state=tk.NORMAL))
                    self.root.after(0, lambda: self.img_save_model_btn.config(state=tk.NORMAL))
                    self.root.after(0, lambda c=n: self.img_param_label.config(
                        text=f"Parameters: {c:,}"))
                    self.root.after(0, self._refresh_image_classes)
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror(
                    "Image Train Error", f"Error during training:\n{err}"))
            finally:
                self.root.after(0, self._reset_image_buttons)

        threading.Thread(target=run, daemon=True).start()


    def _refresh_image_classes(self):
        """Fill the class picker from the engine (a no-op for unconditional models)."""
        names = list(getattr(self.image_engine, "class_names", None) or [])
        values = (_ANY_CLASS, _ALL_CLASSES, *names) if names else (_ANY_CLASS,)
        combo = getattr(self, "img_class_combo", None)
        if combo is not None:
            combo.config(values=values)
        if self.img_class_var.get() not in values:
            self.img_class_var.set(_ANY_CLASS)
        return names

    def _selected_image_class(self):
        choice = self.img_class_var.get()
        if choice in ("", _ANY_CLASS):
            return None
        return "all" if choice == _ALL_CLASSES else choice

    @staticmethod
    def _image_summary_text(summary):
        """One status-line suffix from `ImageGenEngine.last_train_summary`."""
        parts = []
        if summary.get("steps") is not None:
            parts.append(f"{summary['steps']} steps")
        if summary.get("seconds") is not None:
            parts.append(_fmt_duration(float(summary["seconds"])))
        if summary.get("best_val_loss") is not None:
            parts.append(f"best val {summary['best_val_loss']:.4f} "
                         f"(epoch {summary.get('best_epoch')})")
        if summary.get("stopped_early"):
            parts.append("early stop" + (", best weights restored"
                                         if summary.get("restored_best") else ""))
        return (" · " + " · ".join(parts)) if parts else ""


    def _reset_image_buttons(self):
        self.img_train_btn.config(state=tk.NORMAL)
        self.img_folder_btn.config(state=tk.NORMAL)
        self.img_stop_btn.config(state=tk.DISABLED)


    def update_image_progress(self, epoch, total, loss, val=None, info=None):
        info = dict(info or {})
        if val is not None and info.get("val_loss") is None:
            info["val_loss"] = val
        now = time.time()
        important = bool(info.get("is_epoch_end"))
        if (not important) and now - self._img_progress_ui_last_ts < 0.12:
            return
        self._img_progress_ui_last_ts = now
        self.root.after(0, self._apply_image_progress, epoch, total, loss, info)


    def _apply_image_progress(self, epoch, total, loss, info):
        percent = info.get("percent")
        if percent is not None:
            self.img_progress_var.set(max(0, min(100, float(percent))))
        eta = info.get("eta_seconds")
        eta_str = f" · ETA {_fmt_duration(eta)}" if eta else ""
        lr = info.get("lr")
        lr_str = f" · lr {lr:.2e}" if lr else ""
        self.img_progress_detail.set(
            f"{info.get('message', '')} · loss {loss:.4f}{lr_str}{eta_str}")
        if info.get("is_epoch_end"):
            val = info.get("val_loss")
            val_str = f" · val {val:.4f}" if val is not None else ""
            self.img_status_label.config(
                text=f"Status: epoch {epoch}/{total} · loss {loss:.4f}{val_str}")
            self.image_loss_history.append(loss)
            val_hist = getattr(self, "image_val_history", None)
            if val_hist is None:
                val_hist = self.image_val_history = []
            if val is not None:
                val_hist.append(val)
            if gui_common.HAS_MATPLOTLIB:
                self.img_ax.clear()
                self.img_ax.set_xlabel("Epoch"); self.img_ax.set_ylabel("Loss")
                self.img_ax.set_title("Diffusion training loss")
                self.img_ax.plot(range(1, len(self.image_loss_history) + 1),
                                 self.image_loss_history, marker="o", ms=3, label="train")
                if val_hist:
                    self.img_ax.plot(range(1, len(val_hist) + 1), val_hist,
                                     marker="s", ms=3, label="val (EMA)")
                    self.img_ax.legend(loc="upper right", fontsize=8)
                self.img_ax.grid(True, alpha=0.3)
                self.img_canvas.draw_idle()


    def generate_images(self):
        if self.image_engine is None or self.image_engine.model is None:
            return
        try:
            n = max(1, min(36, int(self.img_n_var.get())))
            steps = max(1, int(self.img_steps_var.get()))
            eta = float(self.img_eta_var.get())
            sampler = self.img_sampler_var.get()
            if sampler not in IMAGE_SAMPLERS:
                sampler = "ddim"
            guidance = float(self.img_guidance_var.get() or 1.0)
            class_label = self._selected_image_class()
            if sampler == "ddpm" and eta == 0.0:
                eta = 1.0  # ancestral DDPM
            seed_txt = self.img_seed_gen_var.get().strip()
            seed = int(seed_txt) if seed_txt else None
            use_ema = bool(self.img_gen_ema_var.get())
        except ValueError:
            messagebox.showerror("Params Error", "Enter valid numbers for samples/steps/η/seed.")
            return

        self.img_gen_btn.config(state=tk.DISABLED)
        self.img_status_label.config(text="Status: sampling… 🎨")

        def run():
            try:
                extra = {}
                if class_label is not None:
                    extra = {"class_label": class_label, "guidance_scale": guidance}
                imgs = self.image_engine.generate(n=n, steps=steps, eta=eta,
                                                  seed=seed, use_ema=use_ema,
                                                  sampler=sampler, **extra)
                self._last_generated_images = imgs
                self.root.after(0, lambda: self._show_generated(imgs))
                self.root.after(0, lambda: self.img_save_samples_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.img_status_label.config(
                    text=f"Status: generated {len(imgs)} image(s). ✅"))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror("Generate Error", err))
            finally:
                self.root.after(0, lambda: self.img_gen_btn.config(state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()


    def _show_generated(self, imgs):
        if not gui_common.HAS_MATPLOTLIB:
            return
        import math as _m
        self.img_preview_fig.clear()
        n = len(imgs)
        cols = min(n, 4)
        rows = _m.ceil(n / cols)
        for i, im in enumerate(imgs):
            ax = self.img_preview_fig.add_subplot(rows, cols, i + 1)
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
            ax.axis("off")
        self.img_preview_fig.tight_layout()
        self.img_preview_canvas.draw_idle()


    def save_generated_images(self):
        if not self._last_generated_images:
            return
        d = filedialog.askdirectory(title="Save generated images to…")
        if not d:
            return
        try:
            from PIL import Image as _Image
            for i, im in enumerate(self._last_generated_images):
                _Image.fromarray(im).save(os.path.join(d, f"sample_{i:03d}.png"))
            extra = ""
            if len(self._last_generated_images) > 1:
                from auralite.image_engine import save_image_grid
                save_image_grid(self._last_generated_images, os.path.join(d, "grid.png"))
                extra = " + grid.png"
            self.img_status_label.config(
                text=f"Status: saved {len(self._last_generated_images)} image(s){extra} to {d}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))


    def save_image_model(self):
        if self.image_engine is None or self.image_engine.model is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save image generator", defaultextension=".pt",
            filetypes=_IMAGE_CKPT_TYPES[:2])
        if not path:
            return
        try:
            self.image_engine.save_model(path)
            self.img_status_label.config(text=f"Status: model saved to {path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))


    def load_image_model(self):
        if self.image_engine is None:
            return
        path = filedialog.askopenfilename(
            title="Load image generator", filetypes=_IMAGE_CKPT_TYPES)
        if not path:
            return
        try:
            cfg = self.image_engine.load_model(path)
            self.image_is_trained = True
            self.img_gen_btn.config(state=tk.NORMAL)
            self.img_save_model_btn.config(state=tk.NORMAL)
            self.img_param_label.config(
                text=f"Parameters: {self.image_engine.count_parameters():,}")
            names = self._refresh_image_classes()
            classes = f", {len(names)} classes" if names else ""
            self.img_status_label.config(
                text=f"Status: loaded {cfg.get('img_size')}px {cfg.get('channels')}ch model "
                     f"(pred={cfg.get('prediction_type')}, {cfg.get('schedule')}{classes}).")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

