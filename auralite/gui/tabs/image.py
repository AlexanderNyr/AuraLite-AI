"""gui/tabs/image.py — ImageTabMixin for the AuraLite GUI.

Extracted from `gui_app.py` in v2.6.5. 2.10 adds the validation split and
early-stopping fields, the val-loss curve, the DPM-Solver++ sampler and
`.safetensors` save/load, and class-conditional training (one sub-folder per
class) with a class picker and classifier-free guidance in the Generate row.
2.14.2 adds the model type (diffusion, flow matching, GAN) with the GAN's
settings, the samplers of each type, the GAN's truncation, a quality
measurement (patch SWD) and the stages of the setup on the progress bar.
"""
from .. import common as gui_common
from ..common import (AMP_DTYPES, FLOW_TIME_SAMPLING, GAN_AUGMENTS, GAN_LOSSES, Figure, FigureCanvasTkAgg, IMAGE_PRESETS, LR_SCHEDULES, NOISE_SCHEDULES, PREDICTION_TYPES, SAMPLERS_BY_TYPE, _fmt_duration, filedialog, messagebox, os, threading, time, tk, ttk, validate_image_params)
from .training import _stage_text

#: Sampler choices offered in the Generate row (engine names), per model type.
IMAGE_SAMPLERS = ("dpm++2m", "ddim", "ddpm")
SAMPLERS_IN_ROW = {"diffusion": IMAGE_SAMPLERS, "flow": ("dpm++2m", "euler", "heun"), "gan": ()}
#: Model types: engine name → label of the radio button, and what it is.
MODEL_TYPE_LABELS = {"diffusion": "Diffusion (DDPM)", "flow": "Flow matching", "gan": "GAN"}
MODEL_TYPE_HINTS = {
    "diffusion": ("Denoising diffusion (DDPM, v-prediction, Min-SNR): the reference — stable, "
                  "good quality; sampling takes 10-50 network calls per image."),
    "flow": ("Rectified flow / flow matching (as in Stable Diffusion 3): a straight path from "
             "noise to the image, logit-normal training times; samples with DPM-Solver++, "
             "Euler or Heun."),
    "gan": ("A modern GAN (R3GAN: relativistic loss with R1 + R2, ResNet without "
            "normalisation, adaptive augmentation, EMA generator): one network call per "
            "image — instant sampling. Uses the GAN settings below."),
}
#: Plot titles per model type.
_PLOT_TITLES = {"diffusion": "Diffusion training loss", "flow": "Flow-matching training loss",
                "gan": "GAN training (G / D loss)"}
#: Class-picker entries besides the class names of a class-conditional model.
_ANY_CLASS, _ALL_CLASSES = "(any)", "(one of each)"
#: Checkpoint file types for the save/load dialogs.
_IMAGE_CKPT_TYPES = [("PyTorch checkpoint", "*.pt"), ("safetensors (no pickle)", "*.safetensors"),
                     ("All files", "*.*")]
#: Images the quality measurement generates (at most; fewer if the folder is smaller).
QUALITY_IMAGES = 256

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
            text=("Train your OWN image generator from scratch — diffusion, flow matching or a "
                  "GAN — no pretrained weights. Full control over the architecture, the modern "
                  "training stack and sampling. Small resolutions train even on CPU."),
            style="Sub.TLabel", wraplength=780, justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=10, pady=(10, 6))

        # ---- Model type (2.14.2) ----
        mf = ttk.LabelFrame(tab, text="  🧬  Model Type  ", padding="8")
        mf.pack(fill=tk.X, pady=(0, 8))
        self.img_model_type_var = tk.StringVar(value="diffusion")
        mrow = ttk.Frame(mf); mrow.pack(fill=tk.X)
        for kind, label in MODEL_TYPE_LABELS.items():
            ttk.Radiobutton(mrow, text=label, value=kind, variable=self.img_model_type_var,
                            command=self._on_image_model_type).pack(side=tk.LEFT, padx=6)
        self.img_model_hint = ttk.Label(mf, text=MODEL_TYPE_HINTS["diffusion"],
                                        style="Sub.TLabel", wraplength=760, justify=tk.LEFT)
        self.img_model_hint.pack(anchor=tk.W, pady=(4, 0))

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
            # flow matching (2.14.2)
            "flow_time_sampling": tk.StringVar(value="logit_normal"),
            # GAN (2.14.2)
            "latent_dim":      tk.StringVar(value="64"),
            "gan_loss":        tk.StringVar(value="rpgan"),
            "gan_lr_g":        tk.StringVar(value="0.001"),
            "gan_lr_d":        tk.StringVar(value="0.001"),
            "gan_gamma":       tk.StringVar(value="0.5"),
            "gan_reg_every":   tk.StringVar(value="1"),
            "gan_augment":     tk.StringVar(value="off"),
            "gan_ada_target":  tk.StringVar(value="0.6"),
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
        # 2.14.2: widgets that only apply to some model types: (widget, enabled state)
        self._img_type_widgets: dict[str, list] = {"diffusion": [], "flow": [], "unet": [],
                                                   "gan": []}

        def _grid_field(parent, r, c, label, var, width=10, combo=None, group=None):
            ttk.Label(parent, text=label).grid(row=r, column=c * 2, sticky=tk.W, padx=6, pady=3)
            if combo is not None:
                w = ttk.Combobox(parent, textvariable=var, values=list(combo),
                                 state="readonly", width=width)
            else:
                w = ttk.Entry(parent, textvariable=var, width=width)
            w.grid(row=r, column=c * 2 + 1, sticky=tk.W, padx=6, pady=3)
            if group is not None:
                self._img_type_widgets[group].append((w, "readonly" if combo is not None
                                                      else "normal"))
            return w

        # ---- Architecture ----
        arch = ttk.LabelFrame(tab, text="  🧱  Architecture (U-Net / GAN networks)  ",
                              padding="10")
        arch.pack(fill=tk.X, pady=(0, 8))
        g = ttk.Frame(arch); g.pack(fill=tk.X)
        _grid_field(g, 0, 0, "Image size (px)", self.img_vars["img_size"], 8,
                    combo=("16", "24", "32", "48", "64", "96", "128"))
        _grid_field(g, 0, 1, "Color", self.img_vars["color"], 10, combo=("RGB", "Grayscale"))
        _grid_field(g, 1, 0, "Base channels", self.img_vars["base_channels"])
        _grid_field(g, 1, 1, "Res-blocks / level", self.img_vars["num_res_blocks"])
        _grid_field(g, 2, 0, "Channel mults", self.img_vars["channel_mults"])
        _grid_field(g, 2, 1, "Attn resolutions", self.img_vars["attn_resolutions"], group="unet")
        _grid_field(g, 3, 0, "Attention heads (0=auto)", self.img_vars["attn_heads"],
                    group="unet")
        _grid_field(g, 3, 1, "Time-emb dim (0=auto)", self.img_vars["time_emb_dim"],
                    group="unet")
        _grid_field(g, 4, 0, "Dropout", self.img_vars["dropout"], group="unet")
        ttk.Label(arch, text="channel mults / attn resolutions accept 'auto' or a comma list "
                             "(e.g. 1,2,4  ·  16,8)", style="Sub.TLabel").pack(anchor=tk.W, pady=(4, 0))

        # ---- Diffusion / flow & modern stack ----
        diff = ttk.LabelFrame(tab, text="  🌫️  Diffusion / Flow & Modern Stack  ", padding="10")
        diff.pack(fill=tk.X, pady=(0, 8))
        g2 = ttk.Frame(diff); g2.pack(fill=tk.X)
        _grid_field(g2, 0, 0, "Diffusion steps (T)", self.img_vars["timesteps"],
                    group="diffusion")
        _grid_field(g2, 0, 1, "Noise schedule", self.img_vars["schedule"], 12,
                    combo=NOISE_SCHEDULES, group="diffusion")
        _grid_field(g2, 1, 0, "Prediction type", self.img_vars["prediction_type"], 12,
                    combo=PREDICTION_TYPES, group="diffusion")
        _grid_field(g2, 1, 1, "Min-SNR γ (0=off)", self.img_vars["min_snr_gamma"],
                    group="diffusion")
        _grid_field(g2, 2, 0, "AMP precision", self.img_vars["amp_dtype"], 12, combo=AMP_DTYPES)
        _grid_field(g2, 2, 1, "EMA decay", self.img_vars["ema_decay"])
        _grid_field(g2, 3, 0, "Flow time sampling", self.img_vars["flow_time_sampling"], 12,
                    combo=FLOW_TIME_SAMPLING, group="flow")
        row = ttk.Frame(diff); row.pack(fill=tk.X, pady=(6, 0))
        ttk.Checkbutton(row, text="EMA weights", variable=self.img_use_ema_var).pack(side=tk.LEFT, padx=4)
        ckpt_box = ttk.Checkbutton(row, text="Gradient checkpointing", variable=self.img_ckpt_var)
        ckpt_box.pack(side=tk.LEFT, padx=8)
        compile_box = ttk.Checkbutton(row, text="torch.compile", variable=self.img_compile_var)
        compile_box.pack(side=tk.LEFT, padx=8)
        self._img_type_widgets["unet"] += [(ckpt_box, "normal"), (compile_box, "normal")]
        ttk.Checkbutton(row, text="Augment (h-flip)", variable=self.img_augment_var).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(row, text="Classes from sub-folders",
                        variable=self.img_classes_var).pack(side=tk.LEFT, padx=8)

        # ---- GAN (2.14.2) ----
        ganf = ttk.LabelFrame(tab, text="  ⚔️  GAN (generator vs discriminator)  ", padding="10")
        ganf.pack(fill=tk.X, pady=(0, 8))
        g4 = ttk.Frame(ganf); g4.pack(fill=tk.X)
        _grid_field(g4, 0, 0, "Latent size", self.img_vars["latent_dim"], group="gan")
        _grid_field(g4, 0, 1, "Loss", self.img_vars["gan_loss"], 14, combo=GAN_LOSSES,
                    group="gan")
        _grid_field(g4, 1, 0, "G learning rate", self.img_vars["gan_lr_g"], group="gan")
        _grid_field(g4, 1, 1, "D learning rate", self.img_vars["gan_lr_d"], group="gan")
        _grid_field(g4, 2, 0, "R1/R2 penalty γ", self.img_vars["gan_gamma"], group="gan")
        _grid_field(g4, 2, 1, "Penalty every N steps", self.img_vars["gan_reg_every"],
                    group="gan")
        _grid_field(g4, 3, 0, "Augmentation", self.img_vars["gan_augment"], 10,
                    combo=GAN_AUGMENTS, group="gan")
        _grid_field(g4, 3, 1, "ADA target", self.img_vars["gan_ada_target"], group="gan")
        ttk.Label(ganf, text="rpgan = relativistic pairing (R3GAN) · augmentation: ada = "
                             "DiffAugment with an adaptive probability (for small folders "
                             "trained long, when the validation SWD rises), diffaug = always · "
                             "the GAN ignores the learning rate, weight decay, grad clip and "
                             "accumulation below",
                  style="Sub.TLabel", wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))

        # ---- Optimization ----
        opt = ttk.LabelFrame(tab, text="  ⚙️  Optimization  ", padding="10")
        opt.pack(fill=tk.X, pady=(0, 8))
        g3 = ttk.Frame(opt); g3.pack(fill=tk.X)
        _grid_field(g3, 0, 0, "Epochs", self.img_vars["epochs"])
        _grid_field(g3, 0, 1, "Batch size", self.img_vars["batch_size"])
        _grid_field(g3, 1, 0, "Learning rate", self.img_vars["lr"], group="unet")
        _grid_field(g3, 1, 1, "LR schedule", self.img_vars["lr_schedule"], 14, combo=LR_SCHEDULES)
        _grid_field(g3, 2, 0, "Warmup steps", self.img_vars["warmup_steps"])
        _grid_field(g3, 2, 1, "Grad accumulation", self.img_vars["accumulation_steps"],
                    group="unet")
        _grid_field(g3, 3, 0, "Weight decay", self.img_vars["weight_decay"], group="unet")
        _grid_field(g3, 3, 1, "Grad clip", self.img_vars["grad_clip"], group="unet")
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
        # One bar: the stages of the setup (decoding the images, ...), then the run.
        self.img_progress_var = tk.DoubleVar()
        self.img_progress_bar = ttk.Progressbar(run, variable=self.img_progress_var, maximum=100)
        self.img_progress_bar.pack(fill=tk.X, pady=4)
        self._img_bar_busy = False
        self.img_progress_detail = tk.StringVar(value="")
        ttk.Label(run, textvariable=self.img_progress_detail, style="Sub.TLabel").pack(fill=tk.X, anchor=tk.W)
        self.img_status_label = ttk.Label(run, text="Status: waiting for a folder…")
        self.img_status_label.pack(pady=(0, 2), anchor=tk.W)
        if gui_common.HAS_MATPLOTLIB:
            self.img_fig = Figure(figsize=(6, 2.2), dpi=80)
            self.img_ax = self.img_fig.add_subplot(111)
            self.img_ax.set_xlabel("Epoch"); self.img_ax.set_ylabel("Loss")
            self.img_ax.set_title(_PLOT_TITLES["diffusion"])
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
        self.img_sampler_combo = ttk.Combobox(grow, textvariable=self.img_sampler_var,
                                              values=IMAGE_SAMPLERS, state="readonly", width=8)
        self.img_sampler_combo.pack(side=tk.LEFT, padx=2)
        ttk.Label(grow, text="Steps").pack(side=tk.LEFT, padx=(10, 2))
        self.img_steps_var = tk.StringVar(value="20")
        self.img_steps_entry = ttk.Entry(grow, textvariable=self.img_steps_var, width=5)
        self.img_steps_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(grow, text="η (stochastic)").pack(side=tk.LEFT, padx=(10, 2))
        self.img_eta_var = tk.StringVar(value="0.0")
        self.img_eta_entry = ttk.Entry(grow, textvariable=self.img_eta_var, width=5)
        self.img_eta_entry.pack(side=tk.LEFT, padx=2)
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
        self.img_guidance_entry = ttk.Entry(crow, textvariable=self.img_guidance_var, width=5)
        self.img_guidance_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(crow, text="Truncation (GAN, 0=off)").pack(side=tk.LEFT, padx=(10, 2))
        self.img_truncation_var = tk.StringVar(value="0")
        self.img_truncation_entry = ttk.Entry(crow, textvariable=self.img_truncation_var,
                                              width=5)
        self.img_truncation_entry.pack(side=tk.LEFT, padx=2)

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
        self.img_quality_btn = ttk.Button(brow, text="📏 Measure Quality", state=tk.DISABLED,
                                          command=self.measure_image_quality)
        self.img_quality_btn.pack(side=tk.LEFT, padx=4)
        self.img_param_label = ttk.Label(gen, text="Parameters: —", style="Sub.TLabel")
        self.img_param_label.pack(anchor=tk.W, pady=(0, 4))
        if gui_common.HAS_MATPLOTLIB:
            self.img_preview_fig = Figure(figsize=(6, 2.6), dpi=80)
            self.img_preview_canvas = FigureCanvasTkAgg(self.img_preview_fig, master=gen)
            self.img_preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        else:
            ttk.Label(gen, text="(Install matplotlib to preview generated images)",
                      style="Sub.TLabel").pack()
        self._on_image_model_type()


    # ---- model type (2.14.2) ---------------------------------------------
    def _image_model_type(self) -> str:
        kind = str(self.img_model_type_var.get() or "diffusion")
        return kind if kind in MODEL_TYPE_LABELS else "diffusion"

    def _generator_type(self) -> str:
        """Type of the model in memory (what the Generate row samples), else the selection."""
        engine = getattr(self, "image_engine", None)
        if engine is not None and getattr(engine, "model", None) is not None:
            return str(getattr(engine, "model_type", "diffusion") or "diffusion")
        return self._image_model_type()

    def _on_image_model_type(self):
        """Enable the settings of the selected model type, disable the others."""
        kind = self._image_model_type()
        self.img_model_hint.config(text=MODEL_TYPE_HINTS[kind])
        active = {"diffusion": ("diffusion", "unet"), "flow": ("flow", "unet"),
                  "gan": ("gan",)}[kind]
        for group, widgets in self._img_type_widgets.items():
            for widget, enabled in widgets:
                try:
                    widget.config(state=enabled if group in active else tk.DISABLED)
                except Exception:
                    pass
        self._refresh_image_generate_row()
        if gui_common.HAS_MATPLOTLIB and not self.image_loss_history:
            self.img_ax.set_title(_PLOT_TITLES[kind])
            self.img_canvas.draw_idle()

    def _refresh_image_generate_row(self):
        """Samplers of the model that generates; steps, η and guidance only where they apply."""
        kind = self._generator_type()
        samplers = SAMPLERS_IN_ROW.get(kind, IMAGE_SAMPLERS)
        combo = getattr(self, "img_sampler_combo", None)
        if combo is None:
            return
        if samplers:
            combo.config(values=samplers, state="readonly")
            if self.img_sampler_var.get() not in samplers:
                self.img_sampler_var.set(samplers[0])
        else:
            combo.config(state=tk.DISABLED)
        sampling = tk.NORMAL if kind != "gan" else tk.DISABLED
        for name in ("img_steps_entry", "img_eta_entry", "img_guidance_entry"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.config(state=sampling)
        truncation = getattr(self, "img_truncation_entry", None)
        if truncation is not None:
            truncation.config(state=tk.NORMAL if kind == "gan" else tk.DISABLED)

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
            elif key == "model_type":
                self.img_model_type_var.set(str(value))
        self._on_image_model_type()
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
        model_type = self._image_model_type()
        try:
            params = {
                "model_type":      model_type,
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
                "flow_time_sampling": self.img_vars["flow_time_sampling"].get(),
                "latent_dim":      int(self.img_vars["latent_dim"].get()),
                "gan_loss":        self.img_vars["gan_loss"].get(),
                "gan_lr_g":        float(self.img_vars["gan_lr_g"].get()),
                "gan_lr_d":        float(self.img_vars["gan_lr_d"].get()),
                "gan_gamma":       float(self.img_vars["gan_gamma"].get()),
                "gan_reg_every":   int(self.img_vars["gan_reg_every"].get()),
                "gan_augment":     self.img_vars["gan_augment"].get(),
                "gan_ada_target":  float(self.img_vars["gan_ada_target"].get()),
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

        # continuing keeps the type of the model in memory (the engine fixes it too)
        engine_model = getattr(self.image_engine, "model", None)
        continuing = params["continue_training"] and engine_model is not None
        run_type = self.image_engine.model_type if continuing else model_type
        self._image_run_type = run_type
        self.image_stop_event.clear()
        self.image_loss_history = []
        self.image_val_history = []
        self.image_d_history = []
        self._set_image_bar(0.0)
        self._img_progress_ui_last_ts = 0.0
        note = (f" (continuing the {MODEL_TYPE_LABELS.get(run_type, run_type)} model in memory)"
                if continuing and run_type != model_type else "")
        self.img_status_label.config(text=f"Status: starting… ⏳{note}")
        self.img_train_btn.config(state=tk.DISABLED)
        self.img_folder_btn.config(state=tk.DISABLED)
        self.img_stop_btn.config(state=tk.NORMAL)
        self.img_gen_btn.config(state=tk.DISABLED)
        self.img_quality_btn.config(state=tk.DISABLED)
        if gui_common.HAS_MATPLOTLIB:
            self.img_ax.clear()
            self.img_ax.set_xlabel("Epoch"); self.img_ax.set_ylabel("Loss")
            self.img_ax.set_title(_PLOT_TITLES.get(run_type, _PLOT_TITLES["diffusion"]))
            self.img_canvas.draw()

        def run():
            try:
                self.image_engine.train(
                    self.image_dir, params,
                    progress_callback=self.update_image_progress,
                    stop_event=self.image_stop_event,
                )
                stopped = self.image_stop_event.is_set()
                summary = getattr(self.image_engine, "last_train_summary", None)
                if isinstance(summary, dict) and summary.get("stopped_during"):
                    msg = ("Status: stopped while decoding the images 🛑 — the model in memory "
                           "is unchanged.")
                elif stopped:
                    msg = "Status: stopped. 🛑 Weights kept — you can generate or save."
                else:
                    msg = "Status: training complete! ✅"
                if not stopped and isinstance(summary, dict) and summary:
                    msg += self._image_summary_text(summary)
                self.root.after(0, lambda m=msg: self.img_status_label.config(text=m))
                if self.image_engine.model is not None:
                    self.image_is_trained = True
                    self.root.after(0, self._image_model_ready)
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror(
                    "Image Train Error", f"Error during training:\n{err}"))
            finally:
                self.root.after(0, self._reset_image_buttons)

        threading.Thread(target=run, daemon=True).start()


    def _image_model_ready(self):
        """After training or loading: enable the Generate row for the model in memory."""
        engine = self.image_engine
        self.img_gen_btn.config(state=tk.NORMAL)
        self.img_save_model_btn.config(state=tk.NORMAL)
        self.img_quality_btn.config(state=tk.NORMAL)
        summary = getattr(engine, "parameter_summary", None)
        text = summary() if callable(summary) else f"{engine.count_parameters():,}"
        kind = str(getattr(engine, "model_type", "diffusion") or "diffusion")
        self.img_param_label.config(
            text=f"Parameters: {text} · {MODEL_TYPE_LABELS.get(kind, kind)}")
        self._refresh_image_classes()
        self._refresh_image_generate_row()


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
            metric = "SWD" if summary.get("val_metric") == "swd" else ""
            parts.append(f"best val {metric + ' ' if metric else ''}"
                         f"{summary['best_val_loss']:.4f} (epoch {summary.get('best_epoch')})")
        if summary.get("stopped_early"):
            parts.append("early stop" + (", best weights restored"
                                         if summary.get("restored_best") else ""))
        if summary.get("augment") == "ada":
            parts.append(f"augmentation p {float(summary['augment_p']):.2f}")
        elif summary.get("augment") == "diffaug":
            parts.append("DiffAugment")
        return (" · " + " · ".join(parts)) if parts else ""


    def _reset_image_buttons(self):
        self.img_train_btn.config(state=tk.NORMAL)
        self.img_folder_btn.config(state=tk.NORMAL)
        self.img_stop_btn.config(state=tk.DISABLED)
        if getattr(self, "_img_bar_busy", False):
            self._set_image_bar(0.0)


    # ---- progress ----------------------------------------------------------
    def _set_image_bar(self, percent) -> None:
        """2.14.2: the one bar — a percentage, or (``None``) a busy animation."""
        bar = getattr(self, "img_progress_bar", None)
        if percent is None:
            if not getattr(self, "_img_bar_busy", False):
                self._img_bar_busy = True
                try:
                    if bar is not None:
                        bar.config(mode="indeterminate")
                        bar.start(15)
                except Exception:
                    pass
            return
        if getattr(self, "_img_bar_busy", False):
            self._img_bar_busy = False
            try:
                if bar is not None:
                    bar.stop()
                    bar.config(mode="determinate")
            except Exception:
                pass
        try:
            self.img_progress_var.set(max(0.0, min(100.0, float(percent))))
        except (TypeError, ValueError):
            pass

    def update_image_progress(self, epoch, total, loss, val=None, info=None):
        info = dict(info or {})
        if val is not None and info.get("val_loss") is None:
            info["val_loss"] = val
        now = time.time()
        important = bool(info.get("is_epoch_end")) or info.get("phase") in ("setup", "val")
        if (not important) and now - self._img_progress_ui_last_ts < 0.12:
            return
        self._img_progress_ui_last_ts = now
        self.root.after(0, self._apply_image_progress, epoch, total, loss, info)


    def _apply_image_progress(self, epoch, total, loss, info):
        phase = str(info.get("phase") or ("epoch_end" if info.get("is_epoch_end") else "train"))
        if phase == "setup":
            # 2.14.2: before the first step the bar follows the stage of the setup
            percent = info.get("stage_percent")
            try:
                percent = None if percent is None else float(percent)
            except (TypeError, ValueError):
                percent = None
            self._set_image_bar(percent)
            text = _stage_text(info) if info.get("stage") else str(info.get("message") or "")
            self.img_progress_detail.set(text or "Preparing…")
            self.img_status_label.config(text=f"Status: {info.get('message') or 'preparing…'}")
            return
        percent = info.get("percent")
        if percent is not None:
            try:
                self._set_image_bar(float(percent))
            except (TypeError, ValueError):
                pass
        if phase == "val":
            self.img_progress_detail.set(_stage_text(info) or "Validation…")
            return
        try:
            loss_f = float(loss)
        except (TypeError, ValueError):
            loss_f = 0.0
        eta = info.get("eta_seconds")
        eta_str = f" · ETA {_fmt_duration(eta)}" if eta else ""
        lr = info.get("lr")
        lr_str = f" · lr {lr:.2e}" if lr else ""
        gan = info.get("d_loss") is not None
        if gan:
            aug = info.get("aug_p")
            aug_str = f" · aug p {float(aug):.2f}" if aug is not None else ""
            loss_str = f"G {loss_f:.4f} · D {float(info['d_loss']):.4f}{aug_str}"
        else:
            loss_str = f"loss {loss_f:.4f}"
        rate = info.get("images_per_second")
        rate_str = f" · {float(rate):.0f} img/s" if rate else ""
        self.img_progress_detail.set(
            f"{info.get('message', '')} · {loss_str}{lr_str}{rate_str}{eta_str}")
        if info.get("is_epoch_end"):
            val = info.get("val_loss")
            swd = info.get("val_metric") == "swd"
            val_str = (f" · val {'SWD ' if swd else ''}{val:.4f}" if val is not None else "")
            head = (f"G {loss_f:.4f} · D {float(info['d_loss']):.4f}" if gan
                    else f"loss {loss_f:.4f}")
            self.img_status_label.config(
                text=f"Status: epoch {epoch}/{total} · {head}{val_str}")
            self.image_loss_history.append(loss_f)
            val_hist = getattr(self, "image_val_history", None)
            if val_hist is None:
                val_hist = self.image_val_history = []
            if val is not None:
                val_hist.append(val)
            d_hist = getattr(self, "image_d_history", None)
            if d_hist is None:
                d_hist = self.image_d_history = []
            if gan:
                d_hist.append(float(info["d_loss"]))
            if gui_common.HAS_MATPLOTLIB:
                self._plot_image_history(swd)

    def _plot_image_history(self, swd: bool = False):
        kind = getattr(self, "_image_run_type", None) or self._image_model_type()
        epochs = range(1, len(self.image_loss_history) + 1)
        val_hist = list(getattr(self, "image_val_history", None) or [])
        d_hist = list(getattr(self, "image_d_history", None) or [])
        self.img_ax.clear()
        self.img_ax.set_xlabel("Epoch"); self.img_ax.set_ylabel("Loss")
        title = _PLOT_TITLES.get(kind, _PLOT_TITLES["diffusion"])
        if d_hist:
            self.img_ax.plot(epochs, self.image_loss_history, marker="o", ms=3, label="G")
            self.img_ax.plot(range(1, len(d_hist) + 1), d_hist, marker="s", ms=3, label="D")
            if val_hist:
                best = min(val_hist)
                title += f" — best val SWD {best:.2f} (epoch {val_hist.index(best) + 1})"
            self.img_ax.legend(loc="upper right", fontsize=8)
        else:
            self.img_ax.plot(epochs, self.image_loss_history, marker="o", ms=3, label="train")
            if val_hist and not swd:
                self.img_ax.plot(range(1, len(val_hist) + 1), val_hist,
                                 marker="s", ms=3, label="val (EMA)")
                self.img_ax.legend(loc="upper right", fontsize=8)
        self.img_ax.set_title(title)
        self.img_ax.grid(True, alpha=0.3)
        self.img_canvas.draw_idle()


    # ---- generation ----------------------------------------------------------
    def _generation_kwargs(self):
        """The Generate row as `ImageGenEngine.generate` keyword arguments (ValueError on junk)."""
        kind = self._generator_type()
        n = max(1, min(36, int(self.img_n_var.get())))
        kwargs = {"n": n, "use_ema": bool(self.img_gen_ema_var.get())}
        seed_txt = self.img_seed_gen_var.get().strip()
        kwargs["seed"] = int(seed_txt) if seed_txt else None
        class_label = self._selected_image_class()
        if kind == "gan":
            kwargs["truncation"] = max(0.0, float(self.img_truncation_var.get() or 0.0))
            if class_label is not None:
                kwargs["class_label"] = class_label
            return kwargs
        steps = max(1, int(self.img_steps_var.get()))
        eta = float(self.img_eta_var.get())
        sampler = self.img_sampler_var.get()
        allowed = SAMPLERS_IN_ROW.get(kind, IMAGE_SAMPLERS)
        if sampler not in allowed and sampler not in SAMPLERS_BY_TYPE.get(kind, ()):
            sampler = allowed[0] if allowed else "ddim"
        guidance = float(self.img_guidance_var.get() or 1.0)
        if sampler == "ddpm" and eta == 0.0:
            eta = 1.0  # ancestral DDPM
        kwargs.update(steps=steps, eta=eta, sampler=sampler)
        if class_label is not None:
            kwargs.update(class_label=class_label, guidance_scale=guidance)
        return kwargs

    def generate_images(self):
        if self.image_engine is None or self.image_engine.model is None:
            return
        try:
            kwargs = self._generation_kwargs()
        except ValueError:
            messagebox.showerror("Params Error", "Enter valid numbers for samples/steps/η/seed.")
            return

        self.img_gen_btn.config(state=tk.DISABLED)
        self.img_status_label.config(text="Status: sampling… 🎨")

        def run():
            try:
                imgs = self.image_engine.generate(**kwargs)
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


    def measure_image_quality(self):
        """2.14.2: patch SWD of generated images against the selected folder's images."""
        engine = self.image_engine
        if engine is None or engine.model is None:
            return
        if not self.image_dir:
            messagebox.showinfo("Measure Quality",
                                "Select the folder of real images to compare with "
                                "(📂 Select Image Folder).")
            return
        try:
            kwargs = self._generation_kwargs()
        except ValueError:
            messagebox.showerror("Params Error", "Enter valid numbers for samples/steps/η/seed.")
            return
        for key in ("n", "seed"):
            kwargs.pop(key, None)
        folder = self.image_dir
        self.img_quality_btn.config(state=tk.DISABLED)
        self.img_status_label.config(
            text=f"Status: measuring quality — generating up to {QUALITY_IMAGES} images… 📏")

        def run():
            try:
                result = engine.evaluate(folder, n=QUALITY_IMAGES, **kwargs)
                floor = result.get("real_floor")
                floor_txt = (f"; two sets of real images score {floor:.1f}"
                             if floor is not None else "")
                levels = " / ".join(f"{k}px {v:.1f}" for k, v in result["levels"].items())
                msg = (f"Status: quality — SWD {result['swd']:.1f} (lower is better{floor_txt}) "
                       f"· {levels} · {result['images']} images · "
                       f"{_fmt_duration(float(result['seconds']))}")
                self.root.after(0, lambda m=msg: self.img_status_label.config(text=m))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror("Quality Error", err))
            finally:
                self.root.after(0, lambda: self.img_quality_btn.config(state=tk.NORMAL))

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
            self._image_model_ready()
            names = list(self.image_engine.class_names)
            classes = f", {len(names)} classes" if names else ""
            kind = str(cfg.get("model_type") or "diffusion")
            if kind == "gan":
                detail = (f"GAN, latent {cfg.get('latent_dim')}, "
                          f"{'with' if self.image_engine.discriminator is not None else 'no'} "
                          f"discriminator")
            elif kind == "flow":
                detail = "flow matching"
            else:
                detail = f"pred={cfg.get('prediction_type')}, {cfg.get('schedule')}"
            self.img_status_label.config(
                text=f"Status: loaded {cfg.get('img_size')}px {cfg.get('channels')}ch model "
                     f"({detail}{classes}).")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
