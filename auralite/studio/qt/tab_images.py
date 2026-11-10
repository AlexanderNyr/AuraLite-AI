"""The Images tab: an image generator of one's own - diffusion, flow matching or a GAN -
trained from scratch on a folder of images (2.26: the classic Image tab).

Every setting of the classic tab (:data:`auralite.studio.imagegen.FIELDS`): the model
type and its presets, the network, the diffusion or flow process, the GAN, the
optimization with its validation share, early stop and autosave, classes from
sub-folders; the setup's stages and the losses (the GAN's discriminator too) while it
trains; samples with each type's samplers, a class and guidance, or the GAN's
truncation; the images and the model saved, a model opened, and the quality measured
against the folder (patch SWD).
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QProgressBar,
                               QPushButton, QRadioButton, QSpinBox, QVBoxLayout, QWidget)

from .. import imagegen
from ..i18n import N_, tr
from ..recipes import SettingsError
from .common import (Job, ask_dir, ask_open, ask_save, compact_spins, confirm, heading, hint,
                     package_note, primary, show_error)
from .widgets import CurveChart

if TYPE_CHECKING:
    from .window import MainWindow

SAVE_FILTERS = N_("PyTorch checkpoint (*.pt);;safetensors, no pickle (*.safetensors)")
OPEN_FILTERS = N_("Image generators (*.pt *.safetensors);;All files (*)")
#: The widest the preview grows (pixels): small images are enlarged up to it.
PREVIEW_WIDTH = 640


def to_qimage(image: Any) -> QImage:
    """A uint8 ``H x W`` or ``H x W x C`` array (1, 3 or 4 channels) as a QImage of its own."""
    import numpy as np
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    arr = np.ascontiguousarray(arr)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    data = arr.tobytes()
    if arr.ndim == 2:
        return QImage(data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    if arr.shape[2] == 4:
        return QImage(data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
    return QImage(data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class ImagesTab(QWidget):
    def __init__(self, window: MainWindow):
        super().__init__()
        self.studio = window
        self.job: Job | None = None
        self.job_kind = ""                       # "train", "generate" or "quality"
        self.image_engine: Any = None            # made on first use
        self.folder = ""
        self.images: list[Any] = []              # the last generated images
        self.unsaved = False                     # a trained generator nobody saved yet
        self.run_type = "diffusion"
        self.losses: list[tuple[float, float]] = []
        self.val_losses: list[tuple[float, float]] = []
        self.d_losses: list[tuple[float, float]] = []
        self._last_progress = 0.0
        self._started = 0.0
        #: "" when Pillow is installed; else what reading image files takes
        self.pillow_note = package_note("PIL")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.addWidget(heading(tr("Images")))
        layout.addWidget(hint(tr(
            "Train your own image generator from scratch - diffusion, flow matching or a GAN - "
            "with no pretrained weights: the architecture, the modern training stack and the "
            "sampling are yours to set. Small sizes train even on a CPU.")))
        self.pillow_label = hint(tr("Training and measuring the quality read image files: "
                                    "{note}. Opening a saved generator and generating work "
                                    "without it.", note=self.pillow_note))
        self.pillow_label.setVisible(bool(self.pillow_note))
        layout.addWidget(self.pillow_label)

        # ---- the model type and the presets -----------------------------------------
        type_box = QGroupBox(tr("Model type"))
        type_layout = QVBoxLayout(type_box)
        type_row = QHBoxLayout()
        self.type_group = QButtonGroup(self)
        self.type_buttons: dict[str, QRadioButton] = {}
        for kind, label in imagegen.MODEL_TYPE_LABELS.items():
            button = QRadioButton(tr(label))
            self.type_group.addButton(button)
            self.type_buttons[kind] = button
            type_row.addWidget(button)
        type_row.addStretch(1)
        type_layout.addLayout(type_row)
        self.type_hint = hint("")
        type_layout.addWidget(self.type_hint)
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel(tr("Preset:")))
        self.preset = QComboBox()
        from auralite.image_engine import IMAGE_PRESETS
        for name in IMAGE_PRESETS:                      # 2.28: shown translated
            self.preset.addItem(tr(name), name)
        self.preset.setCurrentIndex(max(0, self.preset.findData("Small (default)")))
        preset_row.addWidget(self.preset)
        self.preset_button = QPushButton(tr("Apply the preset"))
        preset_row.addWidget(self.preset_button)
        preset_row.addStretch(1)
        type_layout.addLayout(preset_row)
        layout.addWidget(type_box)

        # ---- the settings, section by section -------------------------------------------
        self.fields: dict[str, QWidget] = {}
        self.labels: dict[str, QLabel] = {}
        self.sections: dict[str, QGroupBox] = {}
        grids: dict[str, QGridLayout] = {}
        for section, title in imagegen.SECTIONS:
            box = QGroupBox(tr(title))
            grid = grids[section] = QGridLayout(box)
            grid.setHorizontalSpacing(10)
            own = [f for f in imagegen.FIELDS if f.section == section]
            boxes = [f for f in own if f.kind == "bool"]
            others = [f for f in own if f.kind != "bool"]
            for i, field in enumerate(others):
                row_index, column = divmod(i, 2)
                label = QLabel(tr(field.label))
                label.setToolTip(tr(field.tip) if field.tip else "")
                self.labels[field.key] = label
                grid.addWidget(label, row_index, column * 2)
                grid.addWidget(self._make_field(field), row_index, column * 2 + 1)
            if boxes:
                checks = QHBoxLayout()
                for field in boxes:
                    checks.addWidget(self._make_field(field))
                checks.addStretch(1)
                grid.addLayout(checks, (len(others) + 1) // 2, 0, 1, 4)
            grid.setColumnStretch(4, 1)
            self.sections[section] = box
            layout.addWidget(box)
        grids["network"].addWidget(hint(tr("Channel multipliers and attention resolutions "
                                           "take auto or a comma list (1,2,2 / 16,8).")),
                                   99, 0, 1, 4)

        # ---- training -----------------------------------------------------------------
        train_box = QGroupBox(tr("Train"))
        train_layout = QVBoxLayout(train_box)
        buttons = QHBoxLayout()
        self.folder_button = QPushButton(tr("Choose the image folder…"))
        self.start_button = primary(QPushButton(tr("Start training")))
        self.stop_button = QPushButton(tr("Stop"))
        for button in (self.folder_button, self.start_button, self.stop_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        train_layout.addLayout(buttons)
        self.folder_label = hint(tr("No folder yet: a folder of images (PNG, JPEG, BMP, WebP, "
                                    "GIF, TIFF), one sub-folder per class if the model should "
                                    "learn classes."))
        train_layout.addWidget(self.folder_label)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        train_layout.addWidget(self.bar)
        self.detail = hint("")
        train_layout.addWidget(self.detail)
        self.status = QLabel(tr("Choose a folder of images to train on."))
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        train_layout.addWidget(self.status)
        self.chart = CurveChart(empty_text=tr("The loss appears here while the model trains"),
                                x_label=tr("epoch"))
        train_layout.addWidget(self.chart)
        layout.addWidget(train_box)

        # ---- generating ---------------------------------------------------------------
        gen_box = QGroupBox(tr("Generate"))
        gen_layout = QVBoxLayout(gen_box)
        row1 = QHBoxLayout()
        self.count = QSpinBox()
        self.count.setRange(1, imagegen.MAX_SAMPLES)
        self.count.setValue(4)
        self.sampler = QComboBox()
        self.steps = QSpinBox()
        self.steps.setRange(1, 4000)
        self.steps.setValue(20)
        self.eta = QDoubleSpinBox()
        self.eta.setRange(0.0, 1.0)
        self.eta.setSingleStep(0.1)
        self.eta.setToolTip(tr("0: deterministic; 1: as random as DDPM (DDPM with 0 uses 1)"))
        self.seed = QLineEdit()
        self.seed.setPlaceholderText(tr("random"))
        self.seed.setMaximumWidth(90)
        self.use_ema = QCheckBox(tr("EMA weights"))
        self.use_ema.setChecked(True)
        for text, widget in ((tr("Samples:"), self.count), (tr("Sampler:"), self.sampler),
                             (tr("Steps:"), self.steps), (tr("η:"), self.eta),
                             (tr("Seed:"), self.seed)):
            row1.addWidget(QLabel(text))
            row1.addWidget(widget)
        row1.addWidget(self.use_ema)
        row1.addStretch(1)
        gen_layout.addLayout(row1)
        row2 = QHBoxLayout()
        self.class_choice = QComboBox()
        self.class_choice.addItem(tr(imagegen.ANY_CLASS), imagegen.ANY_CLASS)
        self.guidance = QDoubleSpinBox()
        self.guidance.setRange(0.0, 20.0)
        self.guidance.setSingleStep(0.5)
        self.guidance.setValue(2.0)
        self.guidance.setToolTip(tr("Classifier-free guidance for a class: 1 is plain, more "
                                    "follows the class harder (twice the work)"))
        self.truncation = QDoubleSpinBox()
        self.truncation.setRange(0.0, 10.0)
        self.truncation.setSingleStep(0.1)
        self.truncation.setToolTip(tr("The GAN's truncation: above 0 trades variety for "
                                      "quality; 0 is off"))
        for text, widget in ((tr("Class:"), self.class_choice), (tr("Guidance:"), self.guidance),
                             (tr("Truncation (GAN, 0 = off):"), self.truncation)):
            row2.addWidget(QLabel(text))
            row2.addWidget(widget)
        row2.addStretch(1)
        gen_layout.addLayout(row2)
        compact_spins(gen_box, 90)
        row3 = QHBoxLayout()
        self.generate_button = primary(QPushButton(tr("Generate")))
        self.save_images_button = QPushButton(tr("Save the images…"))
        self.save_model_button = QPushButton(tr("Save the model…"))
        self.open_model_button = QPushButton(tr("Open a model…"))
        self.quality_button = QPushButton(tr("Measure the quality"))
        self.quality_button.setToolTip(tr("Generates up to {n} images and compares them with "
                                          "the folder's (patch SWD: lower is better)",
                                          n=imagegen.QUALITY_IMAGES))
        for button in (self.generate_button, self.save_images_button, self.save_model_button,
                       self.open_model_button, self.quality_button):
            row3.addWidget(button)
        row3.addStretch(1)
        gen_layout.addLayout(row3)
        self.params_label = hint(tr("Parameters: —"))
        gen_layout.addWidget(self.params_label)
        self.preview = QLabel(tr("Generated images appear here."))
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(160)
        gen_layout.addWidget(self.preview)
        layout.addWidget(gen_box)
        layout.addStretch(1)

        for button in self.type_buttons.values():
            button.toggled.connect(lambda _checked: self._type_changed())
        self.preset_button.clicked.connect(self.apply_preset)
        self.folder_button.clicked.connect(self.choose_folder)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.generate_button.clicked.connect(self.generate)
        self.save_images_button.clicked.connect(self.save_images)
        self.save_model_button.clicked.connect(self.save_model)
        self.open_model_button.clicked.connect(self.open_model)
        self.quality_button.clicked.connect(self.measure_quality)
        window.busy_changed.connect(lambda _busy: self._update_buttons())
        self.type_buttons["diffusion"].setChecked(True)
        self._type_changed()

    # ---- the settings ------------------------------------------------------------------
    def _make_field(self, field: imagegen.Field) -> QWidget:
        widget: QWidget
        if field.kind == "bool":
            check = QCheckBox(tr(field.label))
            check.setChecked(bool(field.default))
            widget = check
        elif field.kind == "choice":
            combo = QComboBox()
            combo.addItems(list(field.choices))
            combo.setCurrentText(str(field.default))
            widget = combo
        else:
            edit = QLineEdit(str(field.default))
            edit.setMaximumWidth(120)
            widget = edit
        if field.tip:
            widget.setToolTip(tr(field.tip))
        self.fields[field.key] = widget
        return widget

    def values(self) -> dict[str, Any]:
        """Every field's value: a check box's state, a list's choice, a box's text."""
        out: dict[str, Any] = {}
        for key, widget in self.fields.items():
            if isinstance(widget, QCheckBox):
                out[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                out[key] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                out[key] = widget.text()
        return out

    def set_values(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            widget = self.fields.get(key)
            if widget is None:                       # not a field of this tab
                continue
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))

    def model_type(self) -> str:
        for kind, button in self.type_buttons.items():
            if button.isChecked():
                return kind
        return "diffusion"

    def set_model_type(self, kind: str) -> None:
        button = self.type_buttons.get(kind)
        if button is not None:
            button.setChecked(True)
        self._type_changed()

    def generator_type(self) -> str:
        """The type of the generator in memory (what Generate samples), else the choice."""
        engine = self.image_engine
        if engine is not None and getattr(engine, "model", None) is not None:
            return str(getattr(engine, "model_type", "diffusion") or "diffusion")
        return self.model_type()

    def _type_changed(self) -> None:
        kind = self.model_type()
        self.type_hint.setText(tr(imagegen.MODEL_TYPE_HINTS[kind]))
        for field in imagegen.FIELDS:
            applies = field.applies_to(kind)
            self.fields[field.key].setEnabled(applies)
            label = self.labels.get(field.key)
            if label is not None:
                label.setEnabled(applies)
        self.sections["gan"].setEnabled(kind == "gan")
        if not self.losses:
            self.chart.empty_text = self._empty_chart_text(kind)
            self.chart.update()
        self._refresh_generate_row()

    @staticmethod
    def _empty_chart_text(kind: str) -> str:
        return tr("{title}: the loss appears here while the model trains",
                  title=tr(imagegen.PLOT_TITLES.get(kind, imagegen.PLOT_TITLES["diffusion"])))

    def _refresh_generate_row(self) -> None:
        """The samplers of the generator; steps, η and guidance only where they apply."""
        kind = self.generator_type()
        samplers = imagegen.SAMPLERS_IN_ROW.get(kind, imagegen.IMAGE_SAMPLERS)
        current = self.sampler.currentText()
        self.sampler.blockSignals(True)
        self.sampler.clear()
        self.sampler.addItems(list(samplers))
        if current in samplers:
            self.sampler.setCurrentText(current)
        self.sampler.blockSignals(False)
        sampling = kind != "gan"
        self.sampler.setEnabled(sampling)
        for widget in (self.steps, self.eta, self.guidance):
            widget.setEnabled(sampling)
        self.truncation.setEnabled(kind == "gan")

    def _refresh_classes(self) -> None:
        engine = self.image_engine
        names = list(getattr(engine, "class_names", None) or []) if engine is not None else []
        current = self.class_choice.currentData()
        self.class_choice.clear()
        for choice in imagegen.class_choices(names):
            shown = tr(choice) if choice in (imagegen.ANY_CLASS, imagegen.ALL_CLASSES) else choice
            self.class_choice.addItem(shown, choice)
        index = self.class_choice.findData(current)
        self.class_choice.setCurrentIndex(max(0, index))

    def apply_preset(self) -> None:
        name = str(self.preset.currentData() or self.preset.currentText())
        values = imagegen.preset_values(name)
        if not values:
            return
        kind = str(values.pop("model_type", "") or "")
        self.set_values(values)
        if kind:
            self.set_model_type(kind)
        self.status.setText(tr("Preset «{name}» applied.", name=name))

    # ---- the engine and the buttons ---------------------------------------------------
    def engine(self) -> Any:
        if self.image_engine is None:
            from auralite.image_engine import ImageGenEngine
            self.image_engine = ImageGenEngine()
        return self.image_engine

    def has_model(self) -> bool:
        return self.image_engine is not None and getattr(self.image_engine, "model",
                                                         None) is not None

    def _update_buttons(self) -> None:
        idle = self.job is None
        free = idle and not self.studio.busy
        has_model = self.has_model()
        pillow = not self.pillow_note
        self.folder_button.setEnabled(idle)
        self.start_button.setEnabled(bool(self.folder) and free and pillow)
        self.stop_button.setEnabled(not idle and self.job_kind == "train")
        self.preset_button.setEnabled(idle)
        self.generate_button.setEnabled(has_model and free)
        self.quality_button.setEnabled(has_model and free and pillow)
        self.save_model_button.setEnabled(has_model and idle)
        self.open_model_button.setEnabled(free)
        self.save_images_button.setEnabled(bool(self.images) and idle)

    # ---- the folder ---------------------------------------------------------------------
    def choose_folder(self) -> None:
        folder = ask_dir(self, tr("Choose a folder of training images"), self.folder)
        if folder:
            self.set_folder(folder)

    def set_folder(self, folder: str) -> bool:
        from auralite.image_engine import scan_image_folder
        try:
            count = len(scan_image_folder(folder))
        except OSError:
            count = 0
        if not count:
            show_error(self, SettingsError(tr("there are no images in {folder}: PNG, JPEG, BMP, "
                                              "WebP, GIF, TIFF or PPM files",
                                              folder=folder)))
            return False
        self.folder = folder
        name = os.path.basename(folder.rstrip("/\\")) or folder
        self.folder_label.setText(tr("{n} images · {name}", n=f"{count:,}", name=name))
        self.folder_label.setToolTip(folder)
        self.status.setText(tr("Ready: {n} images found.", n=f"{count:,}"))
        self._update_buttons()
        return True

    # ---- training -------------------------------------------------------------------------
    def start(self) -> None:
        if not self.folder:
            show_error(self, SettingsError(tr("choose the folder of images to train on")))
            return
        if self.pillow_note:
            show_error(self, SettingsError(tr("reading image files needs Pillow: {note}",
                                              note=self.pillow_note)))
            return
        kind = self.model_type()
        try:
            params = imagegen.train_params(self.values(), kind)
        except SettingsError as err:
            show_error(self, err)
            return
        if params["autosave_every"] > 0:
            params["autosave_path"] = os.path.join(self.folder, "image_autosave.pt")
        engine = self.engine()
        continuing = bool(params["continue_training"]) and engine.model is not None
        if not continuing and self.unsaved and not confirm(
                self, tr("Replace the image generator?"),
                tr("A new run replaces the image generator in memory, which is not saved. "
                   "Train a new one anyway?")):
            return
        if not self.studio.begin(tr("training images")):
            return
        self.run_type = str(engine.model_type) if continuing else kind
        self.losses, self.val_losses, self.d_losses = [], [], []
        self.chart.set_series([])
        self.chart.empty_text = self._empty_chart_text(self.run_type)
        self._set_bar(0.0)
        self.detail.setText("")
        note = (tr(" (continuing the {kind} model in memory)",
                   kind=tr(imagegen.MODEL_TYPE_LABELS.get(self.run_type, self.run_type)))
                if continuing and self.run_type != kind else "")
        self.status.setText(tr("Starting…") + note)
        folder = self.folder
        self._last_progress = 0.0

        def work(job: Job) -> Any:
            def callback(epoch: Any, total: Any, loss: Any, val: Any = None,
                         info: dict | None = None) -> None:
                info = dict(info or {})
                if val is not None and info.get("val_loss") is None:
                    info["val_loss"] = val
                now = time.monotonic()
                important = bool(info.get("is_epoch_end")) or info.get("phase") in (
                    "setup", "val")
                if not important and now - self._last_progress < 0.12:
                    return
                self._last_progress = now
                job.progress.emit((epoch, total, loss, info))

            engine.train(folder, params, progress_callback=callback, stop_event=job.stop_event)
            return dict(getattr(engine, "last_train_summary", None) or {})

        self._run("train", work, self._trained, tr("training images"))

    def _run(self, kind: str, work: Any, done: Any, what: str) -> None:
        self.job = Job(f"images: {kind}", work, self)
        self.job_kind = kind
        if kind == "train":
            self.job.progress.connect(self._progress)
        self.job.finished.connect(done)
        self.job.failed.connect(lambda exc: self._failed(exc, what))
        self._started = time.time()
        self._update_buttons()
        self.job.start()

    def _set_bar(self, percent: float | None) -> None:
        """The bar in percent, or (None) busy while a stage cannot tell how far it is."""
        if percent is None:
            self.bar.setRange(0, 0)
            return
        self.bar.setRange(0, 1000)
        self.bar.setValue(int(round(10 * max(0.0, min(100.0, float(percent))))))

    def _progress(self, event: tuple) -> None:
        epoch, total, loss, info = event
        view = imagegen.describe_progress(epoch, total, loss, info)
        if "percent" in view:
            self._set_bar(view["percent"])
        if view.get("detail"):
            self.detail.setText(view["detail"])
        if view.get("status"):
            self.status.setText(view["status"])
        if view["phase"] == "setup" or "loss" not in view:
            return
        x = float(len(self.losses) + 1)
        self.losses.append((x, float(view["loss"])))
        if view.get("val") is not None:
            self.val_losses.append((x, float(view["val"])))
        if view.get("d_loss") is not None:
            self.d_losses.append((x, float(view["d_loss"])))
        self._draw_chart(bool(view.get("swd")))

    def _draw_chart(self, swd: bool = False) -> None:
        if self.d_losses:          # a GAN: the generator and the discriminator
            self.chart.set_series([(tr("G (generator)"), self.losses, []),
                                   (tr("D (discriminator)"), self.d_losses, [])])
        else:                      # validation as a patch SWD is another scale: not drawn
            self.chart.set_series([(tr("training"), self.losses,
                                    [] if swd else self.val_losses)])

    def _trained(self, summary: Any) -> None:
        job, self.job = self.job, None
        self.job_kind = ""
        stopped = job is not None and job.stop_event.is_set()
        summary = summary if isinstance(summary, dict) else {}
        engine = self.image_engine
        if summary.get("stopped_during"):
            text = tr("Stopped while reading the images: the generator in memory is unchanged.")
        elif stopped:
            text = tr("Stopped: the weights so far are kept - generate or save them.")
            self.unsaved = True
        else:
            text = tr("Training complete.")
            line = imagegen.summary_text(summary)
            if line:
                text += " " + line
            self.unsaved = True
        if self.val_losses and self.d_losses:
            best = min(self.val_losses, key=lambda p: p[1])
            text += " " + tr("Best validation SWD {v:.2f} (epoch {error}).", v=best[1], error=int(best[0]))
        self._set_bar(100.0 if not stopped else self.bar.value() / 10.0)
        self.status.setText(text)
        if engine is not None and engine.model is not None:
            self._model_ready()
        self.studio.end(model_dirty=None)
        self._update_buttons()

    def _failed(self, exc: BaseException, what: str) -> None:
        kind, self.job, self.job_kind = self.job_kind, None, ""
        if kind == "train":
            self._set_bar(0.0)
            self.status.setText(tr("The training failed - see the message."))
        self.studio.end(model_dirty=None)
        self._update_buttons()
        self.studio.report_error(exc, what)

    def stop(self) -> None:
        if self.job is not None and self.job_kind == "train":
            self.job.stop_event.set()
            self.status.setText(tr("Stopping…"))

    def _model_ready(self) -> None:
        """After training or opening: the Generate row for the generator in memory."""
        self.params_label.setText(imagegen.parameters_text(self.image_engine))
        self._refresh_classes()
        self._refresh_generate_row()
        self._update_buttons()

    # ---- generating -------------------------------------------------------------------------
    def generation_kwargs(self) -> dict[str, Any]:
        return imagegen.generation_kwargs(
            self.generator_type(), n=self.count.value(), sampler=self.sampler.currentText(),
            steps=self.steps.value(), eta=self.eta.value(), seed=self.seed.text(),
            use_ema=self.use_ema.isChecked(),
            class_choice=str(self.class_choice.currentData() or imagegen.ANY_CLASS),
            guidance=self.guidance.value(), truncation=self.truncation.value())

    def generate(self) -> None:
        if not self.has_model():
            return
        try:
            kwargs = self.generation_kwargs()
        except ValueError:
            show_error(self, SettingsError(tr("the seed is a whole number, or empty for a "
                                              "random one")))
            return
        if not self.studio.begin(tr("generating images")):
            return
        engine = self.image_engine
        self.status.setText(tr("Generating {n} images…", n=kwargs["n"]))
        self._run("generate", lambda job: engine.generate(**kwargs), self._generated,
                  tr("generating images"))

    def _generated(self, images: Any) -> None:
        self.job, self.job_kind = None, ""
        self.images = list(images or [])
        self.show_images(self.images)
        self.status.setText(tr("Generated {n} images in {s:.1f} s.", n=len(self.images),
                               s=time.time() - self._started))
        self.studio.end(model_dirty=None)
        self._update_buttons()

    def show_images(self, images: list[Any]) -> None:
        if not images:
            self.preview.setPixmap(QPixmap())
            self.preview.setText(tr("Generated images appear here."))
            return
        from auralite.image_engine import make_image_grid
        grid = make_image_grid(images, cols=min(len(images), 6))
        picture = QPixmap.fromImage(to_qimage(grid))
        scale = max(1, min(8, PREVIEW_WIDTH // max(1, picture.width())))
        if scale > 1:
            picture = picture.scaled(picture.width() * scale, picture.height() * scale,
                                     Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.FastTransformation)
        self.preview.setPixmap(picture)

    def measure_quality(self) -> None:
        """Patch SWD of generated images against the folder's (the classic Measure Quality)."""
        if not self.has_model():
            return
        if not self.folder:
            show_error(self, SettingsError(tr("choose the folder of real images to compare "
                                              "with (Choose the image folder…)")))
            return
        try:
            kwargs = self.generation_kwargs()
        except ValueError:
            show_error(self, SettingsError(tr("the seed is a whole number, or empty for a "
                                              "random one")))
            return
        for key in ("n", "seed"):
            kwargs.pop(key, None)
        if not self.studio.begin(tr("measuring the image quality")):
            return
        engine, folder = self.image_engine, self.folder
        self.status.setText(tr("Measuring the quality: generating up to {n} images…",
                               n=imagegen.QUALITY_IMAGES))
        self._run("quality",
                  lambda job: engine.evaluate(folder, n=imagegen.QUALITY_IMAGES, **kwargs),
                  self._measured, tr("measuring the image quality"))

    def _measured(self, result: Any) -> None:
        self.job, self.job_kind = None, ""
        self.status.setText(imagegen.quality_text(dict(result or {})))
        self.studio.end(model_dirty=None)
        self._update_buttons()

    # ---- files ------------------------------------------------------------------------------
    def save_images(self) -> None:
        if not self.images:
            return
        folder = ask_dir(self, tr("Save the generated images to…"), self.folder)
        if not folder:
            return
        self.write_images(folder)

    def write_images(self, folder: str) -> list[str]:
        """``sample_000.png``… and, for several, ``grid.png`` in `folder` (Qt writes the
        PNGs: no Pillow needed)."""
        written = []
        try:
            for i, image in enumerate(self.images):
                path = os.path.join(folder, f"sample_{i:03d}.png")
                if not to_qimage(image).save(path, "PNG"):
                    raise OSError(tr("could not write {path}", path=path))
                written.append(path)
            if len(self.images) > 1:
                from auralite.image_engine import make_image_grid
                path = os.path.join(folder, "grid.png")
                if not to_qimage(make_image_grid(self.images)).save(path, "PNG"):
                    raise OSError(tr("could not write {path}", path=path))
                written.append(path)
        except OSError as err:
            show_error(self, err, tr("saving the images"))
            return written
        extra = " + grid.png" if len(self.images) > 1 else ""
        self.status.setText(tr("Saved {n} images{extra} in {folder}.", n=len(self.images),
                               extra=extra, folder=folder))
        return written

    def save_model(self) -> None:
        if not self.has_model():
            return
        path = ask_save(self, tr("Save the image generator"), tr(SAVE_FILTERS), "image_model.pt")
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".pt"
        try:
            self.image_engine.save_model(path)
        except Exception as exc:
            show_error(self, exc, tr("saving the image generator"))
            return
        self.unsaved = False
        self.status.setText(tr("Saved the image generator in {path}.", path=path))

    def open_model(self) -> None:
        path = ask_open(self, tr("Open an image generator"), tr(OPEN_FILTERS))
        if path:
            self.load_model(path)

    def load_model(self, path: str) -> bool:
        if self.unsaved and not confirm(
                self, tr("Replace the image generator?"),
                tr("The image generator in memory is not saved. Open another anyway?")):
            return False
        try:
            cfg = self.engine().load_model(path)
        except Exception as exc:
            show_error(self, exc, tr("opening the image generator"))
            return False
        self.unsaved = False
        self.images = []
        self.show_images([])
        self._model_ready()
        self.status.setText(imagegen.loaded_text(dict(cfg or {}), self.image_engine))
        return True
