"""2.14.2 GUI image tab: model types (diffusion / flow matching / GAN), the GAN's
settings and truncation, quality measurement, and the setup stages on the one bar.

tkinter is replaced by the recording doubles of ``tests/_gui_harness.py``; the
image engine behind the buttons is real.
"""
from __future__ import annotations

import os
import sys
import threading

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
pytest.importorskip("tkinter", reason="tkinter not available (headless environment)")
import importlib as _importlib  # noqa: E402

_importlib.import_module("tkinter.filedialog")
_importlib.import_module("tkinter.simpledialog")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._gui_harness import InlineThread, build_gui_app, dialog_double  # noqa: E402


@pytest.fixture()
def app(tmp_path, monkeypatch):
    with build_gui_app(tmp_path, monkeypatch) as instance:
        yield instance


def _folder(tmp_path, count=8, size=16):
    from PIL import Image

    folder = tmp_path / "imgs"
    folder.mkdir(exist_ok=True)
    for index in range(count):
        Image.new("RGB", (size, size), color=(index * 30 % 255, 40, 90)).save(
            folder / f"img{index}.png")
    return folder


def _tiny(app, folder, model_type="gan"):
    app.image_dir = str(folder)
    values = {"img_size": "16", "base_channels": "8", "num_res_blocks": "1", "attn_heads": "2",
              "timesteps": "8", "epochs": "1", "batch_size": "4", "lr": "0.001",
              "ema_decay": "0.9", "amp_dtype": "none", "seed": "0", "latent_dim": "16",
              "val_split": "0.25", "attn_resolutions": "8"}
    for key, value in values.items():
        app.img_vars[key].set(value)
    app.img_model_type_var.set(model_type)
    app._on_image_model_type()


def _status(app) -> str:
    return app.img_status_label.config.call_args.kwargs["text"]


def _state(widget) -> str:
    return widget.config.call_args.kwargs["state"]


def test_model_type_switches_the_settings(app):
    groups = app._img_type_widgets
    assert groups["gan"] and groups["unet"] and groups["diffusion"] and groups["flow"]
    app.img_model_type_var.set("gan")
    app._on_image_model_type()
    assert all(_state(w) == "disabled" for w, _ in groups["unet"] + groups["diffusion"])
    assert all(_state(w) in ("normal", "readonly") for w, _ in groups["gan"])
    assert "GAN" in app.img_model_hint.config.call_args.kwargs["text"]
    assert _state(app.img_sampler_combo) == "disabled"            # nothing to sample with
    assert _state(app.img_truncation_entry) == "normal"
    assert _state(app.img_steps_entry) == "disabled"
    app.img_model_type_var.set("flow")
    app._on_image_model_type()
    assert all(_state(w) == "disabled" for w, _ in groups["gan"] + groups["diffusion"])
    assert all(_state(w) in ("normal", "readonly") for w, _ in groups["flow"] + groups["unet"])
    values = app.img_sampler_combo.config.call_args.kwargs["values"]
    assert values == ("dpm++2m", "euler", "heun") and app.img_sampler_var.get() == "dpm++2m"
    assert _state(app.img_truncation_entry) == "disabled"
    app.img_model_type_var.set("junk")
    assert app._image_model_type() == "diffusion"


def test_gan_presets_select_the_model_type(app):
    app.img_preset_var.set("GAN Tiny (CPU-friendly)")
    app._apply_image_preset()
    assert app.img_model_type_var.get() == "gan"
    assert app.img_vars["gan_lr_d"].get() == "0.001"
    app.img_preset_var.set("Tiny (CPU-friendly)")
    app._apply_image_preset()
    assert app.img_model_type_var.get() == "diffusion"


def test_gan_training_generation_and_quality(app, monkeypatch, tmp_path):
    folder = _folder(tmp_path)
    _tiny(app, folder, "gan")
    app.img_vars["gan_augment"].set("diffaug")
    monkeypatch.setattr(threading, "Thread", InlineThread)
    app.start_image_training()
    engine = app.image_engine
    assert engine.model_type == "gan" and engine.discriminator is not None
    assert engine.config["gan_augment"] == "diffaug" and engine.config["latent_dim"] == 16
    assert "training complete" in _status(app) and "best val SWD" in _status(app)
    param = app.img_param_label.config.call_args.kwargs["text"]
    assert "G " in param and "+ D " in param and "GAN" in param
    assert app.image_d_history and app.image_loss_history
    assert _state(app.img_quality_btn) == "normal"

    seen = {}
    real = engine.generate

    def spy(**kwargs):
        seen.clear()
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(engine, "generate", spy)
    app.img_n_var.set("2")
    app.img_truncation_var.set("1.5")
    app.generate_images()
    assert seen["truncation"] == 1.5 and "steps" not in seen and "sampler" not in seen
    assert len(app._last_generated_images) == 2

    app.measure_image_quality()
    status = _status(app)
    assert "SWD" in status and "lower is better" in status and "real images score" in status
    app.image_dir = ""
    with dialog_double("messagebox.showinfo", None) as info:
        app.measure_image_quality()
    assert info.called


def test_flow_training_offers_the_flow_samplers(app, monkeypatch, tmp_path):
    _tiny(app, _folder(tmp_path), "flow")
    monkeypatch.setattr(threading, "Thread", InlineThread)
    app.start_image_training()
    assert app.image_engine.model_type == "flow"
    app.img_model_type_var.set("diffusion")        # the Generate row follows the model
    app._on_image_model_type()
    values = app.img_sampler_combo.config.call_args.kwargs["values"]
    assert "heun" in values and "ddpm" not in values
    app.img_sampler_var.set("heun")
    app.img_n_var.set("1")
    app.img_steps_var.set("2")
    app.generate_images()
    assert len(app._last_generated_images) == 1
    assert "Flow" in app.img_param_label.config.call_args.kwargs["text"]


def test_setup_stages_run_on_the_one_bar(app):
    app.image_loss_history = []
    app.update_image_progress(0, 3, 0.0, None, {
        "phase": "setup", "stage": "decode", "stage_done": 30, "stage_total": 120,
        "stage_unit": "images", "stage_percent": 25.0,
        "message": "Decoding 120 images (32 px) · 30 / 120 images (25%)"})
    assert app.img_progress_var.get() == pytest.approx(25.0)
    assert app.img_progress_detail.get().startswith("Decoding 120 images")
    assert "Decoding 120 images" in _status(app)
    app.update_image_progress(0, 3, 0.0, None, {"phase": "setup", "stage": "model",
                                                "message": "Building the U-Net…"})
    assert app._img_bar_busy and app.img_progress_bar.start.called
    app._img_progress_ui_last_ts = 0.0
    app.update_image_progress(1, 3, 0.7, None, {
        "phase": "train", "percent": 12.0, "message": "Epoch 1/3 · batch 2/5",
        "g_loss": 0.7, "d_loss": 1.3, "aug_p": 0.25, "images_per_second": 40.0})
    assert not app._img_bar_busy and app.img_progress_var.get() == pytest.approx(12.0)
    detail = app.img_progress_detail.get()
    assert "G 0.7000 · D 1.3000 · aug p 0.25" in detail and "40 img/s" in detail
    app.update_image_progress(1, 3, 0.7, None, {
        "phase": "val", "percent": 33.0, "stage": "val",
        "message": "Validation after epoch 1 (SWD) · generate: 4 / 8 images (50%)"})
    assert app.img_progress_var.get() == pytest.approx(33.0)
    assert "generate: 4 / 8 images" in app.img_progress_detail.get()
    app.update_image_progress(1, 3, 0.7, 12.5, {
        "is_epoch_end": True, "percent": 33.3, "d_loss": 1.2, "val_metric": "swd"})
    assert "val SWD 12.5000" in _status(app) and app.image_d_history == [1.2]


def test_augmentation_is_named_only_when_it_runs(app):
    app.image_loss_history = []
    app._img_progress_ui_last_ts = 0.0
    app.update_image_progress(1, 3, 0.7, None, {
        "phase": "train", "percent": 12.0, "message": "Epoch 1/3 · batch 2/5",
        "g_loss": 0.7, "d_loss": 1.3, "aug_p": None, "images_per_second": 40.0})
    detail = app.img_progress_detail.get()
    assert "G 0.7000 · D 1.3000" in detail and "aug p" not in detail
    text = app._image_summary_text
    assert "augmentation p 0.30" in text({"model_type": "gan", "augment": "ada",
                                          "augment_p": 0.3})
    assert "DiffAugment" in text({"model_type": "gan", "augment": "diffaug", "augment_p": 1.0})
    off = text({"model_type": "gan", "augment": "off", "augment_p": 0.0, "steps": 5})
    assert "5 steps" in off and "augment" not in off.lower()


def test_gan_plot_shows_both_losses_and_the_best_swd(tmp_path, monkeypatch):
    import auralite.gui.tabs.image as image_module
    from tests._gui_harness import FakeFigure, make_mock

    def prepare():
        monkeypatch.setattr(image_module, "Figure", FakeFigure)
        monkeypatch.setattr(image_module, "FigureCanvasTkAgg", make_mock)

    with build_gui_app(tmp_path, monkeypatch, matplotlib=True, prepare=prepare) as app:
        app.img_model_type_var.set("gan")
        app._on_image_model_type()
        assert app.img_ax.title == "GAN training (G / D loss)"
        app.image_loss_history, app.image_val_history, app.image_d_history = [], [], []
        for epoch, (g, d, swd) in enumerate([(0.9, 1.3, 30.0), (0.8, 1.2, 20.0)], start=1):
            app.update_image_progress(epoch, 2, g, swd, {"is_epoch_end": True, "d_loss": d,
                                                         "val_metric": "swd"})
        assert app.img_ax.kinds[-3:] == ["plot", "plot", "legend"]     # G and D
        assert app.image_val_history == [30.0, 20.0]
        assert app.img_ax.title.startswith("GAN training (G / D loss) — best val SWD 20.00")


def test_stop_while_decoding_says_the_model_is_unchanged(app, monkeypatch, tmp_path):
    _tiny(app, _folder(tmp_path), "gan")
    monkeypatch.setattr(threading, "Thread", InlineThread)

    def stopped(*a, **k):
        app.image_engine.last_train_summary = {"stopped_during": "decode", "epochs_run": 0}
        app.image_stop_event.set()
        return []

    monkeypatch.setattr(app.image_engine, "train", stopped)
    app.start_image_training()
    assert "stopped while decoding the images" in _status(app)
    assert app.image_is_trained is False


def test_loading_a_gan_checkpoint(app, monkeypatch, tmp_path):
    from auralite.image_engine import ImageGenEngine

    folder = _folder(tmp_path)
    eng = ImageGenEngine()
    eng.train(str(folder), dict(model_type="gan", img_size=16, base_channels=8, latent_dim=16,
                                epochs=1, batch_size=4, progress_every_seconds=1e9))
    path = tmp_path / "gan.pt"
    eng.save_model(str(path))
    with dialog_double("filedialog.askopenfilename", str(path)):
        app.load_image_model()
    status = _status(app)
    assert "16px 3ch" in status and "GAN, latent 16, with discriminator" in status
    assert _state(app.img_sampler_combo) == "disabled"            # follows the loaded model
    assert app.image_engine.model_type == "gan"


def test_continuing_keeps_the_type_of_the_model_in_memory(app, monkeypatch, tmp_path):
    folder = _folder(tmp_path)
    _tiny(app, folder, "flow")
    monkeypatch.setattr(threading, "Thread", InlineThread)
    app.start_image_training()
    assert app.image_engine.model_type == "flow"
    statuses = []
    real = app.img_status_label.config

    def record(**kwargs):
        statuses.append(kwargs.get("text", ""))
        return real(**kwargs)

    monkeypatch.setattr(app.img_status_label, "config", record)
    app.img_model_type_var.set("gan")
    app.img_continue_var.set(True)
    app.start_image_training()
    assert app.image_engine.model_type == "flow"
    assert any("continuing the Flow matching model in memory" in s for s in statuses)
