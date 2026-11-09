"""2.25: the Studio's Model tab with transformers and PEFT for real (no fakes).

A tiny random Llama on disk: loaded through the tab with LoRA adapters, the adapters saved
and loaded back, a fine-tune on a text file and the model described. Locally it skips
without transformers/peft or PySide6; the CI ``integrations`` job requires them.
"""
from __future__ import annotations

import os
import time
from typing import Any

import pytest

from tests._integration import require

torch = require("torch")
require("transformers")
require("peft")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = require("PySide6.QtWidgets")
QtCore = require("PySide6.QtCore")


@pytest.fixture(scope="module")
def tiny_llama_dir(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    path = tmp_path_factory.mktemp("tiny-llama")
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=128)
    LlamaForCausalLM(config).save_pretrained(path)
    vocab = {chr(97 + i): i for i in range(26)}
    vocab.update({"<unk>": 26, "<s>": 27, "</s>": 28, " ": 29})
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", bos_token="<s>",
                            eos_token="</s>", pad_token="</s>").save_pretrained(path)
    return str(path)


@pytest.fixture()
def dialogs():
    from auralite.studio.qt import common
    seen: dict[str, Any] = {"error": [], "inform": [], "confirm": []}
    answers: dict[str, Any] = {"confirm": True, "save": "", "open": "", "dir": "", "text": ""}
    common.DIALOGS.update(
        error=lambda friendly: seen["error"].append(friendly),
        inform=lambda title, text: seen["inform"].append((title, text)),
        confirm=lambda title, text: seen["confirm"].append((title, text)) or answers["confirm"],
        save=lambda title: answers["save"], open=lambda title: answers["open"],
        dir=lambda title: answers["dir"], text=lambda title: answers["text"])
    seen["answers"] = answers
    yield seen
    common.DIALOGS.clear()


@pytest.fixture()
def studio(tmp_path, monkeypatch, dialogs):
    from auralite.engine import AuraLiteEngine
    from auralite.studio.qt.window import MainWindow
    from auralite.studio.settings import StudioSettings
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    window = MainWindow(engine=AuraLiteEngine(), settings=StudioSettings(first_run_done=True),
                        detect=False, capture_streams=False, save_on_close=False,
                        auto_speed_test=False)
    window.show()
    app.processEvents()
    window.qapp = app                                        # type: ignore[attr-defined]
    yield window
    window.model_dirty = False
    for job in window.running_jobs():
        job.stop_event.set()
        if job.thread is not None:
            job.thread.join(timeout=60)
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()


def wait(app, condition, timeout: float = 300.0) -> None:
    deadline = time.time() + timeout
    while not condition():
        if time.time() > deadline:
            raise AssertionError("timed out waiting for the window")
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def _lora_weights(engine) -> dict:
    return {name: tensor.detach().clone() for name, tensor in
            engine.hf_proxy.model.named_parameters() if "lora_" in name}


def test_a_transformers_model_with_adapters_in_the_model_tab(studio, dialogs, tiny_llama_dir,
                                                              tmp_path):
    tab = studio.model_tab
    assert tab.hf_needs.isHidden() and tab.hf_load_button.isEnabled()
    tab.hf_name.setText(tiny_llama_dir)
    tab.local_only.setChecked(True)
    tab.precision["full"].setChecked(True)
    tab.apply_lora.setChecked(True)
    tab.lora_rank.setValue(4)
    tab.load_transformers()
    wait(studio.qapp, lambda: tab.job is None)
    assert [f.message for f in dialogs["error"]] == []
    engine = studio.engine
    assert engine.is_hf_model() and engine.hf_proxy.is_peft
    info = tab.info.toPlainText()
    assert "Backend         : Hugging Face / transformers" in info
    assert "LoRA / PEFT     : Yes" in info
    assert studio.model_description().endswith("with LoRA adapters")
    assert not studio.quantize_tab.quantize_button.isEnabled()   # AuraLite's models only
    assert tab.hf_save_lora_button.isEnabled() and not tab.hf_lora_button.isEnabled()
    # the adapters, saved and loaded back exactly
    saved = _lora_weights(engine)
    assert saved
    adapter = tmp_path / "adapter"
    dialogs["answers"]["dir"] = str(adapter)
    tab.save_lora()
    assert (adapter / "adapter_config.json").is_file()
    with torch.no_grad():
        for name, tensor in engine.hf_proxy.model.named_parameters():
            if "lora_" in name:
                tensor.add_(1.0)
    tab.load_lora()
    loaded = _lora_weights(engine)
    assert loaded.keys() == saved.keys()
    assert all(torch.equal(loaded[name], saved[name]) for name in saved)   # the file's
    # a fine-tune on a text file: the adapters go where asked
    text = tmp_path / "notes.txt"
    text.write_text("\n\n".join(["abc abc abc abc abc abc abc abc abc abc"] * 6),
                    encoding="utf-8")
    dialogs["answers"]["open"] = str(text)
    dialogs["answers"]["dir"] = str(tmp_path / "tuned")
    tab.ft_epochs.setValue(1)
    tab.ft_batch.setValue(2)
    tab.ft_length.setValue(32)
    tab.ft_accumulation.setValue(1)
    tab.finetune()
    wait(studio.qapp, lambda: tab.job is None)
    assert [f.message for f in dialogs["error"]] == []
    assert dialogs["inform"][-1][0] == "Fine-tuning complete"
    assert (tmp_path / "tuned" / "adapter_config.json").is_file()
    engine.model = None
    engine.backend = "torch"
    engine.hf_proxy = None



@pytest.fixture(scope="module")
def tiny_bpe_llama_dir(tmp_path_factory):
    """A tiny Llama with a BPE tokenizer: what AuraLite imports as its own model."""
    import json

    from transformers import LlamaConfig, LlamaForCausalLM

    from tests.test_hf_v2181 import tiny_tokenizer_json
    path = tmp_path_factory.mktemp("tiny-bpe-llama")
    torch.manual_seed(0)
    LlamaForCausalLM(LlamaConfig(vocab_size=300, hidden_size=32, intermediate_size=64,
                                 num_hidden_layers=2, num_attention_heads=4,
                                 num_key_value_heads=2, max_position_embeddings=128)
                     ).save_pretrained(path)
    (path / "tokenizer.json").write_text(json.dumps(tiny_tokenizer_json()), encoding="utf-8")
    (path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|endoftext|>"}),
                                                encoding="utf-8")
    return str(path)


def test_a_downloaded_model_imported_into_auralite(studio, dialogs, tiny_bpe_llama_dir,
                                                   tiny_llama_dir):
    tab = studio.model_tab
    tab.hf_name.setText(tiny_llama_dir)                      # a WordLevel tokenizer: said
    tab.import_native()
    wait(studio.qapp, lambda: tab.job is None)
    assert "BPE tokenizers only" in dialogs["error"][-1].message
    assert tab.status.text().startswith("Failed") and studio.engine.model is None
    dialogs["error"].clear()
    tab.hf_name.setText(tiny_bpe_llama_dir)
    tab.precision["full"].setChecked(True)
    tab.import_native()
    wait(studio.qapp, lambda: tab.job is None)
    assert [f.message for f in dialogs["error"]] == []
    engine = studio.engine
    assert engine.model is not None and not engine.is_hf_model()
    assert tab.status.text().startswith("Imported: LlamaForCausalLM")
    assert "d_model         : 32" in tab.info.toPlainText()
    assert studio.quantize_tab.quantize_button.isEnabled()      # now AuraLite's own
    assert tab.save_button.isEnabled() and not studio.model_dirty
