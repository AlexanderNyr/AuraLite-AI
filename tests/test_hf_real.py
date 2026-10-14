"""Hugging Face integration against the real transformers / peft (v2.6.6).

Everything else in the suite drives `hf_integration` through fakes, which is how
a whole class of bugs stayed invisible: an unused `import bitsandbytes` that
switched the integration off, fp16+bf16 both on, a manual loop that ignored
`output_dir`, adapters that never loaded. These tests build a tiny random Llama
locally (no network, ~25k parameters) and run the real code paths.

Run locally with:  pip install "transformers>=4.56" peft accelerate
CI: the `integrations` job (AURALITE_REQUIRE_INTEGRATIONS=1 → no skipping).
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import warnings

import pytest

from tests._integration import require

torch = require("torch")
transformers = require("transformers")
peft = require("peft")
require("accelerate")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXTS = ["abc abc abc " * 8] * 16


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
def engine(tiny_llama_dir):
    from auralite.engine import AuraLiteEngine

    engine = AuraLiteEngine()
    engine.load_hf_model(tiny_llama_dir, device_map="cpu", local_files_only=True,
                         verbose=False)
    return engine


def test_the_integration_does_not_need_bitsandbytes():
    """`sys.modules[name] = None` makes `import bitsandbytes` fail like a missing package."""
    code = ("import sys; sys.modules['bitsandbytes'] = None; "
            "import auralite.hf_integration as h; "
            "print(h.HAS_HF_SUPPORT, h.bitsandbytes_available())")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True,
                         text=True, encoding="utf-8", timeout=180, check=True).stdout.split()
    assert out[0] == "True", "HF support must not depend on bitsandbytes"


def test_load_uses_the_current_dtype_argument(tiny_llama_dir, capfd):
    from auralite.engine import AuraLiteEngine

    engine = AuraLiteEngine()
    engine.load_hf_model(tiny_llama_dir, device_map="cpu", local_files_only=True, verbose=False)
    assert engine.is_hf_model()
    assert engine.hf_proxy.model.dtype == torch.float32
    assert "torch_dtype` is deprecated" not in capfd.readouterr().err
    text = engine.hf_proxy.generate("abc", max_new_tokens=4, temperature=0.0)
    assert text.startswith("abc")
    # Greedy streaming joins to exactly the greedy continuation (a random tiny
    # model may well emit EOS first, so the content itself is not asserted).
    streamed = "".join(engine.hf_proxy.generate_streaming("abc", max_new_tokens=4,
                                                          temperature=0.0))
    assert streamed == text[len("abc"):]


def _lora_tensors(model) -> dict:
    """LoRA weights keyed like the saved adapter (without the `.default` adapter name)."""
    return {key.replace(".default", ""): value.detach().clone()
            for key, value in model.state_dict().items() if "lora_" in key}


def test_trainer_finetune_progress_stop_and_output_dir(engine, tmp_path):
    engine.apply_lora_to_hf(rank=4, target_modules=["q_proj", "v_proj"])
    stop = threading.Event()
    progress = []

    def on_progress(epoch, total, loss, val_loss):
        progress.append((epoch, total, loss))
        stop.set()                         # stop after the first report

    out = tmp_path / "adapter"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        saved = engine.hf_proxy.finetune(TEXTS, output_dir=str(out), epochs=3, batch_size=2,
                                         max_length=32, gradient_accumulation_steps=1,
                                         warmup_steps=0, logging_steps=1, save_steps=10_000,
                                         progress_callback=on_progress, stop_event=stop)
    assert saved == str(out)
    assert (out / "adapter_config.json").is_file()
    assert progress and progress[0][1] == 3
    fork = [w for w in caught if "fork()" in str(w.message)]
    assert fork == [], "Trainer must not fork DataLoader workers from a threaded process"


def test_manual_loop_saves_where_asked(engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine.apply_lora_to_hf(rank=4, target_modules=["q_proj", "v_proj"])
    out = tmp_path / "manual"
    saved = engine.hf_proxy.finetune(TEXTS[:4], output_dir=str(out), epochs=1, batch_size=2,
                                     max_length=32, gradient_accumulation_steps=1,
                                     warmup_steps=0, use_trainer=False)
    assert saved == str(out) and (out / "adapter_config.json").is_file()
    assert not (tmp_path / "hf_lora_manual").exists()


def test_reloading_an_adapter_restores_exactly_the_saved_weights(tiny_llama_dir, tmp_path):
    from safetensors.torch import load_file

    from auralite.engine import AuraLiteEngine

    first = AuraLiteEngine()
    first.load_hf_model(tiny_llama_dir, device_map="cpu", local_files_only=True,
                        apply_lora=True, lora_rank=4, lora_target_modules=["q_proj", "v_proj"],
                        verbose=False)
    with torch.no_grad():                  # make the adapter distinguishable from a fresh one
        for name, param in first.hf_proxy.model.named_parameters():
            if "lora_B" in name:
                param.uniform_(-0.5, 0.5)
    first.save_hf_lora(str(tmp_path / "adapter"))
    saved = load_file(str(tmp_path / "adapter" / "adapter_model.safetensors"))

    # The GUI default: "Apply LoRA on load" → a fresh adapter is already there.
    second = AuraLiteEngine()
    second.load_hf_model(tiny_llama_dir, device_map="cpu", local_files_only=True,
                         apply_lora=True, lora_rank=4, lora_target_modules=["q_proj", "v_proj"],
                         verbose=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        second.load_hf_lora(str(tmp_path / "adapter"))
    messages = " ".join(str(w.message) for w in caught)
    assert "missing adapter keys" not in messages.lower()
    assert "multiple adapters" not in messages.lower()

    loaded = _lora_tensors(second.hf_proxy.model)
    assert set(saved) <= set(loaded), sorted(set(saved) - set(loaded))[:3]
    for key, value in saved.items():
        assert torch.equal(loaded[key], value), key
    # Still trainable, so fine-tuning can continue from the loaded adapter.
    assert second.hf_proxy.count_trainable_parameters() > 0
    assert second.hf_proxy.lora_config["rank"] == 4


def test_finetune_hf_accepts_one_short_text(engine, tmp_path):
    saved = engine.finetune_hf("abc abc", output_dir=str(tmp_path / "short"), epochs=1,
                               batch_size=1, max_length=16, gradient_accumulation_steps=1)
    assert os.path.isfile(os.path.join(saved, "adapter_config.json"))
