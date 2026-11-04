"""2.18.1 against the real thing: transformers and llama.cpp (the CI integrations job).

Tiny random Llama / Llama 3.1 / Qwen2 / Mistral / Qwen3 models are made with
transformers itself, imported into AuraLite's engine and compared logit by
logit; exported folders are loaded back by transformers, exported GGUF files
by llama.cpp. Locally the module skips without the packages; the CI job sets
AURALITE_REQUIRE_INTEGRATIONS=1, where a missing package fails.
"""
from __future__ import annotations

import json

import pytest
import torch

from tests._integration import require

transformers = require("transformers")

from auralite.engine.gguf_writer import export_gguf  # noqa: E402
from auralite.engine.hf_models import export_hf, import_hf  # noqa: E402
from tests.test_hf_v2181 import tiny_tokenizer_json  # noqa: E402

COMMON = dict(vocab_size=300, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)


def _configs():
    llama3 = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
              "high_freq_factor": 4.0, "original_max_position_embeddings": 64}
    made = {
        "llama": transformers.LlamaConfig(**COMMON, rms_norm_eps=1e-5, rope_theta=10000.0,
                                          tie_word_embeddings=False),
        "llama3": transformers.LlamaConfig(**COMMON, rms_norm_eps=1e-5, rope_theta=500000.0,
                                           tie_word_embeddings=True, rope_scaling=llama3),
        "qwen2": transformers.Qwen2Config(**COMMON, rms_norm_eps=1e-6, rope_theta=1e6,
                                          tie_word_embeddings=True),
        "mistral": transformers.MistralConfig(**COMMON, rms_norm_eps=1e-5, rope_theta=1e6,
                                              sliding_window=16, tie_word_embeddings=False),
    }
    if hasattr(transformers, "Qwen3Config"):          # transformers >= 4.51
        made["qwen3"] = transformers.Qwen3Config(**COMMON, head_dim=32, rms_norm_eps=1e-6,
                                                 rope_theta=1e6, tie_word_embeddings=True)
    return made


def _model(hf_models, name):
    if name not in hf_models:
        pytest.skip(f"this transformers has no {name} models")
    return hf_models[name]


@pytest.fixture(scope="module")
def hf_models(tmp_path_factory):
    root = tmp_path_factory.mktemp("hf")
    made = {}
    for name, config in _configs().items():
        torch.manual_seed(0)
        model = transformers.AutoModelForCausalLM.from_config(config)
        with torch.no_grad():
            for key, param in model.named_parameters():
                if "norm" in key or key.endswith("bias"):
                    param.add_(torch.randn_like(param) * 0.1)
        path = root / name
        model.save_pretrained(path)
        (path / "tokenizer.json").write_text(json.dumps(tiny_tokenizer_json()), encoding="utf-8")
        (path / "tokenizer_config.json").write_text(json.dumps(
            {"eos_token": "<|endoftext|>"}), encoding="utf-8")
        made[name] = path
    return made


def _reference(path, ids):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path, attn_implementation="eager").float().eval()
    with torch.no_grad():
        return model(ids).logits


IDS = torch.randint(0, 270, (2, 40), generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("name", ["llama", "llama3", "qwen2", "mistral", "qwen3"])
def test_imported_logits_match_transformers(hf_models, name):
    path = _model(hf_models, name)
    result = import_hf(path)
    with torch.no_grad():
        ours = result.model(IDS)
    ref = _reference(path, IDS)
    assert torch.allclose(ours, ref, atol=1e-5, rtol=1e-4), float((ours - ref).abs().max())


@pytest.mark.parametrize("name", ["llama", "qwen2", "mistral", "qwen3"])
def test_exported_folder_loads_in_transformers(hf_models, tmp_path, name):
    result = import_hf(_model(hf_models, name), quantize="nf4")
    result.model.enable_lora(rank=4, targets="all")
    with torch.no_grad():
        for _name, _base, adapter in result.model.lora_pairs():
            adapter.lora_B.normal_(0, 0.05)
        ours = result.model(IDS)
    export_hf(result.model, result.tokenizer, tmp_path / "out", source=result.source,
              dtype="float32")
    ref = _reference(tmp_path / "out", IDS)
    assert torch.allclose(ours, ref, atol=1e-5, rtol=1e-4), float((ours - ref).abs().max())
    tokenizer = transformers.AutoTokenizer.from_pretrained(tmp_path / "out")
    text = "the theatre of the world"
    assert tokenizer(text, add_special_tokens=False)["input_ids"] == \
        result.tokenizer.encode(text)
    chat = tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False,
                                         add_generation_prompt=True)
    assert chat == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"


@pytest.mark.parametrize("name", ["llama", "qwen2", "qwen3", "mistral"])
def test_exported_gguf_runs_in_llama_cpp(hf_models, tmp_path, name):
    llama_cpp = require("llama_cpp")
    import numpy as np
    result = import_hf(_model(hf_models, name))
    ids = IDS[0, :16]                       # within Mistral's window: llama.cpp has none
    with torch.no_grad():
        ours = result.model(ids[None])[0].numpy()
    path = tmp_path / f"{name}.gguf"
    export_gguf(result.model, result.tokenizer, path, quant="f32", source=result.source)
    llm = llama_cpp.Llama(model_path=str(path), n_ctx=64, logits_all=True, verbose=False)
    llm.eval(ids.tolist())
    theirs = np.array(llm.scores[:16])
    assert np.abs(theirs - ours).max() < 2e-3
    assert llm.tokenize(b"the theatre", add_bos=False, special=False) == \
        result.tokenizer.encode("the theatre")
