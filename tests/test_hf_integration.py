"""Unit tests for Hugging Face integration using fakes (no downloads)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

import hf_integration  # noqa: E402
from hf_integration import HFDataset, HFNotAvailableError, HuggingFaceProxy, _check_hf_support  # noqa: E402


class Batch(dict):
    def to(self, device):
        self["device"] = device
        return self


class FakeTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.pad_token_id = None
        self.eos_token_id = 9
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        length = kwargs.get("max_length", 3)
        data = {
            "input_ids": torch.arange(length).unsqueeze(0),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
        }
        if kwargs.get("return_tensors") == "pt":
            return Batch(data)
        return data

    def decode(self, ids, skip_special_tokens=True):
        return "decoded text"

    def save_pretrained(self, path):
        self.saved_to = path


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.ones(1))
        self.eval_called = False
        self.generate_calls = []
        self.dtype = torch.float32

    def eval(self):
        self.eval_called = True
        return super().eval()

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return torch.tensor([[1, 2, 3, 4]])


class FakeAutoTokenizer:
    calls = []
    tokenizer = FakeTokenizer()

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        cls.calls.append((model_name_or_path, kwargs))
        return cls.tokenizer


class FakeAutoModelForCausalLM:
    calls = []
    model = FakeModel()

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        cls.calls.append((model_name_or_path, kwargs))
        return cls.model


class TestAvailability:
    def test_check_hf_support_raises_with_original_import_error(self, monkeypatch):
        monkeypatch.setattr(hf_integration, "HAS_HF_SUPPORT", False)
        monkeypatch.setattr(hf_integration, "_HF_IMPORT_ERROR", "missing transformers", raising=False)
        with pytest.raises(HFNotAvailableError) as exc:
            _check_hf_support()
        assert "Hugging Face" in str(exc.value)
        assert "missing transformers" in str(exc.value)

    def test_check_hf_support_noops_when_available(self, monkeypatch):
        monkeypatch.setattr(hf_integration, "HAS_HF_SUPPORT", True)
        _check_hf_support()


class TestHFDataset:
    def test_dataset_tokenizes_and_creates_labels_clone(self):
        tokenizer = FakeTokenizer()
        ds = HFDataset(["alpha", "beta"], tokenizer, max_length=5)
        item = ds[0]
        assert len(ds) == 2
        assert set(item) == {"input_ids", "attention_mask", "labels"}
        assert item["input_ids"].shape == (5,)
        assert torch.equal(item["labels"], item["input_ids"])
        assert item["labels"].data_ptr() != item["input_ids"].data_ptr()
        assert tokenizer.calls[0][1]["padding"] == "max_length"
        assert tokenizer.calls[0][1]["truncation"] is True


class TestHuggingFaceProxyBasics:
    def test_initial_state_and_info(self):
        proxy = HuggingFaceProxy()
        info = proxy.get_info()
        assert proxy.model is None
        assert proxy.tokenizer is None
        assert proxy.count_parameters() == 0
        assert proxy.count_trainable_parameters() == 0
        assert info["backend"] == "huggingface"
        assert info["parameters"] == 0
        assert info["is_peft"] is False
        proxy.reset_cache()  # no-op compatibility method

    def test_disable_lora_noops_without_adapter(self):
        proxy = HuggingFaceProxy()
        proxy.disable_lora()
        assert proxy.is_peft is False

    def test_disable_lora_merges_when_model_supports_it(self):
        class Mergeable:
            def __init__(self):
                self.merged = False

            def merge_and_unload(self):
                self.merged = True
                return "merged-model"

        proxy = HuggingFaceProxy()
        proxy.model = Mergeable()
        proxy.is_peft = True
        proxy.lora_config = {"rank": 4}
        proxy.disable_lora()
        assert proxy.model == "merged-model"
        assert proxy.is_peft is False
        assert proxy.lora_config is None

    def test_apply_lora_requires_loaded_model(self, monkeypatch):
        monkeypatch.setattr(hf_integration, "HAS_HF_SUPPORT", True)
        with pytest.raises(ValueError, match="Load a model first"):
            HuggingFaceProxy().apply_lora()

    def test_load_lora_adapter_requires_base_model(self):
        with pytest.raises(ValueError, match="Base model"):
            HuggingFaceProxy().load_lora_adapter("adapter")

    def test_save_lora_adapter_requires_adapter(self):
        with pytest.raises(ValueError, match="No LoRA adapter"):
            HuggingFaceProxy().save_lora_adapter("adapter")


class TestHuggingFaceProxyLoadingAndGeneration:
    def setup_fake_hf(self, monkeypatch):
        FakeAutoTokenizer.calls = []
        FakeAutoTokenizer.tokenizer = FakeTokenizer()
        FakeAutoModelForCausalLM.calls = []
        FakeAutoModelForCausalLM.model = FakeModel()
        monkeypatch.setattr(hf_integration, "HAS_HF_SUPPORT", True)
        monkeypatch.setattr(hf_integration, "AutoTokenizer", FakeAutoTokenizer, raising=False)
        monkeypatch.setattr(hf_integration, "AutoModelForCausalLM", FakeAutoModelForCausalLM, raising=False)

    def test_load_model_rejects_4bit_and_8bit_together_before_downloads(self, monkeypatch):
        monkeypatch.setattr(hf_integration, "HAS_HF_SUPPORT", True)
        with pytest.raises(ValueError, match="either 4-bit or 8-bit"):
            HuggingFaceProxy().load_model("model", load_in_4bit=True, load_in_8bit=True)

    def test_load_model_with_fake_classes_sets_tokenizer_model_and_info(self, monkeypatch):
        self.setup_fake_hf(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        proxy = HuggingFaceProxy()
        proxy.load_model("local-model", local_files_only=True, verbose=False, max_seq_len=123)

        assert proxy.model is FakeAutoModelForCausalLM.model
        assert proxy.tokenizer is FakeAutoTokenizer.tokenizer
        assert proxy.tokenizer.pad_token == "<eos>"
        assert proxy.model.eval_called is True
        assert proxy.model_name_or_path == "local-model"
        assert proxy.base_model_name == "local-model"
        assert proxy.max_seq_len == 123
        assert FakeAutoTokenizer.calls[0][1]["local_files_only"] is True
        assert FakeAutoModelForCausalLM.calls[0][1]["local_files_only"] is True
        assert FakeAutoModelForCausalLM.calls[0][1]["torch_dtype"] == torch.float32

    def test_load_model_quantization_falls_back_on_cpu(self, monkeypatch, capsys):
        self.setup_fake_hf(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        proxy = HuggingFaceProxy()
        proxy.load_model("model", load_in_4bit=True, verbose=True)

        assert proxy.is_quantized is False
        assert "requires CUDA" in capsys.readouterr().out
        assert "quantization_config" not in FakeAutoModelForCausalLM.calls[0][1]

    def test_generate_forwards_sampling_parameters(self):
        proxy = HuggingFaceProxy()
        proxy.model = FakeModel()
        proxy.tokenizer = FakeTokenizer()
        proxy.device = torch.device("cpu")

        text = proxy.generate(
            "prompt",
            max_new_tokens=5,
            temperature=0.0,
            top_p=0.5,
            top_k=7,
            repetition_penalty=1.2,
        )

        assert text == "decoded text"
        call = proxy.model.generate_calls[0]
        assert call["max_new_tokens"] == 5
        assert call["temperature"] == 0.0
        assert call["top_p"] == 0.5
        assert call["top_k"] == 7
        assert call["repetition_penalty"] == 1.2
        assert call["do_sample"] is False
        assert call["pad_token_id"] == proxy.tokenizer.eos_token_id

    def test_generate_requires_loaded_model_and_tokenizer(self):
        with pytest.raises(ValueError, match="Model not loaded"):
            HuggingFaceProxy().generate("prompt")

    def test_generate_streaming_requires_loaded_model_and_tokenizer(self):
        with pytest.raises(ValueError, match="Model not loaded"):
            list(HuggingFaceProxy().generate_streaming("prompt"))
