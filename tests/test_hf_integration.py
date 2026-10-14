"""Unit tests for Hugging Face integration using fakes (no downloads)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

import auralite.hf_integration as hf_integration  # noqa: E402
from auralite.hf_integration import HFDataset, HFNotAvailableError, HuggingFaceProxy, _check_hf_support  # noqa: E402


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
        # Pin the transformers version the dtype argument depends on (4.56 renamed
        # `torch_dtype` to `dtype`); individual tests override it.
        monkeypatch.setattr(hf_integration, "_TRANSFORMERS_VERSION", "4.46.0")
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
        assert "requires CUDA" in capsys.readouterr().err  # logged, not printed
        assert "quantization_config" not in FakeAutoModelForCausalLM.calls[0][1]

    def test_generate_forwards_sampling_parameters(self):
        proxy = HuggingFaceProxy()
        proxy.model = FakeModel()
        proxy.tokenizer = FakeTokenizer()
        proxy.device = torch.device("cpu")

        text = proxy.generate(
            "prompt",
            max_new_tokens=5,
            temperature=0.7,
            top_p=0.5,
            top_k=7,
            repetition_penalty=1.2,
        )

        # Prompt is preserved verbatim; only the continuation ids are decoded.
        assert text == "prompt" + "decoded text"
        call = proxy.model.generate_calls[0]
        assert call["max_new_tokens"] == 5
        assert call["temperature"] == 0.7
        assert call["top_p"] == 0.5
        assert call["top_k"] == 7
        assert call["repetition_penalty"] == 1.2
        assert call["do_sample"] is True
        assert call["pad_token_id"] == proxy.tokenizer.eos_token_id

    def test_greedy_generate_does_not_send_sampling_knobs(self):
        """transformers warns about temperature/top_p/top_k when do_sample=False."""
        proxy = HuggingFaceProxy()
        proxy.model = FakeModel()
        proxy.tokenizer = FakeTokenizer()
        proxy.device = torch.device("cpu")
        proxy.generate("prompt", max_new_tokens=3, temperature=0.0, top_p=0.5, top_k=7,
                       repetition_penalty=1.3)
        call = proxy.model.generate_calls[0]
        assert call["do_sample"] is False
        assert call["repetition_penalty"] == 1.3          # applies to greedy too
        assert not {"temperature", "top_p", "top_k"} & set(call)

    def test_pad_token_id_zero_is_a_real_id(self):
        proxy = HuggingFaceProxy()
        proxy.model = FakeModel()
        proxy.tokenizer = FakeTokenizer()
        proxy.tokenizer.pad_token_id = 0
        proxy.device = torch.device("cpu")
        proxy.generate("prompt", max_new_tokens=1)
        assert proxy.model.generate_calls[0]["pad_token_id"] == 0

    def test_generate_requires_loaded_model_and_tokenizer(self):
        with pytest.raises(ValueError, match="Model not loaded"):
            HuggingFaceProxy().generate("prompt")

    def test_generate_streaming_requires_loaded_model_and_tokenizer(self):
        with pytest.raises(ValueError, match="Model not loaded"):
            list(HuggingFaceProxy().generate_streaming("prompt"))


# ----------------------------------------------------------------------
#  v2.6.6 — the problems a real transformers 5 / peft 0.21 run exposed
# ----------------------------------------------------------------------
import ast  # noqa: E402
import threading  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

HF_SOURCE = Path(hf_integration.__file__).read_text(encoding="utf-8")


class TestOptionalBitsandbytes:
    def test_bitsandbytes_is_not_part_of_the_import_gate(self):
        """A missing bitsandbytes used to switch off the whole integration."""
        tree = ast.parse(HF_SOURCE)
        gate = next(node for node in tree.body if isinstance(node, ast.Try)
                    and "HAS_HF_SUPPORT" in ast.unparse(node))
        imported = {alias.name.split(".")[0] for node in ast.walk(gate)
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                    for alias in node.names}
        imported |= {node.module.split(".")[0] for node in ast.walk(gate)
                     if isinstance(node, ast.ImportFrom) and node.module}
        assert "bitsandbytes" not in imported
        assert {"transformers", "peft"} <= imported

    def test_4bit_on_cuda_without_bitsandbytes_explains_the_fix(self, monkeypatch):
        monkeypatch.setattr(hf_integration, "HAS_HF_SUPPORT", True)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(hf_integration, "bitsandbytes_available", lambda: False)
        with pytest.raises(HFNotAvailableError, match="bitsandbytes") as exc:
            HuggingFaceProxy().load_model("model", load_in_4bit=True, verbose=False)
        assert "auralite-ai[qlora]" in str(exc.value)

    def test_install_hint_points_at_the_extras(self, monkeypatch):
        monkeypatch.setattr(hf_integration, "HAS_HF_SUPPORT", False)
        monkeypatch.setattr(hf_integration, "_HF_IMPORT_ERROR", "No module named 'peft'")
        with pytest.raises(HFNotAvailableError) as exc:
            _check_hf_support()
        assert "auralite-ai[hf]" in str(exc.value)
        assert "requirements.txt" not in str(exc.value)


class TestMixedPrecision:
    @pytest.mark.parametrize("cuda,bf16,expected", [
        (False, False, {"fp16": False, "bf16": False}),
        (True, True, {"fp16": False, "bf16": True}),
        (True, False, {"fp16": True, "bf16": False}),
    ])
    def test_never_both_fp16_and_bf16(self, monkeypatch, cuda, bf16, expected):
        seen = {}

        def is_bf16_supported(including_emulation=True):
            seen["including_emulation"] = including_emulation
            return bf16

        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", is_bf16_supported)
        flags = hf_integration.mixed_precision_flags()
        assert flags == expected
        assert not (flags["fp16"] and flags["bf16"])
        if cuda:   # emulated bf16 (pre-Ampere) must not count as support
            assert seen["including_emulation"] is False

    def test_old_torch_without_including_emulation(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
        assert hf_integration.mixed_precision_flags() == {"fp16": False, "bf16": True}


class TestDtypeArgument:
    @pytest.mark.parametrize("version,name", [
        ("4.40.2", "torch_dtype"), ("4.55.4", "torch_dtype"), ("4.56.0", "dtype"),
        ("4.57.0.dev0", "dtype"), ("5.19.0", "dtype"), (None, "torch_dtype"),
        ("not-a-version", "torch_dtype"),
    ])
    def test_name_follows_the_transformers_version(self, version, name):
        assert hf_integration._dtype_load_kwarg(version or "") == name

    def test_load_model_uses_dtype_on_new_transformers(self, monkeypatch):
        TestHuggingFaceProxyLoadingAndGeneration().setup_fake_hf(monkeypatch)
        monkeypatch.setattr(hf_integration, "_TRANSFORMERS_VERSION", "5.19.0")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        HuggingFaceProxy().load_model("m", verbose=False)
        kwargs = FakeAutoModelForCausalLM.calls[0][1]
        assert kwargs["dtype"] == torch.float32 and "torch_dtype" not in kwargs


class TestRemoteCodeAndHubArguments:
    def test_remote_code_is_off_by_default_and_hub_args_are_forwarded(self, monkeypatch):
        TestHuggingFaceProxyLoadingAndGeneration().setup_fake_hf(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        proxy = HuggingFaceProxy()
        proxy.load_model("org/model", verbose=False)
        tok_kwargs = FakeAutoTokenizer.calls[0][1]
        model_kwargs = FakeAutoModelForCausalLM.calls[0][1]
        assert tok_kwargs["trust_remote_code"] is False
        assert model_kwargs["trust_remote_code"] is False
        assert "revision" not in model_kwargs and "token" not in model_kwargs

        proxy.from_pretrained("org/model", revision="v1", token="hf_x", verbose=False)
        tok_kwargs = FakeAutoTokenizer.calls[-1][1]
        model_kwargs = FakeAutoModelForCausalLM.calls[-1][1]
        assert tok_kwargs["revision"] == model_kwargs["revision"] == "v1"
        assert tok_kwargs["token"] == model_kwargs["token"] == "hf_x"

    def test_push_to_hub_passes_the_token_without_a_global_login(self, monkeypatch):
        monkeypatch.setattr(hf_integration, "HAS_HF_SUPPORT", True)
        monkeypatch.setattr(hf_integration, "_huggingface_hub_available", lambda: True)
        pushed = []

        class Pushable:
            def push_to_hub(self, **kwargs):
                pushed.append(kwargs)

        proxy = HuggingFaceProxy()
        proxy.model, proxy.tokenizer = Pushable(), Pushable()
        proxy.push_to_hub("user/repo", token="hf_secret", private=True)
        assert [call["token"] for call in pushed] == ["hf_secret", "hf_secret"]
        calls = {getattr(node.func, "id", getattr(node.func, "attr", ""))
                 for node in ast.walk(ast.parse(HF_SOURCE)) if isinstance(node, ast.Call)}
        assert "login" not in calls, "huggingface_hub.login() persists the token on disk"


class TestDatasetLabels:
    def test_padding_is_ignored_by_the_loss(self):
        class PaddingTokenizer(FakeTokenizer):
            def __call__(self, text, **kwargs):
                data = super().__call__(text, **kwargs)
                data["attention_mask"][0, -2:] = 0      # last two positions = padding
                return data

        item = HFDataset(["alpha"], PaddingTokenizer(), max_length=5)[0]
        assert item["labels"].tolist()[-2:] == [-100, -100]
        assert item["labels"].tolist()[:3] == item["input_ids"].tolist()[:3]


class _Model(torch.nn.Module):
    """Trainable stand-in: forward returns an object with a differentiable loss."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))
        self.saved_to = []

    def forward(self, **batch):
        return types.SimpleNamespace(loss=(self.w * 2.0).sum())

    def save_pretrained(self, path):
        self.saved_to.append(path)


class TestFineTuneOutputAndStop:
    def test_manual_loop_saves_to_the_requested_folder(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        proxy = HuggingFaceProxy()
        proxy.model, proxy.tokenizer = _Model(), FakeTokenizer()
        proxy.device = torch.device("cpu")
        dataset = HFDataset(["alpha", "beta"], FakeTokenizer(), max_length=4)
        target = tmp_path / "chosen"
        out = proxy._finetune_manual_loop(dataset, epochs=1, learning_rate=1e-3, batch_size=2,
                                          max_length=4, gradient_accumulation_steps=1,
                                          output_dir=str(target))
        assert out == str(target)
        assert proxy.model.saved_to == [str(target)]
        assert not (tmp_path / "hf_lora_manual").exists()

    def test_trainer_callback_reports_progress_and_honours_stop(self, monkeypatch):
        monkeypatch.setattr(hf_integration, "TrainerCallback", object, raising=False)
        progress, stop = [], threading.Event()
        callback = HuggingFaceProxy._progress_callback_for_trainer(
            lambda *a: progress.append(a), stop, total_epochs=3)
        state = types.SimpleNamespace(epoch=1.4)
        control = types.SimpleNamespace(should_training_stop=False)

        callback.on_log(None, state, control, logs={"loss": 0.75})
        callback.on_epoch_end(None, types.SimpleNamespace(epoch=2.0), control)
        assert progress == [(2, 3, 0.75, None), (2, 3, 0.75, None)]

        callback.on_step_end(None, state, control)
        assert control.should_training_stop is False
        stop.set()
        callback.on_step_end(None, state, control)
        assert control.should_training_stop is True


class TestAdapterReload:
    def test_loading_an_adapter_replaces_the_current_one(self, monkeypatch):
        monkeypatch.setattr(hf_integration, "HAS_HF_SUPPORT", True)
        seen = {}

        class FakePeftModel:
            @classmethod
            def from_pretrained(cls, base, path, **kwargs):
                seen.update(base=base, path=path, **kwargs)
                loaded = cls()
                loaded.peft_config = {"default": types.SimpleNamespace(
                    r=8, lora_alpha=16, lora_dropout=0.1, target_modules={"v_proj", "q_proj"})}
                return loaded

        class AppliedLoRA:
            def unload(self):
                return "plain-base-model"

        monkeypatch.setattr(hf_integration, "PeftModel", FakePeftModel, raising=False)
        proxy = HuggingFaceProxy()
        proxy.model, proxy.is_peft = AppliedLoRA(), True
        proxy.load_lora_adapter("adapter-dir")

        assert seen["base"] == "plain-base-model"        # not wrapped a second time
        assert seen["path"] == "adapter-dir" and seen["is_trainable"] is True
        assert proxy.lora_config == {"rank": 8, "alpha": 16, "dropout": 0.1,
                                     "target_modules": ["q_proj", "v_proj"]}


class TestStreamingErrors:
    def test_a_failing_generate_raises_instead_of_hanging(self, monkeypatch):
        import queue

        class FakeStreamer:
            def __init__(self, tokenizer, **kwargs):
                self.q = queue.Queue()

            def end(self):
                self.q.put(None)

            def __iter__(self):
                while (item := self.q.get(timeout=10)) is not None:
                    yield item

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.TextIteratorStreamer = FakeStreamer
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        class Exploding(FakeModel):
            def generate(self, **kwargs):
                raise RuntimeError("CUDA out of memory")

        proxy = HuggingFaceProxy()
        proxy.model, proxy.tokenizer = Exploding(), FakeTokenizer()
        proxy.device = torch.device("cpu")
        outcome = {}

        def consume():
            try:
                list(proxy.generate_streaming("prompt", max_new_tokens=2))
            except RuntimeError as exc:
                outcome["error"] = str(exc)

        worker = threading.Thread(target=consume, daemon=True)
        worker.start()
        worker.join(timeout=15)
        assert not worker.is_alive(), "generate_streaming hung after a failing generate()"
        assert outcome["error"] == "CUDA out of memory"


class TestGenerationInputs:
    def test_only_ids_and_mask_reach_generate(self):
        """transformers 4.x rejects `token_type_ids` for decoder-only models."""
        class TypedTokenizer(FakeTokenizer):
            def __call__(self, text, **kwargs):
                data = super().__call__(text, **kwargs)
                data["token_type_ids"] = torch.zeros_like(data["input_ids"])
                return data

        proxy = HuggingFaceProxy()
        proxy.model, proxy.tokenizer = FakeModel(), TypedTokenizer()
        proxy.device = torch.device("cpu")
        proxy.generate("prompt", max_new_tokens=2)
        call = proxy.model.generate_calls[0]
        assert "token_type_ids" not in call
        assert {"input_ids", "attention_mask"} <= set(call)
