"""
hf_integration.py — Hugging Face model support for AuraLite AI

Adds the ability to:
- Load ANY Hugging Face causal LM (Llama, Mistral, Qwen, Gemma, Phi, etc.)
- Load in 4-bit / 8-bit (QLoRA ready via bitsandbytes — optional, only for 4/8-bit)
- Apply LoRA / QLoRA adapters using PEFT
- Fine-tune with LoRA/QLoRA (full or parameter-efficient)
- Generate text (with streaming support)
- Save / load LoRA adapters

This module is designed to work alongside the native AuraLite ModernTransformer
and GGUF backends. It exposes a similar interface where possible.
"""

from __future__ import annotations

import importlib.util
import math
import os
import re
import threading
from typing import Iterator, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger("auralite.hf")


# Optional heavy dependencies — gracefully degrade if missing.
#
# v2.6.6: `bitsandbytes` is deliberately NOT part of this gate. It is only needed
# for 4-/8-bit loading and is checked there (`bitsandbytes_available()`). Until
# 2.6.5 an unused `import bitsandbytes` sat in this block, so a missing
# bitsandbytes switched off the whole integration — on macOS (where the `full`
# extra never installs it) and with the `[hf]` extra (which never listed it).
try:
    import transformers as _transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        PeftModel,
        prepare_model_for_kbit_training,
    )
    HAS_HF_SUPPORT = True
    _HF_IMPORT_ERROR = ""
    _TRANSFORMERS_VERSION: str | None = getattr(_transformers, "__version__", None)
except ImportError as e:
    HAS_HF_SUPPORT = False
    _HF_IMPORT_ERROR = str(e)
    _TRANSFORMERS_VERSION = None

_INSTALL_HINT = (
    "Install them with:\n"
    "  pip install \"auralite-ai[hf]\"        (or, from a checkout: pip install -e \".[hf]\")\n"
    "4-/8-bit QLoRA loading additionally needs bitsandbytes:\n"
    "  pip install \"auralite-ai[qlora]\""
)


class HFNotAvailableError(ImportError):
    """Raised when Hugging Face / PEFT dependencies are missing."""
    pass


def _check_hf_support():
    if not HAS_HF_SUPPORT:
        raise HFNotAvailableError(
            "Hugging Face + LoRA support requires transformers and peft.\n"
            f"{_INSTALL_HINT}\n\n"
            f"Original import error: {_HF_IMPORT_ERROR}"
        )


def bitsandbytes_available() -> bool:
    """True when `bitsandbytes` (needed only for 4-/8-bit loading) is installed."""
    return importlib.util.find_spec("bitsandbytes") is not None


def _huggingface_hub_available() -> bool:
    return importlib.util.find_spec("huggingface_hub") is not None


def _dtype_load_kwarg(version: str | None = None) -> str:
    """Name of the dtype argument of `from_pretrained`.

    transformers 4.56 renamed `torch_dtype` to `dtype` (the old name logs a
    deprecation warning in 5.x). Older releases do not know `dtype` and would
    silently store it on the config instead, hence the explicit version gate.
    """
    version = _TRANSFORMERS_VERSION if version is None else version
    match = re.match(r"(\d+)\.(\d+)", version or "")
    if not match:
        return "torch_dtype"
    return "dtype" if (int(match.group(1)), int(match.group(2))) >= (4, 56) else "torch_dtype"


def _cuda_bf16_native() -> bool:
    """bf16 supported *natively* by the GPU (Ampere+), not merely emulated.

    `torch.cuda.is_bf16_supported()` defaults to `including_emulation=True` and
    then answers True on pre-Ampere GPUs, where bf16 is slow.
    """
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:  # torch < 2.3: no `including_emulation`
        return bool(torch.cuda.is_bf16_supported())


def mixed_precision_flags() -> dict[str, bool]:
    """`TrainingArguments(fp16=..., bf16=...)` — never both True.

    Until 2.6.5 the Trainer got `fp16=cuda_available, bf16=bf16_supported`; on
    any bf16-capable GPU both were True and transformers raised "At most one of
    fp16 and bf16 can be True" before the first step.
    """
    if not torch.cuda.is_available():
        return {"fp16": False, "bf16": False}
    bf16 = _cuda_bf16_native()
    return {"fp16": not bf16, "bf16": bf16}


class HFDataset(Dataset):
    """Simple text dataset for HF fine-tuning (tokenizes on the fly)."""

    def __init__(self, texts: list[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        # For causal LM, labels = input_ids (shifted inside the model). Padding
        # positions are -100 so the loss ignores them — otherwise the manual
        # loop trains the model to emit pad/EOS tokens (the Trainer path's
        # collator did this already; the manual loop did not).
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class HuggingFaceProxy:
    """
    Proxy for any Hugging Face causal language model with full LoRA / QLoRA support.

    Provides a similar high-level interface to AuraLiteEngine's native models
    so the GUI and other code can treat it uniformly where possible.
    """

    backend = "huggingface"

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.model_name_or_path: str = ""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_len: int = 4096
        self.is_quantized: bool = False
        self.is_peft: bool = False
        self.lora_config: Optional[dict] = None
        self.base_model_name: str = ""
        self._original_dtype = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_model(
        self,
        model_name_or_path: str,
        *,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: str | dict = "auto",
        trust_remote_code: bool = False,
        use_fast_tokenizer: bool = True,
        max_seq_len: int = 4096,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Load any causal LM from Hugging Face Hub **or from a local folder**.

        Perfect for "already downloaded" models:
        - Use `local_files_only=True` for completely offline loading.
        - `model_name_or_path` can be a local path like:
          ~/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/xxxxxxx
          or any folder that contains config.json + model weights.

        Supports 4-bit (QLoRA) and 8-bit loading via bitsandbytes (optional
        dependency, only needed for those two modes).

        Security: `trust_remote_code` is **off by default** (v2.6.6). Turning it on
        executes Python code shipped inside the model repository — only do that
        for repositories you trust. (Until 2.6.5 it defaulted to True and nothing
        in the GUI or CLI overrode it.)
        """
        _check_hf_support()

        self.model_name_or_path = model_name_or_path
        self.max_seq_len = max_seq_len
        self.is_quantized = False

        if load_in_4bit and load_in_8bit:
            raise ValueError("Choose either 4-bit or 8-bit loading, not both at once.")

        is_local = local_files_only or os.path.isdir(model_name_or_path)

        if verbose:
            if is_local:
                logger.info(f"[AuraLite-HF] Loading LOCAL / already downloaded model from: {model_name_or_path}")
            else:
                logger.info(f"[AuraLite-HF] Loading model from Hub: {model_name_or_path}")
        if trust_remote_code:
            logger.warning("[AuraLite-HF] trust_remote_code=True: Python code from %s "
                           "will be executed.", model_name_or_path)

        # Determine quantization config
        bnb_config = None
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "device_map": device_map,
        }
        hub_kwargs = {}
        if revision:
            hub_kwargs["revision"] = revision
        if token:
            hub_kwargs["token"] = token

        if load_in_4bit or load_in_8bit:
            if not torch.cuda.is_available():
                logger.warning("[AuraLite-HF] WARNING: 4/8-bit quantization requires CUDA. Falling back to full-precision CPU loading.")
                load_in_4bit = False
                load_in_8bit = False
            elif not bitsandbytes_available():
                raise HFNotAvailableError(
                    "4-bit/8-bit loading needs the optional `bitsandbytes` package:\n"
                    "  pip install \"auralite-ai[qlora]\"   (or: pip install bitsandbytes)\n"
                    "Full-precision loading and LoRA fine-tuning work without it."
                )
            else:
                compute_dtype = torch.float16
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
                load_kwargs["quantization_config"] = bnb_config
                self.is_quantized = True
                if verbose:
                    logger.info(f"[AuraLite-HF] Using {'4-bit' if load_in_4bit else '8-bit'} quantization (QLoRA ready)")

        if torch_dtype is None:
            if load_in_4bit or load_in_8bit:
                torch_dtype = torch.float16
            elif torch.cuda.is_available():
                torch_dtype = torch.bfloat16 if _cuda_bf16_native() else torch.float16
            else:
                torch_dtype = torch.float32

        # `dtype` on transformers >= 4.56, `torch_dtype` before (see helper).
        load_kwargs[_dtype_load_kwarg()] = torch_dtype
        load_kwargs.update(hub_kwargs)

        # Load tokenizer (support local folders + offline mode)
        tokenizer_kwargs = {
            "use_fast": use_fast_tokenizer,
            "trust_remote_code": trust_remote_code,
            **hub_kwargs,
        }
        if local_files_only:
            tokenizer_kwargs["local_files_only"] = True

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, **tokenizer_kwargs
            )
        except Exception as e:
            logger.info(f"[AuraLite-HF] Tokenizer warning: {e}. Trying without fast tokenizer...")
            tokenizer_kwargs["use_fast"] = False
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, **tokenizer_kwargs
            )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|endoftext|>"
            if verbose:
                logger.info("[AuraLite-HF] Set pad_token = eos_token")

        # Load model (with local/offline support)
        if local_files_only:
            load_kwargs["local_files_only"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **load_kwargs,
        )

        self._original_dtype = self.model.dtype if hasattr(self.model, "dtype") else None

        # Move to device if not using device_map="auto"
        if device_map not in ("auto", "balanced", "balanced_low_0"):
            self.model = self.model.to(self.device)

        self.model.eval()
        self.base_model_name = model_name_or_path

        if verbose:
            n_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"[AuraLite-HF] Model loaded. Parameters: {n_params / 1e6:.1f}M")
            logger.info(f"[AuraLite-HF] Device: {self.device}, dtype: {self.model.dtype}")

    # ------------------------------------------------------------------
    # LoRA / QLoRA
    # ------------------------------------------------------------------

    def apply_lora(
        self,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
        verbose: bool = True,
    ):
        """
        Apply LoRA (or QLoRA if model is already 4-bit) using PEFT.

        Works on both full-precision and quantized (4-bit) models.
        """
        _check_hf_support()

        if self.model is None:
            raise ValueError("Load a model first with load_model()")

        if self.is_peft:
            logger.info("[AuraLite-HF] LoRA already applied. Skipping.")
            return

        if verbose:
            logger.info(f"[AuraLite-HF] Applying LoRA (rank={rank}, alpha={alpha})...")

        # Prepare for k-bit training if quantized
        if self.is_quantized:
            self.model = prepare_model_for_kbit_training(self.model)
            if verbose:
                logger.info("[AuraLite-HF] Model prepared for k-bit (QLoRA) training")

        if target_modules is None:
            # Common targets for modern decoder-only models
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias=bias,
            target_modules=target_modules,
            task_type=task_type,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.is_peft = True
        self.lora_config = {
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "target_modules": target_modules,
        }

        if verbose:
            self.model.print_trainable_parameters()

    def disable_lora(self):
        """Merge LoRA weights back into base model (if possible) and disable adapters."""
        if not self.is_peft or self.model is None:
            return
        try:
            if hasattr(self.model, "merge_and_unload"):
                self.model = self.model.merge_and_unload()
            self.is_peft = False
            self.lora_config = None
            logger.info("[AuraLite-HF] LoRA adapters merged and removed.")
        except Exception as e:
            logger.warning(f"[AuraLite-HF] Could not merge LoRA: {e}")

    # ------------------------------------------------------------------
    # Generation (compatible interface)
    # ------------------------------------------------------------------

    def _pad_token_id(self):
        """pad id, falling back to EOS — `0` is a valid id, so no `or` chain."""
        pad = getattr(self.tokenizer, "pad_token_id", None)
        return self.tokenizer.eos_token_id if pad is None else pad

    def _generation_inputs(self, prompt: str) -> dict:
        """`input_ids` + `attention_mask` on the model device — nothing else.

        Tokenizers without explicit `model_input_names` also return
        `token_type_ids`; transformers 4.x `generate()` rejects that for decoder
        models ("model_kwargs are not used by the model"), found by running the
        integration tests against transformers 4.46.
        """
        encoded = self.tokenizer(prompt, return_tensors="pt")
        return {key: value.to(self.device) for key, value in encoded.items()
                if key in ("input_ids", "attention_mask")}

    @staticmethod
    def _decoding_kwargs(do_sample: bool, temperature: float, top_p: float,
                         top_k: int, repetition_penalty: float) -> dict:
        """Sampling knobs only in sampling mode.

        Greedy decoding ignores temperature/top_p/top_k, and transformers warns
        about every one of them when `do_sample=False` — so they are only sent
        when they matter. `repetition_penalty` applies to both modes.
        """
        kwargs = {"do_sample": do_sample, "repetition_penalty": repetition_penalty}
        if do_sample:
            kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
        return kwargs

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """Generate text. Returns full text (prompt + continuation)."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded")

        inputs = self._generation_inputs(prompt)
        do_sample = bool(do_sample and temperature > 0)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **self._decoding_kwargs(do_sample, temperature, top_p, top_k,
                                        repetition_penalty),
                pad_token_id=self._pad_token_id(),
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode only the continuation ids and prepend the ORIGINAL prompt.
        # Decoding the full sequence with skip_special_tokens=True can drop
        # special tokens that appear in `prompt` (chat-template markers such
        # as <|im_start|>), so the result would NOT start with the prompt and
        # every caller doing `full[len(prompt):]` string-slicing mis-slices.
        n_input = inputs["input_ids"].shape[1]
        continuation = self.tokenizer.decode(outputs[0][n_input:], skip_special_tokens=True)
        return prompt + continuation

    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        **kwargs,
    ) -> Iterator[str]:
        """Token-by-token streaming generation (yields deltas)."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded")

        inputs = self._generation_inputs(prompt)

        # Use transformers TextIteratorStreamer if available
        yielded_any = False
        try:
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                **self._decoding_kwargs(temperature > 0, temperature, top_p, top_k,
                                        repetition_penalty),
                "streamer": streamer,
                "pad_token_id": self._pad_token_id(),
            }

            # v2.6.6: an exception inside `generate` (CUDA OOM, bad kwargs) used
            # to kill the worker thread silently while this generator waited on
            # the streamer forever — the GUI hung. The worker now records the
            # error, ends the stream, and the error is re-raised here.
            failure: list[BaseException] = []

            def _run():
                try:
                    self.model.generate(**generation_kwargs)
                except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
                    failure.append(exc)
                    streamer.end()

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()

            for text in streamer:
                yielded_any = True
                yield text

            thread.join()
            if failure:
                raise failure[0]
            return
        except Exception:
            if yielded_any:
                # Streaming already emitted chunks — re-running generation in
                # the fallback would duplicate the output. Surface the error.
                raise
            # Fallback: non-streaming (yield whole thing at once)
            full = self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                **kwargs,
            )
            generated = full[len(prompt):] if full.startswith(prompt) else full
            yield generated

    # ------------------------------------------------------------------
    # Fine-tuning (LoRA / QLoRA)
    # ------------------------------------------------------------------

    def finetune(
        self,
        texts: list[str],
        output_dir: str = "hf_lora_adapter",
        epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        max_length: int = 512,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        save_steps: int = 500,
        logging_steps: int = 50,
        use_trainer: bool = True,
        progress_callback: Optional[Callable] = None,
        stop_event=None,
    ):
        """
        Fine-tune the loaded model with LoRA/QLoRA.

        Uses Hugging Face Trainer when `use_trainer=True` (recommended).
        Falls back to manual loop otherwise.
        """
        _check_hf_support()

        if self.model is None or self.tokenizer is None:
            raise ValueError("Load a model first!")

        if not self.is_peft:
            logger.info("[AuraLite-HF] No LoRA applied — applying default LoRA before fine-tuning...")
            self.apply_lora(rank=16, alpha=32, dropout=0.05)

        # Prepare dataset
        dataset = HFDataset(texts, self.tokenizer, max_length=max_length)

        if use_trainer:
            return self._finetune_with_trainer(
                dataset,
                output_dir=output_dir,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                save_steps=save_steps,
                logging_steps=logging_steps,
                progress_callback=progress_callback,
                stop_event=stop_event,
            )
        else:
            return self._finetune_manual_loop(
                dataset,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                max_length=max_length,
                gradient_accumulation_steps=gradient_accumulation_steps,
                progress_callback=progress_callback,
                stop_event=stop_event,
                output_dir=output_dir,
            )

    @staticmethod
    def _progress_callback_for_trainer(progress_callback, stop_event, total_epochs):
        """Bridge Trainer events to AuraLite's `(epoch, total, loss, val_loss)` callback.

        Also honours `stop_event`: until 2.6.5 the Trainer path ignored both, so
        the GUI showed no progress and could not stop a fine-tune.
        """
        class _AuraLiteTrainerCallback(TrainerCallback):
            last_loss: float | None = None

            def _epoch(self, state) -> int:
                return min(total_epochs, max(1, math.ceil(state.epoch or 0)))

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    self.last_loss = float(logs["loss"])
                    if progress_callback:
                        progress_callback(self._epoch(state), total_epochs, self.last_loss, None)

            def on_epoch_end(self, args, state, control, **kwargs):
                if progress_callback and self.last_loss is not None:
                    progress_callback(self._epoch(state), total_epochs, self.last_loss, None)

            def on_step_end(self, args, state, control, **kwargs):
                if stop_event is not None and stop_event.is_set():
                    control.should_training_stop = True
                return control

        return _AuraLiteTrainerCallback()

    def _finetune_with_trainer(self, dataset, **kwargs):
        """Fine-tune using Hugging Face Trainer (best experience)."""
        training_args = TrainingArguments(
            output_dir=kwargs["output_dir"],
            per_device_train_batch_size=kwargs["batch_size"],
            gradient_accumulation_steps=kwargs["gradient_accumulation_steps"],
            learning_rate=kwargs["learning_rate"],
            num_train_epochs=kwargs["epochs"],
            warmup_steps=kwargs["warmup_steps"],
            logging_steps=kwargs["logging_steps"],
            save_steps=kwargs["save_steps"],
            save_total_limit=2,
            # v2.6.6: at most one of fp16/bf16 (both True crashed on bf16 GPUs).
            **mixed_precision_flags(),
            report_to="none",           # disable wandb etc.
            remove_unused_columns=False,
            # Worker *processes* fork the caller; the GUI trains from a thread of
            # a multi-threaded Tk process, where fork() can deadlock the child.
            dataloader_num_workers=0,
            dataloader_pin_memory=torch.cuda.is_available(),
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        stop_event = kwargs.get("stop_event")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=[self._progress_callback_for_trainer(
                kwargs.get("progress_callback"), stop_event, kwargs["epochs"])],
        )

        logger.info("[AuraLite-HF] Starting fine-tuning with Hugging Face Trainer...")
        trainer.train()
        if stop_event is not None and stop_event.is_set():
            logger.info("[AuraLite-HF] Training stopped early on request; saving the adapter as is.")
        else:
            logger.info("[AuraLite-HF] Training finished.")

        # Save adapter
        self.model.save_pretrained(kwargs["output_dir"])
        self.tokenizer.save_pretrained(kwargs["output_dir"])
        logger.info(f"[AuraLite-HF] LoRA adapter saved to: {kwargs['output_dir']}")

        return kwargs["output_dir"]

    def _finetune_manual_loop(
        self,
        dataset,
        epochs,
        learning_rate,
        batch_size,
        max_length,
        gradient_accumulation_steps,
        progress_callback=None,
        stop_event=None,
        output_dir: str = "hf_lora_adapter",
    ):
        """Simple manual training loop (useful when Trainer is overkill)."""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
        )

        self.model.train()
        global_step = 0

        for epoch in range(epochs):
            if stop_event and stop_event.is_set():
                break

            total_loss = 0.0
            for step, batch in enumerate(loader):
                if stop_event and stop_event.is_set():
                    break

                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss / gradient_accumulation_steps

                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                global_step += 1

                if progress_callback and global_step % 10 == 0:
                    avg = total_loss / (step + 1)
                    progress_callback(epoch + 1, epochs, avg, None)

            avg_loss = total_loss / max(1, len(loader))
            if progress_callback:
                progress_callback(epoch + 1, epochs, avg_loss, None)

            logger.info(f"[AuraLite-HF] Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}")

        self.model.eval()
        logger.info("[AuraLite-HF] Manual fine-tuning complete.")

        # Save where the caller asked (until 2.6.5 this was a hard-coded
        # "./hf_lora_manual" in the current working directory).
        out_dir = output_dir
        os.makedirs(out_dir, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        logger.info(f"[AuraLite-HF] LoRA adapter saved to: {out_dir}")
        return out_dir

    # ------------------------------------------------------------------
    # Save / Load adapters
    # ------------------------------------------------------------------

    def save_lora_adapter(self, path: str):
        """Save only the LoRA adapter weights (very small file)."""
        if not self.is_peft or self.model is None:
            raise ValueError("No LoRA adapter to save (call apply_lora first)")
        self.model.save_pretrained(path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(path)
        logger.info(f"[AuraLite-HF] LoRA adapter saved to {path}")

    def load_lora_adapter(self, adapter_path: str, *, trainable: bool = True):
        """Load a previously saved LoRA adapter on top of the current base model.

        If an adapter is already applied (e.g. the GUI's default "Apply LoRA on
        load"), it is *replaced*. Until 2.6.5 a second `PeftModel` was wrapped
        around the first one: every key gained a second `base_model.model.`
        prefix, peft reported "missing adapter keys" and the saved weights were
        silently not loaded. `trainable=True` keeps the adapter trainable so
        fine-tuning can continue from it (a plain `from_pretrained` freezes it).
        """
        if self.model is None:
            raise ValueError("Base model must be loaded before loading adapter")

        _check_hf_support()

        base = self.model
        if self.is_peft and hasattr(base, "unload"):
            base = base.unload()   # drop the current LoRA layers (no merge)
        self.model = PeftModel.from_pretrained(base, adapter_path, is_trainable=trainable)
        self.is_peft = True
        config = getattr(self.model, "peft_config", {}).get("default")
        if config is not None:
            targets = getattr(config, "target_modules", None)
            self.lora_config = {
                "rank": getattr(config, "r", None),
                "alpha": getattr(config, "lora_alpha", None),
                "dropout": getattr(config, "lora_dropout", None),
                "target_modules": sorted(targets) if targets else targets,
            }
        logger.info(f"[AuraLite-HF] LoRA adapter loaded from {adapter_path}")

    # ------------------------------------------------------------------
    # Utility / Info
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

    def count_trainable_parameters(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_info(self) -> dict:
        return {
            "backend": "huggingface",
            "model": self.model_name_or_path,
            "is_peft": self.is_peft,
            "is_quantized": self.is_quantized,
            "lora_config": self.lora_config,
            "parameters": self.count_parameters(),
            "trainable_parameters": self.count_trainable_parameters(),
            "max_seq_len": self.max_seq_len,
            "device": str(self.device),
        }

    def reset_cache(self):
        """No-op for compatibility with AuraLite native API."""
        pass

    # ------------------------------------------------------------------
    # Hugging Face Hub push / pull (NEW v2.6)
    # ------------------------------------------------------------------

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload AuraLite HF model",
        private: bool = False,
        token: Optional[str] = None,
        create_pr: bool = False,
    ):
        """
        Push the current model (and tokenizer) to the Hugging Face Hub.

        Works for both base models and LoRA adapters.
        Requires `huggingface_hub` (usually comes with transformers).
        """
        _check_hf_support()

        if self.model is None or self.tokenizer is None:
            raise ValueError("No model loaded to push.")

        if not _huggingface_hub_available():
            raise HFNotAvailableError("huggingface_hub is required for push_to_hub.")

        logger.info(f"[AuraLite-HF] Pushing model to https://huggingface.co/{repo_id} ...")

        # The token goes to these two calls only. Until 2.6.5 it went through
        # `huggingface_hub.login()`, which stores it on disk for every later
        # process — a side effect nobody asked for.
        auth = {"token": token} if token else {}
        self.model.push_to_hub(
            repo_id=repo_id,
            commit_message=commit_message,
            private=private,
            create_pr=create_pr,
            **auth,
        )
        self.tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message=commit_message,
            private=private,
            **auth,
        )

        logger.info(f"[AuraLite-HF] ✅ Successfully pushed to {repo_id}")

    def from_pretrained(
        self,
        repo_id: str,
        *,
        revision: str = "main",
        token: Optional[str] = None,
        **load_kwargs,
    ):
        """
        Load a model directly from the Hugging Face Hub (or a specific revision).

        This is a convenience wrapper around `load_model`. (`revision` and
        `token` were accepted but never forwarded before v2.6.6.)
        """
        self.load_model(
            repo_id,
            local_files_only=False,
            revision=revision,
            token=token,
            **load_kwargs,
        )


# ----------------------------------------------------------------------
# Convenience function for the engine
# ----------------------------------------------------------------------

def create_hf_proxy() -> HuggingFaceProxy:
    """Factory function."""
    _check_hf_support()
    return HuggingFaceProxy()