"""
hf_integration.py — Hugging Face model support for AuraLite AI

Adds the ability to:
- Load ANY Hugging Face causal LM (Llama, Mistral, Qwen, Gemma, Phi, etc.)
- Load in 4-bit / 8-bit (QLoRA ready via bitsandbytes)
- Apply LoRA / QLoRA adapters using PEFT
- Fine-tune with LoRA/QLoRA (full or parameter-efficient)
- Generate text (with streaming support)
- Save / load LoRA adapters

This module is designed to work alongside the native AuraLite ModernTransformer
and GGUF backends. It exposes a similar interface where possible.
"""

from __future__ import annotations

import os
import json
import time
from typing import Any, Iterator, Optional, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optional heavy dependencies — gracefully degrade if missing
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        PeftModel,
        PeftConfig,
        prepare_model_for_kbit_training,
    )
    import bitsandbytes as bnb
    HAS_HF_SUPPORT = True
except ImportError as e:
    HAS_HF_SUPPORT = False
    _HF_IMPORT_ERROR = str(e)


class HFNotAvailableError(ImportError):
    """Raised when Hugging Face / PEFT dependencies are missing."""
    pass


def _check_hf_support():
    if not HAS_HF_SUPPORT:
        raise HFNotAvailableError(
            "Hugging Face + LoRA/QLoRA support requires extra packages.\n"
            "Install them with:\n"
            "  pip install -r requirements.txt\n"
            "(transformers, peft, accelerate, bitsandbytes, sentencepiece, etc.)\n\n"
            f"Original import error: {_HF_IMPORT_ERROR}"
        )


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
        # For causal LM, labels = input_ids (shifted inside the model)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
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
        trust_remote_code: bool = True,
        use_fast_tokenizer: bool = True,
        max_seq_len: int = 4096,
        local_files_only: bool = False,
        verbose: bool = True,
    ):
        """
        Load any causal LM from Hugging Face Hub **or from a local folder**.

        Perfect for "already downloaded" models:
        - Use `local_files_only=True` for completely offline loading.
        - `model_name_or_path` can be a local path like:
          ~/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/xxxxxxx
          or any folder that contains config.json + model weights.

        Supports 4-bit (QLoRA) and 8-bit loading via bitsandbytes.
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
                print(f"[AuraLite-HF] Loading LOCAL / already downloaded model from: {model_name_or_path}")
            else:
                print(f"[AuraLite-HF] Loading model from Hub: {model_name_or_path}")

        # Determine quantization config
        bnb_config = None
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "device_map": device_map,
        }

        if load_in_4bit or load_in_8bit:
            if not torch.cuda.is_available():
                print("[AuraLite-HF] WARNING: 4/8-bit quantization requires CUDA. Falling back to full-precision CPU loading.")
                load_in_4bit = False
                load_in_8bit = False
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
                    print(f"[AuraLite-HF] Using {'4-bit' if load_in_4bit else '8-bit'} quantization (QLoRA ready)")

        if torch_dtype is None:
            if load_in_4bit or load_in_8bit:
                torch_dtype = torch.float16
            elif torch.cuda.is_available():
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                torch_dtype = torch.float32

        load_kwargs["torch_dtype"] = torch_dtype

        # Load tokenizer (support local folders + offline mode)
        tokenizer_kwargs = {
            "use_fast": use_fast_tokenizer,
            "trust_remote_code": trust_remote_code,
        }
        if local_files_only:
            tokenizer_kwargs["local_files_only"] = True

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, **tokenizer_kwargs
            )
        except Exception as e:
            print(f"[AuraLite-HF] Tokenizer warning: {e}. Trying without fast tokenizer...")
            tokenizer_kwargs["use_fast"] = False
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, **tokenizer_kwargs
            )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|endoftext|>"
            if verbose:
                print("[AuraLite-HF] Set pad_token = eos_token")

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
            print(f"[AuraLite-HF] Model loaded. Parameters: {n_params / 1e6:.1f}M")
            print(f"[AuraLite-HF] Device: {self.device}, dtype: {self.model.dtype}")

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
            print("[AuraLite-HF] LoRA already applied. Skipping.")
            return

        if verbose:
            print(f"[AuraLite-HF] Applying LoRA (rank={rank}, alpha={alpha})...")

        # Prepare for k-bit training if quantized
        if self.is_quantized:
            self.model = prepare_model_for_kbit_training(self.model)
            if verbose:
                print("[AuraLite-HF] Model prepared for k-bit (QLoRA) training")

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
            print("[AuraLite-HF] LoRA adapters merged and removed.")
        except Exception as e:
            print(f"[AuraLite-HF] Could not merge LoRA: {e}")

    # ------------------------------------------------------------------
    # Generation (compatible interface)
    # ------------------------------------------------------------------

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

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        do_sample = bool(do_sample and temperature > 0)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

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

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]

        # Use transformers TextIteratorStreamer if available
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
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": temperature > 0,
                "streamer": streamer,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            }

            import threading
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            for text in streamer:
                yield text

            thread.join()
            return
        except Exception:
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
            print("[AuraLite-HF] No LoRA applied — applying default LoRA before fine-tuning...")
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
            )

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
            fp16=torch.cuda.is_available(),
            bf16=torch.cuda.is_bf16_supported(),
            report_to="none",           # disable wandb etc.
            remove_unused_columns=False,
            dataloader_num_workers=2,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        print("[AuraLite-HF] Starting fine-tuning with Hugging Face Trainer...")
        trainer.train()
        print("[AuraLite-HF] Training finished.")

        # Save adapter
        self.model.save_pretrained(kwargs["output_dir"])
        self.tokenizer.save_pretrained(kwargs["output_dir"])
        print(f"[AuraLite-HF] LoRA adapter saved to: {kwargs['output_dir']}")

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

            print(f"[AuraLite-HF] Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}")

        self.model.eval()
        print("[AuraLite-HF] Manual fine-tuning complete.")

        # Save
        out_dir = "hf_lora_manual"
        os.makedirs(out_dir, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
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
        print(f"[AuraLite-HF] LoRA adapter saved to {path}")

    def load_lora_adapter(self, adapter_path: str):
        """Load a previously saved LoRA adapter on top of the current base model."""
        if self.model is None:
            raise ValueError("Base model must be loaded before loading adapter")

        _check_hf_support()

        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.is_peft = True
        print(f"[AuraLite-HF] LoRA adapter loaded from {adapter_path}")

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

        try:
            from huggingface_hub import HfApi, login
        except ImportError:
            raise HFNotAvailableError("huggingface_hub is required for push_to_hub.")

        if token:
            login(token=token)

        print(f"[AuraLite-HF] Pushing model to https://huggingface.co/{repo_id} ...")

        self.model.push_to_hub(
            repo_id=repo_id,
            commit_message=commit_message,
            private=private,
            create_pr=create_pr,
        )
        self.tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message=commit_message,
            private=private,
        )

        print(f"[AuraLite-HF] ✅ Successfully pushed to {repo_id}")

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

        This is a convenience wrapper around `load_model`.
        """
        self.load_model(
            repo_id,
            local_files_only=False,
            **load_kwargs,
        )


# ----------------------------------------------------------------------
# Convenience function for the engine
# ----------------------------------------------------------------------

def create_hf_proxy() -> HuggingFaceProxy:
    """Factory function."""
    _check_hf_support()
    return HuggingFaceProxy()