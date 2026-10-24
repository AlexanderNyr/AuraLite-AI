import os
import time

# -----------------------------------------------------------------------
# CPU multithreading — set BEFORE importing torch / numpy so the OpenMP /
# MKL / OpenBLAS backends pick them up.
# -----------------------------------------------------------------------
_CPU_COUNT = os.cpu_count() or 1


def _cpu_thread_counts() -> tuple[int, int]:
    """(intra-op, inter-op) thread counts with sane CPU defaults and env overrides.

    Historical behaviour spawned `cpu_count` *inter-op* threads, which mostly
    adds scheduling overhead: generation runs tiny sequential ops, so inter-op
    parallelism is wasted. Defaults are now:
      intra-op = AURALITE_NUM_THREADS   (default cpu_count, capped at 64)
      inter-op = AURALITE_INTEROP_THREADS (default 1, capped at 8)
    """
    intra = max(1, min(int(os.environ.get("AURALITE_NUM_THREADS", _CPU_COUNT)), 64))
    interop = max(1, min(int(os.environ.get("AURALITE_INTEROP_THREADS", 1)), 8))
    return intra, interop


_INTRA_THREADS, _INTEROP_THREADS = _cpu_thread_counts()
os.environ.setdefault("OMP_NUM_THREADS", str(_INTRA_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_INTRA_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_INTRA_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_INTRA_THREADS))

# -----------------------------------------------------------------------
# Distributed training (DDP) support — v2.3
# -----------------------------------------------------------------------
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import json
import logging
import math
from collections import Counter, defaultdict
from uuid import uuid4
from pathlib import Path
from typing import Callable, Iterator, Any, List, Dict, Optional, cast

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from auralite.engine.logging_utils import canonical_level, engine_logging

logger = logging.getLogger("auralite")


# The engine used to `print()` its progress to stdout; v2.6.5 routes everything
# through this logger instead. Messages keep their historical "[AuraLite] ..."
# / "[AuraLite] WARNING: ..." wording, so terminal output stays recognisable
# while applications can now silence or redirect it. DynamicStreamHandler
# re-resolves sys.stderr on every record, which keeps pytest's `capsys` and the
# GUI's console redirector working; `engine_logging()` is idempotent, so a
# front-end (CLI, GUI, tests) may reconfigure the logger and restore it later.
engine_logging(level=canonical_level(), stream_name="stderr")


# Safe checkpoint I/O lives in auralite/engine/safe_io.py (shared with
# image_engine). `.pt` files are loaded with weights_only=True by default.
from auralite.engine.safe_io import (  # noqa: E402,F401
    numpy_rng_state_to_builtins as _numpy_rng_state_to_builtins,
    numpy_rng_state_from_builtins as _numpy_rng_state_from_builtins,
    safe_torch_load as _safe_torch_load,
    save_safetensors_checkpoint as _save_safetensors_checkpoint,
    load_safetensors_checkpoint as _load_safetensors_checkpoint,
)


# ---------------------------------------------------------------------------
# Public re-exports (v2.6.5 decomposition)
#
# The classes below used to live in this file; they now have their own modules
# (`auralite/engine/tokenizers.py`, `layers.py`, `model.py`, `dataset.py`,
# `scheduler.py`, `optimizers.py`, `gguf_proxy.py`). They are imported back here
# so `auralite.engine._legacy` — and therefore `auralite.engine` — keeps exposing
# exactly the same names as before.
# ---------------------------------------------------------------------------
from auralite.engine.tokenizers import (  # noqa: E402,F401
    UNK_TOKEN, BPETokenizer, CharTokenizer, tokenizer_from_dict,
)
# 2.14.1: plugged-in BPE tokenizers, tokenizer files, training only the tokenizer
from auralite.engine.hf_bpe import HFBPETokenizer, TokenizerFormatError  # noqa: E402,F401
from auralite.engine.tokenizer_io import (  # noqa: E402,F401
    describe_tokenizer, load_tokenizer, read_tokenizer_file, save_tokenizer, train_tokenizer,
    with_characters, with_special_tokens,
)
from auralite.engine.layers import (  # noqa: E402,F401
    _FLEX_COMPILED, Attention, FeedForward, HeadwiseRMSNorm, RMSNorm, Top2MoE,
    TransformerBlock, _flex_attention_available, _flex_block_mask,
    _get_flex_attention, _sdpa_supports_gqa,
)
from auralite.engine.model import (  # noqa: E402,F401
    LoRALayer, ModernTransformer, estimate_n_params,
)
from auralite.engine.dataset import (  # noqa: E402,F401
    CharDataset, WindowBatches, spread_indices, validation_split_point,
)
# 2.15: training data from files, folders and globs; dialogs for chat fine-tuning
from auralite.engine.corpus import (  # noqa: E402,F401
    DataFormatError, TextCorpus, collect_files, data_kind, load_training_data,
)
from auralite.engine.dialogs import DialogSet  # noqa: E402,F401
from auralite.engine.context import ContextWindowMixin  # noqa: E402
from auralite.engine.decode_slots import DecodeSlot, DecodeSlotsMixin  # noqa: E402
from auralite.engine.inference_bench import InferenceBenchMixin  # noqa: E402
from auralite.engine.generation import GenerationMixin, _DraftSession  # noqa: E402,F401
from auralite.engine.trainer import TrainerMixin, _remove_file_quietly  # noqa: E402,F401
from auralite.engine import sampling as _sampling  # noqa: E402
from auralite.engine.sampling import SamplingParams  # noqa: E402,F401
from auralite.engine.scheduler import CosineWarmupScheduler, WSDScheduler  # noqa: E402,F401
from auralite.engine.recipe import DEFAULT_RECIPE  # noqa: E402
from auralite.engine.train_control import TrainControl  # noqa: E402
from auralite.engine import compute_thread as _compute  # noqa: E402
from auralite.engine.optimizers import (  # noqa: E402,F401
    Muon, _ChainedOptimizers, _muon_adjust_lr, _newton_schulz_ortho,
    split_parameters_for_muon,
    adamw_param_groups,
)
from auralite.engine.gguf_proxy import (  # noqa: E402,F401
    is_gguf_like,
    GGUFModelProxy, GGUFNotAvailableError, GGUFTokenizerProxy,
    _gguf_rolling_loglikelihood,
)


# Optional Hugging Face + LoRA/QLoRA support
try:
    from auralite.hf_integration import HuggingFaceProxy, HFNotAvailableError, create_hf_proxy, HAS_HF_SUPPORT
except ImportError:
    HuggingFaceProxy = None
    HFNotAvailableError = Exception
    create_hf_proxy = None
    HAS_HF_SUPPORT = False

# Chat interface (NEW v2.3)
try:
    from auralite.chat_interface import (
        ChatMessage, ChatHistory, apply_chat_template,
        CHAT_TEMPLATES, get_stop_tokens, build_chat_prompt
    )
    HAS_CHAT_SUPPORT = True
except ImportError:
    HAS_CHAT_SUPPORT = False
    ChatMessage = None
    ChatHistory = None
    apply_chat_template = None
    CHAT_TEMPLATES = {}
    get_stop_tokens = lambda x: []
    build_chat_prompt = None

def configure_cpu_threads() -> tuple[int, int]:
    """Apply the intra/inter-op thread counts (idempotent-safe).

    Called once at import; exposed so tests and embedders can re-read the
    environment (torch raises if inter-op threads are set after parallel work
    started, which we deliberately swallow here).
    """
    global _INTRA_THREADS, _INTEROP_THREADS
    _INTRA_THREADS, _INTEROP_THREADS = _cpu_thread_counts()
    try:
        torch.set_num_threads(_INTRA_THREADS)
    except (RuntimeError, ValueError):
        pass
    try:
        torch.set_num_interop_threads(_INTEROP_THREADS)
    except (RuntimeError, ValueError):
        pass
    return _INTRA_THREADS, _INTEROP_THREADS


configure_cpu_threads()


# Parameter validation lives in auralite/engine/params.py (2.14).
from auralite.engine.params import (  # noqa: E402,F401
    KNOWN_TRAIN_PARAMS,
    ParamValidationError,
    _TRAIN_PARAM_SYNONYMS,
    describe_unknown_params,
    unknown_train_params,
    validate_params,
)


# ===================================================================
#  Tokenizers — character-level and BPE (Byte/Char Pair Encoding)
# ===================================================================


# ===================================================================
#  Auto-recommendation helpers
# ===================================================================


def recommend_epochs(n_tokens: int, n_params: int,
                     batch_size: int, seq_length: int,
                     tokens_per_param: float = 20.0,
                     min_epochs: int = 5,
                     max_epochs: int = 500) -> int:
    """Recommend a sensible number of training epochs.

    Strategy (tiered, more realistic than pure Chinchilla which assumes
    you can throw arbitrarily many tokens at the model):

      * For HUGE datasets (>= 10× model capacity in tokens) → 3–10 epochs
        is enough to see everything ~enough times.
      * For MEDIUM datasets → aim for Chinchilla-ish 20 tokens/param
        of total exposure, but capped sensibly.
      * For TINY datasets → enough passes to over-fit / memorise, but
        capped at a reasonable wall-clock budget (≈ 100 epochs by default).

    Args:
        n_tokens:        size of the training corpus after tokenisation
        n_params:        total model parameters
        batch_size:      training batch size (currently unused — kept for API)
        seq_length:      context window
        tokens_per_param: target tokens-to-param ratio (default 20)
        min_epochs:      lower clamp
        max_epochs:      upper clamp (hard ceiling, normally not reached)

    Returns:
        recommended epoch count
    """
    if n_tokens <= seq_length:
        return min_epochs

    # How big is the dataset compared to model capacity?
    # ratio < 1  → dataset smaller than the model can memorise comfortably
    # ratio = 1  → ~Chinchilla optimal (one pass = 20 tok/param)
    # ratio > 1  → plenty of data, few passes needed
    target = tokens_per_param * n_params
    ratio = n_tokens / max(1, target)

    import math
    if ratio >= 10:
        # Huge dataset: 3 passes is plenty.
        epochs = 3
    elif ratio >= 1:
        # Comfortable: 3-15 passes, scaling down with dataset size.
        # ratio=10 → 3 epochs ; ratio=1 → 15 epochs (smooth log interp)
        epochs = int(round(15 - 12 * (math.log10(ratio) / 1.0)))
    else:
        # Small dataset: pure Chinchilla says epochs = 1/ratio, which
        # explodes for tiny files (1KB on a 1M-param model → 6000 epochs).
        # Instead use logarithmic saturation:
        #   ratio=0.5  → ~25 epochs
        #   ratio=0.1  → ~50 epochs
        #   ratio=0.01 → ~80 epochs
        #   ratio→0    → 100 epochs (asymptote)
        # Formula: 100 - 100 / (1 + (-log10(ratio))**1.5 * k)
        log_inv = -math.log10(max(ratio, 1e-9))           # 0 .. ~9
        # Map log_inv in [0, 4] to epochs in [15, 100] smoothly:
        # at ratio=1   (log_inv=0)   → 15 epochs (continuity with branch above)
        # at ratio=0.1 (log_inv=1)   → ~50
        # at ratio=0.01(log_inv=2)   → ~75
        # at ratio=1e-4(log_inv=4)   → ~95
        epochs = int(round(15 + 85 * (1 - 1 / (1 + 0.6 * log_inv ** 1.3))))

    return int(max(min_epochs, min(max_epochs, epochs)))


def recommend_gen_length(seed_str: str,
                         tokenizer,
                         max_seq_len: int = 4096,
                         multiplier: float = 8.0,
                         hard_min: int = 30,
                         hard_max: int = 800,
                         *, full_context: bool = False) -> int:
    """Recommend a generation length (in tokens) for a given seed.

    Aims for `multiplier` times the seed length so short prompts still
    produce meaningful output, bounded below by `hard_min` and above by
    max(hard_max, 2*max_seq_len, 1024). Outputs longer than the training
    window are fine because generation attends to the newest `max_seq_len`
    tokens (the 2.11 context window). With ``full_context=True`` (no window)
    everything past `max_seq_len` degrades into noise, so the recommendation
    stops at the room the prompt leaves inside it.

    Examples (multiplier=8, max_seq_len=64):
        seed_tokens=1   → 30    (hard_min)
        seed_tokens=5   → 40
        seed_tokens=10  → 80
        seed_tokens=50  → 400   (full_context: 14)
        seed_tokens=200 → 1024  (the ceiling)

    Args:
        seed_str:   the prompt the user typed
        tokenizer:  trained CharTokenizer/BPETokenizer (None = use chars)
        max_seq_len: model.max_seq_len, the length the model was trained on
        multiplier: how much longer than the seed the output should be
        hard_min/hard_max: absolute clamps
        full_context: generation runs without the context window

    Returns:
        recommended generation length (in tokens)
    """
    if tokenizer is not None:
        try:
            seed_tokens = len(tokenizer.encode(seed_str)) if seed_str else 0
        except Exception:
            seed_tokens = len(seed_str)
    else:
        seed_tokens = len(seed_str)

    seed_tokens = max(1, seed_tokens)
    raw = int(round(seed_tokens * multiplier))
    effective_max = max(hard_max, max_seq_len * 2, 1024)
    raw = max(hard_min, min(effective_max, raw))
    if full_context:
        raw = min(raw, max(1, int(max_seq_len) - seed_tokens))
    return max(1, raw)


# ===================================================================
#  Modern Building Blocks  (LLaMA / Mistral / Qwen heritage, 2025-26)
# ===================================================================


# -------------------------------------------------------------------


# -------------------------------------------------------------------

_SDPA_GQA_SUPPORTED: bool | None = None


# -------------------------------------------------------------------
#  FlexAttention support (v2.6.5) — opt-in via use_flex_attention=True
# -------------------------------------------------------------------


# -------------------------------------------------------------------


# -------------------------------------------------------------------


# ===================================================================


# ===================================================================
#  Dataset
# ===================================================================


# ===================================================================
#  Cosine Learning-Rate Schedule with Linear Warmup
# ===================================================================


# ===================================================================
#  Muon optimizer (Newton–Schulz orthogonalized momentum) — v2.6
# ===================================================================
#
# Muon (Keller Jordan et al., 2024; Moonlight/Kimi K2 adoption) replaces
# AdamW for the *hidden* 2-D weight matrices of the transformer body.
# Idea: take the momentum buffer, orthogonalize the update matrix via a
# quintic Newton–Schulz iteration (i.e. approximate UV^T of the SVD), then
# apply it with a shape-aware learning rate. Embeddings, the LM head and
# all norm/scalar parameters must stay on AdamW — see
# :func:`split_parameters_for_muon`.


# ===================================================================
#  GGUF / llama.cpp backend (inference-only)
# ===================================================================


# ===================================================================
#  Engine
# ===================================================================


class AuraLiteEngine(GenerationMixin, TrainerMixin, ContextWindowMixin,
                     InferenceBenchMixin, DecodeSlotsMixin):
    """Modern training & inference engine for AuraLite AI v2.1.

    IMPROVED (v2.1+):
    - Gradient accumulation for large models on weak hardware
    - cudnn.benchmark + TF32 for CUDA performance
    - generate_streaming() for real-time generation output
    - generate_batch() for parallel generation of multiple prompts
    - validate_params() before training starts
    """

    def __init__(self):
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_threads = torch.get_num_threads()
        self.model: ModernTransformer | GGUFModelProxy | HuggingFaceProxy | None = None
        self.backend     = "torch"           # "torch", "gguf", or "huggingface"
        self.gguf_path: str | None = None
        self.hf_path: str | None = None
        self.hf_proxy: HuggingFaceProxy | None = None
        self.optimizer   = None
        self.scheduler   = None
        self.scaler      = None
        self.tokenizer   = None              # CharTokenizer | BPETokenizer | GGUFTokenizerProxy | HF tokenizer
        self.vocab_size  = 0
        self.params_used: dict = {}          # remember last training/load params
        self._compiled_inference_model = None  # torch.compile for generation (v2.6.5)
        self._compile_seconds = 0.0
        self.last_val_loss: float | None = None
        #: validation loss in bits per UTF-8 byte (2.11): comparable across
        #: tokenizers, unlike the loss per token
        self.last_val_bpb: float | None = None
        #: what the last `train()` call did (2.9) — see `train` for the keys
        self.last_train_summary: dict | None = None
        # v2.6.2: chat template persisted with the checkpoint; generate_chat()
        # uses it when the caller does not pass `chat_template` explicitly.
        self.default_chat_template: str = "chatml"
        # 2.11: "auto" (the trained length), 0 (no window) or a token count —
        # see set_context_window().
        self._context_window_setting: int | str = "auto"
        self._resume_optimizer_state = None
        self._resume_scheduler_state = None
        self._resume_scaler_state = None

        # DDP (DistributedDataParallel) support — v2.3
        self.is_distributed = False
        self.local_rank = 0
        self.world_size = 1
        self._ddp_model = None
        # 2.14: the ClusterSession of a run on several computers (cluster.py)
        self._cluster: Any = None

        # Auto-detect distributed environment (torchrun / torch.distributed.launch)
        self._try_init_distributed_from_env()

        # CUDA performance tuning
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except AttributeError:
                pass  # older PyTorch versions

    def _try_init_distributed_from_env(self):
        """Initialize DDP if running under torchrun or torch.distributed.launch."""
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            try:
                rank = int(os.environ["RANK"])
                world_size = int(os.environ["WORLD_SIZE"])
                local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))

                dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo",
                                        rank=rank, world_size=world_size)

                if torch.cuda.is_available():
                    torch.cuda.set_device(local_rank)
                    self.device = torch.device(f"cuda:{local_rank}")

                self.is_distributed = True
                self.local_rank = local_rank
                self.world_size = world_size

                logger.info(f"[AuraLite] DDP initialized: rank={rank}, world_size={world_size}, "
                            f"local_rank={local_rank}, device={self.device}")
            except Exception as e:
                logger.warning(f"[AuraLite] WARNING: Failed to initialize DDP: {e}")
                self.is_distributed = False

    def is_gguf_model(self) -> bool:
        return self.backend == "gguf" or is_gguf_like(self.model)

    def is_hf_model(self) -> bool:
        return self.backend == "huggingface" or isinstance(self.model, HuggingFaceProxy)

    # ===================================================================
    #  Chat / Instruction Mode (NEW v2.3)
    # ===================================================================

    # ---- Stop-sequence helpers (v2.6.1) --------------------------------
    # Previously the `stop_tokens` argument of generate_chat() was accepted
    # but silently ignored, and template stop markers (e.g. <|im_end|>) were
    # never applied for the native backend, so chat generations ran past the
    # end of the assistant turn and leaked template artifacts.
    def _resolve_stop_strings(self, stop_tokens: Optional[List[str]] | None,
                              chat_template: str) -> list[str]:
        """Explicit stop strings override the chat-template defaults."""
        if stop_tokens:
            return [s for s in stop_tokens if s]
        try:
            return [s for s in get_stop_tokens(chat_template) if s]
        except Exception:
            return []

    def _stop_token_ids(self, stop_strings: list[str]) -> set[int]:
        """Single-token stop strings can match during sampling (early exit)."""
        ids: set[int] = set()
        for s in stop_strings:
            try:
                enc = self.encode(s)
            except Exception:
                continue
            if len(enc) == 1:
                ids.add(enc[0])
        return ids

    @staticmethod
    def _truncate_at_stop(text: str, stop_strings: list[str]) -> str:
        """Cut text at the earliest occurrence of any stop string."""
        best: int | None = None
        for s in stop_strings:
            i = text.find(s)
            if i != -1 and (best is None or i < best):
                best = i
        return text[:best] if best is not None else text

    def generate_chat(
        self,
        messages: List[Dict[str, str]] | "ChatHistory",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        min_p: float = 0.0,
        chat_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stop_tokens: Optional[List[str]] = None,
        *,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        on_token: Any = None,
    ) -> str:
        """
        Generate a response in chat/instruction mode.

        Args:
            messages: List of {"role": "...", "content": "..."} or ChatHistory
            max_new_tokens: how many tokens to generate
            chat_template: one of CHAT_TEMPLATES keys
            system_prompt: optional system message to prepend

        Returns:
            Generated assistant response (without the prompt).
        """
        if not HAS_CHAT_SUPPORT:
            raise RuntimeError("chat_interface.py is required for chat mode.")
        if chat_template is None:
            chat_template = self.default_chat_template

        # Convert to ChatHistory if needed
        if isinstance(messages, list):
            history = ChatHistory.from_list(messages)
        else:
            history = messages

        # Add system prompt if provided
        if system_prompt and not any(m.role == "system" for m in history.messages):
            history.messages.insert(0, ChatMessage(role="system", content=system_prompt))

        prompt = apply_chat_template(history, template_name=chat_template, add_generation_prompt=True)
        stop_strs = self._resolve_stop_strings(stop_tokens, chat_template)
        if on_token is not None:
            self._require_native_for_token_callback()

        # Generate
        if self.is_gguf_model():
            # GGUF chat completion
            if self.model.use_chat_completion:
                # Pass the conversation history as structured messages
                if isinstance(history, ChatHistory):
                    chat_messages = history.to_list()
                else:
                    chat_messages = messages
                result = self.model.create_chat_completion(
                    messages=chat_messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repetition_penalty, **self._gguf_penalty_kwargs(presence_penalty, frequency_penalty),
                    min_p=min_p,
                    stream=False,
                    stop=stop_strs,
                )
                try:
                    content = result["choices"][0]["message"]["content"] or ""
                except Exception:
                    return ""
                # Same contract as every other path (and as streaming): stripped.
                return content.strip()
            else:
                # Fallback to regular completion with formatted prompt
                full = self._gguf_generate_text(
                    prompt, max_new_tokens, temperature, top_k, top_p,
                    repetition_penalty, min_p, stop=stop_strs,
                    presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
                )
                return self._truncate_at_stop(full[len(prompt):], stop_strs).strip()

        elif self.is_hf_model():
            # HF models usually have their own chat template
            kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if min_p > 0:
                kwargs["min_p"] = min_p
            full = self.hf_proxy.generate(prompt, **kwargs)
            text = full[len(prompt):] if full.startswith(prompt) else full
            return self._truncate_at_stop(text, stop_strs).strip()

        else:
            # Native AuraLite model
            ids = self._prepare_prompt_ids(prompt, max_gen_tokens=max_new_tokens)
            result_ids = self._generate_ids(
                ids, max_new_tokens, temperature, top_k, top_p,
                repetition_penalty, min_p=min_p,
                stop_token_ids=self._stop_token_ids(stop_strs),
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                on_token=on_token,
            )
            # Return only the newly generated part (token-based slice),
            # cut at the first stop marker (incl. multi-token ones).
            generated_ids = result_ids[len(ids):]
            text = self._decode_after(ids, generated_ids)
            return self._truncate_at_stop(text, stop_strs).strip()

    def generate_chat_streaming(
        self,
        messages: List[Dict[str, str]] | "ChatHistory",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        min_p: float = 0.0,
        chat_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stop_tokens: Optional[List[str]] = None,
        *,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        on_token: Any = None,
    ) -> Iterator[str]:
        """
        Streaming version of generate_chat — yields text pieces one by one.

        Works with native AuraLite and GGUF (when streaming is supported).

        The concatenated pieces are exactly what `generate_chat()` returns,
        which strips the answer: leading whitespace of the first piece is
        trimmed (v2.6.5), and whitespace is held back until more text follows,
        so whitespace at the very end is never emitted. Before this, an answer
        that ended in a space came out as "Hello. " over streaming but "Hello."
        from `generate_chat()` — visible only on machines whose float rounding
        made the tiny test model end in a space. Internal spacing is untouched.
        """
        started = False
        pending = ""          # trailing whitespace, emitted only if text follows
        for piece in self._generate_chat_streaming_raw(
                messages, max_new_tokens, temperature, top_k, top_p,
                repetition_penalty, min_p, chat_template, system_prompt,
                stop_tokens, presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty, on_token=on_token):
            if not piece:
                continue
            if not started:
                piece = piece.lstrip()
                if not piece:
                    continue
                started = True
            text = pending + piece
            body = text.rstrip()
            pending = text[len(body):]
            if body:
                yield body
        # Whatever is still pending was trailing whitespace: dropped, as strip() does.

    def _generate_chat_streaming_raw(
        self,
        messages: List[Dict[str, str]] | "ChatHistory",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        min_p: float = 0.0,
        chat_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stop_tokens: Optional[List[str]] = None,
        *,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        on_token: Any = None,
    ) -> Iterator[str]:
        """Unprocessed streaming implementation (see generate_chat_streaming)."""
        if not HAS_CHAT_SUPPORT:
            raise RuntimeError("chat_interface.py is required for chat mode.")
        if chat_template is None:
            chat_template = self.default_chat_template

        if isinstance(messages, list):
            history = ChatHistory.from_list(messages)
        else:
            history = messages

        if system_prompt and not any(m.role == "system" for m in history.messages):
            history.messages.insert(0, ChatMessage(role="system", content=system_prompt))

        prompt = apply_chat_template(history, template_name=chat_template, add_generation_prompt=True)
        stop_strs = self._resolve_stop_strings(stop_tokens, chat_template)
        if on_token is not None:
            self._require_native_for_token_callback()
        text_buffer = ""

        def _consume(piece: str) -> tuple[str, bool]:
            """Accumulate decoded text; return (emittable_text, hit_stop).

            Stop strings are cut even when they span multiple yielded pieces;
            a tail that is a prefix of a stop string is held back either way.
            """
            nonlocal text_buffer
            text_buffer += piece
            cut_pos: int | None = None
            for s in stop_strs:
                i = text_buffer.find(s)
                if i != -1 and (cut_pos is None or i < cut_pos):
                    cut_pos = i
            if cut_pos is not None:
                out = text_buffer[:cut_pos]
                text_buffer = ""
                return out, True
            keep = 0
            for s in stop_strs:
                for k in range(min(len(s) - 1, len(text_buffer)), 0, -1):
                    if text_buffer.endswith(s[:k]):
                        keep = max(keep, k)
                        break
            if keep:
                out, text_buffer = text_buffer[:-keep], text_buffer[-keep:]
            else:
                out, text_buffer = text_buffer, ""
            return out, False

        if self.is_gguf_model():
            if self.model.use_chat_completion:
                stream = self.model.create_chat_completion(
                    history.to_list(),
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repetition_penalty, **self._gguf_penalty_kwargs(presence_penalty, frequency_penalty),
                    min_p=min_p,
                    stream=True,
                    stop=stop_strs,
                )
                for chunk in stream:
                    try:
                        delta = chunk["choices"][0].get("delta", {}).get("content", "")
                        if delta:
                            out, hit = _consume(delta)
                            if out:
                                yield out
                            if hit:
                                return
                    except Exception:
                        continue
                if text_buffer:
                    yield text_buffer
            else:
                # Fallback: non-streaming
                full = self.generate_chat(
                    messages, max_new_tokens, temperature, top_k, top_p,
                    repetition_penalty, min_p, chat_template, system_prompt
                )
                yield full
            return

        if self.is_hf_model():
            # HF streaming (if supported by the proxy)
            kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if min_p > 0:
                kwargs["min_p"] = min_p
            try:
                for token in self.hf_proxy.generate_streaming(prompt, **kwargs):
                    out, hit = _consume(token)
                    if out:
                        yield out
                    if hit:
                        return
                if text_buffer:
                    yield text_buffer
            except Exception:
                # Fallback
                result = self.hf_proxy.generate(prompt, **kwargs)
                text = result[len(prompt):] if result.startswith(prompt) else result
                yield self._truncate_at_stop(text, stop_strs)
            return

        # Native AuraLite streaming (the shared decode loop, 2.8)
        ids = self._prepare_prompt_ids(prompt, max_gen_tokens=max_new_tokens)
        stop_ids = self._stop_token_ids(stop_strs)
        if max_new_tokens <= 0:
            return
        sp = self._sampling_params(temperature, top_k, top_p, repetition_penalty, min_p,
                                   presence_penalty, frequency_penalty)
        hit_stop = False
        tokens = self._iter_token_ids(ids, max_new_tokens, sp, on_token=on_token)
        stream = self._stream_decoder(ids)     # 2.14.1: exact for plugged-in BPE too
        try:
            for nxt in tokens:
                if nxt in stop_ids:
                    hit_stop = True
                    break
                out, hit_stop = _consume(stream.step(nxt))
                if out:
                    yield out
                if hit_stop:
                    break
        finally:
            tokens.close()      # frees the KV cache now, not at garbage collection
        if not hit_stop:
            out, hit_stop = _consume(stream.flush())
            if out:
                yield out
        if not hit_stop and text_buffer:
            yield text_buffer

    def load_gguf_model(self, path: str, *, n_ctx: int = 4096,
                        n_threads: int | None = None,
                        n_gpu_layers: int | None = None,
                        seed: int = -1, chat_format: str | None = None,
                        use_chat_completion: bool = False,
                        n_batch: int = 512, use_mmap: bool = True,
                        use_mlock: bool = False, verbose: bool = False,
                        extra_kwargs: dict[str, Any] | None = None):
        """Load a `.gguf` model through llama.cpp for inference.

        Args:
            path: Path to a GGUF file.
            n_ctx: Context window used by llama.cpp.
            n_threads: CPU threads; defaults to AuraLite's detected thread count.
            n_gpu_layers: Layers to offload to GPU. Default `-1` asks llama.cpp
                to offload as much as possible (if a compatible build/GPU exists).
            seed/chat_format/verbose/extra_kwargs: forwarded to llama_cpp.Llama.
            use_chat_completion: use llama.cpp chat templates / chat completion API.
            n_batch/use_mmap/use_mlock: common llama.cpp loading knobs.
        """
        if n_gpu_layers is None:
            n_gpu_layers = -1
        if n_threads is None:
            n_threads = self.num_threads

        # v2.6.5: a running `llama-server` can be used as the backend instead of
        # loading the file in-process — pass an http(s) URL, or point
        # AURALITE_LLAMA_SERVER_URL at one. The proxy exposes the same surface,
        # so the generation paths below do not care which one they talk to.
        server_url = str(path) if str(path).startswith(("http://", "https://")) else \
            os.environ.get("AURALITE_LLAMA_SERVER_URL", "")
        if server_url:
            from auralite.engine.gguf_proxy import LlamaServerProxy
            model = LlamaServerProxy(
                server_url, n_ctx=n_ctx, chat_format=chat_format,
                api_key=os.environ.get("AURALITE_LLAMA_SERVER_API_KEY") or None,
            )
            self.model = model
            self.backend = "gguf"          # same generation paths, remote runtime
            self.gguf_path = model.path or server_url
            self.hf_path = None
            self._ddp_model = None
            self.tokenizer = model.tokenizer
            self.vocab_size = model.vocab_size
            self.params_used = {"backend": "gguf-server", **model.describe()}
            self.optimizer = None
            self.scheduler = None
            self.scaler = None
            self._resume_optimizer_state = None
            self._resume_scheduler_state = None
            self._resume_scaler_state = None
            logger.info("[AuraLite] Using a llama.cpp server backend at %s (n_ctx=%s)",
                        model.base_url, model.max_seq_len)
            return

        model = GGUFModelProxy(
            path, n_ctx=n_ctx, n_threads=n_threads,
            n_gpu_layers=n_gpu_layers, seed=seed,
            chat_format=chat_format,
            use_chat_completion=use_chat_completion,
            n_batch=n_batch, use_mmap=use_mmap,
            use_mlock=use_mlock, verbose=verbose,
            extra_kwargs=extra_kwargs,
        )
        self.model = model
        self.backend = "gguf"
        self.gguf_path = str(path)
        self.hf_path = None
        self._ddp_model = None
        self.tokenizer = model.tokenizer
        self.vocab_size = model.vocab_size
        self.params_used = {
            "backend": "gguf",
            "path": str(path),
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "n_gpu_layers": n_gpu_layers,
            "seed": seed,
            "chat_format": chat_format,
            "use_chat_completion": use_chat_completion,
            "n_batch": n_batch,
            "use_mmap": use_mmap,
            "use_mlock": use_mlock,
        }
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self._resume_optimizer_state = None
        self._resume_scheduler_state = None
        self._resume_scaler_state = None
        chat = ", chat=on" if use_chat_completion else ""
        logger.info(f"[AuraLite] Loaded GGUF model: {Path(path).name} "
                    f"(ctx={n_ctx}, threads={n_threads}, gpu_layers={n_gpu_layers}, "
                    f"batch={n_batch}{chat})")

    # ===================================================================
    #  NEW: Hugging Face + LoRA / QLoRA support (any model)
    # ===================================================================

    def load_hf_model(
        self,
        model_name_or_path: str,
        *,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: str | None = None,
        device_map: str = "auto",
        max_seq_len: int = 4096,
        apply_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        revision: str | None = None,
        token: str | None = None,
        verbose: bool = True,
    ):
        """
        Load ANY Hugging Face causal language model (Llama, Mistral, Qwen, Gemma, Phi, etc.)

        Fully supports **already downloaded / local models**:

        - Pass a local path (e.g. ~/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/...)
        - Set `local_files_only=True` for completely offline loading (no internet).

        Supports:
        - Full precision, FP16, BF16
        - 4-bit (QLoRA ready) and 8-bit quantization via bitsandbytes
        - Automatic application of LoRA adapters

        After loading you can:
        - Generate text
        - Fine-tune with LoRA/QLoRA using `finetune_hf()`
        - Save only the tiny LoRA adapter

        `trust_remote_code` is off by default (v2.6.6): enabling it executes
        Python code from the model repository.
        """
        if not HAS_HF_SUPPORT:
            raise HFNotAvailableError(
                "Hugging Face + LoRA support is not available.\n"
                "Install the extra dependencies:\n"
                "  pip install \"auralite-ai[hf]\"     (4-/8-bit QLoRA: \"auralite-ai[qlora]\")"
            )

        if self.hf_proxy is None:
            self.hf_proxy = create_hf_proxy()

        dtype = None
        if torch_dtype:
            dtype = getattr(torch, torch_dtype, torch.float16)

        self.hf_proxy.load_model(
            model_name_or_path,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            torch_dtype=dtype,
            device_map=device_map,
            max_seq_len=max_seq_len,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            revision=revision,
            token=token,
            verbose=verbose,
        )

        if apply_lora:
            self.hf_proxy.apply_lora(
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=lora_target_modules,
                verbose=verbose,
            )

        # Wire into engine
        self.model = self.hf_proxy
        self.backend = "huggingface"
        self.gguf_path = None
        self.hf_path = model_name_or_path
        self._ddp_model = None
        self.tokenizer = self.hf_proxy.tokenizer
        self.vocab_size = getattr(self.hf_proxy.tokenizer, "vocab_size", 0)

        self.params_used = {
            "backend": "huggingface",
            "model": model_name_or_path,
            "load_in_4bit": load_in_4bit,
            "load_in_8bit": load_in_8bit,
            "lora_applied": apply_lora,
            "lora_rank": lora_rank if apply_lora else 0,
            "local_files_only": local_files_only,
        }

        self.optimizer = None
        self.scheduler = None
        self.scaler = None

        mode = "LOCAL (offline)" if local_files_only else "Hub"
        logger.info(f"[AuraLite] Loaded Hugging Face model ({mode}): {model_name_or_path}")
        if apply_lora:
            logger.info(f"[AuraLite] LoRA applied (rank={lora_rank}) — ready for QLoRA fine-tuning")

    def apply_lora_to_hf(
        self,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: list[str] | None = None,
    ):
        """Apply LoRA to a loaded HF model (for QLoRA fine-tuning)."""
        if not self.is_hf_model():
            raise ValueError("Load a Hugging Face model first with load_hf_model()")
        self.hf_proxy.apply_lora(rank=rank, alpha=alpha, dropout=dropout,
                                 target_modules=target_modules)
        self.params_used["lora_applied"] = True
        self.params_used["lora_rank"] = rank

    def finetune_hf(
        self,
        texts: list[str] | str,
        output_dir: str = "hf_lora_adapter",
        epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        max_length: int = 512,
        gradient_accumulation_steps: int = 4,
        progress_callback: Callable | None = None,
        stop_event=None,
    ):
        """
        Fine-tune a loaded Hugging Face model using LoRA / QLoRA.

        - If the model was loaded with load_in_4bit=True → this is QLoRA
        - If LoRA was not applied yet, it will be applied automatically (rank 16)
        - Uses Hugging Face Trainer under the hood (best experience)

        Args:
            texts: list of strings or a single long text (will be split)
            output_dir: where to save the LoRA adapter
        """
        if not self.is_hf_model():
            raise ValueError("Load a Hugging Face model first with load_hf_model()")

        if isinstance(texts, str):
            # Split into reasonable chunks; a short text is used as one example.
            # (Until 2.6.5 the fallback was `[texts]` *after* `texts` had been
            # rebound to the empty list, i.e. `[[]]`.)
            raw_text = texts
            texts = [t.strip() for t in raw_text.split("\n\n") if len(t.strip()) > 50]
            if not texts and raw_text.strip():
                texts = [raw_text.strip()]
        if not texts:
            raise ValueError("finetune_hf() needs at least one non-empty training text")

        if not self.hf_proxy.is_peft:
            logger.info("[AuraLite] No LoRA adapter found — applying default LoRA for fine-tuning...")
            self.hf_proxy.apply_lora(rank=16)

        logger.info(f"[AuraLite] Starting LoRA/QLoRA fine-tuning on {len(texts)} examples...")

        return self.hf_proxy.finetune(
            texts,
            output_dir=output_dir,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_length=max_length,
            gradient_accumulation_steps=gradient_accumulation_steps,
            progress_callback=progress_callback,
            stop_event=stop_event,
        )

    def save_hf_lora(self, path: str):
        """Save only the LoRA adapter (tiny file, ~few MB)."""
        if not self.is_hf_model():
            raise ValueError("No Hugging Face model loaded")
        self.hf_proxy.save_lora_adapter(path)

    def load_hf_lora(self, adapter_path: str):
        """Load a saved LoRA adapter on top of the current base HF model."""
        if not self.is_hf_model():
            raise ValueError("Load the base Hugging Face model first")
        self.hf_proxy.load_lora_adapter(adapter_path)

    # ===================================================================
    #  Hugging Face Hub integration (NEW v2.6)
    # ===================================================================

    def push_hf_model_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload from AuraLite",
        private: bool = False,
        token: Optional[str] = None,
    ):
        """Push current HF model to the Hub."""
        if not self.is_hf_model():
            raise ValueError("Only Hugging Face models can be pushed to the Hub.")
        self.hf_proxy.push_to_hub(
            repo_id=repo_id,
            commit_message=commit_message,
            private=private,
            token=token,
        )

    def load_hf_model_from_hub(
        self,
        repo_id: str,
        **kwargs,
    ):
        """Load a model directly from the Hugging Face Hub."""
        self.load_hf_model(repo_id, **kwargs)

    # ===================================================================
    #  Evaluation (NEW v2.7)
    # ===================================================================

    def evaluate_model(
        self,
        tasks: List[str] | str = "arc_easy",
        num_fewshot: int = 0,
        batch_size: int = 1,
        limit: Optional[int] = None,
        progress_callback=None,
    ):
        """
        Evaluate the current model using lm-evaluation-harness.

        Returns a dictionary with results.
        """
        from auralite.evaluation import EvaluationEngine, LMEvalNotAvailableError

        try:
            eval_engine = EvaluationEngine(self)
            results = eval_engine.evaluate(
                tasks=tasks,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                limit=limit,
                progress_callback=progress_callback,
            )
            return results
        except LMEvalNotAvailableError as e:
            raise e

    @staticmethod
    def _gguf_prompt_messages(prompt: str) -> list[dict[str, str]]:
        """Wrap a raw prompt into a minimal chat message list for GGUF chat APIs."""
        return [{"role": "user", "content": prompt}]

    # ---- GGUF evaluation (v2.6.5) -------------------------------------
    def gguf_perplexity(self, text: str) -> float:
        """Perplexity of `text` for the loaded GGUF model.

        GGUF models cannot be wrapped by lm-evaluation-harness through the
        native path, so evaluation is exposed here directly: llama.cpp is run
        in `n_ctx - 1` token windows and PPL is computed from per-token
        logprobs (one token per window boundary is skipped — it has no left
        context). Requires llama-cpp-python >= 0.2.20 with score support.
        """
        if not self.is_gguf_model():
            raise ValueError("gguf_perplexity() requires a loaded .gguf model.")
        return self.model.perplexity(text)

    def gguf_loglikelihood_rolling(self, text: str) -> float:
        """Total log-probability of `text` under the loaded GGUF model."""
        if not self.is_gguf_model():
            raise ValueError("gguf_loglikelihood_rolling() requires a loaded .gguf model.")
        return self.model.loglikelihood_rolling(text)

    @staticmethod
    def _gguf_penalty_kwargs(presence_penalty: float = 0.0,
                             frequency_penalty: float = 0.0) -> dict:
        """llama.cpp's presence/frequency penalties, only when requested."""
        if not (presence_penalty or frequency_penalty):
            return {}
        return {"presence_penalty": float(presence_penalty),
                "frequency_penalty": float(frequency_penalty)}

    def _gguf_generate_text(self, prompt: str, length: int = 50,
                            temperature: float = 0.8,
                            top_k: int = 50, top_p: float = 0.9,
                            repetition_penalty: float = 1.0,
                            min_p: float = 0.0,
                            stop: list[str] | None = None, *,
                            presence_penalty: float = 0.0,
                            frequency_penalty: float = 0.0) -> str:
        if not is_gguf_like(self.model):
            raise ValueError("No GGUF model loaded.")
        if self.model.use_chat_completion:
            out = self.model.create_chat_completion(
                self._gguf_prompt_messages(prompt),
                max_tokens=length, temperature=temperature,
                top_k=top_k, top_p=top_p, min_p=min_p,
                repeat_penalty=repetition_penalty, **self._gguf_penalty_kwargs(presence_penalty, frequency_penalty), stream=False,
                stop=stop,
            )
            try:
                suffix = out["choices"][0].get("message", {}).get("content", "")
            except Exception:
                suffix = ""
        else:
            out = self.model.create_completion(
                prompt, max_tokens=length, temperature=temperature,
                top_k=top_k, top_p=top_p, min_p=min_p,
                repeat_penalty=repetition_penalty, **self._gguf_penalty_kwargs(presence_penalty, frequency_penalty), stream=False,
                stop=stop,
            )
            try:
                suffix = out["choices"][0].get("text", "")
            except Exception:
                suffix = ""
        return prompt + suffix

    # ---- Tokenisation -----------------------------------------------
    def encode(self, s: str) -> list[int]:
        if self.tokenizer is None:
            raise ValueError("No tokenizer — train or load a model first!")
        return self.tokenizer.encode(s)

    def decode(self, ids) -> str:
        if self.tokenizer is None:
            raise ValueError("No tokenizer — train or load a model first!")
        return self.tokenizer.decode(ids)

    # ---- Inference speed (v2.6.5) --------------------------------------
    @_compute.on_compute_thread
    def _infer(self, *args, **kwargs):
        """Forward pass used by every generation path.

        2.13: runs on the engine's compute thread, whichever thread generates
        (``auralite/engine/compute_thread.py``).

        When `enable_inference_compile()` was called, *prompt/batch* forwards
        (sequence length > 1, i.e. the compute-heavy prefill) run through the
        compiled module while single-token decode steps stay eager. That split
        is measured, not guessed: on CPU, compilation speeds prefill up by
        ×1.1–1.7 but *slows down* T=1 decoding, because each new KV length
        makes Dynamo re-check (and sometimes re-trace) its guards.

        Training and checkpointing never go through here, so a compiled module
        can never leak into a `.pt` file.

        A compiled forward that raises falls back to the eager model for good
        (2.14.0, see `_eager_after_compile_failure`): Inductor compiles lazily,
        once per new prompt shape, so a missing C++ compiler can surface on
        any later prompt, long after the warm-up.
        """
        compiled = self._compiled_inference_model
        x = args[0] if args else kwargs.get("x")
        keep = kwargs.pop("logits_to_keep", None)
        if keep is not None and self._accepts_logits_to_keep():
            kwargs["logits_to_keep"] = keep
            keep = None
        model: Any = self.model
        if compiled is not None and getattr(x, "ndim", 0) >= 2 and x.shape[1] > 1:
            try:
                out = compiled(*args, **kwargs)
            except Exception as exc:              # noqa: BLE001 - optional feature
                out = self._eager_after_compile_failure(exc, args, kwargs)
        elif getattr(model, "_orig_mod", None) is not None:
            try:                                  # installed by compile_for_inference()
                out = model(*args, **kwargs)
            except Exception as exc:              # noqa: BLE001 - optional feature
                out = self._eager_after_compile_failure(exc, args, kwargs)
        else:
            out = model(*args, **kwargs)
        if keep is not None and out.ndim == 3 and out.shape[1] > keep:
            out = out[:, -int(keep):]           # model without the 2.8 argument
        return out

    def _eager_after_compile_failure(self, exc: Exception, args: tuple, kwargs: dict):
        """Drop the compiled module after it failed and redo the forward eagerly (2.14.0).

        Without a C++ compiler (no MSVC ``cl`` on Windows, no ``g++`` in a slim
        image) Inductor fails on the first call of every new prompt shape.
        Generation must not: the engine logs one warning, forgets the compiled
        module and runs this forward, and every later one, in the eager model.
        Cache entries a half-run compiled forward may have written are rolled
        back first, so the eager pass starts from the same cache.
        """
        self._compiled_inference_model = None
        orig = getattr(self.model, "_orig_mod", None)
        if orig is not None:
            self.model = orig                     # compile_for_inference() undone
        reason = " ".join(line.strip() for line in str(exc).splitlines() if line.strip())
        logger.warning("[AuraLite] torch.compile failed (%s: %s); generation continues "
                       "in eager mode.", type(exc).__name__, reason[:200])
        if kwargs.get("use_cache"):
            truncate = getattr(self._native_model(), "truncate_cache", None)
            if callable(truncate):
                try:
                    truncate(int(kwargs.get("start_pos", 0)))
                except ValueError:
                    raise exc from None           # the cache cannot be rolled back
        model: Any = self.model
        return model(*args, **kwargs)

    def _accepts_logits_to_keep(self) -> bool:
        """Does the loaded model's forward take `logits_to_keep` (2.8)?"""
        model = getattr(self.model, "_orig_mod", self.model)
        cached = getattr(self, "_ltk_support", None)
        if cached is not None and cached[0] is model:
            return cached[1]
        try:
            import inspect
            ok = "logits_to_keep" in inspect.signature(model.forward).parameters
        except (TypeError, ValueError, AttributeError):
            ok = False
        self._ltk_support = (model, ok)
        return ok

    def enable_inference_compile(self, mode: str = "default",
                                 warmup_tokens: int = 2) -> float:
        """Compile the model for generation; returns the compile time in seconds.

        `torch.compile` trades a slow first call for faster decoding afterwards,
        which only pays off for long generations — `bench_generation_modes()`
        measures the actual numbers instead of guessing. Failure is never fatal:
        the engine logs a warning and keeps running eagerly.

        The inductor cache is redirected to `$AURALITE_HOME/torch_inductor`, so
        the second process start reuses the compiled kernels instead of paying
        the ~5–10 s again.
        """
        model = self._native_model()
        if model is None:
            raise RuntimeError("enable_inference_compile needs a native AuraLite "
                               "model (GGUF/Hugging Face backends run in their own runtime)")
        if self._compiled_inference_model is not None:
            return float(getattr(self, "_compile_seconds", 0.0))
        cache_dir = Path(os.environ.get("AURALITE_HOME", Path.home() / ".auralite")) / "torch_inductor"
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))
        if os.environ.get("AURALITE_COMPILE_DEBUG"):
            os.environ.setdefault("TORCH_LOGS", "output_code")
        start = time.perf_counter()
        try:
            compiled = torch.compile(model, mode=mode, dynamic=False)
        except Exception as exc:                  # noqa: BLE001 - optional feature
            logger.warning("[AuraLite] torch.compile unavailable (%s); staying eager.", exc)
            return 0.0
        self._compiled_inference_model = compiled
        if warmup_tokens:
            # 2.14.0: the warm-up is a real prefill (cache on, last logits
            # only), so it runs on the compute thread, like generation. Recent
            # torch keeps Dynamo's settings per thread, so a warm-up on the
            # caller's thread could pass (e.g. under its `suppress_errors`)
            # while the first real prompt still failed.
            tokens = max(2, int(warmup_tokens))
            try:
                with self._exclusive_decode(tokens), torch.inference_mode():
                    self._prefill(torch.zeros((1, tokens), dtype=torch.long,
                                              device=self.device))
            except Exception as exc:              # noqa: BLE001
                self._compiled_inference_model = None
                logger.warning("[AuraLite] torch.compile warm-up failed (%s); staying eager.", exc)
                return 0.0
            if self._compiled_inference_model is None:
                return 0.0                        # `_infer` fell back to eager and said why
        self._compile_seconds = time.perf_counter() - start
        logger.info("[AuraLite] torch.compile enabled (mode=%s) in %.2fs; "
                    "kernels cached in %s", mode, self._compile_seconds, cache_dir)
        return self._compile_seconds

    def disable_inference_compile(self) -> bool:
        """Drop the compiled module (returns True when one was installed)."""
        had = self._compiled_inference_model is not None
        self._compiled_inference_model = None
        return had

    def _native_model(self):
        """The eager `ModernTransformer`, or None for GGUF/HF/external engines."""
        model = getattr(self.model, "module", self.model)      # unwrap DDP
        if model is None or not isinstance(model, nn.Module):
            return None
        if self.is_gguf_model() or getattr(self, "hf_proxy", None) is not None:
            return None
        return model

    @staticmethod
    def _model_arch_kwargs(model) -> dict:
        """Architecture kwargs of `model` — used to clone it with another KV dtype."""
        return {
            "vocab_size": model.embedding.num_embeddings,
            "d_model": model.d_model, "n_heads": model.n_heads,
            "n_layers": model.n_layers, "d_ff": model.d_ff,
            "max_seq_len": model.max_seq_len,
            "n_kv_heads": model.n_kv_heads, "dropout": model.dropout,
            "use_alibi": model.use_alibi, "rope_scaling": model.rope_scaling,
            "sliding_window": model.sliding_window,
            "use_flex_attention": model.use_flex_attention,
            "use_moe": getattr(model, "use_moe", False),
            "num_experts": getattr(model, "num_experts", 4),
            "tie_word_embeddings": getattr(model, "tie_word_embeddings", False),
            "use_qk_norm": getattr(model, "use_qk_norm", False),
        }

    # ---- Diagnostics (GUI visualisations, v2.6.5) ----------------------
    @_compute.on_compute_thread
    def attention_maps(self, text: str, layer: int = 0, head: int | None = None,
                       max_tokens: int = 256) -> "np.ndarray":
        """Attention probabilities of the *native* model for ``text``.

        Returns a ``(n_heads, T, T)`` float32 array (or ``(T, T)`` when ``head``
        is given), where ``T`` is the number of tokens actually fed to the model
        (the last ``max_tokens`` of ``text``). Row ``i`` answers "how much did
        position *i* attend to each earlier position" — the Chat tab's heatmap.

        Implementation: one forward pass with `capture` set on the model, then
        softmax over the captured RoPE-rotated q/k of that layer, in float32.
        Diagnostics only — generation never pays for this (``capture=None``).

        Raises `RuntimeError` for GGUF/Hugging Face models, which have no native
        attention modules to inspect.
        """
        model = getattr(self.model, "module", self.model)   # unwrap DDP
        model = getattr(model, "_orig_mod", model)           # unwrap torch.compile
        if model is None or not isinstance(model, nn.Module):
            raise RuntimeError("attention_maps needs a loaded native model "
                               "(train() or load_model() a .pt checkpoint first)")
        if self.is_gguf_model() or getattr(self, "hf_proxy", None) is not None:
            raise RuntimeError("attention_maps is only available for native "
                               "AuraLite checkpoints, not GGUF/Hugging Face models")
        layers = getattr(model, "layers", None)
        if not layers:
            raise RuntimeError("loaded model exposes no transformer layers")
        layer_index = int(layer)
        if not 0 <= layer_index < len(layers):
            raise ValueError(f"layer must be in [0, {len(layers) - 1}] for this model")

        ids = list(self.encode(text))[-max(1, int(max_tokens)):]
        if not ids:
            raise ValueError("text produced no tokens")
        was_training = model.training
        model.eval()
        capture: dict = {}
        try:
            with torch.inference_mode():
                x = torch.tensor([ids], dtype=torch.long, device=self.device)
                model(x, capture=capture)
        finally:
            if was_training:
                model.train()

        cap = capture.get(layer_index)
        if not cap or "q" not in cap:
            raise RuntimeError(f"layer {layer_index} did not capture attention")
        q, k = cap["q"], cap["k"]                       # (1, H, T, hd) / (1, Hk, T, hd)
        n_rep = int(cap.get("n_rep", 1))
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
        head_dim = q.shape[-1]
        scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(head_dim)
        t_q, t_k = scores.shape[-2], scores.shape[-1]
        mask = torch.triu(torch.ones(t_q, t_k, dtype=torch.bool,
                                     device=scores.device), diagonal=1)
        window = cap.get("sliding_window") or 0
        if window:
            offsets = torch.arange(t_k, device=scores.device).unsqueeze(0) - \
                torch.arange(t_q, device=scores.device).unsqueeze(1)
            mask |= offsets < -(window - 1)
        scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        if head is not None:
            head_index = int(head)
            if not 0 <= head_index < probs.shape[1]:
                raise ValueError(f"head must be in [0, {probs.shape[1] - 1}]")
            probs = probs[0, head_index]
        else:
            probs = probs[0]
        return probs.cpu().numpy()

    def token_frequencies(self, text: str) -> list[tuple[str, int]]:
        """`[(token_text, count), …]` sorted by count — the Chat tab bar chart."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer — train or load a model first!")
        ids = self.encode(text)
        counts = Counter(ids)
        decode = self.decode
        return [(decode([tid]), n) for tid, n in counts.most_common()]

    def compile_for_inference(self, mode: str = "reduce-overhead") -> bool:
        """Compile the native torch model for low-latency inference.

        Returns True on success and False on graceful fallback. The public engine
        API is unchanged; callers can opt in via this method or set
        `compile_mode` in higher-level serving code.
        """
        if self.model is None or self.is_gguf_model() or self.is_hf_model():
            return False
        original_model = self.model
        try:
            # torch.compile() is lazy: Dynamo/Inductor only run on the FIRST
            # forward pass. On machines without a C++ compiler (e.g. Windows CI:
            # "InvalidCxxCompiler: Compiler: cl is not found") that lazy step
            # would otherwise blow up later, inside generate(). Two safeguards:
            #   1) a warm-up on the compute thread, where generation runs, so a
            #      hard failure is caught here and we cleanly revert to the
            #      eager model + report False;
            #   2) a later recompile that fails (a new prompt shape) makes
            #      `_infer` put the eager model back (2.14.0).
            # Up to 2.13 this also set Dynamo's `suppress_errors`. Recent torch
            # keeps that per thread, so it only covered the caller's thread,
            # not the compute thread, and it stayed set for every later
            # torch.compile on that thread.
            compiled = torch.compile(self.model, mode=mode)
            self._warmup_compiled_model(compiled)
            self.model = compiled
            return True
        except Exception as e:
            logger.warning("torch.compile inference disabled (eager fallback): %s", e)
            self.model = original_model
            try:
                torch._dynamo.reset()
            except Exception:
                pass
            return False

    def _warmup_compiled_model(self, compiled_model) -> None:
        """Force lazy torch.compile compilation via a tiny warm-up forward.

        Runs the same call shape generation uses (single token, use_cache=True)
        so any backend-compile error surfaces now rather than mid-generation.
        """
        # A fresh cache of its own under the model lock (2.12): the warm-up
        # must not write into a running generation's cache. It runs on the
        # compute thread (2.14.0), like the generation it stands in for.
        with self._exclusive_decode(), torch.inference_mode():
            dummy = torch.zeros((1, 1), dtype=torch.long, device=self.device)
            _compute.run_on_compute_thread(compiled_model, dummy, start_pos=0, use_cache=True)

    # ---- Save / Load --------------------------------------------------
    def save_model(self, path: str, include_optimizer: bool = True):
        """Save a checkpoint.

        The container is chosen by the file extension:
        - `*.safetensors` — weights + metadata only (no pickle, no optimizer
          state; ideal for sharing model weights safely);
        - anything else (`*.pt`) — full checkpoint v3: weights, optimizer /
          scheduler / scaler state and RNG (needed to resume training).

        Args:
            include_optimizer: .pt only — set False to store a pure-weights
                checkpoint (smaller, still resumable as inference model).
        """
        if self.model is None:
            raise ValueError("No model to save!")
        if str(path).lower().endswith(".safetensors"):
            return self._save_model_safetensors(path)
        if self.is_gguf_model():
            raise ValueError(
                "Loaded .gguf models are inference-only external files and "
                "cannot be saved as AuraLite .pt checkpoints. Copy the .gguf "
                "file itself if you need to move it."
            )
        if self.is_hf_model():
            raise ValueError(
                "Hugging Face models are managed through the HF/PEFT workflow. "
                "Save LoRA adapters with save_hf_lora() or use push_to_hub()/save_pretrained() instead."
            )
        checkpoint = {
            "model_state":  self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if (include_optimizer and self.optimizer is not None) else None,
            "scheduler_state": self.scheduler.state_dict() if (include_optimizer and self.scheduler is not None) else None,
            "scaler_state": self.scaler.state_dict() if (include_optimizer and self.scaler is not None) else None,
            "vocab_size":   self.vocab_size,
            "tokenizer":    self.tokenizer.to_dict() if self.tokenizer else None,
            "params_used":  self.params_used,
            # Store all architecture fields explicitly
            "d_model":      self.model.d_model,
            "d_ff":         self.model.d_ff,
            "n_heads":      self.model.n_heads,
            "n_layers":     self.model.n_layers,
            "n_kv_heads":   self.model.n_kv_heads,
            "dropout":      self.model.dropout,
            "max_seq_len":  self.model.max_seq_len,
            "use_alibi":    self.model.use_alibi,
            "lora_rank":    self.model.lora_rank,
            "rope_scaling": getattr(self.model, 'rope_scaling', None),
            "use_gradient_checkpointing": getattr(self.model, 'use_gradient_checkpointing', False),
            "sliding_window": getattr(self.model, 'sliding_window', None),
            "kv_cache_dtype": getattr(self.model, 'kv_cache_dtype', None),
            "use_flex_attention": getattr(self.model, 'use_flex_attention', False),
            "use_moe": getattr(self.model, 'use_moe', False),
            "num_experts": getattr(self.model, 'num_experts', 4),
            "tie_word_embeddings": getattr(self.model, 'tie_word_embeddings', True),
            "use_qk_norm": getattr(self.model, 'use_qk_norm', False),
            # v2.6.2: chat template used at training time
            "chat_template": self.default_chat_template,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                # v2.6.5: builtins-only representation (safe to load with
                # weights_only=True); old checkpoints still load via a scoped
                # numpy allowlist in _safe_torch_load.
                "numpy": _numpy_rng_state_to_builtins(np.random.get_state()),
            },
            # Format versioning: 1 = pre-2.4 (may carry "chars"), 2 = v2.4..v2.6.1,
            # 3 = v2.6.2+ (adds chat_template; old files load unchanged).
            "format": "auralite",
            "checkpoint_version": 3,
        }
        # Atomic write: save next to the target and os.replace() it in place, so
        # an interrupted save (Ctrl+C, out of disk) can never leave a truncated,
        # unloadable checkpoint at `path`.
        import tempfile
        dir_name = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(path) + ".", suffix=".tmp", dir=dir_name)
        try:
            os.close(fd)
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ---- safetensors (pickle-free) checkpoints ------------------------
    def _checkpoint_metadata(self) -> dict:
        """Architecture + tokenizer fields stored alongside the weights."""
        return {
            "vocab_size":   self.vocab_size,
            "tokenizer":    self.tokenizer.to_dict() if self.tokenizer else None,
            "params_used":  self.params_used,
            "d_model":      self.model.d_model,
            "d_ff":         self.model.d_ff,
            "n_heads":      self.model.n_heads,
            "n_layers":     self.model.n_layers,
            "n_kv_heads":   self.model.n_kv_heads,
            "dropout":      self.model.dropout,
            "max_seq_len":  self.model.max_seq_len,
            "use_alibi":    self.model.use_alibi,
            "lora_rank":    self.model.lora_rank,
            "rope_scaling": getattr(self.model, 'rope_scaling', None),
            "use_gradient_checkpointing": getattr(self.model, 'use_gradient_checkpointing', False),
            "sliding_window": getattr(self.model, 'sliding_window', None),
            "kv_cache_dtype": getattr(self.model, 'kv_cache_dtype', None),
            "use_flex_attention": getattr(self.model, 'use_flex_attention', False),
            "use_moe": getattr(self.model, 'use_moe', False),
            "num_experts": getattr(self.model, 'num_experts', 4),
            "tie_word_embeddings": getattr(self.model, 'tie_word_embeddings', True),
            "use_qk_norm": getattr(self.model, 'use_qk_norm', False),
            "chat_template": self.default_chat_template,
            "format": "auralite",
            "checkpoint_version": 3,
        }

    def _save_model_safetensors(self, path: str):
        """Pickle-free export: tensors + JSON metadata, no optimizer/RNG state.

        `safetensors` files cannot execute code on load, which makes them the
        safest way to share model weights. They carry weights only — training
        resumption needs the `.pt` format.
        """
        tensors = {
            k: v for k, v in self.model.state_dict().items()
            if not (k.endswith(".rope_cos") or k.endswith(".rope_sin"))
        }
        return _save_safetensors_checkpoint(path, tensors, self._checkpoint_metadata())

    def _read_checkpoint(self, path: str, allow_unsafe: bool = False) -> dict:
        """Read a `.pt` / `.safetensors` checkpoint into a unified dict."""
        if str(path).lower().endswith(".safetensors"):
            checkpoint = _load_safetensors_checkpoint(path)
            checkpoint.setdefault("optimizer_state", None)
            checkpoint.setdefault("scheduler_state", None)
            checkpoint.setdefault("scaler_state", None)
            checkpoint.setdefault("rng_state", None)
            return checkpoint
        return _safe_torch_load(path, map_location=self.device, allow_unsafe=allow_unsafe)

    def load_model(self, path: str, cpu_quantize: bool | str | None = None,
                   allow_unsafe: bool = False):
        """Load a checkpoint.

        Args:
            cpu_quantize: on CPU devices, dynamically INT8-quantize Linear layers
                after loading (much faster decode + ~4x smaller resident model,
                at a small quality cost). Truthy values accepted: True, "int8".
                May also be enabled globally via the `AURALITE_CPU_INT8=1`
                environment variable (used by the API server). Inference-only:
                a quantized model cannot be trained further; save it with
                `save_quantized_model()`.
        """
        if str(path).lower().endswith(".gguf"):
            # Optional advanced knobs without complicating the GUI:
            #   AURALITE_GGUF_N_CTX=8192
            #   AURALITE_GGUF_N_GPU_LAYERS=-1
            #   AURALITE_GGUF_N_THREADS=8
            #   AURALITE_GGUF_CHAT_FORMAT=llama-2
            #   AURALITE_GGUF_USE_CHAT=1
            #   AURALITE_GGUF_N_BATCH=512
            n_ctx = int(os.environ.get("AURALITE_GGUF_N_CTX", "4096"))
            n_gpu_layers = int(os.environ.get("AURALITE_GGUF_N_GPU_LAYERS", "-1"))
            n_threads_env = os.environ.get("AURALITE_GGUF_N_THREADS")
            n_threads = int(n_threads_env) if n_threads_env else None
            chat_format = os.environ.get("AURALITE_GGUF_CHAT_FORMAT") or None
            use_chat = os.environ.get("AURALITE_GGUF_USE_CHAT", "0").lower() in {"1", "true", "yes", "on"}
            n_batch = int(os.environ.get("AURALITE_GGUF_N_BATCH", "512"))
            use_mmap = os.environ.get("AURALITE_GGUF_USE_MMAP", "1").lower() not in {"0", "false", "no", "off"}
            use_mlock = os.environ.get("AURALITE_GGUF_USE_MLOCK", "0").lower() in {"1", "true", "yes", "on"}
            self.load_gguf_model(
                path, n_ctx=n_ctx, n_threads=n_threads,
                n_gpu_layers=n_gpu_layers, chat_format=chat_format,
                use_chat_completion=use_chat, n_batch=n_batch,
                use_mmap=use_mmap, use_mlock=use_mlock,
            )
            return

        self.backend = "torch"
        self.gguf_path = None
        self.hf_path = None
        self._ddp_model = None
        checkpoint = self._read_checkpoint(path, allow_unsafe=allow_unsafe)

        ckpt_version = int(checkpoint.get("checkpoint_version", 1))
        if ckpt_version >= 2:
            logger.info("loading AuraLite checkpoint v%s from %s", ckpt_version, path)
        # v2.6.2: restore the chat template the model was trained with
        # (fall back to params_used, then to the historic "chatml" default).
        self.default_chat_template = (
            checkpoint.get("chat_template")
            or checkpoint.get("params_used", {}).get("chat_template")
            or "chatml"
        )

        if checkpoint.get("tokenizer"):
            self.tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
        elif "chars" in checkpoint:
            # backward compatibility with old char-level checkpoints
            self.tokenizer = CharTokenizer.from_dict({"vocab": checkpoint["chars"]})
        else:
            raise ValueError("Checkpoint has no tokenizer information.")

        self.vocab_size  = checkpoint["vocab_size"]
        self.params_used = checkpoint.get("params_used", {})

        self.model = ModernTransformer(
            vocab_size  = self.vocab_size,
            d_model     = checkpoint["d_model"],
            n_heads     = checkpoint["n_heads"],
            n_layers    = checkpoint["n_layers"],
            d_ff        = checkpoint["d_ff"],
            max_seq_len = checkpoint.get("max_seq_len", 4096),
            n_kv_heads  = checkpoint.get("n_kv_heads"),
            dropout     = checkpoint.get("dropout", 0.0),
            use_alibi   = checkpoint.get("use_alibi", False),
            use_gradient_checkpointing = checkpoint.get("use_gradient_checkpointing", False),
            rope_scaling = checkpoint.get("rope_scaling", None),
            sliding_window = checkpoint.get("sliding_window", None),
            kv_cache_dtype = checkpoint.get("kv_cache_dtype", None),
            use_flex_attention = checkpoint.get("use_flex_attention", False),
            use_moe = checkpoint.get("use_moe", False),
            num_experts = checkpoint.get("num_experts", 4),
            tie_word_embeddings = checkpoint.get("tie_word_embeddings", True),
            use_qk_norm = checkpoint.get("use_qk_norm", False),
        ).to(self.device)

        # Re-create LoRA adapters BEFORE loading the state dict so their
        # parameters exist as keys in the model (they are registered via
        # layer.ffn.lora and therefore live inside model_state).
        lora_rank = checkpoint.get("lora_rank", 0)
        if lora_rank > 0:
            self.model.enable_lora(rank=lora_rank)

        # Backward compatibility: drop RoPE cos/sin buffers from the checkpoint.
        # They are non-persistent since v2.6.1 (derived data, rebuilt on demand),
        # and older checkpoints may carry them at a different size after RoPE
        # auto-extension during generation — loading them verbatim would fail
        # with a state_dict size mismatch.
        model_state = {
            k: v for k, v in checkpoint["model_state"].items()
            if not (k.endswith(".rope_cos") or k.endswith(".rope_sin"))
        }
        self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()

        # Stash training state so a later `continue_training=True` call can
        # resume optimizer / scheduler / scaler state as well.
        self._resume_optimizer_state = checkpoint.get("optimizer_state")
        self._resume_scheduler_state = checkpoint.get("scheduler_state")
        self._resume_scaler_state = checkpoint.get("scaler_state")
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        rng_state = checkpoint.get("rng_state")
        if isinstance(rng_state, dict):
            try:
                if rng_state.get("torch") is not None:
                    torch.set_rng_state(rng_state["torch"])
                if torch.cuda.is_available() and rng_state.get("cuda") is not None:
                    torch.cuda.set_rng_state_all(rng_state["cuda"])
                if rng_state.get("numpy") is not None:
                    np.random.set_state(
                        _numpy_rng_state_from_builtins(rng_state["numpy"])
                        or rng_state["numpy"])
            except Exception as e:
                logger.warning("could not restore RNG state: %s", e)

        # v2.6.3: opt-in CPU INT8 dynamic quantization at load time
        env_flag = os.environ.get("AURALITE_CPU_INT8", "").strip().lower()
        want_int8 = cpu_quantize if cpu_quantize is not None else (
            env_flag in {"1", "true", "yes", "on"})
        if want_int8 in (True, "int8", "1") or want_int8 == "True":
            if self.device.type != "cpu":
                logger.info("cpu_quantize requested but device is %s — skipped",
                            self.device.type)
            else:
                try:
                    self.quantize_model("dynamic")
                except Exception as e:
                    logger.warning("CPU INT8 load-time quantization failed, "
                                   "keeping fp32 model: %s", e)
        _compute.release_caller_pool()        # 2.13: generation uses the compute thread

    # ---- Quantization Integration (NEW v2.2) ---------------------------
    def quantize_model(self, method: str = "dynamic", bits: str = "int8",
                       calibration_text: str = "",
                       progress_callback=None,
                       **kwargs) -> "tuple[Any, Any]":
        """Quantize the current model.

        Args:
            method: "dynamic", "static", "qat", "gptq", "awq", "half"
            bits: "int2", "int3", "int4", "int8", "fp16", "bf16"
            calibration_text: text for calibration (needed for static/gptq/awq/qat)
            progress_callback(step, total, message): optional
            **kwargs: extra QuantConfig fields

        Returns:
            (quantized_model, QuantResult)
        """
        if self.model is None:
            raise ValueError("No model to quantize — train or load a model first!")
        if self.is_gguf_model():
            raise ValueError(
                "GGUF models are already quantized externally. "
                "Quantization only works on native AuraLite .pt models.")
        if self.is_hf_model():
            raise ValueError(
                "Hugging Face models should be quantized with their native tooling or bitsandbytes. "
                "AuraLite quantization only works on native AuraLite .pt models.")

        from auralite.quantization import (QuantizationEngine, QuantConfig,
                                  QuantMethod, BitWidth)

        config = QuantConfig(
            method=QuantMethod(method),
            bits=BitWidth(bits),
            calibration_text=calibration_text,
            **kwargs,
        )

        engine = QuantizationEngine()
        q_model, result = engine.quantize(
            self.model, config,
            tokenizer=self.tokenizer,
            device=self.device,
            progress_callback=progress_callback,
        )

        if not result.errors:
            self.model = q_model
            logger.info(f"[AuraLite] Quantization complete: {result.method} {result.bits} "
                        f"({result.original_size_mb:.2f} → {result.quantized_size_mb:.2f} MB, "
                        f"{result.compression_ratio:.2f}×)")
        _compute.release_caller_pool()        # 2.13: generation uses the compute thread
        return q_model, result

    def benchmark_quantization(self, original_model, quantized_model,
                               text: str, seq_length: int = 64,
                               progress_callback=None):
        """Benchmark original vs quantized model."""
        from auralite.quantization import QuantizationEngine
        engine = QuantizationEngine()
        return engine.benchmark(
            original_model, quantized_model,
            self.tokenizer, text, self.device,
            seq_length=seq_length,
            progress_callback=progress_callback,
        )

    def save_quantized_model(self, path: str, config=None, result=None):
        """Save the quantized model."""
        if self.model is None:
            raise ValueError("No model to save!")
        from auralite.quantization import QuantizationEngine, QuantConfig
        if config is None:
            config = QuantConfig()
        QuantizationEngine.save_quantized(
            self.model, path, config,
            tokenizer=self.tokenizer,
            params_used=self.params_used,
            result=result,
        )

    # ---- Config Management (NEW) --------------------------------------
    def save_config(self, path: str, params: dict):
        """Save training configuration to JSON."""
        config = {
            "version": "2.1",
            "params": params,
            "device": str(self.device),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def load_config(self, path: str) -> dict:
        """Load training configuration from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("params", config)
