"""Native generation: :class:`GenerationMixin` (moved out of ``_legacy.py`` in 2.14).

The decode loop every native generation path shares (``_iter_token_ids``),
``generate``, ``generate_streaming``, ``generate_with_thinking``,
``generate_batch``, speculative decoding and the sampling hooks. The code
moved unchanged; the methods are reached through ``AuraLiteEngine`` as
before, and ``auralite.engine.generation.AuraLiteEngine`` still resolves.
"""
from __future__ import annotations

import importlib
import threading
from collections import defaultdict
from typing import Any, Callable, Iterator

import torch

from auralite.engine import compute_thread as _compute
from auralite.engine import sampling as _sampling
from auralite.engine.batching import DecodeBatch, batchable_model, max_batch_rows
from auralite.engine.decode_slots import DecodeSlot, _unwrap
from auralite.engine.gguf_proxy import is_gguf_like
from auralite.engine.sampling import SamplingParams
from auralite.engine import tokenizer_io as _tok_io

#: ``on_token(token_id, logits)``: see :meth:`GenerationMixin.generate` (2.14).
TokenCallback = Callable[[int, torch.Tensor], None]

_BATCH_LOCK = threading.Lock()
_MODEL_REPLACED = "the model was replaced while a generation was running"

__all__ = ["GenerationMixin"]


def __getattr__(name: str) -> Any:
    if name == "AuraLiteEngine":                 # the 2.4-2.13 export of this module
        return importlib.import_module("auralite.engine").AuraLiteEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _TokenByToken:
    """Streaming text of AuraLite's own tokenizers: ``decode([id])`` per id."""

    def __init__(self, decode: Callable[[list[int]], str]):
        self._decode = decode

    def step(self, token_id: int) -> str:
        return self._decode([token_id])

    def flush(self) -> str:
        return ""

class _DraftSession:
    """A draft model with a persistent KV cache, for speculative decoding (2.8).

    Each round it rolls its cache back to the longest prefix it shares with the
    target's tokens, feeds only the new tokens, and proposes `k` greedy tokens.
    The 2.7 loop re-ran the draft over the whole context for every proposal.
    """

    def __init__(self, engine, capacity: int):
        self.engine = engine
        self.model = engine.model
        # The session drives the draft model's cache directly, so the draft
        # engine stays locked until close() (2.12); its suspended streams keep
        # their caches in their own slots meanwhile.
        self._lock = getattr(engine, "model_lock", None)
        if self._lock is not None:
            self._lock.acquire()
        self.cached: list[int] = []
        try:
            self.model.eval()
            sync = getattr(engine, "_sync_context_window", None)
            if callable(sync):               # the draft engine's own window (2.11)
                sync()
            self.supported = callable(getattr(self.model, "truncate_cache", None))
            try:
                self.model.reset_cache(capacity=capacity)
            except TypeError:
                self.model.reset_cache()
        except BaseException:
            self.close()
            raise

    def _step(self, tokens: list[int], start: int) -> torch.Tensor:
        t = torch.tensor([tokens], dtype=torch.long, device=self.engine.device)
        prefill: Any = getattr(self.engine, "_prefill", None)
        if start == 0 and len(tokens) > 1 and prefill is not None:
            return prefill(t)[0, -1]         # chunked for long contexts (2.11)
        logits = self.engine._infer(t, start_pos=start, use_cache=True, logits_to_keep=1)
        return logits[0, -1]

    def propose(self, ids: list[int], k: int) -> list[int]:
        if k <= 0 or not ids:
            return []
        if not self.supported:          # foreign draft model: stateless, as in 2.7
            try:
                out = self.engine._generate_ids(ids, k, temperature=0.0, top_k=1)
                return list(out[len(ids):len(ids) + k])
            except Exception:
                return []
        try:
            common = 0
            limit = min(len(self.cached), len(ids))
            while common < limit and self.cached[common] == ids[common]:
                common += 1
            if common == len(ids):      # nothing new to feed: re-feed the last token
                common -= 1
            if common < len(self.cached):
                self.model.truncate_cache(common)
                self.cached = self.cached[:common]
            logits = self._step(ids[common:], common)
            self.cached = list(ids)
            proposals: list[int] = []
            for i in range(k):
                tok = int(torch.argmax(logits).item())
                proposals.append(tok)
                if i == k - 1:
                    break
                logits = self._step([tok], len(self.cached))
                self.cached.append(tok)
            return proposals
        except Exception:               # a draft failure must never break decoding
            self.cached = []
            try:
                self.model.reset_cache()
            except Exception:
                pass
            return []

    def close(self) -> None:
        try:
            self.model.reset_cache()
        except Exception:
            pass
        self.cached = []
        lock, self._lock = self._lock, None
        if lock is not None:
            lock.release()


class GenerationMixin:
    """Native generation for :class:`~auralite.engine.AuraLiteEngine`."""

    # ---- decode batches (2.14) -------------------------------------------
    def _new_decode_batch(self, model: Any, rows: int, caches: list,
                          shared: bool = False) -> DecodeBatch | None:
        """A :class:`DecodeBatch` for `model` shaped after a prompt's caches."""
        if not caches or getattr(caches[0], "k", None) is None:
            return None
        attn = model.layers[0].attn
        batch = DecodeBatch(model, rows, spec=attn.kv_cache_dtype, window=caches[0].window,
                            dtype=attn.W_k.weight.dtype, device=caches[0].k.device,
                            retire_when_idle=shared)
        return batch if batch.compatible(caches) else None

    def _shared_decode_batch(self, slot: DecodeSlot) -> DecodeBatch | None:
        """The engine's decode batch for the generation in `slot`, or None.

        None means: decode on your own (batching off, compute thread off, a
        model the batch cannot run, or rows busy with another configuration).
        """
        rows = max_batch_rows()
        if rows < 2 or not _compute.offloading():
            return None
        model, caches = slot.model, slot.caches
        if model is None or not caches or not batchable_model(model):
            return None
        with _BATCH_LOCK:
            batch = self.__dict__.get("_decode_batch")
            if batch is not None and batch.model is model and batch.rows == rows \
                    and batch.compatible(caches):
                return batch
            if batch is not None and batch.model is model and batch.active_rows():
                return None              # other settings still decoding: wait for them
            batch = self._new_decode_batch(model, rows, caches, shared=True)
            self.__dict__["_decode_batch"] = batch
            return batch

    def _prepare_prompt_ids(self, start_str: str,
                            reserve_generation_slot: bool = True,
                            max_gen_tokens: int | None = None) -> list[int]:
        if self.model is None:
            raise ValueError("Train or load a model first!")

        ids = self.encode(start_str)
        if not ids:
            ids = [0]

        # Safety cap on the prompt (bounds the prefill work and, without a
        # context window, the KV cache); prompts may exceed max_seq_len.
        safety_cap = max(self.model.max_seq_len, (max_gen_tokens or 0) + self.model.max_seq_len, 8192)
        max_prompt_tokens = safety_cap - (1 if reserve_generation_slot else 0)
        max_prompt_tokens = max(1, max_prompt_tokens)
        if len(ids) > max_prompt_tokens:
            ids = ids[-max_prompt_tokens:]
        return ids

    @staticmethod
    def _sampling_params(temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
                         repetition_penalty: float = 1.0, min_p: float = 0.0,
                         presence_penalty: float = 0.0,
                         frequency_penalty: float = 0.0) -> SamplingParams:
        return SamplingParams(
            temperature=float(temperature), top_k=int(top_k or 0), top_p=float(top_p),
            repetition_penalty=float(repetition_penalty or 1.0), min_p=float(min_p or 0.0),
            presence_penalty=float(presence_penalty or 0.0),
            frequency_penalty=float(frequency_penalty or 0.0))

    def _hard_cap(self, length: int, prompt_len: int = 0) -> int:
        # Safety cap on the total sequence length. Generation may run past the
        # trained max_seq_len: the context window (2.11) keeps attention inside
        # the trained distances. The prompt itself is capped by
        # `_prepare_prompt_ids`; 2.10 left a capped long prompt room for one
        # generated token only.
        return max(self.model.max_seq_len + length, self.model.max_seq_len * 4, 8192,
                   int(prompt_len) + length)

    def _iter_token_ids(self, ids: list[int], length: int, sp: SamplingParams,
                        stop_token_ids: set[int] | None = None, *,
                        on_token: TokenCallback | None = None,
                        constraint: Any = None) -> Iterator[int]:
        """The decode loop every native generation path shares (2.8).

        Yields each newly sampled id (a stop token is yielded, then the loop
        ends). One prefill over the prompt computes logits for the last
        position only, the KV cache is preallocated for prompt + `length`, and
        each further step feeds one token. 2.7 had four copies of this loop.

        Inference mode is entered per step, not across `yield`: a suspended
        generator must not leave the caller's thread in inference mode.

        2.12: the KV cache belongs to this generation (a `DecodeSlot`) and is
        installed into the model for each step under the model lock, so
        generations running at the same time interleave token by token instead
        of overwriting each other's cache. The lock is never held across
        `yield`: an abandoned stream blocks nobody.

        2.14: after the prompt, the cache moves into a row of the engine's
        :class:`~auralite.engine.batching.DecodeBatch`, and the decode steps
        of all generations running at the same time share one forward pass on
        the compute thread. Sampling stays here, outside any lock.

        `on_token(token_id, logits)` (2.14), when given, sees every sampled id
        with the raw logits it was drawn from (the server's ``logprobs``).

        `constraint` (2.17, :class:`~auralite.engine.structured.JsonConstraint`)
        masks the logits before every draw and follows the drawn tokens; the
        loop ends when the JSON is complete. ``on_token`` still sees the raw
        logits (the model's own distribution).
        """
        if self.model is None:
            raise ValueError("Train or load a model first!")
        if length <= 0:
            return
        self.model.eval()
        self._sync_context_window()
        hard_cap = self._hard_cap(length, len(ids))
        stop = stop_token_ids or set()
        history = list(ids)
        slot = DecodeSlot(min(len(ids) + length, hard_cap))
        batch: DecodeBatch | None = None
        row: int | None = None
        try:
            with self._decode_step(slot), torch.inference_mode():
                t = torch.tensor([ids], dtype=torch.long, device=self.device)
                logits = self._prefill(t)
                step_logits = logits[0, -1]
                if constraint is None:
                    nxt = self._sample_with(step_logits, sp, history)
                    if on_token is not None:
                        on_token(nxt, step_logits)
            if constraint is not None:          # masks are built outside the model lock
                with torch.inference_mode():
                    nxt = constraint.advance(self._sample_with(constraint.mask(step_logits),
                                                               sp, history))
                    if on_token is not None:
                        on_token(nxt, step_logits)
            history.append(nxt)
            yield nxt
            if nxt in stop or (constraint is not None and constraint.done):
                return
            engine_model = self.model
            if length > 1 and len(history) < hard_cap:
                if _unwrap(self._native_model()) is not slot.model:
                    raise RuntimeError(_MODEL_REPLACED)     # replaced after the prompt
                batch = self._shared_decode_batch(slot)
                if batch is not None:
                    with torch.inference_mode():
                        row = _compute.run_on_compute_thread(batch.admit, slot.caches,
                                                             slot.capacity)
                    if row is None:
                        batch = None     # every row taken: decode on our own
                    else:
                        slot.close()     # the row holds the cache now
            for _ in range(length - 1):
                pos = len(history) - 1
                if pos + 1 >= hard_cap:
                    break
                if batch is not None and row is not None:
                    if self.model is not engine_model:
                        raise RuntimeError(_MODEL_REPLACED)
                    with torch.inference_mode():
                        step_logits = _compute.compute_thread.run_in_group(
                            batch, [(row, nxt, pos)])[0]
                        if constraint is not None:
                            nxt = constraint.advance(self._sample_with(
                                constraint.mask(step_logits), sp, history))
                        else:
                            nxt = self._sample_with(step_logits, sp, history)
                        if on_token is not None:
                            on_token(nxt, step_logits)
                else:
                    with self._decode_step(slot), torch.inference_mode():
                        t = torch.tensor([[nxt]], dtype=torch.long, device=self.device)
                        logits = self._infer(t, start_pos=pos, use_cache=True)
                        step_logits = logits[0, -1]
                        if constraint is None:
                            nxt = self._sample_with(step_logits, sp, history)
                    if constraint is not None:
                        with torch.inference_mode():
                            nxt = constraint.advance(self._sample_with(
                                constraint.mask(step_logits), sp, history))
                    if on_token is not None:
                        with torch.inference_mode():
                            on_token(nxt, step_logits)
                history.append(nxt)
                yield nxt
                if nxt in stop or (constraint is not None and constraint.done):
                    return
        finally:
            if batch is not None and row is not None:
                batch.release(row)
            slot.close()       # an abandoned stream only drops its own cache

    def _generate_ids(self, ids: list[int], length: int = 50,
                      temperature: float = 0.8,
                      top_k: int = 50, top_p: float = 0.9,
                      repetition_penalty: float = 1.0,
                      min_p: float = 0.0,
                      stop_token_ids: set[int] | None = None,
                      *, presence_penalty: float = 0.0,
                      frequency_penalty: float = 0.0,
                      on_token: TokenCallback | None = None,
                      constraint: Any = None) -> list[int]:
        if self.model is None:
            raise ValueError("Train or load a model first!")
        sp = self._sampling_params(temperature, top_k, top_p, repetition_penalty, min_p,
                                   presence_penalty, frequency_penalty)
        # a constraint only when there is one: overrides of the loop (tests,
        # subclasses) written before 2.17 do not take the keyword
        extra: dict[str, Any] = {"constraint": constraint} if constraint is not None else {}
        return list(ids) + list(self._iter_token_ids(list(ids), length, sp, stop_token_ids,
                                                     on_token=on_token, **extra))

    def _propose_prompt_lookup(self, ids: list[int], max_proposals: int,
                               ngram: int = 2, max_lookback: int = 512) -> list[int]:
        """Prompt-lookup drafting: continue the most recent matching n-gram.

        Finds the previous occurrence of the last `ngram` tokens inside the
        recent context and proposes the tokens that followed it. No second
        model is needed; it shines on repetitive text (code, lists, quotes)
        and costs a few list comparisons.
        """
        if max_proposals <= 0:
            return []
        n = min(ngram, max(0, len(ids) - 1))
        if n <= 0:
            return []
        suffix = ids[-n:]
        search_end = len(ids) - n
        search_start = max(0, search_end - max_lookback)
        for start in range(search_end - 1, search_start - 1, -1):
            if ids[start:start + n] == suffix:
                proposal = ids[start + n:start + n + max_proposals]
                if proposal:
                    return list(proposal)
        return []

    def _propose_draft_model(self, draft_engine, ids: list[int],
                             max_proposals: int) -> list[int]:
        """Greedy proposal from a small draft model with the same vocabulary.

        Stateless variant (one full draft pass); the speculative loop itself
        uses :class:`_DraftSession`, which keeps the draft's KV cache between
        rounds.
        """
        try:
            out = draft_engine._generate_ids(ids, max_proposals, temperature=0.0, top_k=1)
            return list(out[len(ids):len(ids) + max_proposals])
        except Exception:
            return []

    def _sample_corrected_token(self, filtered_logits: torch.Tensor, rejected_id: int) -> int:
        """Sample from the *residual* distribution p\\{rejected_id} (spec-decoding).

        With a deterministic (point-mass) proposal the rejection rule reduces
        to: accept the draft token with probability p(token); on rejection the
        exact correction is a draw from p with that token's mass removed and
        the remainder renormalized. This keeps the output distribution
        identical to plain sampling with the same filters.
        """
        probs = torch.softmax(filtered_logits, dim=-1)
        probs = probs.clone()
        probs[rejected_id] = 0.0
        total = probs.sum()
        if not torch.isfinite(total) or total <= 0:
            return self._sample_token(filtered_logits, 1.0, 0, 1.0)
        return int(torch.multinomial(probs / total, num_samples=1).item())

    def _speculative_generate_ids(self, ids: list[int], length: int,
                                  temperature: float, top_k: int, top_p: float,
                                  repetition_penalty: float, min_p: float,
                                  draft_engine=None, draft_tokens: int = 4,
                                  stop_token_ids: set[int] | None = None,
                                  stats: dict | None = None, *,
                                  presence_penalty: float = 0.0,
                                  frequency_penalty: float = 0.0) -> list[int]:
        """Exact speculative decoding loop (shared by both draft sources).

        Verification is exact for *any* temperature: each proposal is treated as
        a point-mass draft distribution, accepted with probability p_target(x),
        and replaced by a draw from the residual distribution on rejection — so
        the sampled sequence follows the same law as normal generation, while
        accepted tokens cost one batch forward instead of one forward each.

        2.8: a rejected draft token rolls the KV cache back
        (``truncate_cache``) instead of re-running the model over the whole
        prefix, and a draft model keeps its own cache across rounds instead of
        re-reading the full context for every proposal. ``stats`` reports
        ``rollbacks``; ``cache_resets`` (full re-prefills) stays 0 unless the
        model cannot truncate its cache.
        """
        model = self.model
        model.eval()
        result_ids = list(ids)
        if length <= 0:
            return result_ids

        sp = self._sampling_params(temperature, top_k, top_p, repetition_penalty, min_p,
                                   presence_penalty, frequency_penalty)
        self._sync_context_window()
        hard_cap = self._hard_cap(length, len(ids))
        stop = set(stop_token_ids or ())
        can_truncate = callable(getattr(model, "truncate_cache", None))
        n_drafted = n_accepted = n_resets = n_rollbacks = 0
        draft = (_DraftSession(draft_engine, len(ids) + length + draft_tokens + 1)
                 if draft_engine is not None else None)

        def _forward(tokens: list[int], start: int, keep: int | None = None) -> torch.Tensor:
            t = torch.tensor([tokens], dtype=torch.long, device=self.device)
            return self._infer(t, start_pos=start, use_cache=True, logits_to_keep=keep)

        def _rollback(end: int) -> None:
            """Cache must hold exactly result_ids[:end] afterwards."""
            nonlocal n_resets, n_rollbacks
            if can_truncate:
                try:
                    model.truncate_cache(end)
                    n_rollbacks += 1
                    return
                except ValueError:
                    # A sliding-window cache already dropped the history this
                    # rollback needs (drafts longer than its slack): re-prefill.
                    pass
            n_resets += 1                       # 2.7 path (foreign models)
            model.reset_cache()
            if end:
                self._prefill(torch.tensor([result_ids[:end]], dtype=torch.long,
                                           device=self.device))

        try:
            with self._exclusive_decode(min(len(ids) + length + draft_tokens + 1,
                                            hard_cap + draft_tokens)), torch.inference_mode():
                pending = self._prefill(torch.tensor([result_ids], dtype=torch.long,
                                                     device=self.device))[0, -1]
                while len(result_ids) - len(ids) < length:
                    if len(result_ids) + 1 >= hard_cap:
                        break
                    # Never draft more than the remaining budget: a fully accepted
                    # batch must not overshoot `length` by an extra token.
                    remaining = length - (len(result_ids) - len(ids))
                    if draft is not None:
                        proposals = draft.propose(result_ids, draft_tokens)
                    else:
                        proposals = self._propose_prompt_lookup(result_ids, draft_tokens)
                    proposals = [int(t) for t in proposals if 0 <= int(t) < self.vocab_size
                                 and int(t) not in stop][:max(0, min(draft_tokens, remaining))]

                    if not proposals:
                        # No draft available → ordinary single-token step.
                        nxt = self._sample_with(pending, sp, result_ids)
                        result_ids.append(nxt)
                        if nxt in stop:
                            break
                        pending = _forward([nxt], len(result_ids) - 1)[0, -1]
                        continue

                    n_drafted += len(proposals)
                    base = len(result_ids)
                    verif = _forward(proposals, base)   # cache: result_ids + proposals
                    rejected_at = None
                    for i, tok in enumerate(proposals):
                        step_logits = pending if i == 0 else verif[0, i - 1]
                        filtered = self._filter_with(step_logits, sp, result_ids)
                        probs = torch.softmax(filtered, dim=-1)
                        p_tok = float(probs[tok].item()) if torch.isfinite(probs[tok]) else 0.0
                        if torch.rand(1).item() < p_tok:
                            result_ids.append(tok)
                            n_accepted += 1
                        else:
                            result_ids.append(self._sample_corrected_token(filtered, tok))
                            rejected_at = i
                            break

                    if rejected_at is None:
                        pending = verif[0, len(proposals) - 1]
                        continue
                    corrected = result_ids[-1]
                    if corrected in stop:
                        break
                    # Keep the accepted drafts, forget the rejected tail, then
                    # feed the corrected token like a normal decode step.
                    _rollback(base + rejected_at)
                    pending = _forward([corrected], base + rejected_at)[0, -1]
        finally:
            if draft is not None:
                draft.close()
            if stats is not None:
                stats.update({"drafted": n_drafted, "accepted": n_accepted,
                              "acceptance_rate": (n_accepted / n_drafted) if n_drafted else 0.0,
                              "cache_resets": n_resets, "rollbacks": n_rollbacks,
                              "tokens": len(result_ids) - len(ids)})
        return result_ids

    def generate_speculative(self, start_str: str, length: int = 50,
                             draft_engine: "GenerationMixin | None" = None,
                             draft_tokens: int = 4,
                             temperature: float = 0.8,
                             top_k: int = 50, top_p: float = 0.9,
                             repetition_penalty: float = 1.0,
                             min_p: float = 0.0,
                             stats: dict | None = None, *,
                             presence_penalty: float = 0.0,
                             frequency_penalty: float = 0.0) -> str:
        """Speculative decoding with prompt-lookup drafting (v2.6.5 — real).

        Two ways to obtain draft tokens:
        - **prompt-lookup** (default, no extra model): if the last `ngram`
          tokens appeared earlier in the context, propose what followed them —
          very effective on repetitive text;
        - **draft_engine**: a small compatible model proposes its greedy tokens.

        Drafts are verified in a single batched forward pass; acceptance uses
        exact speculative sampling (accept with probability p_target, resample
        from the residual on rejection), so the output distribution matches
        ordinary `generate()` — including temperature, top-k/top-p and
        repetition penalty — while accepted proposals skip forward passes.
        Measured speedup: see `stats` (`acceptance_rate`); on non-repetitive
        text the fallback per-token path keeps it close to plain decoding.

        For GGUF/HF backends this delegates to the backend's own generation.
        """
        if self.is_gguf_model() or self.is_hf_model():
            return self.generate(start_str, length, temperature, top_k, top_p,
                                 repetition_penalty, min_p)

        if self.tokenizer is None:
            return self.generate(start_str, length, temperature, top_k, top_p,
                                 repetition_penalty, min_p)
        # Conservative compatibility check for an external draft model: it must
        # share the target's vocabulary (token ids are compared directly).
        if draft_engine is not None:
            draft_vocab = getattr(draft_engine.tokenizer, "to_dict", lambda: None)()
            target_vocab = getattr(self.tokenizer, "to_dict", lambda: None)()
            if not draft_vocab or draft_vocab != target_vocab or draft_engine.model is None:
                draft_engine = None

        ids = self._prepare_prompt_ids(start_str, max_gen_tokens=length)
        used_prompt = self.decode(ids)
        result_ids = self._speculative_generate_ids(
            ids, length, temperature, top_k, top_p, repetition_penalty, min_p,
            draft_engine=draft_engine, draft_tokens=draft_tokens, stats=stats,
            presence_penalty=presence_penalty, frequency_penalty=frequency_penalty)
        generated_full = self.decode(result_ids)
        return start_str + generated_full[len(used_prompt):]

    # ---- Generation ---------------------------------------------------
    def generate(self, start_str: str, length: int = 50,
                 temperature: float = 0.8,
                 top_k: int = 50, top_p: float = 0.9,
                 repetition_penalty: float = 1.0,
                 min_p: float = 0.0, *,
                 presence_penalty: float = 0.0,
                 frequency_penalty: float = 0.0,
                 on_token: TokenCallback | None = None,
                 grammar: Any = None) -> str:
        """Continue `start_str` by up to `length` tokens.

        `presence_penalty` / `frequency_penalty` (2.8, OpenAI semantics) lower
        the logits of tokens among the last 64 by ``presence + frequency *
        count``; the native engine and GGUF honour them, Hugging Face models
        (which have no equivalent) ignore them.

        `on_token(token_id, logits)` (2.14, native models only) is called for
        every generated token with the logits it was sampled from.

        `grammar` (2.17, a :class:`~auralite.engine.structured.JsonGrammar`):
        the continuation is valid JSON of that grammar.
        """
        if on_token is not None:
            self._require_native_for_token_callback()
        if grammar is not None and self.is_hf_model():
            self.structured_check(grammar)               # type: ignore[attr-defined]
        if self.is_gguf_model():
            return self._gguf_generate_text(
                start_str, length, temperature, top_k, top_p,
                repetition_penalty=repetition_penalty, min_p=min_p,
                presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
                grammar=grammar,
            )

        if self.is_hf_model():
            # HF models use token count directly as max_new_tokens
            kwargs = {
                "max_new_tokens": length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if min_p > 0.0:
                kwargs["min_p"] = min_p
            return self.hf_proxy.generate(start_str, **kwargs)

        ids = self._prepare_prompt_ids(start_str, max_gen_tokens=length)
        constraint = self._json_constraint(grammar, length)  # type: ignore[attr-defined]
        extra: dict[str, Any] = {"constraint": constraint} if constraint is not None else {}
        result_ids = self._generate_ids(
            ids, length, temperature, top_k, top_p,
            repetition_penalty=repetition_penalty, min_p=min_p,
            presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
            on_token=on_token, **extra)
        return start_str + self._decode_after(ids, result_ids[len(ids):])

    # ---- Text of generated ids (2.14.1) --------------------------------
    def _decode_after(self, prompt_ids, new_ids) -> str:
        """The text `new_ids` add after `prompt_ids`.

        ``decode(new_ids)`` for AuraLite's tokenizers; a plugged-in BPE
        decodes with the last prompt tokens (tokenizer_io.decode_continuation).
        """
        if getattr(self.tokenizer, "context_sensitive_decode", False):
            return _tok_io.decode_continuation(self.tokenizer, prompt_ids, new_ids)
        return self.decode(new_ids)

    def _stream_decoder(self, prompt_ids) -> Any:
        """``step(id) -> text`` / ``flush()`` for streaming (tokenizer_io.StreamDecoder)."""
        if getattr(self.tokenizer, "context_sensitive_decode", False):
            return _tok_io.StreamDecoder(self.tokenizer, prompt_ids)
        return _TokenByToken(self.decode)

    def _require_native_for_token_callback(self) -> None:
        if self.model is not None and (self.is_gguf_model() or self.is_hf_model()):
            raise ValueError("on_token (per-token logits) needs a native AuraLite model; "
                             "GGUF and Hugging Face models generate in their own runtime")

    # ---- Thinking Mode (two-pass generation) ---------------------------
    def generate_with_thinking(self, start_str: str, length: int = 50,
                               temperature: float = 0.8,
                               top_k: int = 50, top_p: float = 0.9,
                               repetition_penalty: float = 1.0,
                               thinking_length: int | None = None,
                               thinking_temperature: float | None = None,
                               web_context: str | None = None,
                               min_p: float = 0.0
                               ) -> tuple[str, str]:
        """Two-pass 'thinking' generation.

        Pass 1 ("thinking"): the model free-writes a draft continuation of
        the prompt at a slightly higher temperature — an exploration pass.

        Pass 2 ("answer"): the draft (and optional web-search context) is
        prepended to the prompt as extra conditioning, and the model
        generates the final output at the requested settings.

        Returns (thinking_text, final_text).

        Note: this is an inference-time technique. The small model is not
        trained to reason, but conditioning the second pass on its own
        draft (self-conditioning) plus retrieved web snippets typically
        yields more on-topic continuations.
        """
        if self.model is None:
            raise ValueError("Train or load a model first!")

        if thinking_length is None:
            thinking_length = max(16, length // 2)
        if thinking_temperature is None:
            thinking_temperature = min(2.0, temperature + 0.2)

        if self.is_gguf_model():
            think_prompt = f"{web_context}\n{start_str}" if web_context else start_str
            draft_full = self._gguf_generate_text(
                think_prompt, thinking_length,
                thinking_temperature, top_k, top_p,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
            )
            thinking_text = draft_full[len(think_prompt):].strip()

            ctx_parts = []
            if web_context:
                ctx_parts.append(web_context)
            if thinking_text:
                ctx_parts.append(thinking_text)
            ctx_parts.append(start_str)
            final_prompt = "\n".join(ctx_parts)
            final_full = self._gguf_generate_text(
                final_prompt, length, temperature, top_k, top_p,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
            )
            final_text = start_str + final_full[len(final_prompt):]
            return thinking_text, final_text

        if self.is_hf_model():
            think_prompt = f"{web_context}\n{start_str}" if web_context else start_str
            think_kwargs = {
                "max_new_tokens": thinking_length,
                "temperature": thinking_temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if min_p > 0:
                think_kwargs["min_p"] = min_p
            draft_full = self.hf_proxy.generate(think_prompt, **think_kwargs)
            thinking_text = (draft_full[len(think_prompt):] if draft_full.startswith(think_prompt) else draft_full).strip()

            ctx_parts = []
            if web_context:
                ctx_parts.append(web_context)
            if thinking_text:
                ctx_parts.append(thinking_text)
            ctx_parts.append(start_str)
            final_prompt = "\n".join(ctx_parts)
            final_kwargs = {
                "max_new_tokens": length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if min_p > 0:
                final_kwargs["min_p"] = min_p
            final_full = self.hf_proxy.generate(final_prompt, **final_kwargs)
            continuation = final_full[len(final_prompt):] if final_full.startswith(final_prompt) else final_full
            return thinking_text, start_str + continuation

        # ---- Pass 1: exploration draft ---------------------------------
        think_prompt = start_str
        if web_context:
            think_prompt = f"{web_context}\n{start_str}"

        think_ids = self._prepare_prompt_ids(think_prompt, max_gen_tokens=thinking_length)
        think_prompt_used = self.decode(think_ids)
        draft_full = self.decode(self._generate_ids(
            think_ids, thinking_length,
            temperature=thinking_temperature,
            top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, min_p=min_p))
        # Keep only the newly generated part as the "thoughts"
        thinking_text = draft_full[len(think_prompt_used):].strip()

        # ---- Pass 2: final answer conditioned on the draft -------------
        ctx_parts = []
        if web_context:
            ctx_parts.append(web_context)
        if thinking_text:
            ctx_parts.append(thinking_text)
        ctx_parts.append(start_str)

        final_prompt = "\n".join(ctx_parts)
        final_ids = self._prepare_prompt_ids(final_prompt, max_gen_tokens=length)
        final_prompt_used = self.decode(final_ids)
        final_full = self.decode(self._generate_ids(
            final_ids, length,
            temperature=temperature,
            top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, min_p=min_p))
        final_text = start_str + final_full[len(final_prompt_used):]

        return thinking_text, final_text

    # ---- Streaming Generation (NEW) -----------------------------------
    def generate_streaming(self, start_str: str, length: int = 50,
                           temperature: float = 0.8,
                           top_k: int = 50, top_p: float = 0.9,
                           repetition_penalty: float = 1.0,
                           min_p: float = 0.0, *,
                           presence_penalty: float = 0.0,
                           frequency_penalty: float = 0.0,
                           on_token: TokenCallback | None = None,
                           grammar: Any = None) -> Iterator[str]:
        """Generate text token-by-token, yielding each new token as it's produced.

        Yields individual decoded tokens so the GUI can update in real-time.
        `on_token` and `grammar` as in :meth:`generate`: `on_token` is called
        before the token's text is yielded.
        """
        if self.model is None:
            raise ValueError("Train or load a model first!")
        if on_token is not None:
            self._require_native_for_token_callback()
        if grammar is not None and self.is_hf_model():
            self.structured_check(grammar)               # type: ignore[attr-defined]
        gguf_grammar = (self._gguf_structured_kwargs(grammar)  # type: ignore[attr-defined]
                        if grammar is not None and self.is_gguf_model() else {})

        if self.is_gguf_model():
            if not is_gguf_like(self.model):
                return
            if self.model.use_chat_completion:
                stream = self.model.create_chat_completion(
                    self._gguf_prompt_messages(start_str),
                    max_tokens=length, temperature=temperature,
                    top_k=top_k, top_p=top_p, min_p=min_p,
                    repeat_penalty=repetition_penalty, **self._gguf_penalty_kwargs(presence_penalty, frequency_penalty), stream=True,
                    **gguf_grammar,
                )
                for chunk in stream:
                    try:
                        text = chunk["choices"][0].get("delta", {}).get("content", "")
                    except Exception:
                        text = ""
                    if text:
                        yield text
            else:
                stream = self.model.create_completion(
                    start_str, max_tokens=length, temperature=temperature,
                    top_k=top_k, top_p=top_p, min_p=min_p,
                    repeat_penalty=repetition_penalty, **self._gguf_penalty_kwargs(presence_penalty, frequency_penalty), stream=True,
                    **gguf_grammar,
                )
                for chunk in stream:
                    try:
                        text = chunk["choices"][0].get("text", "")
                    except Exception:
                        text = ""
                    if text:
                        yield text
            return

        if self.is_hf_model():
            kwargs = {
                "max_new_tokens": length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if min_p > 0:
                kwargs["min_p"] = min_p
            try:
                for token in self.hf_proxy.generate_streaming(start_str, **kwargs):
                    yield token
            except Exception:
                full = self.hf_proxy.generate(start_str, **kwargs)
                yield full[len(start_str):] if full.startswith(start_str) else full
            return

        ids = self._prepare_prompt_ids(start_str, max_gen_tokens=length)
        sp = self._sampling_params(temperature, top_k, top_p, repetition_penalty, min_p,
                                   presence_penalty, frequency_penalty)
        stream = self._stream_decoder(ids)
        constraint = self._json_constraint(grammar, length)  # type: ignore[attr-defined]
        extra: dict[str, Any] = {"constraint": constraint} if constraint is not None else {}
        for nxt in self._iter_token_ids(ids, length, sp, on_token=on_token, **extra):
            yield stream.step(nxt)
        rest = stream.flush()
        if rest:
            yield rest

    def _generate_batch_group(self, batch_ids: list[list[int]], length: int = 50,
                              temperature: float = 0.8,
                              top_k: int = 50, top_p: float = 0.9,
                              repetition_penalty: float = 1.0,
                              min_p: float = 0.0, *,
                              presence_penalty: float = 0.0,
                              frequency_penalty: float = 0.0) -> list[list[int]]:
        """Generate in parallel for a batch of prompt ids.

        2.14: the prompts may have different lengths; they decode as one
        batch (:meth:`_generate_rows`). A model the batch cannot run decodes
        one group of equal-length prompts after the other, as in 2.13.
        """
        if self.model is None:
            raise ValueError("Train or load a model first!")
        if not batch_ids:
            return []

        result_ids = [list(ids) for ids in batch_ids]
        if length <= 0:
            return result_ids
        sp = self._sampling_params(temperature, top_k, top_p, repetition_penalty, min_p,
                                   presence_penalty, frequency_penalty)
        rows_out = self._generate_rows(result_ids, length, sp)
        if rows_out is not None:
            return rows_out
        grouped: dict[int, list[int]] = defaultdict(list)
        for i, ids in enumerate(result_ids):
            grouped[len(ids)].append(i)
        for prompt_len in sorted(grouped, reverse=True):
            members = grouped[prompt_len]
            out = self._generate_equal_length(
                [result_ids[i] for i in members], length, sp)
            for i, ids in zip(members, out, strict=True):
                result_ids[i] = ids
        return result_ids

    def _generate_equal_length(self, batch_ids: list[list[int]], length: int,
                               sp: SamplingParams) -> list[list[int]]:
        """The 2.13 batch: prompts of one length, one shared position."""
        result_ids = [list(ids) for ids in batch_ids]
        self._sync_context_window()
        hard_cap = self._hard_cap(length, len(batch_ids[0]))
        B = len(batch_ids)
        # The batch holds the model for the whole call, with its own cache (2.12).
        with self._exclusive_decode(min(len(batch_ids[0]) + length, hard_cap)), \
                torch.inference_mode():
            batch = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
            logits = self._prefill(batch)
            next_tokens = []
            for b in range(B):
                nxt = self._sample_with(logits[b, -1], sp, result_ids[b])
                result_ids[b].append(nxt)
                next_tokens.append(nxt)

            for _ in range(length - 1):
                pos = len(result_ids[0]) - 1
                if pos + 1 >= hard_cap:
                    break
                next_input = torch.tensor(next_tokens, dtype=torch.long,
                                          device=self.device).unsqueeze(1)
                logits = self._infer(next_input, start_pos=pos, use_cache=True)
                next_tokens = []
                for b in range(B):
                    nxt = self._sample_with(logits[b, -1], sp, result_ids[b])
                    result_ids[b].append(nxt)
                    next_tokens.append(nxt)
        return result_ids

    def _generate_rows(self, prompt_ids: list[list[int]], length: int,
                       sp: SamplingParams) -> list[list[int]] | None:
        """Decode every prompt as one batch, each row at its own position (2.14).

        Prompts of equal length run their prefill together (as in 2.13), then
        all rows decode in lockstep, one forward pass per token for the whole
        batch, in a :class:`DecodeBatch` of their own. Returns None when the
        model cannot be batched (the caller falls back to length groups).
        """
        model = _unwrap(self._native_model())
        if model is None or not batchable_model(model):
            return None
        self._sync_context_window()
        n = len(prompt_ids)
        result = [list(ids) for ids in prompt_ids]
        caps = [self._hard_cap(length, len(ids)) for ids in prompt_ids]
        grouped: dict[int, list[int]] = defaultdict(list)
        for i, ids in enumerate(prompt_ids):
            grouped[len(ids)].append(i)
        batch: DecodeBatch | None = None
        rows: list[int] = [0] * n
        for prompt_len in sorted(grouped, reverse=True):
            members = grouped[prompt_len]
            slot = DecodeSlot(min(prompt_len + length, caps[members[0]]))
            try:
                with self._decode_step(slot), torch.inference_mode():
                    t = torch.tensor([prompt_ids[i] for i in members], dtype=torch.long,
                                     device=self.device)
                    logits = self._prefill(t)
                if slot.model is not model:
                    raise RuntimeError(_MODEL_REPLACED)
                if batch is None:
                    batch = self._new_decode_batch(model, n, slot.caches or [])
                    if batch is None:
                        return None
                elif not batch.compatible(slot.caches or []):
                    return None
                with torch.inference_mode():
                    for b, i in enumerate(members):
                        row = _compute.run_on_compute_thread(batch.admit, slot.caches,
                                                             slot.capacity, b)
                        assert row is not None
                        rows[i] = row
                        result[i].append(self._sample_with(logits[b, -1], sp, result[i]))
            finally:
                slot.close()
        assert batch is not None
        for _ in range(length - 1):
            todo = [i for i in range(n) if len(result[i]) < caps[i]]
            if not todo:
                break
            if _unwrap(self._native_model()) is not model:
                raise RuntimeError(_MODEL_REPLACED)
            items = [(rows[i], result[i][-1], len(result[i]) - 1) for i in todo]
            with self.model_lock, torch.inference_mode():
                step_logits = _compute.run_on_compute_thread(batch.run_items, items)
                for i, row_logits in zip(todo, step_logits, strict=True):
                    result[i].append(self._sample_with(row_logits, sp, result[i]))
        return result

    # ---- Batch Generation (NEW) ---------------------------------------
    def generate_batch(self, prompts: list[str], length: int = 50,
                       temperature: float = 0.8,
                       top_k: int = 50, top_p: float = 0.9,
                       repetition_penalty: float = 1.0,
                       min_p: float = 0.0, *,
                       presence_penalty: float = 0.0,
                       frequency_penalty: float = 0.0) -> list[str]:
        """Generate text for multiple prompts in parallel.

        Prompts are first truncated to the model context limit. 2.14: all of
        them decode as one batch, every row at its own position (one forward
        pass per token for the whole batch); prompts of equal length still
        share their prefill. 2.13 decoded each group of equal-length prompts
        separately, one group after the other. Models the batch cannot run
        (quantized or custom modules) keep the 2.13 groups.

        v2.6.5: identical prompts are deduplicated — a batch like
        ["What is X?", "What is X?", "something else"] runs the duplicate once and
        reuses the result, which is the common GUI/API pattern.
        """
        if self.model is None:
            raise ValueError("Train or load a model first!")
        if not prompts:
            return []

        if self.is_gguf_model() or self.is_hf_model():
            return [self.generate(p, length, temperature, top_k, top_p,
                                  repetition_penalty=repetition_penalty, min_p=min_p,
                                  presence_penalty=presence_penalty,
                                  frequency_penalty=frequency_penalty)
                    for p in prompts]

        # --- Deduplicate identical prompts -----------------------------
        unique_prompts: list[str] = []
        first_index: dict[str, int] = {}
        for p in prompts:
            if p not in first_index:
                first_index[p] = len(unique_prompts)
                unique_prompts.append(p)

        self.model.eval()

        prepared = [self._prepare_prompt_ids(prompt, max_gen_tokens=length)
                    for prompt in unique_prompts]

        results: list[str | None] = [None] * len(unique_prompts)
        generated_ids = self._generate_batch_group(
            prepared, length, temperature, top_k, top_p,
            repetition_penalty=repetition_penalty, min_p=min_p,
            presence_penalty=presence_penalty, frequency_penalty=frequency_penalty)
        for orig_idx, (used_ids, out_ids) in enumerate(zip(prepared, generated_ids, strict=True)):
            used_prompt = self.decode(used_ids)
            out_text = self.decode(out_ids)
            results[orig_idx] = unique_prompts[orig_idx] + out_text[len(used_prompt):]

        unique_final = [r if r is not None else "" for r in results]
        return [unique_final[first_index[p]] for p in prompts]

    def _sample_with(self, logits: torch.Tensor, sp: SamplingParams,
                     history: list[int] | None) -> int:
        """Every decode loop samples through `_sample_token`, which stays the
        one hook to override (tests script it, subclasses may replace it)."""
        return self._sample_token(logits, sp.temperature, sp.top_k, sp.top_p,
                                  sp.repetition_penalty, history, sp.min_p,
                                  presence_penalty=sp.presence_penalty,
                                  frequency_penalty=sp.frequency_penalty)

    def _filter_with(self, logits: torch.Tensor, sp: SamplingParams,
                     history: list[int] | None) -> torch.Tensor:
        return self._filtered_logits(logits, sp.temperature, sp.top_k, sp.top_p,
                                     sp.repetition_penalty, history, sp.min_p,
                                     presence_penalty=sp.presence_penalty,
                                     frequency_penalty=sp.frequency_penalty)

    def _filtered_logits(self, logits: torch.Tensor,
                         temperature: float, top_k: int, top_p: float,
                         repetition_penalty: float = 1.0,
                         recent_ids: list[int] | None = None,
                         min_p: float = 0.0, *,
                         presence_penalty: float = 0.0,
                         frequency_penalty: float = 0.0) -> torch.Tensor:
        """Full-vocabulary logits after every sampling filter (``-inf`` = removed).

        Speculative decoding verifies draft tokens against exactly this
        distribution. Implementation: :mod:`auralite.engine.sampling` (2.8).
        """
        sp = self._sampling_params(temperature, top_k, top_p, repetition_penalty, min_p,
                                   presence_penalty, frequency_penalty)
        return _sampling.filtered_logits(logits, sp, recent_ids)

    def _sample_token(self, logits: torch.Tensor,
                      temperature: float, top_k: int, top_p: float,
                      repetition_penalty: float = 1.0,
                      recent_ids: list[int] | None = None,
                      min_p: float = 0.0, *,
                      presence_penalty: float = 0.0,
                      frequency_penalty: float = 0.0) -> int:
        """Sample one token id (penalties, temperature, top-k, min-p, top-p).

        Falls back to a uniform draw if every token was filtered out.
        """
        sp = self._sampling_params(temperature, top_k, top_p, repetition_penalty, min_p,
                                   presence_penalty, frequency_penalty)
        return _sampling.sample(logits, sp, recent_ids)
