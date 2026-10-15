"""Inference benchmarks of the native engine (moved out of `_legacy.py` in 2.12).

`bench_prefill` compares eager and compiled prompt processing,
`bench_generation_modes` decode speed and KV-cache size per cache dtype.
Both are methods of :class:`~auralite.engine.AuraLiteEngine` through
:class:`InferenceBenchMixin`; the code is unchanged apart from the model lock.
"""
from __future__ import annotations

import time
from typing import Any

import torch

from auralite.engine.decode_slots import DecodeSlotsMixin


class InferenceBenchMixin(DecodeSlotsMixin):
    """`AuraLiteEngine.bench_prefill` / `bench_generation_modes`."""

    # Provided by AuraLiteEngine.
    model: Any
    device: Any
    _compiled_inference_model: Any

    def _native_model(self) -> Any:  # pragma: no cover - AuraLiteEngine overrides it
        raise NotImplementedError

    def _infer(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def _model_arch_kwargs(self, *args: Any, **kwargs: Any) -> dict:  # pragma: no cover
        raise NotImplementedError

    def enable_inference_compile(self, *args: Any, **kwargs: Any) -> float:  # pragma: no cover
        raise NotImplementedError

    def disable_inference_compile(self) -> bool:  # pragma: no cover
        raise NotImplementedError

    def generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def bench_prefill(self, shapes: tuple = ((1, 64), (4, 64), (1, 256), (4, 256)),
                      repeats: int = 3) -> list[dict]:
        """Compare eager vs `torch.compile`d prompt processing (prefill).

        Prefill is where compilation actually pays off: one big matmul-heavy
        forward per prompt. Decode steps (T=1) are measured by
        `bench_generation_modes()` and stay eager on purpose.

        Returns `[{shape, batch, tokens, eager_tokens_per_sec,
        compiled_tokens_per_sec, speedup, compile_seconds}]`.
        """
        model = self._native_model()
        if model is None:
            raise RuntimeError("bench_prefill needs a native AuraLite model")
        vocab = model.embedding.num_embeddings
        repeats = max(1, int(repeats))

        def _measure(fn, x):
            best = 1e9
            for _ in range(repeats):
                start = time.perf_counter()
                with torch.inference_mode():
                    fn(x)
                best = min(best, time.perf_counter() - start)
            return best

        rows = []
        self.disable_inference_compile()
        eager_times = {}
        for batch, tokens in shapes:
            x = torch.randint(0, vocab, (int(batch), int(tokens)), device=self.device)
            eager_times[(int(batch), int(tokens))] = _measure(self.model, x)
        compile_seconds = self.enable_inference_compile()
        for batch, tokens in shapes:
            key = (int(batch), int(tokens))
            x = torch.randint(0, vocab, key, device=self.device)
            compiled_time = _measure(self._infer, x)
            eager_time = eager_times[key]
            rows.append({
                "shape": f"B={key[0]} T={key[1]}",
                "batch": key[0], "tokens": key[1],
                "eager_tokens_per_sec": key[0] * key[1] / max(1e-9, eager_time),
                "compiled_tokens_per_sec": key[0] * key[1] / max(1e-9, compiled_time),
                "speedup": eager_time / max(1e-9, compiled_time),
                "compile_seconds": compile_seconds,
            })
        self.disable_inference_compile()
        return rows

    def bench_generation_modes(self, tokens: int = 32, prompt: str = "hello",
                               modes: tuple = ("fp32", "fp16", "int8"),
                               include_compile: bool = False,
                               repeats: int = 2) -> list[dict]:
        """Measure decode speed and KV-cache size per configuration.

        Returns `[{mode, tokens_per_sec, seconds, kv_bytes_per_token, notes}]`
        for every requested KV-cache dtype (the model is rebuilt with the same
        weights and a different `kv_cache_dtype`) and, when `include_compile`,
        for the `torch.compile`d variant. The loaded model is restored before
        returning, so callers can keep using the engine.
        """
        model = self._native_model()
        if model is None:
            raise RuntimeError("bench_generation_modes needs a native AuraLite model")
        tokens = max(1, int(tokens))
        repeats = max(1, int(repeats))
        arch = self._model_arch_kwargs(model)
        head_dim = arch["d_model"] // arch["n_heads"]
        kv_heads = arch["n_kv_heads"] or arch["n_heads"]

        def kv_bytes_per_token(spec: str | None) -> int:
            # Measured on the real storage (2.8): int8/fp8 include their
            # per-(token, head) scales, fp16/bf16 really are 2 bytes.
            from auralite.engine.kv_cache import KVCache
            probe = torch.zeros(1, kv_heads, 1, head_dim)
            cache = KVCache(spec)
            cache.append(probe, probe, start_pos=0)
            return cache.bytes_per_token() * arch["n_layers"]

        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        original = self.model
        results: list[dict] = []
        # The variants replace self.model for the duration: other generations
        # wait for the lock instead of decoding with a temporary model (2.12).
        self.model_lock.acquire()
        try:
            for mode in modes:
                mode_key = str(mode).lower()
                variant = type(model)(**arch, kv_cache_dtype=("none" if mode_key == "fp32" else mode_key))
                variant.load_state_dict(state)
                variant.to(self.device)
                variant.eval()
                self.model = variant
                self._compiled_inference_model = None
                speeds = []
                for _ in range(repeats):
                    self.generate(prompt, 1, temperature=0.0, top_k=1)      # warm-up
                    start = time.perf_counter()
                    self.generate(prompt, tokens, temperature=0.0, top_k=1)
                    speeds.append(tokens / max(1e-9, time.perf_counter() - start))
                results.append({
                    "mode": f"kv={mode_key}",
                    "tokens_per_sec": max(speeds) if repeats > 1 else speeds[0],
                    "seconds": tokens / (max(speeds) if repeats > 1 else speeds[0]),
                    "kv_bytes_per_token": kv_bytes_per_token(
                        None if mode_key == "fp32" else mode_key),
                    "notes": "",
                })
            if include_compile:
                self.model = original
                self.disable_inference_compile()
                compile_seconds = self.enable_inference_compile()
                speeds = []
                for _ in range(repeats):
                    self.generate(prompt, 1, temperature=0.0, top_k=1)
                    start = time.perf_counter()
                    self.generate(prompt, tokens, temperature=0.0, top_k=1)
                    speeds.append(tokens / max(1e-9, time.perf_counter() - start))
                best = max(speeds)
                results.append({
                    "mode": "kv=fp32+compile",
                    "tokens_per_sec": best,
                    "seconds": tokens / best,
                    "kv_bytes_per_token": kv_bytes_per_token(None),
                    "notes": f"compile={compile_seconds:.1f}s (cached for later runs)",
                })
        finally:
            self._compiled_inference_model = None
            self.model = original
            self.model_lock.release()
        return results


__all__ = ["InferenceBenchMixin"]
