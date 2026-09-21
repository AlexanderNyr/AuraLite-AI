"""
evaluation.py — Model evaluation using lm-evaluation-harness (optional)

Provides a clean interface to evaluate AuraLite, GGUF and Hugging Face models
on standard benchmarks (MMLU, GSM8K, ARC, Hellaswag, etc.).

The lm-eval package is optional. If not installed, the module gracefully
degrades and raises a clear error message.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable
import json

try:
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    HAS_LM_EVAL = True
except ImportError:
    HAS_LM_EVAL = False
    LM = object  # placeholder so the module can still be imported cleanly


class LMEvalNotAvailableError(ImportError):
    """Raised when lm-evaluation-harness is not installed."""
    pass


def _check_lm_eval():
    if not HAS_LM_EVAL:
        raise LMEvalNotAvailableError(
            "lm-evaluation-harness is required for model evaluation.\n"
            "Install it with:\n"
            "  pip install lm-eval\n"
            "or\n"
            "  pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git"
        )


# ======================================================================
#  Custom LM wrapper for AuraLite native models
# ======================================================================

class AuraLiteLM(LM):
    """
    lm-eval compatible wrapper around AuraLiteEngine's native model.
    """

    def __init__(self, engine, batch_size: int = 1):
        super().__init__()
        self.engine = engine
        self.batch_size = batch_size
        self.model = engine.model
        self.tokenizer = engine.tokenizer
        self.device = engine.device

    def loglikelihood(self, requests):
        """Compute log-likelihood of continuation given context."""
        import torch
        res = []
        for req in requests:
            context, continuation = req.args
            # Encode
            ctx_ids = self.engine.encode(context)
            cont_ids = self.engine.encode(continuation)
            if not cont_ids:
                # Empty continuation: zero-length logprob, vacuously greedy.
                res.append((0.0, True))
                continue

            full = ctx_ids + cont_ids
            input_ids = full[:-1]

            # Run forward
            with torch.no_grad():
                logits = self.model(
                    torch.tensor([input_ids], device=self.device)
                )
                log_probs = torch.log_softmax(logits[0], dim=-1)

            # Sum logprobs of the continuation tokens, and check argmax agreement.
            # (is_greedy was previously hardcoded True, which silently inflated
            # every metric that depends on it.)
            cont_logprob = 0.0
            is_greedy = True
            for i, tid in enumerate(cont_ids):
                pos = len(ctx_ids) + i - 1
                if pos < 0 or pos >= len(log_probs):
                    # Cannot score this token (e.g. empty context edge case);
                    # treat as impossible rather than borrowing a wrong index.
                    cont_logprob = float("-inf")
                    is_greedy = False
                    continue
                lp_row = log_probs[pos]
                cont_logprob += lp_row[tid].item()
                if int(torch.argmax(lp_row).item()) != int(tid):
                    is_greedy = False

            res.append((cont_logprob, is_greedy))
        return res

    def generate_until(self, requests):
        """Generate a continuation per request (needed by GSM8K-style tasks).

        Previously a stub returning empty strings, which broke every
        generative task. Honours lm-eval gen_kwargs: `until` stop strings and
        `max_gen_toks`.
        """
        res = []
        for req in requests:
            context = req.args[0]
            gen_kwargs = req.args[1] if len(req.args) > 1 else {}
            max_toks = int(gen_kwargs.get("max_gen_toks", 256))
            until = gen_kwargs.get("until", []) or []
            if isinstance(until, str):
                until = [until]
            text = self.engine.generate(context, length=max_toks)
            continuation = text[len(context):] if text.startswith(context) else text
            cut = None
            for stop in until:
                i = continuation.find(stop)
                if i != -1 and (cut is None or i < cut):
                    cut = i
            if cut is not None:
                continuation = continuation[:cut]
            res.append(continuation)
        return res

    def loglikelihood_rolling(self, requests):
        """Rolling log-likelihood for perplexity tasks (e.g. wikitext).

        lm-eval passes single-argument requests here and expects a plain list
        of total logprobs (not (logprob, is_greedy) tuples). The old shim
        delegated to loglikelihood(), which crashed unpacking 1-tuples and
        returned the wrong type. Tokens beyond the context window are scored
        with a sliding window of size max_seq_len (stride = window).
        """
        import torch
        res: list[float] = []
        max_seq = int(getattr(self.model, "max_seq_len", 4096) or 4096)
        for req in requests:
            text = req.args[0]
            ids = self.engine.encode(text)
            if len(ids) < 2:
                res.append(0.0)
                continue
            total = 0.0
            with torch.no_grad():
                start = 0
                # First window scores tokens 1..max_seq; then slide.
                while start < len(ids) - 1:
                    end = min(start + max_seq, len(ids) - 1)
                    window_in = ids[start:end]
                    window_tgt = ids[start + 1:end + 1]
                    logits = self.model(
                        torch.tensor([window_in], device=self.device)
                    )
                    log_probs = torch.log_softmax(logits[0], dim=-1)
                    for j, tid in enumerate(window_tgt):
                        total += log_probs[j, tid].item()
                    start = end
            res.append(total)
        return res


# ======================================================================
#  Evaluation Engine
# ======================================================================

class EvaluationEngine:
    """
    High-level evaluation interface for AuraLite.

    Supports:
    - Native AuraLite models
    - GGUF models (via llama.cpp)
    - Hugging Face models
    """

    def __init__(self, engine):
        self.engine = engine
        self.results = {}

    def evaluate(
        self,
        tasks: List[str] | str = "arc_easy",
        num_fewshot: int = 0,
        batch_size: int = 1,
        limit: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation on one or more tasks.

        Args:
            tasks: Task name or list of tasks (e.g. ["arc_easy", "gsm8k", "mmlu"])
            num_fewshot: Number of few-shot examples
            batch_size: Batch size for evaluation
            limit: Limit number of examples (for quick testing)
            progress_callback: Optional callback

        Returns:
            Dictionary with results
        """
        _check_lm_eval()

        if isinstance(tasks, str):
            tasks = [tasks]

        # Prepare model wrapper
        if self.engine.is_hf_model():
            # Use the native HF model directly
            model = self.engine.hf_proxy.model
            tokenizer = self.engine.hf_proxy.tokenizer
            from lm_eval.models.huggingface import HFLM
            lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        elif self.engine.is_gguf_model():
            raise NotImplementedError(
                "GGUF evaluation via lm-evaluation-harness is not implemented in AuraLite yet. "
                "Use a native AuraLite or Hugging Face model for evaluation."
            )
        else:
            # Native AuraLite → use our wrapper
            lm = AuraLiteLM(self.engine, batch_size=batch_size)

        print(f"[AuraLite-Eval] Evaluating on tasks: {tasks}")

        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=str(self.engine.device),
            limit=limit,
        )

        self.results = results
        return results

    def print_results(self, results: Optional[Dict] = None):
        """Pretty print evaluation results."""
        if results is None:
            results = self.results

        if not results or "results" not in results:
            print("No results to display.")
            return

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        for task, metrics in results["results"].items():
            print(f"\n{task}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric:20s}: {value:.4f}")
                else:
                    print(f"  {metric:20s}: {value}")

        print("\n" + "=" * 60)

    def save_results(self, path: str, results: Optional[Dict] = None):
        """Save results to JSON file."""
        if results is None:
            results = self.results
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[AuraLite-Eval] Results saved to {path}")


# ======================================================================
#  Convenience function
# ======================================================================

def create_evaluator(engine) -> EvaluationEngine:
    """Factory function."""
    return EvaluationEngine(engine)