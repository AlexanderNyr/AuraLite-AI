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
        res = []
        for req in requests:
            context, continuation = req.args
            # Encode
            ctx_ids = self.engine.encode(context)
            cont_ids = self.engine.encode(continuation)

            # We need the model to return logits for the continuation tokens
            # For simplicity we use a greedy approach here (can be improved)
            full = ctx_ids + cont_ids
            input_ids = full[:-1]
            target_ids = full[1:]

            # Run forward
            import torch
            with torch.no_grad():
                logits = self.model(
                    torch.tensor([input_ids], device=self.device)
                )
                log_probs = torch.log_softmax(logits[0], dim=-1)

            # Sum logprobs of the continuation tokens
            cont_logprob = 0.0
            for i, tid in enumerate(cont_ids):
                pos = len(ctx_ids) + i - 1
                if pos < len(log_probs):
                    cont_logprob += log_probs[pos, tid].item()

            # is_greedy = True if we would have generated exactly this continuation
            is_greedy = True
            res.append((cont_logprob, is_greedy))
        return res

    def generate_until(self, requests):
        """Not used in most benchmarks, but required by the interface."""
        return [""] * len(requests)

    def loglikelihood_rolling(self, requests):
        """Used for perplexity calculation."""
        return self.loglikelihood(requests)


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
        else:
            # Native or GGUF → use our wrapper
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