#!/usr/bin/env python3
"""Evaluate an AuraLite AI model checkpoint on a text file.

Usage:
    python -m auralite.eval --model checkpoint.pt --test test.txt --device cuda

(``python eval.py …`` from a source checkout still works until 2.9.)
"""

import argparse
import math
import time

import torch
import torch.nn.functional as F

from auralite.engine import AuraLiteEngine


def _compute_avg_token_loss(engine: AuraLiteEngine, text: str,
                            batch_size: int = 32) -> tuple[float, int]:
    """Return (mean_nll_nats_per_token, n_predicted_tokens).

    We predict each token once using non-overlapping windows of length
    `seq_length`, which avoids the heavy overlap of a sliding-window eval and
    keeps the metric meaning clear for both char and BPE tokenizers.
    """
    ids = engine.encode(text)
    if len(ids) < 2:
        return float("inf"), 0

    seq_length = engine.params_used.get("seq_length", 64)
    seq_length = max(1, int(seq_length))
    encoded = torch.tensor(ids, dtype=torch.long)

    # Start positions for non-overlapping blocks. All blocks except possibly
    # the tail have exactly `seq_length` predictions.
    n_pred_total = len(encoded) - 1
    full_block_count = n_pred_total // seq_length
    full_starts = [i * seq_length for i in range(full_block_count)]
    tail_start = full_block_count * seq_length
    tail_len = n_pred_total - tail_start

    total_nll = 0.0
    total_pred_tokens = 0
    engine.model.eval()

    with torch.inference_mode():
        for i in range(0, len(full_starts), batch_size):
            batch_starts = full_starts[i:i + batch_size]
            if not batch_starts:
                continue

            xb = torch.stack([encoded[s:s + seq_length] for s in batch_starts]).to(engine.device)
            yb = torch.stack([encoded[s + 1:s + 1 + seq_length] for s in batch_starts]).to(engine.device)

            logits = engine.model(xb)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                yb.reshape(-1),
                reduction="sum",
            )
            total_nll += loss.item()
            total_pred_tokens += yb.numel()

        if tail_len > 0:
            xb = encoded[tail_start:tail_start + tail_len].unsqueeze(0).to(engine.device)
            yb = encoded[tail_start + 1:tail_start + 1 + tail_len].unsqueeze(0).to(engine.device)
            logits = engine.model(xb)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                yb.reshape(-1),
                reduction="sum",
            )
            total_nll += loss.item()
            total_pred_tokens += yb.numel()

    if total_pred_tokens == 0:
        return float("inf"), 0
    return total_nll / total_pred_tokens, total_pred_tokens


def compute_perplexity(engine: AuraLiteEngine, text: str, batch_size: int = 32) -> float:
    """Compute perplexity on token predictions (lower is better)."""
    avg_loss, n_pred = _compute_avg_token_loss(engine, text, batch_size)
    if n_pred == 0 or not math.isfinite(avg_loss):
        return float("inf")
    return math.exp(min(avg_loss, 100))  # Cap to prevent overflow


def compute_bpt(engine: AuraLiteEngine, text: str, batch_size: int = 32) -> float:
    """Compute bits-per-token."""
    avg_loss, n_pred = _compute_avg_token_loss(engine, text, batch_size)
    if n_pred == 0 or not math.isfinite(avg_loss):
        return float("inf")
    return avg_loss / math.log(2)


def compute_bpc(engine: AuraLiteEngine, text: str, batch_size: int = 32) -> float:
    """Compute bits-per-character.

    For char tokenizers this is essentially the same as bits-per-token.
    For BPE tokenizers we convert token NLL into bits per original character,
    so the label stays accurate instead of silently reporting bits-per-token.
    """
    avg_loss, n_pred = _compute_avg_token_loss(engine, text, batch_size)
    if n_pred == 0 or not math.isfinite(avg_loss) or len(text) <= 1:
        return float("inf")

    total_nll_bits = (avg_loss / math.log(2)) * n_pred
    return total_nll_bits / max(1, len(text) - 1)


def _bits_per_byte(engine: AuraLiteEngine, text: str, avg_loss: float, n_pred: int) -> float:
    if n_pred == 0 or not math.isfinite(avg_loss):
        return float("inf")
    ids = engine.encode(text)
    # every token after the first is predicted exactly once
    n_bytes = len(engine.decode(ids[1:1 + n_pred]).encode("utf-8"))
    if n_bytes == 0:
        return float("inf")
    return (avg_loss * n_pred) / math.log(2) / n_bytes


def compute_bpb(engine: AuraLiteEngine, text: str, batch_size: int = 32) -> float:
    """Compute bits per UTF-8 byte of the predicted text (2.11).

    The one metric that compares models with different tokenizers: nats per
    token reward a tokenizer whose tokens cover more text, while bits per byte
    measure the information the model needs for the text itself. The training
    summary reports the same number for the validation split.
    """
    avg_loss, n_pred = _compute_avg_token_loss(engine, text, batch_size)
    return _bits_per_byte(engine, text, avg_loss, n_pred)


def generate_sample(engine: AuraLiteEngine, seed: str, length: int = 200) -> str:
    """Generate a sample text for visual inspection."""
    return engine.generate(seed, length=length, temperature=0.8, top_k=50, top_p=0.9)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auralite.eval",
                                     description="Evaluate AuraLite AI model")
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--test", required=True, help="Path to test .txt file")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                        help="Device (default: auto-detect)")
    parser.add_argument("--seed", default="The ", help="Seed phrase for generation")
    parser.add_argument("--gen-length", type=int, default=200,
                        help="Length of generated sample")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    args = parser.parse_args(argv)

    # Load model
    print(f"Loading model from {args.model}...")
    engine = AuraLiteEngine()
    if args.device:
        engine.device = torch.device(args.device)
    engine.load_model(args.model)
    print(f"  Device: {engine.device}")
    print(f"  Parameters: {engine.model.count_parameters():,}")
    print(f"  Vocab size: {engine.vocab_size}")
    print(f"  Tokenizer: {engine.tokenizer.kind}")

    # Load test data
    print(f"\nLoading test data from {args.test}...")
    with open(args.test, "r", encoding="utf-8") as f:
        test_text = f.read()
    print(f"  Test size: {len(test_text):,} characters")

    # Evaluate
    print("\nComputing metrics...")
    start = time.time()

    # One pass over the text; 2.10 ran the model three times (one per metric).
    avg_loss, n_pred = _compute_avg_token_loss(engine, test_text, args.batch_size)
    finite = n_pred > 0 and math.isfinite(avg_loss)
    perplexity = math.exp(min(avg_loss, 100)) if finite else float("inf")
    bpt = avg_loss / math.log(2) if finite else float("inf")
    bpc = ((avg_loss / math.log(2)) * n_pred / max(1, len(test_text) - 1)
           if finite and len(test_text) > 1 else float("inf"))
    bpb = _bits_per_byte(engine, test_text, avg_loss, n_pred)

    elapsed = time.time() - start
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Bits-per-token: {bpt:.4f}")
    print(f"  Bits-per-character: {bpc:.4f}")
    print(f"  Bits-per-byte: {bpb:.4f}  (compares models across tokenizers)")
    if engine.tokenizer.kind != "char":
        print("  Note: BPC is normalized by original character count (not mislabeled BPT).")
    print(f"  Evaluation time: {elapsed:.2f}s")

    # Generate sample
    print(f"\nGenerating sample (seed: '{args.seed}')...")
    start = time.time()
    sample = generate_sample(engine, args.seed, args.gen_length)
    elapsed = time.time() - start
    print(f"  Generation time: {elapsed:.2f}s")
    print(f"  Generated text:\n{'='*50}")
    print(sample)
    print(f"{'='*50}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
