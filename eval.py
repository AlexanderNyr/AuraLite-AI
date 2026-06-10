#!/usr/bin/env python3
"""Evaluate an AuraLite AI model checkpoint on a text file.

Usage:
    python eval.py --model checkpoint.pt --test test.txt --device cuda
"""

import argparse
import math
import time
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_engine import AuraLiteEngine


def compute_perplexity(engine: AuraLiteEngine, text: str, batch_size: int = 32) -> float:
    """Compute perplexity on text (lower is better)."""
    ids = engine.encode(text)
    if not ids:
        return float("inf")

    seq_length = engine.model.max_seq_len if hasattr(engine.model, 'max_seq_len') else 64
    encoded = torch.tensor(ids, dtype=torch.long).to(engine.device)

    n_samples = max(0, len(encoded) - seq_length)
    if n_samples == 0:
        return float("inf")

    total_loss = 0.0
    total_tokens = 0
    engine.model.eval()

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            x = encoded[i:batch_end + seq_length].unsqueeze(0)  # (1, seq_len + n)
            batch_x = []
            batch_y = []
            for j in range(batch_end - i):
                batch_x.append(x[0, j:j + seq_length])
                batch_y.append(x[0, j + 1:j + seq_length + 1])

            xb = torch.stack(batch_x).to(engine.device)
            yb = torch.stack(batch_y).to(engine.device)

            logits = engine.model(xb)  # (B, T, vocab)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                yb.reshape(-1),
            )
            total_loss += loss.item() * len(batch_x)
            total_tokens += len(batch_x)

    avg_loss = total_loss / max(1, total_tokens)
    return math.exp(min(avg_loss, 100))  # Cap to prevent overflow


def compute_bpc(engine: AuraLiteEngine, text: str, batch_size: int = 32) -> float:
    """Compute bits-per-character."""
    ids = engine.encode(text)
    if not ids:
        return float("inf")

    seq_length = engine.model.max_seq_len if hasattr(engine.model, 'max_seq_len') else 64
    encoded = torch.tensor(ids, dtype=torch.long).to(engine.device)

    n_samples = max(0, len(encoded) - seq_length)
    if n_samples == 0:
        return float("inf")

    total_loss = 0.0
    total_tokens = 0
    engine.model.eval()

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            x = encoded[i:batch_end + seq_length].unsqueeze(0)
            batch_x = []
            batch_y = []
            for j in range(batch_end - i):
                batch_x.append(x[0, j:j + seq_length])
                batch_y.append(x[0, j + 1:j + seq_length + 1])

            xb = torch.stack(batch_x).to(engine.device)
            yb = torch.stack(batch_y).to(engine.device)

            logits = engine.model(xb)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                yb.reshape(-1),
            )
            total_loss += loss.item() * len(batch_x)
            total_tokens += len(batch_x)

    avg_loss = total_loss / max(1, total_tokens)
    return avg_loss / math.log(2)  # Convert from nats to bits


def generate_sample(engine: AuraLiteEngine, seed: str, length: int = 200) -> str:
    """Generate a sample text for visual inspection."""
    return engine.generate(seed, length=length, temperature=0.8, top_k=50, top_p=0.9)


def main():
    parser = argparse.ArgumentParser(description="Evaluate AuraLite AI model")
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--test", required=True, help="Path to test .txt file")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                        help="Device (default: auto-detect)")
    parser.add_argument("--seed", default="The ", help="Seed phrase for generation")
    parser.add_argument("--gen-length", type=int, default=200,
                        help="Length of generated sample")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    args = parser.parse_args()

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

    perplexity = compute_perplexity(engine, test_text, args.batch_size)
    bpc = compute_bpc(engine, test_text, args.batch_size)

    elapsed = time.time() - start
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Bits-per-character: {bpc:.4f}")
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


if __name__ == "__main__":
    main()
