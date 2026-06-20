"""Pytest benchmark helpers; run manually on target hardware.

    python -m pytest tests/bench_generation.py --benchmark-only
"""
import time
import pytest

torch = pytest.importorskip("torch")
from model_engine import ModernTransformer


def tokens_per_sec_decode(device="cuda" if torch.cuda.is_available() else "cpu"):
    model = ModernTransformer(512, 128, 4, 2, 256, max_seq_len=256).to(device).eval()
    prompt = torch.randint(0, 512, (1, 32), device=device)
    with torch.no_grad():
        model.reset_cache(); logits = model(prompt, use_cache=True); nxt = int(logits[0, -1].argmax())
        if device.startswith("cuda"): torch.cuda.synchronize()
        t0 = time.perf_counter(); n = 64
        for i in range(n):
            logits = model(torch.tensor([[nxt]], device=device), start_pos=32+i, use_cache=True)
            nxt = int(logits[0, -1].argmax())
        if device.startswith("cuda"): torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def test_decode_tokens_per_sec_smoke():
    assert tokens_per_sec_decode("cpu") > 0
