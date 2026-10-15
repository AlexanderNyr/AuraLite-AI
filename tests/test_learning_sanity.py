"""Does the model actually learn? And do old checkpoints still load? (v2.6.6)

Every other training test asserts `loss > 0` or `loss is not None`. A broken
causal mask, a RoPE sign error or an optimizer that never steps would pass all
of them. These tests pin down *behaviour* instead:

* ~380 optimizer steps (~5 s on a CPU) on a small deterministic corpus must
  take the validation loss from ~ln(vocab) ≈ 3.4 nats to far below 0.5,
  predict the next character correctly almost everywhere, and continue a
  memorised sentence greedily. Since 2.9 an epoch is one pass over the corpus
  (6 batches here), so the budget is 64 epochs — the token count of one 2.8
  epoch, which visited every window;
* a checkpoint written by AuraLite 2.6.4 (before the `weights_only` loading,
  the decomposed engine and the 2.6.5 refactors) must load with identical
  tokenisation and logits. `tests/data/golden_v2_6_4.*` was produced by the
  2.6.4 code itself (commit 26baeb8); see the header of `_forward_logits`.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from auralite.engine import AuraLiteEngine  # noqa: E402
from tests._integration import require  # noqa: E402

DATA = Path(__file__).parent / "data"
CORPUS = ("the quick brown fox jumps over the lazy dog. "
          "a journey of a thousand miles begins with a single step. "
          "to be or not to be, that is the question. ") * 40
# A fixed-length run: early stopping (on by default since 2.12) is switched off.
PARAMS = {"tokenizer": "char", "d_model": 64, "d_ff": 128, "n_heads": 4, "n_layers": 2,
          "n_kv_heads": 2, "seq_length": 64, "batch_size": 16, "lr": 3e-3, "epochs": 64,
          "val_split": 0.1, "seed": 0, "optimizer": "adamw", "early_stopping_patience": 0}


def _forward_logits(engine, ids: list[int], model=None) -> torch.Tensor:
    """Logits `[T, vocab]` of one forward pass over `ids` (no cache, eval mode)."""
    model = (model if model is not None else engine.model).eval()
    with torch.no_grad():
        out = model(torch.tensor([ids]))
    out = out[0] if isinstance(out, (tuple, list)) else out
    return out[0] if out.dim() == 3 else out


@pytest.fixture(scope="module")
def learned():
    engine = AuraLiteEngine()
    engine.train(CORPUS, dict(PARAMS))
    return engine


def test_short_training_learns_the_corpus(learned):
    uniform = math.log(learned.vocab_size)
    assert uniform > 3.0
    assert learned.last_val_loss is not None
    assert learned.last_val_loss < 0.5, (
        f"val loss {learned.last_val_loss:.3f} after training (uniform guess: {uniform:.2f})")
    summary = learned.last_train_summary
    assert summary["epochs_completed"] == 64 and summary["stop_reason"] is None
    assert 300 <= summary["optimizer_steps"] <= 500


def test_next_character_is_predicted_almost_everywhere(learned):
    ids = learned.encode(CORPUS[:300])
    hits = total = 0
    for start in range(0, len(ids) - 1, 63):
        window = ids[start:start + 64]
        if len(window) < 2:
            continue
        predicted = _forward_logits(learned, window[:-1]).argmax(-1).tolist()
        hits += sum(p == t for p, t in zip(predicted, window[1:], strict=True))
        total += len(window) - 1
    assert hits / total >= 0.9, f"teacher-forced accuracy {hits / total:.2%}"


def test_greedy_generation_continues_a_memorised_sentence(learned):
    out = learned.generate("the quick brown fox jumps over the", length=12,
                           temperature=0.0, top_k=1)
    assert "lazy dog" in out, out


def test_the_bar_is_not_reachable_without_training(learned):
    """Negative control: re-initialised weights fail the same measurements."""
    fresh = copy.deepcopy(learned.model)
    torch.manual_seed(1)
    for module in fresh.modules():
        if module is not fresh and hasattr(module, "reset_parameters"):
            module.reset_parameters()
    ids = learned.encode(CORPUS[:300])
    logits = _forward_logits(learned, ids[:-1], model=fresh)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor(ids[1:])).item()
    accuracy = (logits.argmax(-1) == torch.tensor(ids[1:])).float().mean().item()
    assert loss > 2.0 and accuracy < 0.5, (loss, accuracy)


# ----------------------------------------------------------------------
#  Golden checkpoints written by older releases
# ----------------------------------------------------------------------
GOLDEN = sorted(path.stem for path in DATA.glob("golden_v*.json"))


@pytest.mark.parametrize("stem", GOLDEN)
def test_golden_checkpoint_loads_bit_for_bit(stem):
    reference = json.loads((DATA / f"{stem}.json").read_text(encoding="utf-8"))
    checkpoint = next(DATA / f"{stem}{ext}" for ext in (".pt", ".safetensors")
                      if (DATA / f"{stem}{ext}").exists())
    if checkpoint.suffix == ".safetensors":
        require("safetensors")

    engine = AuraLiteEngine()
    engine.load_model(str(checkpoint))

    assert engine.encode(reference["prompt"]) == reference["token_ids"]
    assert sum(p.numel() for p in engine.model.parameters()) == reference["n_params"]
    logits = _forward_logits(engine, reference["token_ids"])
    expected = torch.tensor(reference["logits"])
    assert logits.shape == expected.shape
    assert torch.allclose(logits, expected, atol=1e-4, rtol=1e-4), (
        f"{checkpoint.name}: max |Δ| = {(logits - expected).abs().max().item():.2e}")


def test_the_2_6_4_golden_is_really_from_2_6_4():
    reference = json.loads((DATA / "golden_v2_6_4.json").read_text(encoding="utf-8"))
    assert reference["auralite_version"] == "2.6.4"
    assert "golden_v2_6_4" in GOLDEN
