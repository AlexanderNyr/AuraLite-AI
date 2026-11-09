"""What the Quantize tab offers, without a toolkit (2.25).

The classic interface's Quant tab, as data: the methods and the bit widths each one
takes, their settings (:class:`QuantSettings`, turned into the engine's
:class:`~auralite.quantization.QuantConfig`), and calibration text read from training
data - a text file as it is, dialogs and preference pairs as the words of their turns.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

__all__ = ["BITS", "CALIBRATED", "CALIBRATION_CHARS", "FP8_NOTE", "METHODS", "QuantSettings",
           "bits_for", "bits_note", "calibration_text", "describe_method", "runnable",
           "read_calibration_file", "uses"]

#: method -> its label; the order of the menu
METHODS = (("dynamic", "Dynamic INT8 - no calibration text needed"),
           ("static", "Static INT8 - weights and activations"),
           ("qat", "QAT - trained a little to get used to INT8"),
           ("gptq", "GPTQ - weights in 2 to 8 bits"),
           ("awq", "AWQ - 4 or 8 bits, the important weights protected"),
           ("hqq", "HQQ - 2 to 8 bits, no calibration text needed"),
           ("half", "Half precision - FP16, BF16 or FP8"))

#: bit width -> its label
BITS = {"int2": "2-bit integers", "int3": "3-bit integers", "int4": "4-bit integers",
        "int8": "8-bit integers", "fp16": "FP16", "bf16": "BF16", "fp8_e4m3": "FP8 (E4M3)",
        "fp8_e5m2": "FP8 (E5M2)"}

#: the methods that read calibration text (QAT trains on it)
CALIBRATED = frozenset({"static", "qat", "gptq", "awq"})

#: calibration text is cut here: the methods read a few hundred windows of it
CALIBRATION_CHARS = 1_000_000

#: setting -> the methods it changes (the tab enables a setting for those only)
_USES = {"group_size": {"gptq", "awq", "hqq"}, "percdamp": {"gptq"},
         "calibration_samples": {"static", "qat", "gptq", "awq"},
         "calibration_seq": {"static", "qat", "gptq", "awq"}, "awq_alpha": {"awq"},
         "qat_epochs": {"qat"}, "qat_lr": {"qat"}, "symmetric": {"qat", "gptq", "awq"},
         "per_channel": {"qat"}}


def uses(setting: str, method: str) -> bool:
    """True when `setting` changes what `method` does."""
    return method in _USES.get(setting, ())


def bits_for(method: str) -> list[str]:
    """The bit widths `method` takes, smallest first (the engine's own list)."""
    from auralite.quantization import METHOD_SUPPORTED_BITS, QuantMethod
    return [b.value for b in METHOD_SUPPORTED_BITS.get(QuantMethod(method), [])]


def describe_method(method: str) -> str:
    """The engine's description of `method`."""
    from auralite.quantization import METHOD_DESCRIPTIONS, QuantMethod
    return str(METHOD_DESCRIPTIONS.get(QuantMethod(method), ""))


#: PyTorch stores float8 but a processor cannot compute in it (no matrix products, not even
#: a multiplication): a model in FP8 is a file to keep, not one to run
FP8_NOTE = ("FP8 is for storage only: a processor cannot compute in it, so the model cannot "
            "chat or be measured in FP8 - save it, then go back to the original.")


def runnable(bits: str) -> bool:
    """Whether a model quantized to `bits` still runs here (FP8 does not)."""
    return not str(bits).startswith("fp8")


def bits_note(bits: str) -> str:
    """What to know about `bits` before quantizing ("" for most)."""
    return "" if runnable(bits) else FP8_NOTE


@dataclass
class QuantSettings:
    """The Quantize tab's form (the classic Quant tab's fields)."""

    method: str = "dynamic"
    bits: str = "int8"
    group_size: int = 128
    percdamp: float = 0.01
    calibration_samples: int = 128
    calibration_seq: int = 64
    awq_alpha: float = 0.5
    qat_epochs: int = 5
    qat_lr: float = 1e-5
    symmetric: bool = True
    per_channel: bool = True

    def config(self, calibration: str = "") -> Any:
        """The engine's ``QuantConfig`` (ValueError for a method or width it lacks)."""
        from auralite.quantization import BitWidth, QuantConfig, QuantMethod
        samples = max(1, int(self.calibration_samples))
        return QuantConfig(
            method=QuantMethod(self.method), bits=BitWidth(self.bits),
            calibration_text=calibration, calibration_samples=samples,
            calibration_seq_length=max(8, int(self.calibration_seq)),
            gptq_group_size=max(1, int(self.group_size)),
            gptq_percdamp=float(self.percdamp), awq_alpha=float(self.awq_alpha),
            awq_n_calibration=samples, qat_epochs=max(1, int(self.qat_epochs)),
            qat_lr=float(self.qat_lr), symmetric=bool(self.symmetric),
            per_channel=bool(self.per_channel))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuantSettings:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(data or {}).items() if k in known})


def read_calibration_file(path: str, limit: int = CALIBRATION_CHARS) -> str:
    """The text of a file (bytes that are not UTF-8 skipped), cut at `limit` characters."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read(limit)


def calibration_text(paths: list[str], limit: int = CALIBRATION_CHARS) -> str:
    """Text to calibrate on from training data (a Train or Fine-tune tab's): plain text as
    it is, dialogs and preference pairs as the words of their turns. ValueError for data
    without text (a prepared dataset holds token ids)."""
    from auralite.engine.corpus import load_training_data
    from auralite.engine.dialogs import DialogSet
    from auralite.engine.preference import PreferenceSet
    data = load_training_data(list(paths), stream=False)
    if isinstance(data, PreferenceSet):
        data = data.as_dialogs()
    if isinstance(data, DialogSet):
        parts: list[str] = []
        size = 0
        for dialog in data.dialogs:
            part = "\n".join(str(m.get("content", "")) for m in dialog if m.get("content"))
            parts.append(part)
            size += len(part) + 2
            if size >= limit:
                break
        return "\n\n".join(parts)[:limit]
    text = getattr(data, "text", None)
    if isinstance(text, str):
        return text[:limit]
    raise ValueError("this data holds no text to calibrate on (a prepared dataset is token "
                     "ids): choose a text file")
