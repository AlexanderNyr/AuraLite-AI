"""What the evaluation shows, without a toolkit (2.25: the Studio's Evaluate tab and the
classic Eval tab share it).

* :func:`chat_eval_rows` - a chat model's replies to held-out dialogs, measured
  (:func:`~auralite.engine.chat_eval.evaluate_chat`), as rows of a table;
* :func:`lm_eval_lines` - the scores of lm-evaluation-harness's tasks as text;
* :func:`parse_tasks` - the tasks of a comma-separated list.
"""
from __future__ import annotations

from typing import Any

from .i18n import tr

__all__ = ["COMMON_TASKS", "chat_eval_rows", "lm_eval_lines", "parse_tasks"]

#: the classic Eval tab's quick picks
COMMON_TASKS = ("arc_easy", "arc_challenge", "hellaswag", "winogrande", "gsm8k", "mmlu")


def _fmt(value: Any, pattern: str) -> str:
    return tr("n/a") if value is None else pattern.format(value)


def chat_eval_rows(result: dict[str, Any]) -> list[tuple[str, str]]:
    """``(metric, value)`` rows of a chat evaluation, then a few replies as samples
    (marked ✓ exact, · stopped by itself, … cut at the length)."""
    rows = [
        (tr("Stopped by itself"), _fmt(result.get("stop_rate"), "{:.1%}")),
        (tr("Exact replies"), _fmt(result.get("exact"), "{:.1%}")),
        (tr("No stray template markers"), _fmt(result.get("clean_rate"), "{:.1%}")),
        (tr("Reply loss (nats per token)"), _fmt(result.get("reply_loss"), "{:.4f}")),
        (tr("Reply bits per byte"), _fmt(result.get("reply_bpb"), "{:.4f}")),
        (tr("New tokens per reply"), _fmt(result.get("mean_new_tokens"), "{:.1f}")),
    ]
    for sample in result.get("samples", []) or []:
        mark = "✓" if sample.get("exact") else ("·" if sample.get("stopped") else "…")
        rows.append((f"{mark} {str(sample.get('user', ''))[:40]}",
                     str(sample.get("reply", ""))[:120]))
    return rows


def lm_eval_lines(results: Any) -> list[str]:
    """The scores of an lm-evaluation-harness run: ``=== task ===`` and its metrics."""
    if not isinstance(results, dict) or "results" not in results:
        return [str(results)]
    lines: list[str] = []
    for task, metrics in dict(results["results"]).items():
        lines.append(f"=== {task} ===")
        for key, value in dict(metrics).items():
            is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
            lines.append(f"  {key}: {value:.4f}" if is_number else f"  {key}: {value}")
        lines.append("")
    return lines


def parse_tasks(text: str) -> list[str]:
    """``"arc_easy, hellaswag"`` -> ``["arc_easy", "hellaswag"]`` (each once, in order)."""
    return list(dict.fromkeys(t.strip() for t in str(text or "").split(",") if t.strip()))
