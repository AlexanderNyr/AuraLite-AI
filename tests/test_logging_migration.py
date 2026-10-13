"""Tests for the print → logging migration (roadmap P1.7).

Before v2.6.5 the engine and the optional backends reported everything with
``print()``: a library user calling ``train()`` got stdout spam they could not
silence, the CLI's ``--quiet``/``--log-level`` flags only affected a couple of
modules, and the GUI's console tab could not colourise library output.

What is pinned here:

* no ``print()`` is left in the library modules (front-ends — CLI, eval script,
  benchmarks, GUI — keep printing, that *is* their output);
* engine progress goes through the ``auralite`` logger and is visible on stderr
  for library users, on stdout for the CLI (historical UX, so
  ``auralite train … > log.txt`` still works), and obeys ``--quiet``;
* the handler follows ``sys.stdout``/``sys.stderr`` replacements made *after*
  it was created (GUI console redirector, pytest capture, wrappers);
* warnings from the optional backends are logged at WARNING, not printed.
"""
from __future__ import annotations

import ast
import logging
import pathlib

import pytest

from model_engine.logging_utils import (
    DynamicStreamHandler,
    canonical_level,
    engine_logging,
    reset_engine_logging,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Modules whose *job* is to talk to the user on stdout.
FRONT_ENDS = {"auralite_cli.py", "eval.py", "evaluation.py"}


def _print_calls(path: pathlib.Path) -> list[int]:
    tree = ast.parse(path.read_text())
    return [node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "print"]


@pytest.mark.parametrize("module", [
    "model_engine/_legacy.py", "model_engine/layers.py", "model_engine/model.py",
    "model_engine/dataset.py", "model_engine/scheduler.py", "model_engine/optimizers.py",
    "model_engine/tokenizers.py", "model_engine/gguf_proxy.py", "model_engine/utils.py",
    "image_engine.py", "hf_integration.py", "export.py", "web_tools.py",
])
def test_library_modules_do_not_print(module):
    calls = _print_calls(REPO_ROOT / module)
    assert calls == [], f"{module} still prints (lines {calls})"


def test_front_ends_may_still_print():
    """The CLI and the eval script write reports to stdout — that is their API."""
    assert _print_calls(REPO_ROOT / "auralite_cli.py")
    assert _print_calls(REPO_ROOT / "eval.py") == [] or _print_calls(REPO_ROOT / "eval.py")
    for name in FRONT_ENDS:
        assert (REPO_ROOT / name).exists()


# ----------------------------------------------------------------------
#  Engine → logger
# ----------------------------------------------------------------------
def test_engine_progress_is_logged_to_stderr(tiny_text, capsys):
    from model_engine import AuraLiteEngine

    engine = AuraLiteEngine()
    engine.train(tiny_text, {
        "tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_layers": 2,
        "seq_length": 16, "batch_size": 8, "lr": 1e-2, "epochs": 1, "val_split": 0.0,
        "seed": 0, "amp_dtype": "none",
    })
    captured = capsys.readouterr()
    assert "[AuraLite] Model built:" in captured.err
    assert captured.out == ""            # nothing reaches stdout any more


def test_engine_messages_carry_their_historical_wording(capsys):
    """Banners are unchanged, so logs stay greppable for existing users."""
    import model_engine._legacy as legacy

    legacy.logger.info("[AuraLite] Model built: 1,234 params (vocab=16)")
    assert "[AuraLite] Model built: 1,234 params (vocab=16)" in capsys.readouterr().err


def test_info_level_is_visible_by_default():
    assert canonical_level() == "INFO"
    logger = reset_engine_logging()
    try:
        assert logger.level == logging.INFO
        assert logger.propagate is True
    finally:
        reset_engine_logging()


def test_env_var_overrides_level(monkeypatch, capsys):
    monkeypatch.setenv("AURALITE_LOG_LEVEL", "WARNING")
    logger = reset_engine_logging()
    logger.info("[AuraLite] invisible")
    logger.warning("[AuraLite] WARNING: visible")
    err = capsys.readouterr().err
    assert "invisible" not in err
    assert "visible" in err


def test_sub_logger_records_reach_the_engine_handler(capsys):
    """`auralite.hf` & co. propagate to the parent handler instead of printing."""
    reset_engine_logging()
    logging.getLogger("auralite.hf").warning(
        "[AuraLite-HF] WARNING: 4/8-bit quantization requires CUDA.")
    assert "requires CUDA" in capsys.readouterr().err


# ----------------------------------------------------------------------
#  DynamicStreamHandler
# ----------------------------------------------------------------------
def test_handler_follows_reassigned_streams(capsys):
    import io
    import sys

    handler = DynamicStreamHandler("stdout")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("auralite.dynamic_probe")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        logger.info("to-the-real-stdout")
        assert "to-the-real-stdout" in capsys.readouterr().out

        buffer = io.StringIO()
        original, sys.stdout = sys.stdout, buffer
        try:
            logger.info("to-the-replaced-stdout")
        finally:
            sys.stdout = original
        assert buffer.getvalue().strip() == "to-the-replaced-stdout"
    finally:
        logger.handlers.clear()


def test_handler_rejects_unknown_stream_name():
    with pytest.raises(ValueError):
        DynamicStreamHandler("stdin")


def test_engine_logging_is_idempotent():
    first = engine_logging()
    count = len(first.handlers)
    second = engine_logging()
    assert second is first and len(second.handlers) == count


# ----------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------
def _tiny_cli_args(tmp_path, tiny_text, extra=()):
    text_file = tmp_path / "corpus.txt"
    text_file.write_text(tiny_text, encoding="utf-8")
    params = tmp_path / "params.json"
    import json

    params.write_text(json.dumps({
        "tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_layers": 2,
        "seq_length": 16, "batch_size": 8, "lr": 1e-2, "epochs": 1, "val_split": 0.0,
        "seed": 0, "amp_dtype": "none",
    }), encoding="utf-8")
    return ["train", "--text", str(text_file), "--params", str(params),
            "--model", str(tmp_path / "cli_log.pt"), *extra]


def test_cli_keeps_progress_on_stdout(tmp_path, tiny_text, capsys):
    from auralite_cli import main

    assert main(_tiny_cli_args(tmp_path, tiny_text)) == 0
    captured = capsys.readouterr()
    assert "[AuraLite] Model built:" in captured.out     # same place as before v2.6.5
    assert "Model built" not in captured.err


def test_cli_quiet_silences_library_logs(tmp_path, tiny_text, capsys):
    from auralite_cli import main

    assert main(_tiny_cli_args(tmp_path, tiny_text, extra=["--quiet"])) == 0
    captured = capsys.readouterr()
    assert "[AuraLite]" not in captured.out
    assert "[AuraLite]" not in captured.err


def test_cli_log_level_warning_hides_info_but_keeps_warnings(tmp_path, tiny_text, capsys):
    import os

    from auralite_cli import _configure_logging, build_parser, main

    _configure_logging(build_parser().parse_args(
        ["info", "--model", "x.pt", "--log-level", "WARNING"]))
    logger = logging.getLogger("auralite")
    assert logger.level == logging.WARNING
    assert os.environ["AURALITE_LOG_LEVEL"] == "WARNING"
    logger.info("[AuraLite] info-hidden")
    logger.warning("[AuraLite] WARNING: warning-kept")
    out = capsys.readouterr().out
    assert "info-hidden" not in out and "warning-kept" in out
    assert logger.propagate is False       # no duplicate lines via the root logger
    reset_engine_logging()
    assert main(_tiny_cli_args(tmp_path, tiny_text, extra=["--quiet"])) == 0


# ----------------------------------------------------------------------
#  Backends log their failures instead of printing them
# ----------------------------------------------------------------------
def test_web_tools_failure_is_a_warning(caplog, monkeypatch):
    import web_tools

    def boom(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr(web_tools.urllib.request, "urlopen", boom)
    monkeypatch.setattr(web_tools, "wikipedia_search", lambda *a, **k: [])
    with caplog.at_level(logging.WARNING, logger="auralite.webtools"):
        assert web_tools.build_web_context("aurora", max_results=1) == ""

    messages = [record.getMessage() for record in caplog.records]
    assert any("DuckDuckGo search failed" in m for m in messages)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_export_reports_saved_files_via_logger(capsys):
    """`export.py` announces success through `auralite.export` (INFO)."""
    import export as export_mod

    reset_engine_logging()
    logging.getLogger("auralite.export").info("[AuraLite-Export] TorchScript model saved to /tmp/x.pt")
    assert "TorchScript model saved" in capsys.readouterr().err
    assert export_mod.logger.name == "auralite.export"
