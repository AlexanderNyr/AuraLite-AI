"""Unit tests for optional lm-evaluation-harness integration."""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluation
from evaluation import (
    AuraLiteLM,
    EvaluationEngine,
    LMEvalNotAvailableError,
    _check_lm_eval,
    create_evaluator,
)


class FakeNativeEngine:
    def __init__(self):
        self.model = object()
        self.tokenizer = object()
        self.device = "cpu"

    def is_hf_model(self):
        return False

    def is_gguf_model(self):
        return False


class FakeGGUFEngine(FakeNativeEngine):
    def is_gguf_model(self):
        return True


class FakeHFProxy:
    def __init__(self):
        self.model = object()
        self.tokenizer = object()


class FakeHFEngine(FakeNativeEngine):
    def __init__(self):
        super().__init__()
        self.hf_proxy = FakeHFProxy()

    def is_hf_model(self):
        return True


class TestAvailability:
    def test_check_lm_eval_raises_clear_error_when_missing(self, monkeypatch):
        monkeypatch.setattr(evaluation, "HAS_LM_EVAL", False)
        with pytest.raises(LMEvalNotAvailableError) as exc:
            _check_lm_eval()
        assert "lm-evaluation-harness" in str(exc.value)
        assert "pip install" in str(exc.value)

    def test_check_lm_eval_noops_when_available(self, monkeypatch):
        monkeypatch.setattr(evaluation, "HAS_LM_EVAL", True)
        _check_lm_eval()


class TestEvaluationEngine:
    def test_create_evaluator_factory(self):
        engine = FakeNativeEngine()
        evaluator = create_evaluator(engine)
        assert isinstance(evaluator, EvaluationEngine)
        assert evaluator.engine is engine

    def test_native_evaluate_wraps_string_task_and_forwards_options(self, monkeypatch):
        calls = {}

        def fake_simple_evaluate(**kwargs):
            calls.update(kwargs)
            return {"results": {"arc_easy": {"acc": 0.25}}}

        monkeypatch.setattr(evaluation, "HAS_LM_EVAL", True)
        monkeypatch.setattr(evaluation, "evaluator", types.SimpleNamespace(simple_evaluate=fake_simple_evaluate), raising=False)

        engine = FakeNativeEngine()
        results = EvaluationEngine(engine).evaluate(
            tasks="arc_easy",
            num_fewshot=5,
            batch_size=3,
            limit=7,
        )

        assert results["results"]["arc_easy"]["acc"] == 0.25
        assert isinstance(calls["model"], AuraLiteLM)
        assert calls["tasks"] == ["arc_easy"]
        assert calls["num_fewshot"] == 5
        assert calls["batch_size"] == 3
        assert calls["device"] == "cpu"
        assert calls["limit"] == 7

    def test_native_evaluate_keeps_task_list(self, monkeypatch):
        monkeypatch.setattr(evaluation, "HAS_LM_EVAL", True)
        monkeypatch.setattr(
            evaluation,
            "evaluator",
            types.SimpleNamespace(simple_evaluate=lambda **kwargs: kwargs),
            raising=False,
        )
        result = EvaluationEngine(FakeNativeEngine()).evaluate(tasks=["mmlu", "gsm8k"])
        assert result["tasks"] == ["mmlu", "gsm8k"]

    def test_gguf_evaluation_is_explicitly_not_implemented(self, monkeypatch):
        monkeypatch.setattr(evaluation, "HAS_LM_EVAL", True)
        with pytest.raises(NotImplementedError, match="GGUF evaluation"):
            EvaluationEngine(FakeGGUFEngine()).evaluate("arc_easy")

    def test_hf_evaluate_uses_hflm_wrapper(self, monkeypatch):
        created = {}

        class FakeHFLM:
            def __init__(self, pretrained, tokenizer, batch_size):
                created["pretrained"] = pretrained
                created["tokenizer"] = tokenizer
                created["batch_size"] = batch_size

        hf_module = types.ModuleType("lm_eval.models.huggingface")
        hf_module.HFLM = FakeHFLM
        lm_eval_module = types.ModuleType("lm_eval")
        lm_eval_module.__path__ = []
        models_module = types.ModuleType("lm_eval.models")
        models_module.__path__ = []

        monkeypatch.setitem(sys.modules, "lm_eval", lm_eval_module)
        monkeypatch.setitem(sys.modules, "lm_eval.models", models_module)
        monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", hf_module)
        monkeypatch.setattr(evaluation, "HAS_LM_EVAL", True)
        monkeypatch.setattr(
            evaluation,
            "evaluator",
            types.SimpleNamespace(simple_evaluate=lambda **kwargs: {"model": kwargs["model"]}),
            raising=False,
        )

        engine = FakeHFEngine()
        result = EvaluationEngine(engine).evaluate(tasks="arc_easy", batch_size=4)

        assert isinstance(result["model"], FakeHFLM)
        assert created["pretrained"] is engine.hf_proxy.model
        assert created["tokenizer"] is engine.hf_proxy.tokenizer
        assert created["batch_size"] == 4

    def test_print_results_handles_empty_results(self, capsys):
        EvaluationEngine(FakeNativeEngine()).print_results({})
        assert "No results" in capsys.readouterr().out

    def test_print_results_formats_numbers_and_strings(self, capsys):
        results = {"results": {"task": {"acc": 0.123456, "alias": "demo"}}}
        EvaluationEngine(FakeNativeEngine()).print_results(results)
        out = capsys.readouterr().out
        assert "EVALUATION RESULTS" in out
        assert "acc" in out and "0.1235" in out
        assert "alias" in out and "demo" in out

    def test_save_results_writes_json(self, tmp_path, capsys):
        path = tmp_path / "results.json"
        results = {"results": {"task": {"acc": 1.0}}}
        EvaluationEngine(FakeNativeEngine()).save_results(str(path), results)
        assert '"acc": 1.0' in path.read_text(encoding="utf-8")
        assert str(path) in capsys.readouterr().out
