"""Regression tests for the v2.6.5 improvements (roadmap implementation).

Covered here:
  * safe checkpoint I/O (`model_engine.safe_io`, `weights_only` + scoped numpy
    allowlist, `.safetensors` checkpoints, RNG builtins) — roadmap P0;
  * AST-whitelisted `safe_calculate` and the hardened agent sandbox;
  * OpenAI-compatible server auth (multi-key hmac), lifespan and `/metrics`;
  * real FlexAttention and real speculative decoding (id-equivalence, budget,
    stats), the streaming `lstrip` fix and `generate_batch` deduplication;
  * GGUF perplexity math (fake llama object, no llama-cpp-python required);
  * agent env knobs, the "Model built" size report, the GUI split and
    packaging hygiene (extras, pytest config, PEP 561 marker).

Tests that need third-party packages use ``importorskip``; the ones that need
real weights use the shared fixtures from ``conftest.py``.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import types

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:  # tests are also run from arbitrary cwd
    sys.path.insert(0, REPO_ROOT)

from tests._integration import require  # noqa: E402


# ======================================================================
#  1. Safe checkpoint I/O
# ======================================================================
class _Hostile:
    """A pickle payload that tries to drop a file when unpickled."""

    def __init__(self, marker_path: str):
        self.marker_path = marker_path

    def __reduce__(self):
        return (open, (self.marker_path, "w"))


def test_safe_torch_load_roundtrip_of_plain_weights(tmp_path):
    from auralite.engine.safe_io import numpy_rng_state_to_builtins, safe_torch_load

    path = tmp_path / "plain.pt"
    torch.save({
        "model_state": {"w": torch.ones(3)},
        "rng_state": numpy_rng_state_to_builtins(np.random.get_state()),
    }, path)

    loaded = safe_torch_load(str(path))
    assert torch.equal(loaded["model_state"]["w"], torch.ones(3))
    assert loaded["rng_state"]["kind"] == "MT19937"
    assert isinstance(loaded["rng_state"]["keys"], list)
    # No allowlist needed for the new format: plain weights_only load works too
    assert torch.load(path, weights_only=True)["model_state"]["w"].shape == (3,)


def test_safe_torch_load_rejects_hostile_pickle(tmp_path, monkeypatch):
    from auralite.engine.safe_io import safe_torch_load

    marker = tmp_path / "pwned"
    path = tmp_path / "evil.pt"
    torch.save(_Hostile(str(marker)), path)     # a real .pt with a hostile payload

    with pytest.raises(ValueError, match="Refusing to load"):
        safe_torch_load(str(path))
    assert not marker.exists(), "arbitrary code executed during a safe load!"

    # Explicit opt-in is still possible for trusted files — and only there the
    # payload runs (that is exactly why the default refuses it).
    safe_torch_load(str(path), allow_unsafe=True)
    assert marker.exists()

    # Same via the environment variable, and a bare (non-torch) pickle is
    # refused as well instead of crashing with a confusing error.
    marker.unlink()
    monkeypatch.setenv("AURALITE_UNSAFE_LOAD", "1")
    safe_torch_load(str(path))
    assert marker.exists()
    monkeypatch.delenv("AURALITE_UNSAFE_LOAD")

    junk = tmp_path / "junk.pt"
    junk.write_bytes(pickle.dumps({"not": "a torch file"}))
    with pytest.raises(ValueError, match="Refusing to load"):
        safe_torch_load(str(junk))


def test_safe_torch_load_reads_legacy_numpy_rng_state(tmp_path):
    """Pre-2.6.5 checkpoints stored `np.random.get_state()` verbatim."""
    from auralite.engine.safe_io import (
        numpy_rng_state_from_builtins, safe_torch_load,
    )

    path = tmp_path / "legacy.pt"
    state = np.random.get_state()
    torch.save({"rng_state": state, "model_state": {"w": torch.zeros(2)}}, path)

    with pytest.raises(Exception):
        torch.load(path, weights_only=True)  # ndarray → rejected by default

    loaded = safe_torch_load(str(path))
    restored = numpy_rng_state_from_builtins(loaded["rng_state"])
    np.random.set_state(restored)  # must be a valid numpy state tuple
    assert loaded["model_state"]["w"].shape == (2,)


def test_numpy_rng_state_builtins_are_json_serializable():
    from auralite.engine.safe_io import (
        numpy_rng_state_from_builtins, numpy_rng_state_to_builtins,
    )

    payload = numpy_rng_state_to_builtins(np.random.get_state())
    json.dumps(payload)                                # no numpy objects inside
    restored = numpy_rng_state_from_builtins(payload)
    assert restored[1].dtype == np.uint32
    assert numpy_rng_state_to_builtins(None) is None
    assert numpy_rng_state_from_builtins(None) is None


def test_safetensors_checkpoint_roundtrip(tmp_path):
    safetensors = require("safetensors")
    from auralite.engine.safe_io import (
        load_safetensors_checkpoint, save_safetensors_checkpoint,
    )
    assert safetensors is not None

    path = tmp_path / "ckpt.safetensors"
    tensors = {"a": torch.randn(4), "b": torch.zeros(2, 2)}
    save_safetensors_checkpoint(str(path), tensors, {"version": 3, "tokenizer": "char"})

    loaded = load_safetensors_checkpoint(str(path))
    assert loaded["version"] == 3 and loaded["tokenizer"] == "char"
    for key, value in tensors.items():
        assert torch.equal(loaded["model_state"][key], value)

    with pytest.raises(ValueError, match="Not an AuraLite safetensors checkpoint"):
        load_safetensors_checkpoint(str(path), meta_key="auralite-image")


def test_safetensors_engine_roundtrip_preserves_generation(tmp_path, trained_tiny_engine):
    """A `.safetensors` save/load must reproduce identical generations."""
    require("safetensors")
    engine, _ = trained_tiny_engine
    path = tmp_path / "tiny.safetensors"
    engine.save_model(str(path))          # .safetensors → pickle-free container
    assert path.exists()

    from auralite.engine import AuraLiteEngine
    reloaded = AuraLiteEngine()
    reloaded.load_model(str(path))

    assert reloaded.model.count_parameters() == engine.model.count_parameters()
    ids_a = engine.generate("AuraLite", length=6, temperature=0.0, top_k=1)
    ids_b = reloaded.generate("AuraLite", length=6, temperature=0.0, top_k=1)
    assert ids_a == ids_b


# ======================================================================
#  2. Sandboxed calculator
# ======================================================================
@pytest.mark.parametrize("expr,expected", [
    ("2+2*2", 6),
    ("(1+2)**3", 27),
    ("max(3,7,5)", 7),
    ("factorial(6)", 720),
    ("abs(-4) + 1", 5),
])
def test_safe_calculate_valid(expr, expected):
    from auralite.agent.tools import safe_calculate
    assert safe_calculate(expr) == expected


def test_safe_calculate_math_functions_and_constants():
    import math
    from auralite.agent.tools import safe_calculate
    assert safe_calculate("sqrt(16) + pi") == pytest.approx(4 + math.pi)
    assert safe_calculate("sin(pi/2)") == pytest.approx(1.0)
    assert safe_calculate("log(e)") == pytest.approx(1.0)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('id')",
    "().__class__.__bases__[0].__subclasses__()",
    "open('/etc/passwd').read()",
    "lambda: 1",
    "[x for x in range(3)]",
    "2**2**100",
    "'a' * 10",
    "factorial(10**9)",
    "print(1)",
])
def test_safe_calculate_blocks_attacks(expr):
    from auralite.agent.tools import safe_calculate
    with pytest.raises((ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError)):
        safe_calculate(expr)


# ======================================================================
#  3. Hardened agent sandbox
# ======================================================================
def test_sandbox_strips_secret_env_and_redirects_home(tmp_path, monkeypatch):
    from auralite.agent.sandbox import Sandbox

    monkeypatch.setenv("MY_TEST_API_KEY", "super-secret")
    with Sandbox(workdir=str(tmp_path)) as box:
        result = box.run_python(
            "import os\n"
            "print('key:', os.environ.get('MY_TEST_API_KEY'))\n"
            "print('home_is_workdir:', os.path.samefile(os.environ['HOME'], '.'))\n"
        )
    assert result.returncode == 0, result.combined_output()
    assert "key: None" in result.stdout
    assert "home_is_workdir: True" in result.stdout


def test_sandbox_network_denied_drops_network_commands():
    from auralite.agent.sandbox import Sandbox
    box = Sandbox(allow_network=False)
    try:
        assert "curl" not in box.shell_whitelist
        assert "wget" not in box.shell_whitelist
    finally:
        box.cleanup()


# ======================================================================
#  4. Agent configuration via environment
# ======================================================================
def test_agent_env_knobs(monkeypatch):
    from auralite.agent.agent import AuraLiteAgent

    monkeypatch.setenv("AURALITE_AGENT_MAX_ITERATIONS", "12")
    monkeypatch.setenv("AURALITE_AGENT_TOOL_TIMEOUT", "30")
    agent = AuraLiteAgent(engine=object())
    try:
        assert agent.max_iterations == 12
        assert agent.timeout_per_tool == pytest.approx(30.0)
    finally:
        agent.cleanup()


def test_agent_env_knobs_fall_back_on_garbage(monkeypatch):
    from auralite.agent.agent import AuraLiteAgent

    monkeypatch.setenv("AURALITE_AGENT_MAX_ITERATIONS", "many")
    monkeypatch.setenv("AURALITE_AGENT_TOOL_TIMEOUT", "")
    agent = AuraLiteAgent(engine=object(), max_iterations=3, timeout_per_tool=1.5)
    try:
        assert agent.max_iterations == 3                 # explicit arg wins
        assert agent.timeout_per_tool == pytest.approx(1.5)
    finally:
        agent.cleanup()


# ======================================================================
#  5. OpenAI-compatible server
# ======================================================================
@pytest.fixture()
def server_app(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    monkeypatch.delenv("AURALITE_MODEL_PATH", raising=False)
    from auralite.server import openai_server
    return openai_server


def test_server_health_and_metrics(server_app):
    from fastapi.testclient import TestClient

    with TestClient(server_app.app) as client:  # runs lifespan
        health = client.get("/health")
        assert health.status_code == 200

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert metrics.headers["content-type"].startswith("text/plain")
        body = metrics.text
        for key in ("auralite_uptime_seconds", "auralite_requests_total",
                    "auralite_tokens_generated_total", "auralite_errors_total",
                    "auralite_generations_inflight", "auralite_model_loaded"):
            assert key in body


def test_server_multi_key_auth(server_app, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("AURALITE_API_KEY", "key-one,key-two")
    with TestClient(server_app.app) as client:
        assert client.get("/v1/models").status_code == 401
        assert client.get("/v1/models", headers={
            "Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/v1/models", headers={
            "Authorization": "Bearer key-one"}).status_code == 200
        assert client.get("/v1/models", headers={
            "Authorization": "Bearer key-two"}).status_code == 200
        # /health stays public even with keys configured
        assert client.get("/health").status_code == 200


def test_server_metrics_token_counter(server_app):
    """`_metric`/`render_metrics` expose Prometheus-style counters."""
    server_app._metric("requests_total", key="/v1/completions")
    server_app._metric("tokens_generated_total", 17)
    text = server_app.render_metrics()

    assert 'auralite_requests_total{path="/v1/completions"} ' in text
    line = next(l for l in text.splitlines() if l.startswith("auralite_tokens_generated_total "))
    assert int(line.split()[-1]) >= 17


# ======================================================================
#  6. Speculative decoding (real implementation)
# ======================================================================
def _ids(engine, text: str) -> list[int]:
    return [int(i) for i in engine.tokenizer.encode(text)]


def test_speculative_greedy_matches_plain_decoding(trained_engine):
    engine = trained_engine
    for prompt in ("AuraLite", "small"):
        for length in (8, 16):
            plain = engine.generate(prompt, length=length, temperature=0.0, top_k=1)
            spec = engine.generate_speculative(prompt, length=length, temperature=0.0,
                                               top_k=1, draft_tokens=4)
            assert _ids(engine, plain) == _ids(engine, spec), (prompt, length)


def test_speculative_respects_length_budget(trained_engine):
    engine = trained_engine
    stats: dict = {}
    text = engine.generate_speculative("AuraLite", length=10, temperature=0.0, top_k=1,
                                       stats=stats)
    prompt_ids = _ids(engine, "AuraLite")
    assert len(_ids(engine, text)) - len(prompt_ids) <= 10
    assert set(stats) >= {"drafted", "accepted", "acceptance_rate",
                          "cache_resets", "tokens"}
    assert stats["drafted"] >= stats["accepted"] >= 0


def test_speculative_sampling_smoke(trained_engine):
    engine = trained_engine
    torch.manual_seed(0)
    text = engine.generate_speculative("AuraLite", length=12, temperature=2.0,
                                       top_k=0, top_p=1.0, draft_tokens=3)
    assert len(_ids(engine, text)) > len(_ids(engine, "AuraLite"))


# ======================================================================
#  7. FlexAttention parity
# ======================================================================
def test_flex_attention_matches_sdpa():
    # The flex-attention helpers (and their caches) live in model_engine.layers
    # since the v2.6.5 decomposition; _legacy re-exports them.
    from auralite.engine import ModernTransformer
    from auralite.engine import layers as layers_mod

    if layers_mod._get_flex_attention() is False:
        pytest.skip("FlexAttention is not available in this runtime")

    kwargs = dict(vocab_size=48, d_model=32, n_heads=4, n_kv_heads=2,
                  n_layers=2, d_ff=64, max_seq_len=32)
    torch.manual_seed(0)
    base = ModernTransformer(**kwargs).eval()
    flex = ModernTransformer(**kwargs, use_flex_attention=True).eval()
    flex.load_state_dict(base.state_dict())

    ids = torch.randint(0, 48, (1, 16))
    with torch.inference_mode():
        out_base = base(ids)
        out_flex = flex(ids)
    if id(flex) in layers_mod._FLEX_FAILED:
        pytest.skip("FlexAttention raised at runtime; SDPA fallback was used")

    delta = (out_base.float() - out_flex.float()).abs().max().item()
    assert delta < 1e-4, f"FlexAttention vs SDPA mismatch: {delta}"


# ======================================================================
#  8. GGUF perplexity math
# ======================================================================
class _FakeLlama:
    """Deterministic stand-in for a llama-cpp-python model with `logits_all`."""

    def __init__(self, vocab: int = 8):
        self.vocab = vocab
        self.logits_all = True
        self.evals: list[list[int]] = []
        self.resets = 0
        self._scores = np.zeros((1, vocab), dtype=np.float32)

    def eval(self, tokens):
        self.evals.append(list(tokens))
        self._scores = np.zeros((len(tokens), self.vocab), dtype=np.float32)

    @property
    def scores(self):
        return self._scores

    def reset(self):
        self.resets += 1


class _FakeTokenizer:
    kind = "gguf"

    def encode(self, text):
        return [ord(c) % 8 for c in text]

    def decode(self, ids):
        return "".join(chr(int(i) % 8) for i in ids)


def test_gguf_rolling_loglikelihood_window_math():
    from auralite.engine._legacy import _gguf_rolling_loglikelihood

    llama, tokenizer = _FakeLlama(), _FakeTokenizer()
    total, scored = _gguf_rolling_loglikelihood(llama, "x" * 30, 8, tokenizer,
                                                return_count=True)
    assert scored == 25                       # 30 tokens, 5 windows of 7 → 25 scored
    assert total == pytest.approx(25 * (-np.log(8)), abs=1e-4)  # uniform logits
    assert len(llama.evals) == 5


def test_gguf_proxy_perplexity_and_counts():
    from auralite.engine._legacy import GGUFModelProxy

    proxy = object.__new__(GGUFModelProxy)
    proxy.llama = _FakeLlama()
    proxy.tokenizer = _FakeTokenizer()
    proxy.max_seq_len = 8
    proxy._load_kwargs = {}
    proxy.metadata = {}

    assert proxy.count_parameters() == 0
    assert proxy.perplexity("x" * 30) == pytest.approx(8.0, rel=1e-3)
    assert proxy.loglikelihood_rolling("x" * 30) == pytest.approx(25 * (-np.log(8)), abs=1e-4)
    # too short to score → nan, not a crash
    assert np.isnan(proxy.perplexity("x"))

    llama, owns = proxy._perplexity_llama()
    assert llama is proxy.llama and owns is False  # logits_all already set


def test_gguf_perplexity_requires_gguf_model(trained_engine):
    with pytest.raises(ValueError, match="requires a loaded .gguf model"):
        trained_engine.gguf_perplexity("text")
    with pytest.raises(ValueError, match="requires a loaded .gguf model"):
        trained_engine.gguf_loglikelihood_rolling("text")


# ======================================================================
#  9. Chat streaming parity + 10. batch dedup
# ======================================================================
def test_chat_streaming_matches_batch_output(trained_engine):
    engine = trained_engine
    messages = [{"role": "user", "content": "Hi"}]
    batch = engine.generate_chat(messages, max_new_tokens=12, temperature=0.0,
                                 top_k=1, chat_template="chatml")
    streamed = "".join(engine.generate_chat_streaming(
        messages, max_new_tokens=12, temperature=0.0, top_k=1, chat_template="chatml"))
    assert streamed == batch
    assert streamed and not streamed[0].isspace()   # v2.6.5 lstrip fix


def test_generate_batch_deduplicates_identical_prompts(trained_engine, monkeypatch):
    engine = trained_engine
    calls: list[int] = []
    original = engine._generate_batch_group

    def spy(batch_ids, *args, **kwargs):
        calls.append(len(batch_ids))
        return original(batch_ids, *args, **kwargs)

    monkeypatch.setattr(engine, "_generate_batch_group", spy)
    prompts = ["AuraLite", "AuraLite", "AuraLite"]
    out = engine.generate_batch(prompts, length=6, temperature=0.0, top_k=1)
    assert len(out) == 3
    assert out[0] == out[1] == out[2]
    assert calls == [1], "duplicate prompts were not deduplicated"


# ======================================================================
#  11. "Model built" size report
# ======================================================================
def test_train_reports_actual_model_size(capsys, tiny_text):
    from auralite.engine import AuraLiteEngine

    engine = AuraLiteEngine()
    engine.train(tiny_text, {
        "tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4,
        "n_layers": 2, "n_kv_heads": 2, "seq_length": 16, "batch_size": 8,
        "lr": 1e-2, "epochs": 1, "val_split": 0.0, "seed": 0,
        "optimizer": "adamw", "lr_schedule": "cosine", "amp_dtype": "none",
        "weight_decay": 0.0,
    })
    # v2.6.5: the engine reports through the `auralite` logger (stderr by
    # default), not with print() — see tests/test_logging_migration.py.
    out = capsys.readouterr().err
    assert "[AuraLite] Model built:" in out
    assert "params" in out
    actual = engine.model.count_parameters()
    assert f"{actual:,} params" in out


# ======================================================================
#  12. GUI split (gui_app.py → gui/ package)
# ======================================================================
def test_gui_entrypoint_is_a_shim_over_the_package():
    with pytest.warns(DeprecationWarning, match="auralite.gui.app"):
        sys.modules.pop("gui_app", None)
        import gui_app
    from auralite.gui.app import AIApp, main

    assert gui_app.AIApp is AIApp           # old import path keeps working (2.7.x)
    assert gui_app.main is main
    assert callable(main)


def test_gui_tabs_are_single_source_mixins():
    import ast
    import pathlib

    from auralite.gui.app import AIApp

    core = pathlib.Path(REPO_ROOT) / "auralite" / "gui"
    mixin_files = sorted((core / "tabs").glob("*.py"))
    assert len(mixin_files) >= 10, "expected one module per tab"

    owners: dict[str, list[str]] = {}
    for path in mixin_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for func in node.body:
                    if isinstance(func, ast.FunctionDef):
                        owners.setdefault(func.name, []).append(path.name)

    duplicates = {name: files for name, files in owners.items() if len(files) > 1}
    assert not duplicates, f"methods defined twice across mixins: {duplicates}"

    missing = [name for name in owners if not hasattr(AIApp, name)]
    assert not missing, f"mixin methods not reachable from AIApp: {missing}"

    for tab in ("training", "image", "generation", "model", "quantization",
                "evaluation", "export", "console", "chat"):
        assert hasattr(AIApp, f"_build_{tab}_tab"), f"{tab} tab builder missing"
    assert len([m for m in dir(AIApp) if not m.startswith("__")]) >= 100


def test_gui_main_guards_multiprocessing():
    gui_main = open(os.path.join(REPO_ROOT, "auralite", "gui", "app.py"),
                    encoding="utf-8").read()
    assert "freeze_support()" in gui_main
    assert "__main__" in gui_main
    assert "class AIApp" in gui_main
    assert "class AIApp" not in open(os.path.join(REPO_ROOT, "gui_app.py"),
                                     encoding="utf-8").read()


# ======================================================================
#  13. Packaging hygiene
# ======================================================================
def _pyproject() -> dict:
    import tomllib
    with open(os.path.join(REPO_ROOT, "pyproject.toml"), "rb") as fh:
        return tomllib.load(fh)


def test_extras_are_real_and_lean():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert {"test", "lint", "full", "serve", "dev"} <= set(extras)
    for dead in ("auto-gptq", "auto_gptq", "optimum", "chromadb", "vllm-server"):
        payload = json.dumps(extras)
        assert dead not in payload, f"dead extra {dead!r} is still declared"
    assert any("safetensors" in dep for dep in extras["safetensors"])
    assert any("bitsandbytes" in dep for dep in extras["full"])
    assert any(dep.startswith("auralite-ai[") for dep in extras["dev"])


def test_pytest_configuration():
    cfg = _pyproject()["tool"]["pytest"]["ini_options"]
    assert "-q" not in cfg["addopts"]
    markers = " ".join(cfg["markers"])
    assert "integration" in markers


def test_packaging_targets_exist_on_disk():
    """2.7.0: one package, discovered — no hand-kept module lists to forget."""
    import pathlib

    cfg = _pyproject()["tool"]["setuptools"]
    assert "py-modules" not in cfg
    find = cfg["packages"]["find"]
    assert find["include"] == ["auralite", "auralite.*"]
    assert find["namespaces"] is False
    root = pathlib.Path(REPO_ROOT) / "auralite"
    for init_dir in [p for p in root.rglob("*") if p.is_dir() and p.name != "__pycache__"]:
        assert (init_dir / "__init__.py").exists(), f"{init_dir} would not be packaged"


def test_pep561_marker_is_packaged():
    data = _pyproject()["tool"]["setuptools"]["package-data"]
    assert "py.typed" in data["auralite"]
    assert os.path.exists(os.path.join(REPO_ROOT, "auralite", "py.typed"))


def test_ci_runs_the_extended_hygiene_gate():
    ci = open(os.path.join(REPO_ROOT, ".github", "workflows", "ci.yml"),
              encoding="utf-8").read()
    assert "--select E9,F,B" in ci
    assert "optional-deps" in ci
    assert "workflow_dispatch" in ci


# ======================================================================
#  14. Quantization backends (2.7.0: auto = torch.ao while it exists, else torchao)
# ======================================================================
def test_quant_backend_default_is_auto_and_resolves_to_torch_ao(monkeypatch):
    from auralite.quantization_backends import describe_backends, resolve_backend

    monkeypatch.delenv("AURALITE_QUANT_BACKEND", raising=False)
    assert resolve_backend() == "torch.ao"          # torch 2.14 still ships it
    info = describe_backends()
    assert info["default_backend"].startswith("auto")
    assert info["selected_backend"] == "torch.ao"
    assert info["pytorch"]
    assert info["torch.ao.quantization"] == "available"


@pytest.mark.parametrize("torch_ao, torchao, expected", [
    (True, True, "torch.ao"),        # faster eager CPU kernels: torchao is not preferred
    (True, False, "torch.ao"),
    (False, True, "torchao"),        # a PyTorch without torch.ao.quantization
    (False, False, RuntimeError),
])
def test_quant_backend_auto_resolution(monkeypatch, torch_ao, torchao, expected):
    import auralite.quantization_backends as qb

    monkeypatch.setattr(qb, "torch_ao_available", lambda: torch_ao)
    monkeypatch.setattr(qb, "torchao_available", lambda: torchao)
    if expected is RuntimeError:
        with pytest.raises(RuntimeError, match="pip install torchao"):
            qb.resolve_backend("auto")
    else:
        assert qb.resolve_backend("auto") == expected
        assert qb.resolve_backend("") == expected     # unset env var = auto


def test_quant_backend_unknown_value_is_rejected():
    from auralite.quantization_backends import resolve_backend
    with pytest.raises(ValueError, match="Unknown quantization backend"):
        resolve_backend("nonsense")


def test_quant_backend_torchao_missing_raises_clear_error(monkeypatch):
    import auralite.quantization_backends as qb

    monkeypatch.setitem(sys.modules, "torchao", None)     # behaves like "not installed"
    assert qb.torchao_available() is False and qb.torchao_version() is None
    monkeypatch.delenv("AURALITE_QUANT_BACKEND", raising=False)
    with pytest.raises(RuntimeError, match="pip install torchao"):
        qb.resolve_backend("torchao")


def test_quant_backend_torchao_stub_is_used(monkeypatch):
    """Old torchao releases: `quantize_(model, int8_weight_only())`."""
    import auralite.quantization_backends as qb

    calls: list = []
    fake_quant = types.ModuleType("torchao.quantization")

    def quantize_(model, config):
        calls.append((model, config))

    fake_quant.quantize_ = quantize_
    fake_quant.int8_weight_only = lambda: "int8-weight-only-config"
    fake_torchao = types.ModuleType("torchao")
    fake_torchao.quantization = fake_quant
    monkeypatch.setitem(sys.modules, "torchao", fake_torchao)
    monkeypatch.setitem(sys.modules, "torchao.quantization", fake_quant)

    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    assert qb.resolve_backend("torchao") == "torchao"
    returned = qb.dynamic_int8(model, "torchao")
    assert returned is model
    assert calls == [(model, "int8-weight-only-config")]


def test_quant_backend_guard_when_torch_ao_is_gone(monkeypatch):
    """Future PyTorch without torch.ao.quantization → descriptive RuntimeError."""
    import auralite.quantization_backends as qb

    monkeypatch.setitem(sys.modules, "torch.ao.quantization", None)
    monkeypatch.setitem(sys.modules, "torch.quantization", None)
    with pytest.raises(RuntimeError, match="torchao"):
        qb.torch_ao_namespace()


def test_dynamic_int8_real_path_produces_quantized_layers():
    pytest.importorskip("torch")
    from auralite.quantization_backends import dynamic_int8

    model = torch.nn.Sequential(torch.nn.Linear(8, 4))
    quantized = dynamic_int8(model)      # default: auto → torch.ao (also with torchao installed)
    weight_dtypes = set()
    for module in quantized.modules():
        weight = getattr(module, "weight", None)
        if callable(weight):                 # quantized dynamic Linear exposes weight()
            try:
                weight = weight()
            except Exception:  # noqa: BLE001 - not every module can report it
                weight = None
        if isinstance(weight, torch.Tensor):
            weight_dtypes.add(weight.dtype)
    assert torch.qint8 in weight_dtypes, weight_dtypes  # INT8 weights, really
    with torch.inference_mode():
        out = quantized(torch.randn(2, 8))
    assert out.shape == (2, 4)


# ======================================================================
#  15. CLI: --quiet / --log-level / safe info / --bench / --backends
# ======================================================================
def test_cli_global_and_subcommand_flags_parse():
    from auralite.cli import build_parser

    parser = build_parser()
    sub = parser.parse_args(["info", "--model", "x.pt", "--quiet", "--bench", "5"])
    assert sub.quiet is True and sub.bench == 5
    root = parser.parse_args(["--quiet", "--log-level", "INFO", "info", "--model", "x.pt"])
    assert root.quiet is True and root.log_level == "INFO"
    default = parser.parse_args(["generate", "--model", "m.pt", "--prompt", "hi"])
    # v2.6.5: engine progress is logged at INFO and the CLI shows it by default
    assert default.quiet is False and default.log_level == "INFO"


def test_cli_logging_configuration(monkeypatch):
    import logging

    from auralite.cli import _configure_logging, build_parser

    args = build_parser().parse_args(["info", "--model", "x.pt", "--log-level", "DEBUG"])
    _configure_logging(args)
    logger = logging.getLogger("auralite")
    assert logger.level == logging.DEBUG
    assert os.environ["AURALITE_LOG_LEVEL"] == "DEBUG"
    assert logger.propagate is False          # no duplicated records via root

    quiet = build_parser().parse_args(["info", "--model", "x.pt", "--quiet"])
    _configure_logging(quiet)
    assert logging.getLogger("auralite").level == logging.CRITICAL


def test_cli_info_uses_safe_load_and_reports_backends(trained_ckpt_path, capsys):
    from auralite.cli import main

    assert main(["info", "--model", trained_ckpt_path, "--backends", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "checkpoint_version: 3" in out
    assert "quantization backends:" in out
    assert "selected_backend:" in out


def test_cli_info_fails_cleanly_on_junk(tmp_path, capsys):
    from auralite.cli import main

    junk = tmp_path / "junk.pt"
    junk.write_text("definitely not a checkpoint", encoding="utf-8")
    assert main(["info", "--model", str(junk)]) == 2
    assert "cannot read" in capsys.readouterr().err

    assert main(["info", "--model", str(tmp_path / "missing.pt")]) == 2
    assert "not found" in capsys.readouterr().err


def test_cli_bench_reports_tokens_per_second(trained_ckpt_path, capsys):
    from auralite.cli import main

    assert main(["info", "--model", trained_ckpt_path, "--bench", "4", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "tokens/s" in out and "weights_mb:" in out


# ======================================================================
#  16. Documentation of the feature status
# ======================================================================
def test_readme_documents_feature_status():
    readme = open(os.path.join(REPO_ROOT, "README.md"), encoding="utf-8").read()
    assert "Feature status" in readme
    for topic in ("Speculative decoding", "FlexAttention", "sandbox",
                  "Quantization backends", "single worker"):
        assert topic in readme, topic


def test_architecture_map_exists():
    arch = open(os.path.join(REPO_ROOT, "ARCHITECTURE.md"), encoding="utf-8").read()
    for topic in ("auralite/engine/", "auralite/gui/", "tabs/", "safe_io",
                  "auralite/server/openai_server.py", "quantization_backends.py",
                  "_compat.py"):
        assert topic in arch, topic


# ======================================================================
#  17. Server lifecycle + legacy GUI namespace
# ======================================================================
def test_server_restart_clears_shutdown_flag(server_app):
    """A restarted server must stream again (the flag used to stay set)."""
    from fastapi.testclient import TestClient

    with TestClient(server_app.app):
        pass
    assert server_app._shutdown.is_set(), "shutdown should be flagged on exit"

    with TestClient(server_app.app) as client:      # simulated restart
        assert not server_app._shutdown.is_set()
        assert client.get("/health").status_code == 200


def test_gui_shim_exposes_the_legacy_namespace():
    """`gui_app.<name>` keeps working (deprecated) for scripts that monkeypatch it."""
    import warnings

    from auralite.gui import common

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import gui_app

    for name in ("messagebox", "filedialog", "tk", "ttk", "ConsoleRedirector"):
        assert getattr(gui_app, name) is getattr(common, name), name
    assert callable(gui_app._fmt_duration)
    if hasattr(common, "HAS_MATPLOTLIB"):
        assert gui_app.HAS_MATPLOTLIB == common.HAS_MATPLOTLIB
    assert gui_app.AIApp is __import__("auralite.gui.app", fromlist=["AIApp"]).AIApp


# ======================================================================
#  18. Server: reproducible seeded generation on a real engine
# ======================================================================
def test_seeded_server_requests_are_reproducible(trained_engine, monkeypatch):
    """The request-level `seed` makes sampling reproducible on the real engine."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import auralite.server.openai_server as srv

    monkeypatch.setattr(srv, "_engine", trained_engine)
    body = {"prompt": "AuraLite", "max_tokens": 6, "temperature": 1.5, "seed": 99}
    with TestClient(srv.app, raise_server_exceptions=False) as client:
        first = client.post("/v1/completions", json=body).json()
        second = client.post("/v1/completions", json=body).json()
    assert first["choices"][0]["text"] == second["choices"][0]["text"]
    assert first["usage"]["completion_tokens"] > 0


# ======================================================================
#  19. Low-RAM corpora: token chunks + PagedDataset in train()
# ======================================================================
def test_token_chunks_are_exact_for_char_tokenizer(tiny_tokenizer, tiny_text):
    """Chunked tokenization must reproduce whole-text encoding exactly."""
    from auralite.engine import AuraLiteEngine

    engine = AuraLiteEngine()
    engine.tokenizer = tiny_tokenizer
    chunks = list(engine._iter_token_chunks(tiny_text, chunk_chars=97, overlap_chars=16))
    assert len(chunks) > 3, "chunking did not actually split the corpus"
    assert sum(len(c) for c in chunks) == len(tiny_text)
    assert [i for c in chunks for i in c] == list(tiny_tokenizer.encode(tiny_text))


def test_write_tokens_memmap_and_paged_view(tmp_path):
    from auralite.engine.dataset import PagedDataset, write_tokens_memmap

    path = tmp_path / "tokens.bin"
    total = write_tokens_memmap(path, [list(range(100)), list(range(100, 160))],
                                dtype="uint32")
    assert total == 160
    assert path.stat().st_size == 160 * 4          # exact, no reserved tail

    full = PagedDataset(path, seq_length=8)
    assert len(full) == 160 - 8
    x, y = full[0]
    assert x.tolist() == list(range(8)) and y.tolist() == list(range(1, 9))

    head = PagedDataset(path, seq_length=8, offset=0, limit=100)
    tail = PagedDataset(path, seq_length=8, offset=100, limit=60)
    assert head.n_tokens == 100 and tail.n_tokens == 60
    assert len(head) == 92 and len(tail) == 52
    assert tail[0][0].tolist() == list(range(100, 108))   # second half starts at 100


def test_train_with_mmap_corpus_matches_ram(tiny_text, tmp_path):
    """Same seed, same tokens → identical weights, loss history and output."""
    from auralite.engine import AuraLiteEngine

    params = {
        "tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_layers": 2,
        "n_kv_heads": 2, "seq_length": 16, "batch_size": 8, "lr": 2e-2, "epochs": 1,
        "val_split": 0.0, "seed": 0, "optimizer": "adamw", "lr_schedule": "cosine",
        "amp_dtype": "none", "weight_decay": 0.0,
    }
    ram = AuraLiteEngine()
    ram.train(tiny_text, dict(params))

    cache = tmp_path / "corpus.uint32.bin"
    mmap_engine = AuraLiteEngine()
    mmap_engine.train(tiny_text, {**params, "corpus_cache": "mmap",
                                  "corpus_cache_path": str(cache)})

    assert cache.exists() and cache.stat().st_size > 0
    assert mmap_engine._corpus_cache_path == str(cache)
    ram_state = ram.model.state_dict()
    mmap_state = mmap_engine.model.state_dict()
    assert ram_state.keys() == mmap_state.keys()
    for key in ram_state:
        assert torch.equal(ram_state[key], mmap_state[key]), key
    assert ram.generate("AuraLite", 8, temperature=0.0, top_k=1) == \
        mmap_engine.generate("AuraLite", 8, temperature=0.0, top_k=1)


def test_train_mmap_reports_memmap_and_honours_env(tiny_text, tmp_path, monkeypatch, capsys):
    from auralite.engine import AuraLiteEngine

    monkeypatch.setenv("AURALITE_CORPUS_CACHE", "mmap")   # env switch, no params key
    cache = tmp_path / "env_corpus.bin"
    engine = AuraLiteEngine()
    engine.train(tiny_text, {
        "tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_layers": 2,
        "seq_length": 16, "batch_size": 8, "lr": 2e-2, "epochs": 1, "val_split": 0.0,
        "seed": 0, "amp_dtype": "none", "corpus_cache_path": str(cache),
    })
    out = capsys.readouterr().err        # library messages now go through logging
    assert "uint32 memmap" in out and cache.exists()


def test_cli_train_mmap_flag(tiny_text, tmp_path, capsys):
    from auralite.cli import main

    text_file = tmp_path / "corpus.txt"
    text_file.write_text(tiny_text, encoding="utf-8")
    params = tmp_path / "params.json"
    params.write_text(json.dumps({
        "tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_layers": 2,
        "seq_length": 16, "batch_size": 8, "lr": 2e-2, "epochs": 1, "val_split": 0.0,
        "seed": 0, "amp_dtype": "none",
    }), encoding="utf-8")
    out_ckpt = tmp_path / "mmap_cli.pt"
    cache = tmp_path / "cli_corpus.bin"
    rc = main(["train", "--text", str(text_file), "--params", str(params),
               "--model", str(out_ckpt), "--corpus-cache", str(cache), "--quiet"])
    assert rc == 0, capsys.readouterr().err
    assert out_ckpt.exists() and cache.exists()


def test_cli_rejects_unknown_training_parameter(tiny_text, tmp_path, capsys):
    from auralite.cli import main

    text_file = tmp_path / "corpus.txt"
    text_file.write_text(tiny_text, encoding="utf-8")
    params = tmp_path / "bad.json"
    params.write_text(json.dumps({"epoch": 3, "d_model": "big"}), encoding="utf-8")
    rc = main(["train", "--text", str(text_file), "--params", str(params),
               "--model", str(tmp_path / "x.pt")])
    assert rc == 2
    assert "invalid training parameters" in capsys.readouterr().err
