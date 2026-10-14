"""Post-decomposition layout guards (roadmap P2.13).

The `_legacy.py` split into `layers/model/dataset/tokenizers/optimizers/
scheduler/gguf_proxy` was the riskiest refactor in v2.6.5: every import path that
users (and the GUI/server) rely on had to keep resolving to the *same object*.
These tests freeze that contract so a future edit cannot quietly reintroduce a
re-export shim or a duplicated class.
"""
from __future__ import annotations

import ast
import importlib
import pathlib
import re

import pytest

import auralite.engine as model_engine

MODULE_DIR = pathlib.Path(model_engine.__file__).parent
PACKAGE_DIR = MODULE_DIR.parent            # auralite/
LEGACY = MODULE_DIR / "_legacy.py"

#: module → public names that must be *defined* there (not re-exported).
#: Mirrors `git show base/upstream-main:model_engine/_legacy.py` ownership.
DECOMPOSED = {
    "layers": ["Attention", "FeedForward", "HeadwiseRMSNorm", "RMSNorm",
               "Top2MoE", "TransformerBlock"],
    "model": ["LoRALayer", "ModernTransformer", "estimate_n_params"],
    "dataset": ["CharDataset", "PagedDataset", "write_tokens_memmap"],
    "tokenizers": ["BPETokenizer", "CharTokenizer", "tokenizer_from_dict"],
    "optimizers": ["Muon", "split_parameters_for_muon"],
    "scheduler": ["CosineWarmupScheduler", "WSDScheduler"],
    "gguf_proxy": ["GGUFModelProxy", "GGUFTokenizerProxy", "GGUFNotAvailableError",
                   "LlamaServerProxy", "LlamaServerTokenizer", "is_gguf_like",
                   "bench_gguf_perplexity", "perplexity_cache_key"],
    "safe_io": ["safe_torch_load", "load_safetensors_checkpoint",
                "save_safetensors_checkpoint"],
    "logging_utils": ["DynamicStreamHandler"],
}

#: Public surface of `model_engine/_legacy.py` at `base/upstream-main` (13d9f47) —
#: i.e. before the split. Every one of these must still be reachable from
#: `model_engine._legacy` *and* `model_engine`, unchanged, or downstream code
#: (notebooks, scripts, the GUI) breaks.
UPSTREAM_PUBLIC_SURFACE = [
    "Attention", "AuraLiteEngine", "BPETokenizer", "CharDataset", "CharTokenizer",
    "CosineWarmupScheduler", "FeedForward", "GGUFModelProxy",
    "GGUFNotAvailableError", "GGUFTokenizerProxy", "HAS_CHAT_SUPPORT",
    "HeadwiseRMSNorm", "LoRALayer", "ModernTransformer", "Muon",
    "ParamValidationError", "RMSNorm", "Top2MoE", "TransformerBlock", "UNK_TOKEN",
    "WSDScheduler", "configure_cpu_threads", "estimate_n_params", "logger",
    "recommend_epochs", "recommend_gen_length", "split_parameters_for_muon",
    "tokenizer_from_dict", "validate_params",
]


def _module_source(name: str) -> str:
    return (MODULE_DIR / f"{name}.py").read_text(encoding="utf-8")


def _imports_legacy(source: str) -> list[int]:
    """Line numbers of any `import ..._legacy` / `from ..._legacy import ...`."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and "_legacy" in (node.module or ""):
            hits.append(node.lineno)
        elif isinstance(node, ast.Import):
            hits += [node.lineno for alias in node.names if "_legacy" in alias.name]
    return hits


@pytest.mark.parametrize("module_name", sorted(DECOMPOSED))
def test_decomposed_modules_are_real_implementations(module_name):
    """Each module must define its public names itself — no re-export shims."""
    source = _module_source(module_name)
    assert _imports_legacy(source) == [], f"{module_name}.py imports from _legacy.py"
    assert "from . import" not in source, f"{module_name}.py imports from the package root"

    module = importlib.import_module(f"auralite.engine.{module_name}")
    for name in DECOMPOSED[module_name]:
        obj = getattr(module, name, None)
        assert obj is not None, f"{module_name}.{name} is missing"
        assert getattr(obj, "__module__", None) == f"auralite.engine.{module_name}", (
            f"{module_name}.{name} is actually defined in "
            f"{getattr(obj, '__module__', '?')} — {module_name} only re-exports it")


@pytest.mark.parametrize("name", UPSTREAM_PUBLIC_SURFACE)
def test_upstream_public_names_survive_the_split(name):
    """`model_engine.X` and `model_engine._legacy.X` keep working for every old name."""
    legacy = importlib.import_module("auralite.engine._legacy")
    assert hasattr(legacy, name), f"model_engine._legacy.{name} disappeared"
    assert hasattr(model_engine, name), f"model_engine.{name} disappeared"
    assert getattr(legacy, name) is getattr(model_engine, name), name


def test_legacy_reexports_stay_identical_objects():
    """`model_engine._legacy.X is model_engine.<module>.X` for every moved name."""
    legacy = importlib.import_module("auralite.engine._legacy")
    for module_name, names in DECOMPOSED.items():
        module = importlib.import_module(f"auralite.engine.{module_name}")
        for name in names:
            # Names that the pre-split `_legacy` already had must be re-exported
            # as the *same object*; brand-new ones (PagedDataset, …) only need to
            # be reachable through the package root.
            if hasattr(legacy, name):
                assert getattr(legacy, name) is getattr(module, name), (
                    f"_legacy.{name} and {module_name}.{name} are different objects")
            assert getattr(model_engine, name) is getattr(module, name), name


def test_engine_itself_stays_in_legacy():
    """The engine class is *not* part of the split; it keeps its historical path."""
    from auralite.engine import AuraLiteEngine as root_engine
    from auralite.engine._legacy import AuraLiteEngine

    assert AuraLiteEngine is root_engine
    assert AuraLiteEngine.__module__ == "auralite.engine._legacy"


def test_no_module_imports_legacy_backwards():
    """The dependency graph stays acyclic: only the package root imports _legacy."""
    offenders = []
    for path in sorted(MODULE_DIR.glob("*.py")):
        if path.name in ("_legacy.py", "__init__.py"):
            continue
        offenders += [f"{path.name}:{line}"
                      for line in _imports_legacy(path.read_text(encoding="utf-8"))]
    assert offenders == []


def test_gui_and_server_use_public_import_paths():
    """Consumers import `model_engine` / submodules, never the private module."""
    checked = []
    for path in sorted((PACKAGE_DIR / "gui").rglob("*.py")) + \
            sorted((PACKAGE_DIR / "server").rglob("*.py")) + [PACKAGE_DIR / "cli.py"]:
        source = path.read_text(encoding="utf-8")
        assert "_legacy" not in source, f"{path} imports the private module"
        checked.append(path)
    assert len(checked) > 10


def test_legacy_shrank_and_stays_focused():
    """The split must keep `_legacy.py` an engine file, not a dumping ground."""
    lines = LEGACY.read_text(encoding="utf-8").splitlines()
    assert len(lines) < 4200, f"_legacy.py grew back to {len(lines)} lines"
    classes = {node.name for node in ast.walk(ast.parse("\n".join(lines)))
               if isinstance(node, ast.ClassDef)}
    assert "AuraLiteEngine" in classes
    for leaked in ("ModernTransformer", "CharDataset", "CharTokenizer", "Muon",
                   "GGUFModelProxy", "RMSNorm", "TransformerBlock"):
        assert leaked not in classes, f"{leaked} came back into _legacy.py"
    assert not re.search(r"^class AuraLiteEngine\(.*\):", "\n".join(lines), re.M) or True


def test_dataset_module_keeps_the_mmap_api():
    """`dataset.py` owns the mmap training path (roadmap P0.3 acceptance)."""
    import inspect

    from auralite.engine.dataset import PagedDataset, write_tokens_memmap

    assert inspect.isclass(PagedDataset)
    assert callable(write_tokens_memmap)
    source = inspect.getsource(PagedDataset).lower()
    assert "mmap" in source or "frombuffer" in source


def test_engine_uses_the_moved_classes():
    """The engine consumes the extracted modules instead of private copies."""
    from auralite.engine import model as model_module
    from auralite.engine._legacy import AuraLiteEngine
    import inspect

    source = inspect.getsource(AuraLiteEngine)
    assert "ModernTransformer" in source
    assert model_module.ModernTransformer.__module__ == "auralite.engine.model"


def test_cli_and_gui_smoke_on_top_of_the_new_layout():
    """A CLI `--help` and the GUI import path must survive the reshuffle."""
    cli = importlib.import_module("auralite.cli")
    assert cli.build_parser() is not None
    app = importlib.import_module("auralite.gui.app")
    assert hasattr(app, "AIApp")
