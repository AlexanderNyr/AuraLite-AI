"""Release engineering checks (roadmap P2.14).

These tests keep the *distribution* honest: one version number, a release
workflow that cannot publish a tag it did not verify, a Docker target that
exists, and documentation links that point at real files. They are cheap — no
network, no build — but they catch the class of mistake that is only discovered
by the people downloading a broken release.
"""
from __future__ import annotations

import importlib.util
import pathlib
import re

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
READMES = [README, REPO_ROOT / "README.ru.md"]
DOC_FILES = [p for p in (REPO_ROOT / "docs").glob("*.md")]
WORKFLOWS = {p.name: yaml.safe_load(p.read_text(encoding="utf-8"))
             for p in (REPO_ROOT / ".github" / "workflows").glob("*.yml")}


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def version_check():
    return _load_script("check_version_consistency")


# ----------------------------------------------------------------------
#  Version consistency (the bug that started v2.6.5)
# ----------------------------------------------------------------------
def test_version_strings_agree(version_check):
    assert version_check.main([]) == 0


def test_version_check_rejects_a_foreign_tag(version_check, capsys):
    assert version_check.main(["v99.99.99"]) == 1
    assert "does not match" in capsys.readouterr().err


def test_version_check_accepts_its_own_tag(version_check):
    assert version_check.main([f"v{version_check._module_version()}"]) == 0


def test_server_has_no_hardcoded_version():
    source = (REPO_ROOT / "server" / "openai_server.py").read_text(encoding="utf-8")
    assert not re.search(r'^_AURALITE_VERSION\s*=\s*"', source, re.M)
    from server.openai_server import _AURALITE_VERSION
    from model_engine import __version__

    assert _AURALITE_VERSION == __version__


def test_pyproject_and_package_versions_match(version_check):
    assert version_check._pyproject_version() == version_check._module_version()
    assert version_check._changelog_version() == version_check._pyproject_version()


# ----------------------------------------------------------------------
#  The release workflow
# ----------------------------------------------------------------------
def test_release_workflow_triggers_on_tags():
    workflow = WORKFLOWS["release.yml"]
    triggers = workflow.get("on") or workflow.get(True)
    assert triggers["push"]["tags"] == ["v*"]
    assert "workflow_dispatch" in triggers


def test_release_workflow_jobs_and_permissions():
    workflow = WORKFLOWS["release.yml"]
    jobs = workflow["jobs"]
    assert {"verify", "build", "release", "image", "pypi"} <= set(jobs)

    # Least privilege by default, escalated only where unavoidable.
    assert workflow["permissions"] == {"contents": "read"}
    assert jobs["release"]["permissions"]["contents"] == "write"
    assert jobs["image"]["permissions"]["packages"] == "write"

    # Nothing publishes before verify/build succeed.
    assert set(jobs["release"]["needs"]) == {"verify", "build"}
    assert jobs["image"]["needs"] == ["verify", "build"]


def test_release_workflow_verifies_version_and_gates_on_lint():
    verify = WORKFLOWS["release.yml"]["jobs"]["verify"]
    step_names = [step.get("name", "") for step in verify["steps"]]
    assert any("Version strings must agree" in name for name in step_names)
    assert any("Ruff gate" in name for name in step_names)
    assert any("Fast test suite" in name for name in step_names)

    joined = "\n".join(str(step) for step in verify["steps"])
    assert "scripts/check_version_consistency.py" in joined
    # The version comes from the tag, not from a developer's guess.
    assert "github.ref_name" in joined


def _smoke_installs_the_built_wheel(steps) -> None:
    joined = "\n".join(str(step) for step in steps)
    assert "python -m build" in joined
    assert "twine check" in joined
    # The wheel goes into a clean venv *with* the documented `[serve]` extra
    # (the server module needs FastAPI) and the smoke script runs outside the
    # checkout, so nothing can be imported from the source tree by accident.
    assert '"$(ls dist/*.whl)[serve]"' in joined
    assert "python -m venv /tmp/wheel-venv" in joined
    assert 'cd /tmp && /tmp/wheel-venv/bin/python "$GITHUB_WORKSPACE/scripts/smoke_installed_wheel.py"' in joined


def test_release_workflow_builds_and_checks_artifacts():
    _smoke_installs_the_built_wheel(WORKFLOWS["release.yml"]["jobs"]["build"]["steps"])


def test_ci_builds_and_smoke_tests_the_wheel_on_every_push():
    """Editable installs hide packaging gaps, so CI must install the real wheel."""
    jobs = WORKFLOWS["ci.yml"]["jobs"]
    assert "package" in jobs
    _smoke_installs_the_built_wheel(jobs["package"]["steps"])


def test_pypi_publishing_is_tokenless_and_opt_in():
    jobs = WORKFLOWS["release.yml"]["jobs"]
    pypi = jobs["pypi"]
    # Opt-in: nothing is uploaded until the maintainer sets the repo variable.
    assert pypi["if"] == "vars.PYPI_PUBLISH == 'true'"
    # Only artifacts that were verified, built and smoke-tested are published.
    assert set(pypi["needs"]) == {"verify", "build"}
    # Trusted Publishing (OIDC): the job may mint an identity token, nothing else.
    assert pypi["permissions"] == {"id-token": "write"}
    assert pypi["environment"]["name"] == "pypi"
    uses = [str(step.get("uses", "")) for step in pypi["steps"]]
    assert "pypa/gh-action-pypi-publish@release/v1" in uses
    assert any(use.startswith("actions/download-artifact@") for use in uses)
    publish = next(step for step in pypi["steps"]
                   if "gh-action-pypi-publish" in str(step.get("uses", "")))
    assert "password" not in (publish.get("with") or {})      # no long-lived token
    text = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "PYPI_API_TOKEN" not in text


#: First major of each action that runs on Node 24 (Node 20 is deprecated on
#: GitHub-hosted runners; upstream CI already prints the deprecation warning).
NODE24_MAJORS = {
    "actions/checkout": 5, "actions/setup-python": 6,
    "actions/upload-artifact": 6, "actions/download-artifact": 7,
    "docker/setup-buildx-action": 4, "docker/login-action": 4,
    "docker/metadata-action": 6, "docker/build-push-action": 7,
    "softprops/action-gh-release": 3,
}


@pytest.mark.parametrize("name", sorted(WORKFLOWS))
def test_workflows_use_node24_action_majors(name):
    for job in WORKFLOWS[name]["jobs"].values():
        for step in job.get("steps", []):
            uses = step.get("uses")
            if not uses or "@" not in uses:
                continue
            action, ref = uses.split("@", 1)
            if action not in NODE24_MAJORS:
                continue
            major = int(re.match(r"v(\d+)", ref).group(1))
            assert major >= NODE24_MAJORS[action], f"{name}: {uses} still runs on Node 20"


def test_workflow_dispatch_inputs_that_jobs_read_are_declared():
    workflow = WORKFLOWS["ci.yml"]
    triggers = workflow.get("on") or workflow.get(True)
    declared = set((triggers["workflow_dispatch"] or {}).get("inputs", {}))
    text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    used = set(re.findall(r"inputs\.([A-Za-z_][\w-]*)", text))
    assert used <= declared, f"undeclared workflow_dispatch inputs: {sorted(used - declared)}"


def test_release_workflow_attaches_artifacts_and_pushes_image():
    jobs = WORKFLOWS["release.yml"]["jobs"]
    release_step = next(step for step in jobs["release"]["steps"]
                        if "action-gh-release" in str(step.get("uses", "")))
    assert release_step["with"]["files"] == "dist/*"
    assert release_step["with"]["generate_release_notes"] is True
    assert release_step["with"]["tag_name"].startswith("v${{")
    # The workflow must build the release from the artifacts CI produced.
    assert any("download-artifact" in str(step.get("uses", ""))
               for step in jobs["release"]["steps"])

    image_uses = [str(step.get("uses", "")) for step in jobs["image"]["steps"]]
    assert any("docker/login-action" in use for use in image_uses)
    build_step = next(step for step in jobs["image"]["steps"]
                      if "build-push-action" in str(step.get("uses", "")))
    assert build_step["with"]["target"] == "runtime-cpu"
    assert build_step["with"]["tags"].startswith("${{ steps.meta.outputs.tags }}")
    meta_step = next(step for step in jobs["image"]["steps"]
                     if "metadata-action" in str(step.get("uses", "")))
    assert meta_step["with"]["images"] == "ghcr.io/${{ github.repository }}"
    assert "latest" in meta_step["with"]["tags"]


# ----------------------------------------------------------------------
#  Container + CI gates
# ----------------------------------------------------------------------
def test_docker_targets_used_by_ci_exist():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^FROM .* AS (\S+)", dockerfile, re.M))
    assert {"runtime-cpu", "runtime-cuda"} <= targets

    for name in ("ci.yml", "release.yml"):
        joined = (REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        for used in re.findall(r"target: (\S+)", joined):
            assert used in targets, f"{name} builds the missing Docker target {used}"


def test_ci_coverage_job_measures_the_user_facing_surface():
    coverage = WORKFLOWS["ci.yml"]["jobs"]["coverage"]
    joined = "\n".join(str(step) for step in coverage["steps"])
    for package in ("--cov=gui", "--cov=image_engine", "--cov=auralite_cli"):
        assert package in joined
    assert "--cov-fail-under=" in joined
    assert "import tkinter" in joined      # GUI tests must not silently skip


# ----------------------------------------------------------------------
#  Documentation
# ----------------------------------------------------------------------
@pytest.mark.parametrize("path", READMES + DOC_FILES, ids=lambda p: p.name)
def test_markdown_relative_links_resolve(path):
    text = path.read_text(encoding="utf-8")
    missing = []
    for target in re.findall(r"\]\((?!https?://|#|mailto:)([^)#\s]+)", text):
        candidate = (path.parent / target).resolve()
        if not candidate.exists():
            missing.append(target)
    assert missing == [], f"{path.name} links to missing files: {missing}"


def test_readme_advertises_the_documentation_set():
    readme = README.read_text(encoding="utf-8")
    for target in ("docs/quickstart.md", "docs/faq.md", "docs/quantization.md",
                   "README.ru.md"):
        assert target in readme, target
    # The badges must not promise a version the code does not have.
    from model_engine import __version__

    assert f"version-{__version__}-orange" in readme


def test_docs_keep_the_honest_claims():
    """The FAQ/quantization docs must keep the measured, unflattering numbers."""
    faq = (REPO_ROOT / "docs" / "faq.md").read_text(encoding="utf-8")
    assert "slower" in faq.lower()
    assert "931" in faq and "798" in faq          # eager vs compiled decode
    quant = (REPO_ROOT / "docs" / "quantization.md").read_text(encoding="utf-8")
    assert "memory" in quant.lower() and "usually not" in quant.lower()


def test_scripts_are_executable_and_documented():
    script = REPO_ROOT / "scripts" / "check_version_consistency.py"
    assert script.exists()
    assert script.read_text(encoding="utf-8").startswith("#!/usr/bin/env python3")
    arch = (REPO_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "release.yml" in arch
    assert "check_version_consistency" in arch
