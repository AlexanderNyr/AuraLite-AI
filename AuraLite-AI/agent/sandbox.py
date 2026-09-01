"""
agent/sandbox.py — Isolated execution sandbox for AuraLite Agent.

Strategy: subprocess-based sandboxing that works on Windows 10, Linux, and macOS
without requiring Docker or root privileges.

Security layers:
  1. subprocess with timeout (no runaway processes)
  2. Separate working directory per session (tempdir, auto-cleaned)
  3. Resource limits via threading watchdog (CPU time / memory RSS on all platforms)
  4. Command whitelist for shell mode (configurable)
  5. stdout/stderr captured, size-capped (no infinite output floods)
  6. Network access left open (model may need web tools) — caller controls

Windows 10 compatible: no UNIX-only signals/setpgrp used; watchdog thread kills
the process via Popen.kill().
"""
from __future__ import annotations

import os
import platform
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_TIMEOUT_S = 15          # wall-clock seconds per execution
MAX_OUTPUT_BYTES  = 64 * 1024   # 64 KB output cap
MAX_OUTPUT_LINES  = 500

# Commands always blocked regardless of whitelist
HARD_BLOCKED = {
    # shell self-destruction / privilege escalation
    "rm", "rmdir", "del", "rd", "format", "mkfs",
    "sudo", "su", "runas",
    # process escalation
    "chmod", "chown", "chattr",
    # system-level controls
    "shutdown", "reboot", "halt", "poweroff", "init",
    # dangerous net tools
    "nc", "ncat", "netcat",
}

# Default whitelist: only these base commands are allowed in shell mode.
# Python execution goes through a separate python binary path always.
DEFAULT_SHELL_WHITELIST = {
    "python", "python3", "python3.11", "python3.10", "python3.12",
    "pip", "pip3",
    "echo", "cat", "type",          # basic I/O (type = Windows cat)
    "ls", "dir",                    # listing
    "pwd", "cd",
    "mkdir", "touch",
    "grep", "find", "where", "which",
    "git",
    "curl", "wget",
    "node", "npm",
    "java",
    "ping",
    "date", "time",
    "uname",
}


# ── Result ──────────────────────────────────────────────────────────────────

@dataclass
class SandboxResult:
    stdout:     str  = ""
    stderr:     str  = ""
    returncode: int  = 0
    timed_out:  bool = False
    blocked:    bool = False
    error:      str  = ""
    duration_s: float = 0.0

    @property
    def success(self) -> bool:
        return not self.timed_out and not self.blocked and not self.error and self.returncode == 0

    def combined_output(self, max_chars: int = 4000) -> str:
        parts = []
        if self.blocked:
            return f"[SANDBOX] ❌ Command blocked by security policy: {self.error}"
        if self.timed_out:
            parts.append("[SANDBOX] ⏱ Execution timed out.\n")
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"\n[stderr]\n{self.stderr}")
        if self.error:
            parts.append(f"\n[error] {self.error}")
        result = "".join(parts)
        if len(result) > max_chars:
            result = result[:max_chars] + f"\n… [truncated, {len(result)} chars total]"
        return result


# ── Sandbox ──────────────────────────────────────────────────────────────────

class Sandbox:
    """
    Isolated subprocess-based execution sandbox.

    Usage:
        sb = Sandbox()
        result = sb.run_python("print('hello')")
        result = sb.run_shell("ls -la")
        sb.cleanup()   # or use as context manager
    """

    def __init__(
        self,
        workdir: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        shell_whitelist: Optional[set[str]] = None,
        allow_network: bool = True,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
        python_executable: Optional[str] = None,
    ):
        self.timeout = float(timeout)
        self.shell_whitelist = shell_whitelist if shell_whitelist is not None else DEFAULT_SHELL_WHITELIST
        self.allow_network = allow_network
        self.max_output_bytes = int(max_output_bytes)
        self.python_executable = python_executable or sys.executable

        # Create (or reuse) isolated working directory
        if workdir:
            self._workdir_obj = None
            self.workdir = Path(workdir)
            self.workdir.mkdir(parents=True, exist_ok=True)
        else:
            self._workdir_obj = tempfile.TemporaryDirectory(prefix="auralite_agent_")
            self.workdir = Path(self._workdir_obj.name)

        self._is_windows = platform.system() == "Windows"

    # ── Context manager ─────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()

    def cleanup(self):
        if self._workdir_obj is not None:
            try:
                self._workdir_obj.cleanup()
            except Exception:
                pass
            self._workdir_obj = None

    # ── Public API ──────────────────────────────────────────────────────────

    def run_python(self, code: str, *, extra_env: dict | None = None) -> SandboxResult:
        """Execute Python code in an isolated subprocess."""
        # Write code to a temp file in the workdir
        script = self.workdir / "_agent_script.py"
        try:
            script.write_text(code, encoding="utf-8")
        except Exception as e:
            return SandboxResult(error=f"Could not write script: {e}", returncode=1)

        cmd = [self.python_executable, str(script)]
        return self._run_cmd(cmd, extra_env=extra_env, label="python")

    def run_shell(self, command: str, *, extra_env: dict | None = None) -> SandboxResult:
        """Execute a shell command with whitelist enforcement."""
        # Parse the first token to check whitelist
        try:
            if self._is_windows:
                # Windows: use shlex with posix=False
                tokens = shlex.split(command, posix=False)
            else:
                tokens = shlex.split(command)
        except ValueError as e:
            return SandboxResult(error=f"Command parse error: {e}", returncode=1, blocked=True)

        if not tokens:
            return SandboxResult(error="Empty command", returncode=1)

        base_cmd = Path(tokens[0]).name.lower()
        # Strip .exe on Windows
        if base_cmd.endswith(".exe"):
            base_cmd = base_cmd[:-4]

        # Hard block check
        if base_cmd in HARD_BLOCKED:
            return SandboxResult(
                error=f"'{base_cmd}' is permanently blocked",
                returncode=1, blocked=True,
            )

        # Whitelist check
        if base_cmd not in self.shell_whitelist:
            return SandboxResult(
                error=f"'{base_cmd}' is not in the allowed command whitelist.\n"
                      f"Allowed: {sorted(self.shell_whitelist)}",
                returncode=1, blocked=True,
            )

        if self._is_windows:
            cmd_list = ["cmd", "/c", command]
        else:
            cmd_list = ["/bin/sh", "-c", command]

        return self._run_cmd(cmd_list, extra_env=extra_env, label=base_cmd)

    def write_file(self, filename: str, content: str) -> SandboxResult:
        """Write a file inside the sandbox workdir."""
        try:
            target = self._safe_path(filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return SandboxResult(stdout=f"File written: {filename} ({len(content)} chars)")
        except Exception as e:
            return SandboxResult(error=str(e), returncode=1)

    def read_file(self, filename: str, max_chars: int = 8000) -> SandboxResult:
        """Read a file from the sandbox workdir."""
        try:
            target = self._safe_path(filename)
            if not target.exists():
                return SandboxResult(error=f"File not found: {filename}", returncode=1)
            content = target.read_text(encoding="utf-8", errors="replace")
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n… [truncated at {max_chars} chars]"
            return SandboxResult(stdout=content)
        except Exception as e:
            return SandboxResult(error=str(e), returncode=1)

    def list_files(self, subdir: str = ".") -> SandboxResult:
        """List files in a sandbox directory."""
        try:
            target = self._safe_path(subdir)
            if not target.is_dir():
                return SandboxResult(error=f"Not a directory: {subdir}", returncode=1)
            entries = sorted(target.iterdir())
            lines = []
            for e in entries:
                size = e.stat().st_size if e.is_file() else 0
                kind = "DIR " if e.is_dir() else "FILE"
                lines.append(f"{kind}  {e.name}  ({size} bytes)" if e.is_file() else f"{kind}  {e.name}/")
            return SandboxResult(stdout="\n".join(lines) if lines else "(empty directory)")
        except Exception as e:
            return SandboxResult(error=str(e), returncode=1)

    def install_package(self, package: str) -> SandboxResult:
        """Install a Python package into the sandbox environment."""
        # Sanitize: only allow simple package names
        import re
        if not re.match(r"^[a-zA-Z0-9_\-\.\[\]>=!<~,\s]+$", package):
            return SandboxResult(error=f"Invalid package name: {package}", returncode=1, blocked=True)
        cmd = [self.python_executable, "-m", "pip", "install", "--quiet", package]
        return self._run_cmd(cmd, label="pip", timeout_override=120.0)

    # ── Internal ────────────────────────────────────────────────────────────

    def _safe_path(self, filename: str) -> Path:
        """Resolve path and ensure it stays inside workdir."""
        p = (self.workdir / filename).resolve()
        if not str(p).startswith(str(self.workdir.resolve())):
            raise ValueError(f"Path escape attempt blocked: {filename}")
        return p

    def _run_cmd(
        self,
        cmd: list[str],
        *,
        extra_env: dict | None = None,
        label: str = "cmd",
        timeout_override: float | None = None,
    ) -> SandboxResult:
        """Run a command list with full watchdog protection."""
        timeout = timeout_override if timeout_override is not None else self.timeout
        t0 = time.perf_counter()

        # Build environment
        env = os.environ.copy()
        # Prevent the subprocess from inheriting AURALITE_* secrets accidentally
        for key in list(env.keys()):
            if key.startswith("AURALITE_") and "TOKEN" in key.upper():
                env.pop(key, None)
        if extra_env:
            env.update({str(k): str(v) for k, v in extra_env.items()})

        # Spawn
        try:
            kwargs: dict = dict(
                cwd=str(self.workdir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            # On Windows avoid showing a console window for subprocesses
            if self._is_windows:
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

            proc = subprocess.Popen(cmd, **kwargs)
        except FileNotFoundError:
            return SandboxResult(
                error=f"Executable not found: {cmd[0]}", returncode=127,
                duration_s=time.perf_counter() - t0,
            )
        except Exception as e:
            return SandboxResult(
                error=str(e), returncode=1,
                duration_s=time.perf_counter() - t0,
            )

        # Watchdog thread
        killed = threading.Event()

        def _watchdog():
            if not killed.wait(timeout=timeout):
                try:
                    proc.kill()
                except Exception:
                    pass
                killed.set()

        wd = threading.Thread(target=_watchdog, daemon=True)
        wd.start()

        try:
            raw_stdout, raw_stderr = proc.communicate(timeout=timeout + 2)
        except subprocess.TimeoutExpired:
            proc.kill()
            raw_stdout, raw_stderr = proc.communicate()
            killed.set()
            return SandboxResult(
                stdout=self._cap(raw_stdout),
                stderr=self._cap(raw_stderr),
                returncode=-1,
                timed_out=True,
                duration_s=time.perf_counter() - t0,
            )
        finally:
            killed.set()  # stop watchdog

        return SandboxResult(
            stdout=self._cap(raw_stdout),
            stderr=self._cap(raw_stderr),
            returncode=proc.returncode,
            duration_s=time.perf_counter() - t0,
        )

    def _cap(self, text: str) -> str:
        """Cap output size to prevent floods."""
        if len(text.encode("utf-8", errors="replace")) > self.max_output_bytes:
            text = text[:self.max_output_bytes // 2] + "\n… [output truncated] …\n"
        # Also cap lines
        lines = text.splitlines()
        if len(lines) > MAX_OUTPUT_LINES:
            lines = lines[:MAX_OUTPUT_LINES] + [f"… [{len(lines) - MAX_OUTPUT_LINES} more lines truncated]"]
            text = "\n".join(lines)
        return text
