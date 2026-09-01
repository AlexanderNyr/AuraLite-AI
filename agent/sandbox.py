"""
agent/sandbox.py — Isolated execution sandbox for AuraLite Agent.

Strategy: subprocess-based sandboxing that works on Windows 10, Linux, and macOS
without requiring Docker or root privileges.

Security layers:
  1. subprocess with timeout (no runaway processes)
  2. Separate working directory per session (tempdir, auto-cleaned)
  3. Watchdog thread kills the process via Popen.kill()
  4. Command whitelist for shell mode
  5. stdout/stderr captured, size-capped
  6. Path escape prevention
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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_TIMEOUT_S = 15
MAX_OUTPUT_BYTES  = 64 * 1024
MAX_OUTPUT_LINES  = 500

HARD_BLOCKED = {
    "rm", "rmdir", "del", "rd", "format", "mkfs",
    "sudo", "su", "runas",
    "chmod", "chown", "chattr",
    "shutdown", "reboot", "halt", "poweroff", "init",
    "nc", "ncat", "netcat",
}

DEFAULT_SHELL_WHITELIST = {
    "python", "python3", "python3.11", "python3.10", "python3.12",
    "pip", "pip3",
    "echo", "cat", "type",
    "ls", "dir",
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


@dataclass
class SandboxResult:
    stdout:     str   = ""
    stderr:     str   = ""
    returncode: int   = 0
    timed_out:  bool  = False
    blocked:    bool  = False
    error:      str   = ""
    duration_s: float = 0.0

    @property
    def success(self) -> bool:
        return not self.timed_out and not self.blocked and not self.error and self.returncode == 0

    def combined_output(self, max_chars: int = 4000) -> str:
        parts = []
        if self.blocked:
            return f"[SANDBOX] Command blocked by security policy: {self.error}"
        if self.timed_out:
            parts.append("[SANDBOX] Execution timed out.\n")
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"\n[stderr]\n{self.stderr}")
        if self.error:
            parts.append(f"\n[error] {self.error}")
        result = "".join(parts)
        if len(result) > max_chars:
            result = result[:max_chars] + f"\n[truncated, {len(result)} chars total]"
        return result


class Sandbox:
    """Isolated subprocess-based execution sandbox."""

    def __init__(
        self,
        workdir: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        shell_whitelist: Optional[set] = None,
        allow_network: bool = True,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
        python_executable: Optional[str] = None,
    ):
        self.timeout = float(timeout)
        self.shell_whitelist = shell_whitelist if shell_whitelist is not None else DEFAULT_SHELL_WHITELIST
        self.allow_network = allow_network
        self.max_output_bytes = int(max_output_bytes)
        self.python_executable = python_executable or sys.executable

        if workdir:
            self._workdir_obj = None
            self.workdir = Path(workdir)
            self.workdir.mkdir(parents=True, exist_ok=True)
        else:
            self._workdir_obj = tempfile.TemporaryDirectory(prefix="auralite_agent_")
            self.workdir = Path(self._workdir_obj.name)

        self._is_windows = platform.system() == "Windows"

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

    def run_python(self, code: str, *, extra_env=None) -> SandboxResult:
        script = self.workdir / "_agent_script.py"
        try:
            script.write_text(code, encoding="utf-8")
        except Exception as e:
            return SandboxResult(error=f"Could not write script: {e}", returncode=1)
        cmd = [self.python_executable, str(script)]
        return self._run_cmd(cmd, extra_env=extra_env, label="python")

    def run_shell(self, command: str, *, extra_env=None) -> SandboxResult:
        try:
            if self._is_windows:
                tokens = shlex.split(command, posix=False)
            else:
                tokens = shlex.split(command)
        except ValueError as e:
            return SandboxResult(error=f"Command parse error: {e}", returncode=1, blocked=True)

        if not tokens:
            return SandboxResult(error="Empty command", returncode=1)

        base_cmd = Path(tokens[0]).name.lower()
        if base_cmd.endswith(".exe"):
            base_cmd = base_cmd[:-4]

        if base_cmd in HARD_BLOCKED:
            return SandboxResult(error=f"'{base_cmd}' is permanently blocked", returncode=1, blocked=True)

        if base_cmd not in self.shell_whitelist:
            return SandboxResult(
                error=f"'{base_cmd}' is not in the allowed command whitelist. Allowed: {sorted(self.shell_whitelist)}",
                returncode=1, blocked=True,
            )

        if self._is_windows:
            cmd_list = ["cmd", "/c", command]
        else:
            cmd_list = ["/bin/sh", "-c", command]

        return self._run_cmd(cmd_list, extra_env=extra_env, label=base_cmd)

    def write_file(self, filename: str, content: str) -> SandboxResult:
        try:
            target = self._safe_path(filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return SandboxResult(stdout=f"File written: {filename} ({len(content)} chars)")
        except Exception as e:
            return SandboxResult(error=str(e), returncode=1)

    def read_file(self, filename: str, max_chars: int = 8000) -> SandboxResult:
        try:
            target = self._safe_path(filename)
            if not target.exists():
                return SandboxResult(error=f"File not found: {filename}", returncode=1)
            content = target.read_text(encoding="utf-8", errors="replace")
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n[truncated at {max_chars} chars]"
            return SandboxResult(stdout=content)
        except Exception as e:
            return SandboxResult(error=str(e), returncode=1)

    def list_files(self, subdir: str = ".") -> SandboxResult:
        try:
            target = self._safe_path(subdir)
            if not target.is_dir():
                return SandboxResult(error=f"Not a directory: {subdir}", returncode=1)
            entries = sorted(target.iterdir())
            lines = []
            for e in entries:
                if e.is_dir():
                    lines.append(f"DIR   {e.name}/")
                else:
                    lines.append(f"FILE  {e.name}  ({e.stat().st_size} bytes)")
            return SandboxResult(stdout="\n".join(lines) if lines else "(empty directory)")
        except Exception as e:
            return SandboxResult(error=str(e), returncode=1)

    def install_package(self, package: str) -> SandboxResult:
        import re
        if not re.match(r"^[a-zA-Z0-9_\-\.\[\]>=!<~,\s]+$", package):
            return SandboxResult(error=f"Invalid package name: {package}", returncode=1, blocked=True)
        cmd = [self.python_executable, "-m", "pip", "install", "--quiet", package]
        return self._run_cmd(cmd, label="pip", timeout_override=120.0)

    def _safe_path(self, filename: str) -> Path:
        p = (self.workdir / filename).resolve()
        if not str(p).startswith(str(self.workdir.resolve())):
            raise ValueError(f"Path escape attempt blocked: {filename}")
        return p

    def _run_cmd(self, cmd, *, extra_env=None, label="cmd", timeout_override=None) -> SandboxResult:
        timeout = timeout_override if timeout_override is not None else self.timeout
        t0 = time.perf_counter()

        env = os.environ.copy()
        for key in list(env.keys()):
            if key.startswith("AURALITE_") and "TOKEN" in key.upper():
                env.pop(key, None)
        if extra_env:
            env.update({str(k): str(v) for k, v in extra_env.items()})

        try:
            kwargs = dict(
                cwd=str(self.workdir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if self._is_windows:
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            proc = subprocess.Popen(cmd, **kwargs)
        except FileNotFoundError:
            return SandboxResult(error=f"Executable not found: {cmd[0]}", returncode=127,
                                 duration_s=time.perf_counter() - t0)
        except Exception as e:
            return SandboxResult(error=str(e), returncode=1, duration_s=time.perf_counter() - t0)

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

        _did_timeout = False
        try:
            raw_stdout, raw_stderr = proc.communicate(timeout=timeout + 2)
        except subprocess.TimeoutExpired:
            proc.kill()
            raw_stdout, raw_stderr = proc.communicate()
            _did_timeout = True
        finally:
            killed.set()

        # Also detect watchdog kill (returncode == -9 on Linux, 1 on Windows)
        actual_timed_out = _did_timeout or (proc.returncode in (-9, -15) and
                                             time.perf_counter() - t0 >= timeout - 0.5)
        return SandboxResult(stdout=self._cap(raw_stdout), stderr=self._cap(raw_stderr),
                             returncode=-1 if actual_timed_out else proc.returncode,
                             timed_out=actual_timed_out,
                             duration_s=time.perf_counter() - t0)

    def _cap(self, text: str) -> str:
        if len(text.encode("utf-8", errors="replace")) > self.max_output_bytes:
            text = text[:self.max_output_bytes // 2] + "\n[output truncated]\n"
        lines = text.splitlines()
        if len(lines) > MAX_OUTPUT_LINES:
            lines = lines[:MAX_OUTPUT_LINES] + [f"[{len(lines) - MAX_OUTPUT_LINES} more lines truncated]"]
            text = "\n".join(lines)
        return text
