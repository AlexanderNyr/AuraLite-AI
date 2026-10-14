"""
agent/sandbox.py — Execution sandbox for AuraLite Agent.

Strategy: subprocess-based sandboxing that works on Windows 10, Linux, and macOS
without requiring Docker or root privileges.

.. warning::
    This is **risk reduction, not a security boundary**. The agent's subprocesses
    run as your user: they can read files outside the workdir (e.g. /etc/passwd),
    reach the network, and bypass the shell whitelist by executing code through
    `python -c`. Real isolation requires a container/VM with dropped privileges,
    a read-only filesystem and ``--network none``. What this module *does* give
    you:

  1. subprocess with timeout (no runaway processes) + POSIX resource limits
     (CPU time, file size, optional address-space cap) and process-group kill
     so orphaned grandchildren cannot linger;
  2. a separate working directory per session (tempdir, auto-cleaned), used as
     TMPDIR/HOME so temporary and cache files do not leak into your real home;
  3. a watchdog thread that kills the process (and its group) on timeout;
  4. a command whitelist for shell mode, with network tools removed when
     ``allow_network=False`` (best-effort: proxies are also pointed at a dead
     port, but a direct socket call can still connect);
  5. stdout/stderr captured and size-capped;
  6. path-escape prevention for the file tools (real containment, no prefix
     tricks);
  7. secret-looking environment variables (``*TOKEN*``, ``*API_KEY*``,
     ``*SECRET*``, ``*PASSWORD*``, ``*CREDENTIAL*``) are stripped from the
     child environment by default.
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

# POSIX resource limits (best-effort; ignored on Windows).
# Memory limit defaults to disabled (0) because torch-based code legitimately
# needs a few hundred MB; set AURALITE_SANDBOX_MEM_MB to cap it.
DEFAULT_FILE_LIMIT_MB = 64
_SECRET_ENV_MARKERS = ("TOKEN", "SECRET", "API_KEY", "PASSWORD", "CREDENTIAL", "PRIVATE_KEY")
_NETWORK_COMMANDS = {"curl", "wget", "ping", "nc", "ncat", "netcat", "ssh", "scp", "ftp"}

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
        memory_limit_mb: int | None = None,
        file_limit_mb: int | None = None,
        strip_secret_env: bool = True,
    ):
        self.timeout = float(timeout)
        self.shell_whitelist = shell_whitelist if shell_whitelist is not None else DEFAULT_SHELL_WHITELIST
        self.allow_network = allow_network
        self.max_output_bytes = int(max_output_bytes)
        self.python_executable = python_executable or sys.executable
        self.memory_limit_mb = int(memory_limit_mb if memory_limit_mb is not None
                                   else os.environ.get("AURALITE_SANDBOX_MEM_MB", "0") or 0)
        self.file_limit_mb = int(file_limit_mb if file_limit_mb is not None
                                 else os.environ.get("AURALITE_SANDBOX_FILE_MB", DEFAULT_FILE_LIMIT_MB) or 0)
        self.strip_secret_env = strip_secret_env

        if workdir:
            self._workdir_obj = None
            self.workdir = Path(workdir)
            self.workdir.mkdir(parents=True, exist_ok=True)
        else:
            self._workdir_obj = tempfile.TemporaryDirectory(prefix="auralite_agent_")
            self.workdir = Path(self._workdir_obj.name)

        self._is_windows = platform.system() == "Windows"
        if not allow_network:
            # Best-effort network denial: drop network tools from the whitelist
            # and point proxy variables at a dead local port. A direct socket
            # connection from python is still possible — see module docstring.
            self.shell_whitelist = set(self.shell_whitelist) - _NETWORK_COMMANDS

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
        """Resolve `filename` inside the sandbox workdir, rejecting escapes.

        NOTE: the previous check was a plain string-prefix test
        (`str(p).startswith(str(self.workdir))`), which also accepted sibling
        directories that merely *share the prefix* — e.g. workdir
        "/tmp/sbx" allowed reading "/tmp/sbx_secret/..." because
        "/tmp/sbx_secret/x".startswith("/tmp/sbx") is True. Use real path
        containment instead.
        """
        root = self.workdir.resolve()
        p = (self.workdir / filename).resolve()
        try:
            p.relative_to(root)
        except ValueError:
            raise ValueError(f"Path escape attempt blocked: {filename}") from None
        return p

    def _child_env(self, extra_env=None) -> dict:
        """Sanitized environment for the child process."""
        env = os.environ.copy()
        if self.strip_secret_env:
            for key in list(env.keys()):
                upper = key.upper()
                if any(marker in upper for marker in _SECRET_ENV_MARKERS):
                    env.pop(key, None)
        # Keep temporary files and byte-code caches inside the sandbox dir.
        tmp_dir = self.workdir / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        env.update({
            "TMPDIR": str(tmp_dir), "TEMP": str(tmp_dir), "TMP": str(tmp_dir),
            "PYTHONPYCACHEPREFIX": str(self.workdir / ".pycache"),
        })
        if not os.environ.get("AURALITE_SANDBOX_KEEP_HOME"):
            env["HOME"] = str(self.workdir)
            env["USERPROFILE"] = str(self.workdir)
        if not self.allow_network:
            # Requests-based libraries fail fast against a dead proxy; raw
            # sockets can still escape (documented limitation).
            env.update({
                "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9",
                "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
                "no_proxy": "", "NO_PROXY": "",
            })
        if extra_env:
            env.update({str(k): str(v) for k, v in extra_env.items()})
        return env

    def _preexec(self):
        """POSIX child setup: resource limits + own process group (None on Windows)."""
        if self._is_windows:
            return None
        cpu_limit = int(self.timeout) + 5
        file_limit = self.file_limit_mb * 1024 * 1024 if self.file_limit_mb > 0 else 0
        mem_limit = self.memory_limit_mb * 1024 * 1024 if self.memory_limit_mb > 0 else 0
        if not (cpu_limit or file_limit or mem_limit):
            return None

        def _apply():
            try:
                os.setsid()  # own process group → we can kill the whole tree
            except OSError:
                pass
            try:
                import resource
                if cpu_limit:
                    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 2))
                if file_limit:
                    resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
                if mem_limit:
                    resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
            except Exception:
                pass  # limits are best-effort; never block execution over them
        return _apply

    def _kill_tree(self, proc) -> None:
        """Kill the child and (on POSIX) its whole process group."""
        if not self._is_windows:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
                return
            except Exception:
                pass
        try:
            proc.kill()
        except Exception:
            pass

    def _run_cmd(self, cmd, *, extra_env=None, label="cmd", timeout_override=None) -> SandboxResult:
        timeout = timeout_override if timeout_override is not None else self.timeout
        t0 = time.perf_counter()

        env = self._child_env(extra_env)

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
            preexec = self._preexec()
            if preexec is not None:
                kwargs["preexec_fn"] = preexec
            if self._is_windows:
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            proc = subprocess.Popen(cmd, **kwargs)
        except FileNotFoundError:
            return SandboxResult(error=f"Executable not found: {cmd[0]}", returncode=127,
                                 duration_s=time.perf_counter() - t0)
        except Exception as e:
            return SandboxResult(error=str(e), returncode=1, duration_s=time.perf_counter() - t0)

        finished = threading.Event()         # set when the process completes/returns
        watchdog_killed = threading.Event()  # set only if the watchdog kills it

        def _watchdog():
            # Wait up to `timeout`; if the process has not finished by then, kill it.
            if not finished.wait(timeout=timeout):
                watchdog_killed.set()
                self._kill_tree(proc)

        wd = threading.Thread(target=_watchdog, daemon=True)
        wd.start()

        _did_timeout = False
        try:
            raw_stdout, raw_stderr = proc.communicate(timeout=timeout + 2)
        except subprocess.TimeoutExpired:
            self._kill_tree(proc)
            raw_stdout, raw_stderr = proc.communicate()
            _did_timeout = True
        finally:
            finished.set()

        # Cross-platform timeout detection: the watchdog sets its own flag when it
        # kills the process, regardless of the platform-specific return code
        # (POSIX reports -9/-15, Windows reports 1), so we never rely on the code.
        actual_timed_out = _did_timeout or watchdog_killed.is_set()
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
