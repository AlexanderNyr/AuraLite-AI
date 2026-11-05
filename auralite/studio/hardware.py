"""What this computer can do for training: memory, processor, GPU, disk (2.19).

:func:`detect_hardware` reads it without extra packages (``psutil`` is used
when it is installed); :func:`measure_gflops` times a matrix product with the
threads AuraLite trains with, which is what the speed estimates of
:mod:`auralite.studio.advisor` are based on.
"""
from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Hardware", "auralite_home", "detect_hardware", "format_gb", "measure_gflops",
           "process_memory"]

_GB = 1 << 30


def auralite_home() -> Path:
    """AuraLite's own folder (``$AURALITE_HOME``, default ``~/.auralite``)."""
    root = os.environ.get("AURALITE_HOME")
    return Path(root).expanduser() if root else Path.home() / ".auralite"


def format_gb(n: float | None) -> str:
    """``"7.8 GB"`` (one decimal), ``"?"`` when unknown."""
    if n is None:
        return "?"
    return f"{n / _GB:.1f} GB"


@dataclass
class Hardware:
    """The parts of a computer that decide what it can train, and how fast."""

    system: str = ""
    cpu: str = ""
    cores: int = 1                      # physical cores (threads when unknown)
    threads: int = 1                    # logical processors
    simd: str = ""                      # "AVX512", "AVX2", "AVX", "SSE4.2", "NEON", "basic"
    x86_64_v2: bool | None = None       # SSE4.2 + POPCNT (what Qt 6.10+ needs); None: not x86
    memory_total: int = 0               # bytes
    memory_available: int = 0           # bytes free now
    gpu: str = ""                       # the GPU training uses ("" = none)
    gpu_memory: int = 0                 # bytes (0 = none, or shared with the CPU)
    accelerator: str = "cpu"            # "cuda", "mps" or "cpu": where training runs
    disk_free: int = 0                  # bytes free where AuraLite keeps models
    train_threads: int = 1              # the threads AuraLite trains with
    gflops: float | None = None         # measured (measure_gflops); None = not measured
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hardware:
        known = {k: v for k, v in dict(data).items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def summary_lines(self) -> list[str]:
        """The computer in a few lines (the Studio's Start page, ``auralite doctor``)."""
        cpu = self.cpu or "unknown processor"
        lines = [f"Processor: {cpu} - {self.cores} cores / {self.threads} threads"
                 + (f", {self.simd}" if self.simd else "")]
        memory = f"Memory: {format_gb(self.memory_total)}"
        if self.memory_available:
            memory += f" ({format_gb(self.memory_available)} free now)"
        lines.append(memory)
        if self.gpu:
            vram = f", {format_gb(self.gpu_memory)}" if self.gpu_memory else ""
            lines.append(f"GPU: {self.gpu}{vram} - training runs on it ({self.accelerator})")
        else:
            lines.append("GPU: none that PyTorch can train on - training runs on the processor")
        if self.disk_free:
            lines.append(f"Disk: {format_gb(self.disk_free)} free for models ({auralite_home()})")
        if self.gflops:
            lines.append(f"Speed test: {self.gflops:,.0f} GFLOP/s "
                         f"({'GPU' if self.accelerator != 'cpu' else f'{self.train_threads} threads'})")
        if self.system:
            lines.append(f"System: {self.system}")
        return lines + [f"Note: {note}" for note in self.notes]


# ----------------------------------------------------------------------
#  Pieces
# ----------------------------------------------------------------------
def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _system_name() -> str:
    system = platform.system()
    if system == "Linux":
        try:
            pretty = platform.freedesktop_os_release().get("PRETTY_NAME", "")
        except OSError:
            pretty = ""
        return f"Linux {platform.release()}" + (f" ({pretty})" if pretty else "")
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    if system == "Windows":
        return f"Windows {platform.release()} ({platform.version()})"
    return f"{system} {platform.release()}".strip()


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True,
                              timeout=5, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _cpu_name(cpuinfo: str) -> str:
    match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
    if match:
        return " ".join(match.group(1).split())
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return " ".join(str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).split())
        except OSError:
            pass
    if sys.platform == "darwin":
        name = _sysctl("machdep.cpu.brand_string")
        if name:
            return name
    return platform.processor() or platform.machine()


def _cores(cpuinfo: str, threads: int) -> int:
    try:
        import psutil  # optional
        physical = psutil.cpu_count(logical=False)
        if physical:
            return int(physical)
    except Exception:
        pass
    pairs = set()
    physical_id = core_id = ""
    for line in cpuinfo.splitlines() + [""]:
        if line.startswith("physical id"):
            physical_id = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            core_id = line.split(":", 1)[1].strip()
        elif not line.strip() and core_id:
            pairs.add((physical_id, core_id))
            physical_id = core_id = ""
    if pairs:
        return len(pairs)
    if sys.platform == "darwin":
        value = _sysctl("hw.physicalcpu")
        if value.isdigit():
            return int(value)
    return max(1, threads)


def _windows_feature(feature: int) -> bool:
    try:
        return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(feature))  # type: ignore[attr-defined]
    except Exception:
        return False


#: IsProcessorFeaturePresent knows SSE4.2, AVX, AVX2 and AVX-512 (38-41) from Windows 10
#: 2004 (build 19041) on; older Windows answers "no" for each of them, on any processor.
WINDOWS_FEATURES_BUILD = 19041


def _windows_build() -> int:
    """Windows' build number (19045 = Windows 10 22H2); 0 elsewhere."""
    if sys.platform != "win32":
        return 0
    try:
        return int(sys.getwindowsversion().build)
    except Exception:
        return 0


def _numpy_cpu_features() -> dict[str, bool]:
    """The processor's features as NumPy read them with CPUID for its own dispatch
    (``{}`` when NumPy cannot say). Works on every system, every Windows version."""
    import importlib
    for name in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        features = getattr(module, "__cpu_features__", None)
        if isinstance(features, dict) and features:
            return {str(key): bool(value) for key, value in features.items()}
    return {}


def _simd(cpuinfo: str, *, system: str = "") -> tuple[str, bool | None]:
    """The widest vector instructions and whether the CPU is x86-64-v2 (SSE4.2 + POPCNT);
    ``None`` for the second when nothing can tell."""
    system = system or sys.platform
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64") or machine.startswith("arm"):
        return "NEON", None
    match = re.search(r"^flags\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
    if match:
        flags = set(match.group(1).split())
        v2 = {"sse4_2", "popcnt"} <= flags
        for name, flag in (("AVX512", "avx512f"), ("AVX2", "avx2"), ("AVX", "avx"),
                           ("SSE4.2", "sse4_2")):
            if flag in flags:
                return name, v2
        return "basic", v2
    features = _numpy_cpu_features()
    if "SSE2" in features:                                 # NumPy's CPUID, on x86
        v2 = features.get("SSE42", False) and features.get("POPCNT", False)
        for name, key in (("AVX512", "AVX512F"), ("AVX2", "AVX2"), ("AVX", "AVX"),
                          ("SSE4.2", "SSE42")):
            if features.get(key):
                return name, v2
        return "basic", v2
    if system == "win32" and _windows_build() >= WINDOWS_FEATURES_BUILD:
        # PF_AVX512F 41, PF_AVX2 40, PF_AVX 39, PF_SSE4_2 38. No POPCNT flag: every
        # processor with SSE4.2 has it.
        v2 = _windows_feature(38)
        for name, feature in (("AVX512", 41), ("AVX2", 40), ("AVX", 39), ("SSE4.2", 38)):
            if _windows_feature(feature):
                return name, v2
        return "basic", v2
    if system == "darwin":
        for name, key in (("AVX512", "hw.optional.avx512f"), ("AVX2", "hw.optional.avx2_0"),
                          ("AVX", "hw.optional.avx1_0"), ("SSE4.2", "hw.optional.sse4_2")):
            if _sysctl(key) == "1":
                return name, _sysctl("hw.optional.sse4_2") == "1"
    try:
        import torch
        capability = str(torch.backends.cpu.get_cpu_capability())
        if capability in ("AVX512", "AVX2"):
            return capability, True
    except Exception:
        pass
    return "", None


def _memory_total(meminfo: str) -> int:
    match = re.search(r"^MemTotal:\s*(\d+)\s*kB", meminfo, re.MULTILINE)
    if match:
        return int(match.group(1)) * 1024
    if sys.platform == "win32":
        class _Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = _Status()
        status.dwLength = ctypes.sizeof(_Status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return int(status.ullTotalPhys)
        except Exception:
            pass
    if sys.platform == "darwin":
        value = _sysctl("hw.memsize")
        if value.isdigit():
            return int(value)
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return 0


def _cuda_arch_supported(torch: Any, major: int, minor: int) -> bool:
    """Whether this PyTorch build has kernels for compute capability `major.minor`."""
    try:
        archs = list(torch.cuda.get_arch_list())
    except Exception:
        return True
    if not archs:
        return True
    capability = major * 10 + minor
    for arch in archs:
        kind, _, number = str(arch).partition("_")
        if not number.isdigit():
            continue
        value = int(number)
        if kind == "sm" and value // 10 == major and value <= capability:
            return True
        if kind == "compute" and value <= capability:      # PTX compiles for newer GPUs
            return True
    return False


def _accelerator(notes: list[str]) -> tuple[str, str, int]:
    """``(accelerator, gpu name, gpu bytes)`` of what training would run on."""
    try:
        import torch
    except Exception:
        return "cpu", "", 0
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            if _cuda_arch_supported(torch, int(props.major), int(props.minor)):
                return "cuda", str(props.name), int(props.total_memory)
            notes.append(f"{props.name} (compute capability {props.major}.{props.minor}) is too "
                         f"old for this PyTorch build: training runs on the processor")
            return "cpu", "", 0
    except Exception:
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps", "Apple GPU (Metal)", 0
    except Exception:
        pass
    return "cpu", "", 0


def _disk_free(folder: Path) -> int:
    probe = folder
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return int(shutil.disk_usage(probe).free)
    except OSError:
        return 0


def _train_threads() -> int:
    env = os.environ.get("AURALITE_NUM_THREADS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    try:
        import torch
        return max(1, int(torch.get_num_threads()))
    except Exception:
        return max(1, os.cpu_count() or 1)


def detect_hardware(*, gflops: float | None = None) -> Hardware:
    """The computer as AuraLite sees it (fast: no benchmark unless `gflops` is given)."""
    from auralite.engine.sizing import available_memory

    cpuinfo = _read("/proc/cpuinfo") if sys.platform.startswith("linux") else ""
    meminfo = _read("/proc/meminfo") if sys.platform.startswith("linux") else ""
    threads = max(1, os.cpu_count() or 1)
    notes: list[str] = []
    simd, v2 = _simd(cpuinfo)
    accelerator, gpu, gpu_memory = _accelerator(notes)
    hw = Hardware(system=_system_name(), cpu=_cpu_name(cpuinfo), cores=_cores(cpuinfo, threads),
                  threads=threads, simd=simd, x86_64_v2=v2, memory_total=_memory_total(meminfo),
                  memory_available=int(available_memory("cpu") or 0), gpu=gpu,
                  gpu_memory=gpu_memory, accelerator=accelerator,
                  disk_free=_disk_free(auralite_home()), train_threads=_train_threads(),
                  gflops=gflops, notes=notes)
    if hw.cores > hw.threads:
        hw.cores = hw.threads
    if simd in ("SSE4.2", "basic") and accelerator == "cpu":
        hw.notes.append("no AVX: training works, but several times slower than on a CPU with AVX2")
    if v2 is False:
        hw.notes.append("no SSE4.2 / POPCNT: AuraLite Studio needs PySide6 6.9.3 here "
                        "(Qt 6.10+ needs an x86-64-v2 processor; pip install "
                        "\"PySide6-Essentials<6.10\"); the classic GUI works")
    return hw


def measure_gflops(device: str = "cpu", *, seconds: float = 1.0, size: int = 768,
                   threads: int | None = None) -> float:
    """Float32 matrix-product speed in GFLOP/s, with AuraLite's training threads.

    About a second. On a GPU the product runs in bfloat16 (what training uses
    there) when the GPU supports it.
    """
    import torch

    previous = torch.get_num_threads()
    if threads:
        torch.set_num_threads(int(threads))
    try:
        dtype = torch.float32
        if str(device).startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            size = max(size, 2048)
        elif str(device) == "mps":
            dtype = torch.float16
            size = max(size, 1024)
        a = torch.randn(size, size, device=device, dtype=dtype)
        b = torch.randn(size, size, device=device, dtype=dtype)

        def sync() -> None:
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            elif str(device) == "mps":
                torch.mps.synchronize()

        torch.mm(a, b)
        sync()
        best = 0.0
        for _ in range(3):                 # the best of three: other programs only slow it
            count = 0
            start = time.perf_counter()
            elapsed = 0.0
            while elapsed < seconds / 3 or count < 2:
                torch.mm(a, b)
                count += 1
                if str(device) != "cpu" and count % 4 == 0:
                    sync()
                elapsed = time.perf_counter() - start
            sync()
            elapsed = time.perf_counter() - start
            best = max(best, 2.0 * size ** 3 * count / max(elapsed, 1e-9) / 1e9)
        return best
    finally:
        torch.set_num_threads(previous)


def process_memory() -> int:
    """Bytes of RAM this process uses now (0 when unknown)."""
    if sys.platform.startswith("linux"):
        match = re.search(r"^VmRSS:\s*(\d+)\s*kB", _read("/proc/self/status"), re.MULTILINE)
        return int(match.group(1)) * 1024 if match else 0
    if sys.platform == "win32":
        class _Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        try:
            process = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
            if ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
                    process, ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        except Exception:
            return 0
        return 0
    try:
        import resource
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
    except Exception:
        return 0
