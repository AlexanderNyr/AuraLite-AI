"""PyInstaller run-time hook of the AuraLite app (2.21, Windows): load the part of the Visual
C++ runtime the app brings for PyTorch before anything else asks for it.

PyTorch 2.13+'s torch_cpu.dll links vcruntime140_threads.dll. The PyTorch wheel does not
ship it (it expects the Visual C++ Redistributable, which an older Windows lacks), and
PyInstaller takes no DLL from Windows' folder that is not on its list, this one included:
auralite_app.spec puts the build machine's copy into the app's folder. torch loads its DLLs
with search flags that skip that folder - but a DLL loaded once by its full path serves every
later request for its name, so it is loaded here, first. (msvcp140.dll and vcruntime140*.dll
need nothing: torch loads them through ctypes, which PyInstaller points at the app's folder.)
"""
import os
import sys

if sys.platform == "win32" and getattr(sys, "frozen", False):
    import ctypes

    for _name in ("vcruntime140_threads.dll",):
        _path = os.path.join(getattr(sys, "_MEIPASS", ""), _name)
        if os.path.isfile(_path):
            try:
                ctypes.WinDLL(_path)
            except OSError:            # the self-test says which copy is loaded, if any
                pass
