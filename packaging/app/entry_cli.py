"""The app's console program (``auralite.exe``): the ``auralite`` command frozen by
PyInstaller (2.21). See packaging/app/build_app.py."""
import multiprocessing
import sys

if __name__ == "__main__":
    # Worker processes (spawn) start this very program again; this turns them into workers.
    multiprocessing.freeze_support()
    from auralite.cli import main

    sys.exit(main())
