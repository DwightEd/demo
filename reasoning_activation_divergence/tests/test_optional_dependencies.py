from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_raw_cli_imports_when_sklearn_is_unavailable() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    code = """
import builtins
real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'sklearn' or name.startswith('sklearn.'):
        raise ModuleNotFoundError("blocked sklearn for raw-path regression test")
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
import functional_divergence.raw_residual_experiment
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)

    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
