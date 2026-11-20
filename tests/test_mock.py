"""Pytest entry for the CPU mock harness. No GPU, no network, no model weights."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_notebook_end_to_end_with_mocks():
    proc = subprocess.run([sys.executable, str(ROOT / "tests" / "run_mock.py")], capture_output=True, text=True, cwd=ROOT, timeout=1500)
    tail = (proc.stdout + proc.stderr)[-3000:]
    assert proc.returncode == 0, tail
    assert "ALL CELLS OK" in proc.stdout, tail


def test_selftest_stage():
    proc = subprocess.run([sys.executable, str(ROOT / "gr00th_stress.py"), "--stage", "selftest"], capture_output=True, text=True, cwd=ROOT, timeout=900)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
