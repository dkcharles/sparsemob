"""Pytest collection config.

The A3 tests (tests/test_torchcoder.py, tests/test_synthgen.py) exercise the PyTorch
port. They run on whatever torch build is importable: a CPU build is enough for every
test except the single CUDA smoke test, which self-guards (``test_runs_on_cuda_if_available``
returns early when no GPU is present). We therefore collect them whenever torch imports
at all, so the core (CPU) venv covers the GPU-port logic too, and only skip when torch is
entirely unavailable.
"""


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


collect_ignore = []
if not _has_torch():
    collect_ignore += ["tests/test_torchcoder.py", "tests/test_synthgen.py"]
