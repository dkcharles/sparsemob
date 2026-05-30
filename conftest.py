"""Pytest collection config.

The A3 torch/GPU tests (tests/test_torchcoder.py) require a CUDA build of PyTorch and
are meant to run only in the isolated A3 venv. The core suite is pure NumPy, so we skip
collecting the torch tests unless a CUDA-capable torch is importable.
"""


def _has_cuda_torch() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


collect_ignore = []
if not _has_cuda_torch():
    collect_ignore += ["tests/test_torchcoder.py", "tests/test_synthgen.py"]
