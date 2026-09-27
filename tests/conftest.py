"""Shared test configuration.

Tests must not leak torch's *default dtype* between modules (a leaked float64 default silently changes every later
test). The autouse fixture restores it after each test; modules that need float64 request ``double`` explicitly.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    prev = torch.get_default_dtype()
    yield
    torch.set_default_dtype(prev)


@pytest.fixture()
def double():
    torch.set_default_dtype(torch.float64)
    yield


def pytest_collection_modifyitems(config, items):
    import torch as _t

    if not _t.cuda.is_available():
        skip = pytest.mark.skip(reason="no CUDA device")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip)
