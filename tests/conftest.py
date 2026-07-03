"""Shared pytest fixtures."""

import pytest

from vca.data_processing.synthetic import make_synthetic_nsclc


@pytest.fixture(scope="session")
def synthetic_td():
    """A small synthetic NSCLC dataset (session-scoped for speed)."""
    return make_synthetic_nsclc(200, seed=123)


@pytest.fixture(scope="session")
def split(synthetic_td):
    train, test = synthetic_td.train_test_split(0.3, seed=0)
    return train, test


def _has_numpyro() -> bool:
    try:
        import jax  # noqa: F401
        import numpyro  # noqa: F401

        return True
    except ImportError:
        return False


requires_numpyro = pytest.mark.skipif(not _has_numpyro(), reason="needs [bayes] extra")
