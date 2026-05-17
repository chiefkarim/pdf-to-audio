import numpy as np
import pytest

from app.services.tts import get_sample_rate, synthesise


@pytest.mark.slow
def test_synthesise_returns_ndarray() -> None:
    result = synthesise("Hello world.")
    assert isinstance(result, np.ndarray)


@pytest.mark.slow
def test_synthesise_dtype_float32() -> None:
    result = synthesise("Hello world.")
    assert result.dtype == np.float32


@pytest.mark.slow
def test_synthesise_nonempty_output() -> None:
    result = synthesise("Hello world.")
    assert len(result) > 0


@pytest.mark.slow
def test_synthesise_empty_input() -> None:
    result = synthesise("")
    assert isinstance(result, np.ndarray)
    assert len(result) == 0


@pytest.mark.slow
def test_sample_rate_positive() -> None:
    assert get_sample_rate() > 0
