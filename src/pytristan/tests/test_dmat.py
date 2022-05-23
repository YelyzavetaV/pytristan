import pytest
from pytristan import Derivative


def test_derivative_invalid_args():
    with pytest.raises(TypeError):
        Derivative(disc=None, order=1.0)

    with pytest.raises(TypeError):
        Derivative(disc="fourier", order=1.0)

    with pytest.raises(ValueError):
        Derivative(disc="fourier", order=-1)
