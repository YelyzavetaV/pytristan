import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pytristan import nkron


def test_nkron_nd():
    x = np.random.rand(2, 3)
    y = np.random.rand(3, 2)
    z = np.random.rand(1, 2)
    t = np.random.rand(3, 3)

    assert_array_equal(x, nkron(x))
    assert_array_almost_equal(np.kron(y, x), nkron(x, y))
    assert_array_almost_equal(np.kron(z, np.kron(y, x)), nkron(x, y, z))
    assert_array_almost_equal(np.kron(t, np.kron(z, np.kron(y, x))), nkron(x, y, z, t))
