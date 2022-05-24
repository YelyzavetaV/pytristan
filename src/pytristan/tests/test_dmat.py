import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from pytristan import (
    nkron,
    cheb,
    Grid,
    fd_diff_mat,
    cheb_diff_mat,
    Derivative,
    NDDiffMat,
)


def test_derivative_invalid_args():
    with pytest.raises(TypeError):
        Derivative(axis=0, order=1, disc=None)

    with pytest.raises(TypeError):
        Derivative(axis=0, order=1.0, disc="four")

    with pytest.raises(ValueError):
        Derivative(axis=0, order=-1, disc="four")


uniform_3D_grid = Grid.from_bounds([-1.0, 1.0, 16], [0.0, 1.0, 12], [0.0, 2.0, 10])
mixed_non_period_2D_grid = Grid.from_bounds(
    (-1.0, 1.0, 16), (0.0, 1.0, 12), axes=[0], mappers=[cheb]
)


def test_nd_diff_mat_mixed():
    fd_d1x = fd_diff_mat(uniform_3D_grid.axpoints(0), 1)
    fd_d1y = fd_diff_mat(uniform_3D_grid.axpoints(1), 1)
    fd_d2xy = NDDiffMat(
        uniform_3D_grid,
        Derivative(disc="findiff", axis=0, order=1),
        Derivative(disc="findiff", axis=1, order=1),
    )
    assert_array_almost_equal(
        nkron(fd_d1x, fd_d1y, np.eye(uniform_3D_grid.npts[2])), fd_d2xy
    )

    fd_d3z = fd_diff_mat(uniform_3D_grid.axpoints(2), 3)
    fd_d1xd3z = NDDiffMat(
        uniform_3D_grid,
        Derivative(disc="findiff", axis=0, order=1),
        Derivative(disc="findiff", axis=2, order=3),
    )
    assert_array_almost_equal(
        nkron(fd_d1x, np.eye(uniform_3D_grid.npts[1]), fd_d3z), fd_d1xd3z
    )

    cheb_d1x = cheb_diff_mat(mixed_non_period_2D_grid.axpoints(0), 1)
    mixed_non_period_d2xy = NDDiffMat(
        mixed_non_period_2D_grid,
        Derivative(disc="cheb", axis=0, order=1),
        Derivative(disc="findiff", axis=1, order=1),
    )
    assert_array_almost_equal(nkron(cheb_d1x, fd_d1y), mixed_non_period_d2xy)
