import pytest
import numpy as np
from numpy.polynomial.chebyshev import chebpts2
from numpy.testing import assert_almost_equal, assert_array_equal
from pytristan import cheb, Grid, _get_grid_manager, get_grid, drop_grid, drop_last_grid


def test_cheb_raises():
    with pytest.raises(TypeError):
        cheb(10.0)
    with pytest.raises(ValueError):
        cheb(-10)
    with pytest.raises(TypeError):  # Error if only one bound value is provided
        cheb(10, 0.0)
        cheb(10, xmax=2.0)


def test_cheb_pts():
    assert_almost_equal(cheb(10), chebpts2(10)[::-1])


def test_cheb_bounds():
    x = cheb(10, 0.0, 10.0)
    assert x[0] == 0.0 and x[-1] == 10.0


def test_grid_from_bounds_raises():
    with pytest.raises(ValueError):
        Grid.from_bounds(0, [1, 1], 2)
        Grid.from_bounds(0, 1, 2, axes=0, mappers=[])
        Grid.from_bounds([[0], [0]], [[1], [1]], [[2], [2]])
    with pytest.raises(TypeError):
        Grid.from_bounds([0], [1], [2], axes=[0], mappers=["bad_mapper"])


def test_grid_from_arrs_raises():
    with pytest.raises(TypeError):
        Grid.from_arrs(1)
    # Error test (when coordinate array in wrong format).
    with pytest.raises(ValueError):
        Grid.from_arrs([np.arange(4).reshape(2, 2)])
        Grid.from_arrs([0])
    with pytest.warns(RuntimeWarning):
        Grid.from_arrs([np.zeros(2)])


def test_grid_from_arrs():
    arrs = [[0.0], [-1.0, 1.0]]
    assert_array_equal(
        Grid.from_arrs(arrs),
        Grid(arrs),
    )


def test_get_grid_num():
    # Grid retrieval test
    grid = get_grid(
        num=1, npts=[20, 10], geom="polar", fornberg=True, axes=[1], mappers=[cheb]
    )

    grid0 = get_grid(num=1)
    assert grid.num == grid0.num and id(grid) == id(grid0)
    drop_last_grid()


def test_drop_grid_raises():
    # Error test (check if nitem is a positive integer and a permissive check for num).
    with pytest.raises(TypeError):
        drop_grid(nitem=5.5)
        drop_grid(nitem=-1)
        drop_grid(num="10")
    # Warning test (no arguments passed).
    with pytest.warns(
        RuntimeWarning, match="No grids were dropped because num is None and nitem=0."
    ):
        drop_grid()
    # Error test (num and nitem passed in the same call and nitem != 0).
    with pytest.raises(ValueError):
        drop_grid(num=2, nitem=1)
    # Warning test (trying to drop a grid that is not registered).
    with pytest.warns(RuntimeWarning):
        drop_grid(num=10)


def test_drop_grid_num():
    for _ in range(5):
        get_grid(npts=[20, 10], geom="polar", fornberg=True, axes=[1], mappers=[cheb])

    grid_manager = _get_grid_manager()

    assert grid_manager.nums() == [0, 1, 2, 3, 4]

    # drop grids with identifiers 0 and 3
    drop_grid(num=[0, 3])
    assert grid_manager.nums() == [1, 2, 4]

    # Drop last grid
    drop_last_grid()
    assert grid_manager.nums() == [1, 2]

    # Drop last two grids
    drop_grid(nitem=2)
    assert grid_manager.nums() == []
