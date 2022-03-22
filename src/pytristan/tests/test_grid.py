import pytest
import numpy as np
from numpy.polynomial.chebyshev import chebpts2
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_array_almost_equal,
)
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
        Grid.from_bounds(0, (0.0, 1.0, 2.0))
        Grid.from_bounds((0.0, 1.0), (0.0, 1.0))


def test_grid_from_arrs_raises():
    # Error test (when coordinate array in wrong format).
    with pytest.raises(ValueError):
        Grid.from_arrs(np.arange(4).reshape(2, 2))
        Grid.from_arrs(1, 2, 3)
    with pytest.warns(RuntimeWarning):
        Grid.from_arrs(np.zeros(2))


arr = np.array([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0], [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0]])


def test_grid_from_bounds_raw():
    grid = Grid.from_bounds((0.0, 1.0, 2), (-1.0, 1.0, 3))
    assert_array_equal(arr, grid)


def test_grid_from_bounds_cheb():
    x = chebpts2(8)[::-1]
    x = (1.0 - x) / 2.0
    y = chebpts2(6)[::-1]
    y = (1.0 - y) / 2.0 * 4.0 - 2.0

    grid1 = Grid.from_arrs(x, np.array([-1.0, 0.0, 1.0]))
    grid2 = Grid.from_bounds((0.0, 1.0, 8), (-1.0, 1.0, 3), axes=[0], mappers=[cheb])

    assert_array_almost_equal(grid1, grid2)

    grid3 = Grid.from_arrs(x, y)
    grid4 = Grid.from_bounds(
        (0.0, 1.0, 8), (-2.0, 2.0, 6), axes=[0, 1], mappers=[cheb, cheb]
    )

    assert_array_almost_equal(grid3, grid4)


def test_grid_from_arrs():
    grid = Grid.from_arrs(np.arange(2.0), np.linspace(-1.0, 1.0, 3))
    assert_array_equal(arr, grid)


def test_get_grid_from_bounds():
    grid1 = Grid.from_bounds((0.0, 1.0, 2), (-2.0, 2.0, 4))
    grid2 = get_grid((0.0, 1.0, 2), (-2.0, 2.0, 4), from_bounds=True)

    assert_array_almost_equal(grid1, grid2)

    drop_last_grid()


def test_get_grid_from_arrs():
    grid1 = Grid.from_arrs(np.arange(3), np.linspace(-1, 1, 4))
    grid2 = get_grid(np.arange(3), np.linspace(-1, 1, 4))

    assert_array_almost_equal(grid1, grid2)

    drop_last_grid()


def test_get_grid_num():
    # Grid retrieval test.
    grid1 = get_grid(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5), num=99)
    grid2 = get_grid(num=99)

    assert grid1.num == grid2.num and id(grid1) == id(grid2)

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
        get_grid(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3))

    grid_manager = _get_grid_manager()

    assert grid_manager.nums() == [0, 1, 2, 3, 4]
    # Drop grids with identifiers 0 and 3.
    drop_grid(num=[0, 3])

    assert grid_manager.nums() == [1, 2, 4]
    # Drop last grid.
    drop_last_grid()

    assert grid_manager.nums() == [1, 2]
    # Drop last two grids.
    drop_grid(nitem=2)

    assert grid_manager.nums() == []
