import pytest
import numpy as np
from numpy.polynomial.chebyshev import chebpts2
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_array_almost_equal,
)
from pytristan import (
    cheb,
    Grid,
    _get_grid_manager,
    get_grid,
    get_polar_grid,
    drop_grid,
    drop_last_grid,
    drop_all_grids,
)


@pytest.mark.parametrize(
    "npts, xmin, xmax",
    [
        (10.0, None, None),
        (-10, None, None),
        (10, 0.0, None),
        (10, None, 0.0),
    ],
)
def test_cheb_invalid_args(npts, xmin, xmax):
    with pytest.raises((ValueError, TypeError)):
        cheb(npts, xmin, xmax)


def test_cheb_pts():
    assert_almost_equal(cheb(10), chebpts2(10)[::-1])


def test_cheb_bounds():
    x = cheb(10, 0.0, 10.0)
    assert x[0] == 0.0 and x[-1] == 10.0


@pytest.mark.parametrize(
    "bounds, axes, mappers",
    [
        (([0.0, 1.0],), [], []),  # Bound must be an array-like containing 3 elements.
        (([0.0, 1.0, 2.0],), [], []),  # Number of points must be an integer.
        (([0.0, 1.0, 2.0],), [0], []),  # axes and mappers must be of the same size.
        (([0.0, 1.0, 2.0],), [0.0], [cheb]),  # axes must be integers.
        (([0.0, 1.0, 2.0],), [1], [cheb]),  # axis out of bounds.
        (([0.0, 1.0, 2.0],), [-3], [cheb]),  # (Negative indexing) axis out of bounds.
    ],
)
def test_grid_from_bounds_invalid_args(bounds, axes, mappers):
    with pytest.raises((ValueError, TypeError, IndexError)):
        Grid.from_bounds(*bounds, axes=axes, mappers=mappers)


def test_grid_from_arrs_invalid_args():
    # Error test (when coordinate array in wrong format).
    with pytest.raises(ValueError):
        Grid.from_arrs(np.arange(4).reshape(2, 2))

    with pytest.raises(ValueError):
        Grid.from_arrs(1, 2, 3)


arr = np.array([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0], [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0]])


def test_grid_from_bounds_raw():
    grid = Grid.from_bounds((0.0, 1.0, 2), (-1.0, 1.0, 3))
    assert_array_equal(arr, grid)


def test_grid_from_bounds_mappers():
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
    grid5 = Grid.from_bounds(
        (0.0, 1.0, 8), (-2.0, 2.0, 6), axes=[1, 0], mappers=[cheb, cheb]
    )

    assert_array_almost_equal(grid3, grid4)
    assert_array_almost_equal(grid3, grid5)

    grid6 = Grid.from_arrs(np.linspace(0.0, 1.0, 8), y)
    grid7 = Grid.from_bounds((0.0, 1.0, 8), (-2.0, 2.0, 6), axes=[1], mappers=[cheb])

    assert_array_almost_equal(grid6, grid7)
    # Check that negative indices are properly converted to the positive ones.
    grid8 = Grid.from_bounds((0.0, 1.0, 8), (-2.0, 2.0, 6), axes=[-1], mappers=[cheb])
    grid9 = Grid.from_bounds((0.0, 1.0, 8), (-1.0, 1.0, 3), axes=[-2], mappers=[cheb])

    assert_array_almost_equal(grid7, grid8)
    assert_array_almost_equal(grid2, grid9)


def test_grid_from_arrs():
    grid = Grid.from_arrs(np.arange(2.0), np.linspace(-1.0, 1.0, 3))
    assert_array_equal(arr, grid)


def test_grid_axpoints_raises():
    grid = Grid.from_bounds([0.0, 1.0, 2])

    with pytest.raises(TypeError):
        grid.axpoints(0.0)  # axis must be integer.

    with pytest.raises(IndexError):
        grid.axpoints(1)  # axis out of bounds.

    with pytest.raises(IndexError):
        grid.axpoints(-2)  # (Negative indexing) axis out of bounds.


def test_grid_axpoints():
    grid = Grid.from_bounds([0.0, 1.0, 2], [-1.0, 1.0, 3])

    assert_array_almost_equal(grid.axpoints(0), np.asarray([0.0, 1.0]))
    assert_array_almost_equal(grid.axpoints(1), np.asarray([-1.0, 0.0, 1.0]))
    assert_array_almost_equal(grid.axpoints(-2), np.asarray([0.0, 1.0]))


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


def test_get_grid_raises():
    with pytest.raises(TypeError):
        get_grid(num=1.0)  # num must be integer.

    with pytest.raises(ValueError):
        get_grid(num=1)  # Does not exist and no grid data provided.


def test_get_grid_num():
    # Grid retrieval test.
    grid1 = get_grid(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5), num=99)
    grid2 = get_grid(num=99)

    assert grid1.num == grid2.num and id(grid1) == id(grid2)

    drop_last_grid()


def test_get_grid_overwrite():
    # Initial grid
    grid1 = get_grid((0.0, 1.0, 5), from_bounds=True, num=2)

    # Retrieval of previous grid
    grid2 = get_grid(num=2)
    assert id(grid1) == id(grid2)

    # Overwriting of previous grid and check allocation
    grid3 = get_grid((0.0, 2.0, 6), from_bounds=True, num=2, overwrite=True)
    assert grid1.num == grid3.num and id(grid1) != id(grid3)
    assert_array_almost_equal(grid3, get_grid(num=2))

    # Drop all grids
    drop_all_grids()


def test_get_grid_polar_coords():
    grid1 = get_polar_grid(4, 6)

    assert_array_almost_equal(
        grid1.axpoints(0), np.arange(-np.pi, np.pi, 2 * np.pi / 4)
    )
    assert_array_almost_equal(grid1.axpoints(1), np.linspace(0, 1, 6))

    grid2 = get_polar_grid(4, 6, axes=[1], mappers=[cheb])

    assert_array_almost_equal(grid2.axpoints(1), cheb(6, 0.0, 1.0))

    grid3 = get_polar_grid(4, 6, fornberg=True, axes=[1], mappers=[cheb])

    assert_array_almost_equal(grid3.axpoints(1), cheb(12, -1.0, 1.0))

    drop_grid(nitem=3)


@pytest.mark.parametrize(
    "num, nitem",
    [
        (None, 1.0),  # nitem must be integer.
        ([1.0, 0], 0),  # All nums must be integer.
        (None, -1),  # nitem can't be negative.
        (0, 1),  # Forbidden to supply num and nitem != 0 at the same time.
        ([[0], [1]], 0),  # Wrong format of num.
    ],
)
def test_drop_grid_invalid_args(num, nitem):
    with pytest.raises((ValueError, TypeError)):
        drop_grid(num, nitem)


def test_drop_grid_warns():
    with pytest.warns(RuntimeWarning):
        drop_grid()  # Do nothing?

    with pytest.warns(RuntimeWarning):
        drop_grid(num=10)  # Grid with such num does not exist.


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


def test_drop_all_grids():
    for _ in range(5):
        get_grid(np.linspace(0.0, 1.0, 3))
    drop_all_grids()

    assert _get_grid_manager().nums() == []
