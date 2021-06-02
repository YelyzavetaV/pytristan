import pytest
import numpy as np
from numpy.testing import assert_almost_equal

import pytristan as ts

gridname_1D = 'test_grid1D'
gridname_2D = 'test_grid2D'


def test_axpoints_exception():
    # Test exceptions when passing wrong axis to axpoints.
    my_grid = ts.Grid.from_bounds((-1.0, 1.0), (2.0, 4.0), (5, 5), gridname_2D)

    with pytest.raises(IndexError):
        my_grid.axpoints(3)


def test_gausslob_if_cases():
    axes = (0,)
    mappers = ('gausslob',)

    # Test npoints = 0.
    with pytest.raises(ValueError):
        my_grid = ts.Grid.from_bounds(
            (-1.0,), (2.0,), (0,), gridname_1D, axes=axes, mappers=mappers
        )

    # Test npoints = 1.
    my_grid = ts.Grid.from_bounds(
        (-1.0,), (2.0,), (1,), gridname_1D, axes=axes, mappers=mappers
    )
    assert_almost_equal(my_grid, np.array([[1.0]]), decimal=15)


def test_mismatch():
    # Test exceptions when passing mismatching dimensions to from_bounds.

    # Test len(xmin) != len(xmax).
    with pytest.raises(ValueError):
        ts.Grid.from_bounds((-1.0, 1.0), (2.0,), (5, 5), gridname_2D)

    # Test len(xmin) != len(npoints).
    with pytest.raises(ValueError):
        ts.Grid.from_bounds((-1.0, 1.0), (2.0, 4), (5,), gridname_2D)

    # Test len(xmin) != len(npoints).
    with pytest.raises(ValueError):
        ts.Grid.from_bounds((-1.0, 1.0), (2.0, 4), (5,), gridname_2D)

    # Test len(axes) != len(mappers).
    with pytest.raises(ValueError):
        ts.Grid.from_bounds(
            (-1.0, 2.0),
            (2.0, 3.0),
            (4, 3),
            gridname_2D,
            axes=(1,),
            mappers=('gausslob', 'gausslob'),
        )


def test_grid1D_uniform():
    my_grid = ts.Grid.from_bounds(
        (-1.0,),
        (2.0,),
        (6,),
        gridname_1D,
    )
    assert_almost_equal(
        my_grid, np.array([[-1.0, -0.4, 0.2, 0.8, 1.4, 2.0]]), decimal=15
    )
    assert my_grid.name == gridname_1D


def test_grid1D_chebyshev(num_regression):
    axes = (0,)
    mappers = ('gausslob',)

    my_grid = ts.Grid.from_bounds(
        (-1.0,), (2.0,), (20,), gridname_1D, axes=axes, mappers=mappers
    )
    assert_almost_equal(my_grid, [ts.gausslob(20, xlim=[-1.0, 2.0])], decimal=15)


def test_grid2D_uniform():
    my_grid = ts.Grid.from_bounds(
        (-1.0, 2.0),
        (1.0, 3.5),
        (3, 4),
        gridname_2D,
    )
    assert_almost_equal(
        my_grid,
        np.array(
            [
                [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
                [2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 3.0, 3.0, 3.0, 3.5, 3.5, 3.5],
            ]
        ),
        decimal=15,
    )
    assert my_grid.name == gridname_2D


def test_grid2D_chebyshev():
    axes = (1,)
    mappers = ('gausslob',)

    my_grid = ts.Grid.from_bounds(
        (-1.0, 2.0), (2.0, 3.0), (4, 3), gridname_2D, axes=axes, mappers=mappers
    )
    assert_almost_equal(
        my_grid,
        np.array(
            [
                [-1.0, 0.0, 1.0, 2.0, -1.0, 0.0, 1.0, 2.0, -1.0, 0.0, 1.0, 2.0],
                [2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 2.5, 3.0, 3.0, 3.0, 3.0],
            ]
        ),
        decimal=15,
    )
    assert my_grid.name == gridname_2D


def test_keyword_mappers():
    def dummy(npoints):
        return np.linspace(-1.0, 2.0, npoints)

    axes = (0, 1)
    mappers = ((dummy, 4), 'gausslob')

    my_grid = ts.Grid.from_bounds(
        (np.nan, 2.0),
        (np.nan, 3.0),
        (np.nan, 3),
        gridname_2D,
        axes=axes,
        mappers=mappers,
    )
    assert_almost_equal(
        my_grid,
        np.array(
            [
                [-1.0, 0.0, 1.0, 2.0, -1.0, 0.0, 1.0, 2.0, -1.0, 0.0, 1.0, 2.0],
                [2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 2.5, 3.0, 3.0, 3.0, 3.0],
            ]
        ),
        decimal=15,
    )


def test_from_arrs_3D():

    x = np.linspace(-1.0, 1.0, 2)
    y = np.linspace(2.0, 3.0, 2)
    z = np.linspace(-4, -5, 2)
    my_grid = ts.Grid.from_arrs((x, y, z), gridname_2D)
    assert_almost_equal(
        my_grid,
        np.array(
            [
                [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
                [2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0],
                [-4.0, -4.0, -4.0, -4.0, -5.0, -5.0, -5.0, -5.0],
            ]
        ),
        decimal=15,
    )


def test_axpoints():
    my_grid = ts.Grid.from_bounds(
        (-1.0, 2.0, -4.0),
        (1.0, 5.0, -2.0),
        (3, 4, 5),
        gridname_2D,
    )
    assert_almost_equal(my_grid.axpoints(0), np.array([-1.0, 0, 1]), decimal=15)
    assert_almost_equal(my_grid.axpoints(1), np.array([2.0, 3.0, 4.0, 5.0]), decimal=15)
    assert_almost_equal(
        my_grid.axpoints(2), np.array([-4.0, -3.5, -3.0, -2.5, -2.0]), decimal=15
    )


def test_mgrids():
    my_grid = ts.Grid.from_bounds(
        (-1.0, 2.0, -4.0),
        (1.0, 4.0, -1.0),
        (2, 3, 4),
        gridname_2D,
    )
    my_meshgrid = np.meshgrid(
        np.linspace(-1.0, 1.0, 2),
        np.linspace(2.0, 4.0, 3),
        np.linspace(-4.0, -1.0, 4),
        indexing='ij',
    )
    assert_almost_equal(my_grid.mgrids(), my_meshgrid, decimal=15)
