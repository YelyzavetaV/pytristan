import pytest
from pytristan import get_grid, cheb, drop_grid, drop_last_grid, _get_grid_manager


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
        UserWarning, match="No grids were dropped because num is None and nitem=0."
    ):
        drop_grid()
    # Error test (num and nitem passed in the same call and nitem != 0).
    with pytest.raises(ValueError):
        drop_grid(num=2, nitem=1)
    # Warning test (trying to drop a grid that is not registered).
    with pytest.warns(UserWarning):
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
