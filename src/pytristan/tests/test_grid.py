import pytest
from pytristan import get_grid, cheb, drop_grid, drop_last_grid, _get_grid_manager


def test_grid_num():
    # Grid retrieval test
    grid = get_grid(
        num=1, npts=[20, 10], geom="polar", fornberg=True, axes=[1], mappers=[cheb]
    )

    grid0 = get_grid(num=1)
    assert grid.num == grid0.num and id(grid) == id(grid0)
    drop_last_grid()


def test_drop_grid():

    for _ in range(5):
        get_grid(npts=[20, 10], geom="polar", fornberg=True, axes=[1], mappers=[cheb])

    grid_manager = _get_grid_manager()

    assert grid_manager.nums() == [0, 1, 2, 3, 4]

    drop_grid(num=[0, 3])
    assert grid_manager.nums() == [1, 2, 4]

    with pytest.warns(
        UserWarning, match="No grids were dropped because num=None and nitem=0"
    ):
        drop_grid()  # should raise a warning

    drop_last_grid()
    assert grid_manager.nums() == [1, 2]

    drop_grid(nitem=2)
    assert grid_manager.nums() == []
