from pytristan import get_grid, cheb

# 2D polar grid
grid = get_grid(
    id=0, npts=[20, 10], geom="polar", fornberg=True, axes=[1], mappers=[cheb]
)
