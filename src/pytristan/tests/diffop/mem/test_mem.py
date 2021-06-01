# For code reproducibility. Needs to be executed before numpy imports
try:
    import mkl

    mkl.cbwr_set(branch="avx")
    mkl.set_num_threads(4)
except ImportError:
    print("Running without mkl code reproducibility.")

import pytest
import yaml
from pathlib import Path

import pytristan as ts

# Get current working directory.
cwd = Path(__file__).resolve().parent

infile = str(cwd) + '/infile.yaml'

with open(infile) as filename:
    config = yaml.load(filename, Loader=yaml.FullLoader)

cases = []
ids = []

for geom in config['geometries']:
    for disc, res in config['discretisations'].items():
        for check in config['checks']:
            case = {}
            case['name'] = '-'.join((geom, disc, check))

            case['discretisation'] = disc

            case['grid_params'] = {
                'geometry': geom,
                'r_min': 0.0,
                'r_max': 1.0,
                'num_points': res,
            }
            cases.append(case)
            ids.append(case['name'])


@pytest.mark.parametrize('case', cases, ids=ids)
def test(case):
    """Testing memory usage by DiffopManager"""

    discr = case['discretisation']

    # We create two Tristan grids with different identifiers.
    grids = (
        ts.Grid.from_bounds(
            (case['grid_params']['r_min'],),
            (case['grid_params']['r_max'],),
            (case['grid_params']['num_points'],),
            'grid0',
            case['grid_params']['geometry'],
        ),
        ts.Grid.from_bounds(
            (case['grid_params']['r_min'],),
            (case['grid_params']['r_max'],),
            (case['grid_params']['num_points'],),
            'grid1',
            case['grid_params']['geometry'],
        ),
    )

    # Create three sets of differential operators of orders 1-4 on each grid.
    nruns = 3
    diffops = [[], []]
    for i, grid in enumerate(grids):
        for _ in range(nruns):
            for order in range(1, 5):
                diffops[i].append(ts.diffop(order, grid, disc=discr))

    # Store ids of all operators.
    diffops_ids = [[], []]
    for n, i in enumerate(diffops):
        for j in i:
            diffops_ids[n].append(id(j))

    for nrun in range(nruns - 1):
        for order in range(1, 5):
            assert (
                diffops[0][order - 1 + 4 * nrun] is diffops[0][order - 1 + 2 * 4 * nrun]
                and diffops[1][order - 1 + 4 * nrun]
                is diffops[1][order - 1 + 2 * 4 * nrun]
                and not set(diffops_ids[0]).intersection(set(diffops_ids[1]))
            )
