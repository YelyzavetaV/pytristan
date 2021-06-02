"""Differential Operator Getter Module

Module provides a getter function for the discretized differential operators. Getter is
based on a single instance of DiffopManager that serves a static storage for the diffe-
rential operators matrices.

Relies on numpy, tristan.fin_diff_mat_mod and .cheb_mat_mod.

* DiffopManager - "dummy" Python class used as a container for differential operators
    matrices.
* get_diffop_manager - getter for the instance of DiffopManager. Responsible for that
    there is the only instance of DiffopManager reused by diffop at the runtime.
* diffop - returns a matrix of a differential operator of requested order, on a given
    grid, over a given axis in n dimensions, discretized using Finite Differences or
    Chebyshev Spectral Method. In the case of finite differences discretization, diffe-
    rential operator of requested accuracy is constructed.
"""

import numpy as np
from .fin_diff_mat_mod import DiffMat
from .cheb_mat_mod import DiffMatCheb

__all__ = ['diffop']


class DiffopManager(object):
    """Dummy container for the differential operator matrices."""

    pass


def get_diffop_manager(diffop_manager_instance=DiffopManager()):
    """Getter for the DiffopManager one and only instance.

    This is where the magic happens. Python’s default arguments are evaluated once when
    the function is defined, not each time the function is called.
    See https://docs.python-guide.org/writing/gotchas/.
    """
    return diffop_manager_instance


def diffop(
    order: int,
    grid: np.ndarray,
    axis: int = 0,
    disc: str = 'findiff',
    acc: int = 2,
):
    """Getter for the differential operator.

    Parameters
    ----------
    order: int
        Order of differential operator.
    grid: np.ndarray or tristan.Grid
        Grid to base differential operator on. If np.ndarray is passed, matrices stored
        by the instance of DiffopManager cannot be reused - grid is considered "anony-
        mous". If tristan.Grid is passed, differential matrix can be identified and
        reused if it has already been constructed at the runtime.
    axis: int
        Coordinate axis defining the direction of differentiation. Default is 0.
    disc: str
        Derivative discretization scheme: 'findiff' for Finite Differences, 'cheb' for
        Chebyshev Spectral Method. Default id 'findiff'.
    acc: int
        Accuracy of approximation in Finite Differences. Default is 2.

    Returns
    -------
    _diffop: np.ndarray
        Differential operator matrix.

    Raises
    ------
    ValueError
        When disc is not recognized.
    """
    # Get an "active" instance of DiffopManager.
    diffop_manager = get_diffop_manager()

    # Each differential operator is identified by the kind of discretization, order of
    # derivative and, if bound to the tristan.Grid, name of the grid.
    name = disc + str(order)
    # Try getting the name of the grid. If it is the "plain" np.ndarray, gridname is an
    # empty string.
    gridname = getattr(grid, 'name', '')
    name += gridname

    _diffop = getattr(diffop_manager, name, None)
    if _diffop is None:
        if 'findiff' in disc:
            _diffop = DiffMat(grid, acc, axis).getmat(order)
        elif 'cheb' in disc:
            _diffop = DiffMatCheb(grid, axis).getmat(order)
        else:
            raise ValueError(f'Invalid discretization "{disc}"')

        # Differential operators will only be stored by DiffopManager if they are not
        # "anonymous". "Anonymous" operators are those not bound to any tristan.Grid,
        # as in this case it is unknown whether they were already created.
        if gridname:
            setattr(diffop_manager, name, _diffop)

    return _diffop
