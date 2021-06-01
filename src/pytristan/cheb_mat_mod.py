import numpy as np
import numpy.linalg as nla
from .def_mod import WP_real


def cheb_mat(x):
    npoints = len(x)
    x = x.reshape((npoints, 1))

    n = np.arange(npoints)

    c = (
        np.hstack(([2.0], np.ones(npoints - 2, dtype=WP_real), [2.0])) * (-1) ** n
    ).reshape(npoints, 1)
    X = np.tile(x, (1, npoints))
    dX = X - X.T
    # The np.eye term is to avoid division by zero on the diagonal;
    # the diagonal part of D is properly computed in the subsequent line.
    D = c * (1.0 / c.T) / (dX + np.eye(npoints))
    D -= np.diag(np.sum(D, axis=1))

    return D


class DiffMatCheb:
    def __init__(self, grid, axis=0):
        self.x = grid.axpoints(axis)
        self.geometry = grid.geom
        self.npoints = grid.npoints[axis]

    def getmat(self, order):
        D = cheb_mat(self.x)

        return nla.matrix_power(D, order)
