"""Chebyshev Differential Matrix"""

import numpy as np
from ._base_dmat import BaseDiffMat


class ChebDiffMat(BaseDiffMat):
    def __new__(cls, grid, order, axis):
        return super().__new__(cls, grid, order, axis)

    def _mat(self, grid, order, axis):
        """Computes 1D Chebyshev differential matrix.

        References
        ----------
        [1] L.N. Trefethen, "Spectral Methods in MATLAB". SIAM, Philadelphia, 2000.
        """
        npts = grid.npts[axis]
        x = grid.axpoints(axis)

        x = x.reshape((npts, 1))
        n = np.arange(npts)

        c = (np.hstack(([2.0], np.ones(npts - 2), [2.0])) * (-1) ** n).reshape(npts, 1)
        X = np.tile(x, (1, npts))
        dX = X - X.T
        # The np.eye term is to avoid division by zero on the diagonal; the diagonal
        # part of D is properly computed in the subsequent line.
        mat = c * (1.0 / c.T) / (dX + np.eye(npts))
        mat -= np.diag(np.sum(mat, axis=1))

        return np.linalg.matrix_power(mat, order)
