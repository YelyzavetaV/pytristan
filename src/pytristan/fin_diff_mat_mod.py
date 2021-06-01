import math
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
from .def_mod import WP_real


class DiffMat:
    def __init__(self, grid, accuracy, axis=0):
        self.accuracy = accuracy
        # extract params
        self.npoints = grid.npoints[axis]
        # Only works for the uniform grid.
        self.h = (grid[axis][-1] - grid[axis][0]) / (self.npoints - 1)

    @staticmethod
    def fin_coeffs(diff_order, stencil_length, stencil_shift=0):

        stencil = np.arange(0, stencil_length) + stencil_shift

        mat = np.zeros((stencil_length, stencil_length), dtype=WP_real)

        for i in range(stencil_length):
            mat[i, :] = stencil ** i

        rhs = np.zeros(stencil_length)
        rhs[diff_order] = math.factorial(diff_order)

        return la.solve(mat, rhs)

    def getmat(self, order):
        stencil_len_central = 2 * ((order - 1) // 2) + 1 + self.accuracy
        stencil_len = order + self.accuracy
        stencil_shift = -(stencil_len_central - 1) // 2

        if self.npoints < stencil_len:
            raise ValueError(
                'Number of grid points too low for accuracy required'
            )

        # Build diagonal skeleton
        coeffs = self.fin_coeffs(order, stencil_len_central, stencil_shift)
        mat = sparse.diags(
            coeffs,
            range(stencil_shift, stencil_len_central + stencil_shift),
            shape=(self.npoints, self.npoints),
        ).toarray()

        # Build top rows and bottoms rows
        for i in range(0, -stencil_shift):
            coeffs = self.fin_coeffs(order, stencil_len, -i)
            mat[i, 0:stencil_len] = coeffs
            mat[-i - 1, -stencil_len:] = np.flip(coeffs) * (-1) ** order

        return mat / self.h ** order
