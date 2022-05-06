"""
Base Differential Matrix
"""
import numpy as np
from .matutils import nkron


class BaseDiffMat(np.ndarray):
    """Base class for differential operators.

    Inherits from numpy.ndarray - can be therefore treated as any numpy.ndarray.
    Cannot be instatiated (abstract class).

    Parameters
    ----------
    grid: Grid
        Numerical grid.
    order: int
        Order of derivative.
    axis: int
        Number of an axis, with respect to which to derive. Must vary in the interval
        0 <= axis <= grid.num_dim - 1.
    parity: 1, -1 or None
        In the case if Fornberg derivative is requested, parity of a function to act on
        with a differential operator. 1 stands for even and -1 - for odd. Default is
        None.
    """

    def __new__(cls, grid, order, axis, *args, **kwargs):
        obj = cls._ndmat(grid, order, axis, *args, **kwargs).view(cls)

        obj.num = None  # Unique identifier of the differential operator
        obj.order = order  # Order of derivative
        obj.axis = axis  # Axis, along which direvative is computed

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.num = None
        self.order = None
        self.axis = None

    def _mat(self, grid, order, axis, *args, **kwargs) -> np.ndarray:
        """Method to provide implementation of a differential matrix in 1D.

        Must be overwritten by each child of BaseMat, as entries of differential matrix
        are discretization dependant.
        """
        raise TypeError(
            f"Can't instantiate class {self.__name__} - implementation for the entries"
            f" of the matrix not provided."
        )

    @classmethod
    def _ndmat(cls, grid, order: int, axis: int, *args, **kwargs) -> np.ndarray:
        """Computes multidimensional differential operator from 1D operator (if needed).

        This method must not be overwritten by children of BaseMat.
        In the case if Fornberg derivative is requested, polar (or spherical) grid
        symmetry condition will be applied.

        References
        ----------
        [1] Bengt Fornberg, "A Pseudospectral Approach for Polar and Spherical
        Geometries", SIAM J. Sci. Comp. 16 (5), 1995.

        [2] Lloyd. N. Trefethen, "Spectral Methods in MATLAB". SIAM, Philadelphia, 2000.
        """
        mat = cls._mat(cls, grid, order, axis, *args, **kwargs)

        if grid.num_dim > 1:
            mats = [np.eye(grid.npts[ax]) for ax in range(grid.num_dim)]
            mats[axis] = mat
            mat = nkron(*mats)

        return mat
