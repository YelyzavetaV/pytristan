"""Differential Matrix"""
from dataclasses import dataclass, field
from typing import Union, Callable
from operator import index
import warnings
import numpy as np
from scipy.linalg import solve, toeplitz
from scipy.sparse import diags
from scipy.fft import fft, ifft
from functools import wraps
from .matutils import nkron
from ._manager import ObjectManager, _drop_items

__all__ = [
    "findiff_coeffs",
    "findiff_matrix",
    "chebyshev_matrix",
    "fourier_matrix",
    "default_discs",
    "Derivative",
    "DifferentialMatrix",
    "dmat",
    "drop_dmat",
]


def findiff_coeffs(order, stencil_len, stencil_shift=0):
    pows = np.arange(stencil_len)
    stencil = pows + stencil_shift
    lhs = np.tile(stencil, (stencil_len, 1)) ** pows[:, np.newaxis]

    rhs = np.zeros(stencil_len)
    rhs[order] = np.math.factorial(order)

    return solve(lhs, rhs)


def findiff_matrix(x, order, *, accuracy=2):
    npts = len(x)
    # TO FIX
    h = x[1] - x[0]

    stencil_len_central = 2 * ((order - 1) // 2) + 1 + accuracy
    stencil_len = order + accuracy
    stencil_shift = -(stencil_len_central - 1) // 2

    if npts < stencil_len:
        raise ValueError("Number of grid points too low for accuracy required")

    # Build diagonal skeleton.
    coeffs = findiff_coeffs(order, stencil_len_central, stencil_shift)
    mat = diags(
        coeffs,
        range(stencil_shift, stencil_len_central + stencil_shift),
        shape=(npts, npts),
    ).A

    # Build top rows and bottoms rows
    for i in range(0, -stencil_shift):
        coeffs = findiff_coeffs(order, stencil_len, -i)
        mat[i, :stencil_len] = coeffs
        mat[-i - 1, -stencil_len:] = np.flip(coeffs) * (-1) ** order

    return mat / h**order


def chebyshev_matrix(x, order):
    """Computes 1D Chebyshev differential matrix.

    References
    ----------
    [1] L.N. Trefethen, "Spectral Methods in MATLAB". SIAM, Philadelphia, 2000.
    """
    npts = len(x)
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


def _fourier_matrix(x, order):
    """Computes 1D Fourier differential matrix.

    References
    ----------
    [1] L.N. Trefethen, "Spectral Methods in MATLAB". SIAM, Philadelphia, 2000.
    """
    nt = len(x)
    # Modes should be arranged in the following order to be passed to ifft: zero
    # harmonic, positive harmonics, negative harmonics in the ascending order.
    if nt % 2:
        col = (
            np.hstack(
                (
                    1j * np.arange(int((nt - 1) / 2) + 1),
                    1j * np.arange(-int((nt - 1) / 2), 0, -1),
                )
            )
            ** order
        )
    else:
        col = 1j * np.tile(np.arange(int(nt / 2)), 2)
        col[int(nt / 2) + 1 :] = -np.flip(col[int(nt / 2) + 1 :])
        if not order % 2:
            col[int(nt / 2)] = col[int(nt / 2) + 1] - 1j
        col **= order

    return ifft(col * fft(np.eye(nt))).T.real


def fourier_matrix(x, order):
    """Computes 1D Fourier differential matrix.

    References
    ----------
    [1] L.N. Trefethen, "Spectral Methods in MATLAB". SIAM, Philadelphia, 2000.
    """
    nx = len(x)
    dx = 2 * np.pi / nx
    if order == 1:
        if nx % 2 == 0:
            col = np.hstack([0, 0.5 / np.tan(np.arange(1, nx) * 0.5 * dx)])
        else:
            raise NotImplementedError()
        col[1::2] *= -1
        row = col[[0, *np.arange(nx - 1, 0, -1)]]
        return toeplitz(col, row)
    elif order == 2:
        if nx % 2 == 0:
            col = np.hstack(
                [
                    np.pi**2 / 3 / dx**2 + 1 / 6,
                    0.5 / np.sin(np.arange(1, nx) * 0.5 * dx) ** 2,
                ]
            )
        else:
            raise NotImplementedError()
        col[::2] *= -1
        return toeplitz(col)
    else:
        raise NotImplementedError()


def custom_matrix(constructor):
    @wraps(constructor)
    def do_checks(x, order, **kwargs):
        print("Doing checks...")
        return constructor(x, order, **kwargs)

    return do_checks


_default_discs = dict(
    findiff=findiff_matrix, chebyshev=chebyshev_matrix, fourier=fourier_matrix
)


def default_discs():
    return list(_default_discs.keys())


@dataclass
class Derivative:
    axis: int
    order: int
    disc: Union[str, Callable]
    accuracy: Union[None, int] = None
    parity: Union[None, int] = None
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.disc, str) and not callable(self.disc):
            raise TypeError(
                "Derivative's discretisation must be a string or a callable."
            )
        try:
            index(self.order)
        except TypeError as e:
            raise TypeError("Derivatives' order must be an integer.") from e
        if self.order < 0:
            raise ValueError("Derivative's order must be a positive number or 0.")

    def __call__(self, grid):
        return dmat(self, grid=grid)


class DifferentialMatrix(np.ndarray):
    def __new__(cls, grid, *derivs: Derivative):
        if not derivs:
            warnings.warn(
                "Requesting a differential matrix with no derivatives specified will "
                "result into the identity matrix.",
                RuntimeWarning,
            )

        orders = [0] * grid.num_dim
        discs = [None] * grid.num_dim
        parities = [None] * grid.num_dim
        mats = [np.eye(grid.npts[axis]) for axis in range(grid.num_dim)]

        for d in derivs:
            if not isinstance(d, Derivative):
                raise TypeError(
                    "Each derivatives' parameters must be specified using the "
                    "Derivative object (see documentation)."
                )
            axis, order = d.axis, d.order
            if order:
                disc, accuracy, parity, kwargs = d.disc, d.accuracy, d.parity, d.kwargs
                try:
                    orders[axis], discs[axis], parities[axis], f = (
                        (order, disc.__name__, parity, disc)
                        if callable(disc)
                        else (order, disc, parity, _default_discs[disc])
                    )
                except KeyError as e:
                    raise ValueError(
                        f"Unknown discretisation {disc}. Available default "
                        f"discretisations are {default_discs()}."
                    ) from e
                # If the user specified the accuracy, pass it as a keyword argument.
                if accuracy is not None:
                    kwargs["accuracy"] = accuracy

                mats[axis] = f(grid.axpoints(axis), order, **kwargs)
            else:
                warnings.warn(
                    f"Requested a 0-th-order derivative along axis {axis} - this is "
                    "equivalent to acting with an identity matrix on a vector when "
                    "working in this dimension.",
                    RuntimeWarning,
                )

        # Treat special cases!
        # TODO: tempting to use match operator...
        # WIP. Should this be separated in a function?
        if "polar" in grid.geom:
            # TODO: r-dimension must come last. Do we generalize for circular
            # coordinate systems?
            rmid = int(grid.npts[1] / 2)
            if orders[1]:
                parity = parities[1]
                if parity is None:
                    warnings.warn(
                        "Radial derivative's parity not specified, using "
                        "default value 1.",
                        RuntimeWarning,
                    )
                    parity = 1

                # pmats contain a slice of a radial differential matrix related to a
                # "positive" half of an extended radial domain.
                pmats, nmats = mats.copy(), mats.copy()
                pmats[1] = pmats[1][rmid:, rmid:]

                nmats[1] = nmats[1][rmid:, rmid - 1 :: -1]
                nmats[0] = np.roll(nmats[0], int(grid.npts[0] / 2), axis=1)
                # Assemble the bits.
                obj = nkron(*pmats) + parity * nkron(*nmats)
            else:
                # No special care except that we only use "positive" 1/4 of a radial
                # differential matrix (which is identity matrix here).
                mats[1] = mats[1][rmid:, rmid:]
        # If not a special case.
        if "obj" not in locals():
            obj = nkron(*mats)

        obj = obj.view(cls)
        obj.num = None  # "Register" differential operator using diffop.
        obj.orders = orders
        obj.discs = discs
        obj.parities = parities

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.num = None
        self.orders = None
        self.discs = None
        self.parities = None


def _get_dmat_manager(_dmat_manager_instance=ObjectManager()):
    """Getter for the DiffopManager one and only instance.

    This is where the magic happens. Python’s default arguments are evaluated once when
    the function is defined, not each time the function is called.
    See https://docs.python-guide.org/writing/gotchas/.
    """
    return _dmat_manager_instance


def dmat(*derivs, num=None, overwrite=False, grid=None):
    dmat_manager = _get_dmat_manager()
    if num is None:
        nums = dmat_manager.nums()
        num = max(nums) + 1 if nums else 0
    else:
        try:
            index(num)
        except TypeError as e:
            raise TypeError("Unique identifier num must be an integer") from e

    dmat = getattr(dmat_manager, str(num), None)
    if dmat is None or overwrite:
        dmat = DifferentialMatrix(grid, *derivs)
        dmat.num = num
        setattr(dmat_manager, str(num), dmat)

    return dmat


def drop_dmat(num=None, nitem=0):
    dmat_manager = _get_dmat_manager()
    _drop_items(dmat_manager, num, nitem)
