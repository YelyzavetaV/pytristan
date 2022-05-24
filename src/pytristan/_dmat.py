"""Differential Matrix"""
from dataclasses import dataclass, field
from typing import Union, Callable
from operator import index
import warnings
import numpy as np
from scipy.linalg import solve
from scipy.sparse import diags
from scipy.fft import fft, ifft
from .matutils import nkron

__all__ = [
    "fd_coeffs",
    "fd_diff_mat",
    "cheb_diff_mat",
    "four_diff_mat",
    "default_discs",
    "Derivative",
    "NDDiffMat",
]


def fd_coeffs(order, stencil_len, stencil_shift=0):
    pows = np.arange(stencil_len)
    stencil = pows + stencil_shift
    lhs = np.tile(stencil, (stencil_len, 1)) ** pows[:, np.newaxis]

    rhs = np.zeros(stencil_len)
    rhs[order] = np.math.factorial(order)

    return solve(lhs, rhs)


def fd_diff_mat(x, order, accuracy=2):
    npts = len(x)
    # TO FIX
    h = x[1] - x[0]

    stencil_len_central = 2 * ((order - 1) // 2) + 1 + accuracy
    stencil_len = order + accuracy
    stencil_shift = -(stencil_len_central - 1) // 2

    if npts < stencil_len:
        raise ValueError("Number of grid points too low for accuracy required")

    # Build diagonal skeleton.
    coeffs = fd_coeffs(order, stencil_len_central, stencil_shift)
    mat = diags(
        coeffs,
        range(stencil_shift, stencil_len_central + stencil_shift),
        shape=(npts, npts),
    ).A

    # Build top rows and bottoms rows
    for i in range(0, -stencil_shift):
        coeffs = fd_coeffs(order, stencil_len, -i)
        mat[i, :stencil_len] = coeffs
        mat[-i - 1, -stencil_len:] = np.flip(coeffs) * (-1) ** order

    return mat / h**order


def cheb_diff_mat(x, order):
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


def four_diff_mat(x, order):
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


DEFAULT_DISCS = dict(findiff=fd_diff_mat, cheb=cheb_diff_mat, four=four_diff_mat)


def default_discs():
    return list(DEFAULT_DISCS.keys())


@dataclass
class Derivative:
    axis: int
    order: int
    disc: Union[str, Callable]
    accuracy: Union[None, int] = None
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


class NDDiffMat(np.ndarray):
    def __new__(cls, grid, *derivs: Derivative):
        if not derivs:
            warnings.warn(
                "Requesting a differential matrix with no derivatives specified will "
                "result into the identity matrix.",
                RuntimeWarning,
            )

        orders = [0] * grid.num_dim
        discs = [None] * grid.num_dim
        mats = [np.eye(grid.npts[axis]) for axis in range(grid.num_dim)]

        for d in derivs:
            if not isinstance(d, Derivative):
                raise TypeError(
                    "Each derivatives' parameters must be specified using the "
                    "Derivative object (see documentation)."
                )
            axis, order = d.axis, d.order
            if order:
                disc, accuracy, kwargs = d.disc, d.accuracy, d.kwargs
                try:
                    orders[axis], discs[axis], f = (
                        (order, disc.__name__, disc)
                        if callable(disc)
                        else (order, disc, DEFAULT_DISCS[disc])
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
                    f"Requested a 0-th-order derivative along axis {axis}.",
                    RuntimeWarning,
                )

        obj = nkron(*mats).view(cls)

        obj.num = None
        obj.orders = orders
        obj.discs = discs

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.num = None
        self.orders = None
        self.discs = None
