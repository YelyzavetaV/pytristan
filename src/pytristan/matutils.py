from string import ascii_lowercase
from functools import reduce
from operator import add
import numpy as np

__all__ = ["nkron"]


def nkron(*mats):
    """Computes chain Kronecker product of n matrices.

    In the chain of Kronecker products, matrices appear in the order reversed to that
    of mats. For instance, for four input matrices a1, a2, a3, a4, the output is
    kron(a4, kron(a3, kron(a2, a1))).

    Parameters
    ----------
    mats: numpy.ndarrays
        Chain Kronecker product input matrices.

    Returns
    -------
    mat: numpy.ndarray
        2D numpy.ndarray of a shape (m1 * m2 * ... * mN) x (n1 * n2 * ... * nN), where
        mi and ni are a number of rows and a number of columns in i-th input array,
        respectively.
    """
    nrow, ncol = np.prod([mat.shape[0] for mat in mats]), np.prod(
        [mat.shape[1] for mat in mats]
    )
    ndim = len(mats)

    mats = mats[::-1]

    xyz = ascii_lowercase
    # Construct subscripts string for einsum, i.e. "ab,cd->acbd" for 2D grid,
    # "ab,cd,ef->acegbdfh" - for 3D grid etc.
    subscripts = ",".join(xyz[i : i + 2] for i in range(0, ndim * 2, 2)) + "->"

    ranges = [xyz[i : i + 2] for i in range(0, ndim * 2, 2)]
    for ijk in zip(*ranges):
        subscripts = reduce(add, ijk, subscripts)

    return np.einsum(subscripts, *mats).reshape(nrow, ncol)
