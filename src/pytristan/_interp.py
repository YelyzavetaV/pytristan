import numpy as np
from .grid import cheb_bary_weights

__all__ = ["fft_interp", "cheb_bary_interp"]


def fft_interp(f: np.ndarray, x: np.ndarray, xnew: np.ndarray):
    """Interpolation using fast Fourier transform.

    Parameters
    ----------
    f: np.ndarray
        A field defined at points x.
    x: np.ndarray
        Old grid points.
    xnew: np.ndarray
        Evaluation points.
    """
    n, m = x.size, xnew.size
    # Get Fourier frequencies ordered as: 0, positive, negative.
    k = np.fft.fftfreq(n, 1 / n)
    # Fast Fourier transform of f: these are Fourier coefficients.
    fk = (
        np.sum(
            f[:, np.newaxis] * np.exp(-1j * x[:, np.newaxis] * np.tile(k, (n, 1))),
            axis=0,
        )
        * 2.0
        * np.pi
        / n
    )
    # Perform ifft to get physical data at new grid points.
    return (
        np.sum(
            fk[:, np.newaxis] * np.exp(1j * xnew[:, np.newaxis] * np.tile(k, (m, 1))).T,
            axis=0,
        )
        / 2
        / np.pi
    ).real


def cheb_bary_interp(f: np.ndarray, x: np.ndarray, xnew: np.ndarray):
    """Interpolation using barycentric formula.

    Parameters
    ----------
    f: np.ndarray
        A field defined at points x.
    x: np.ndarray
        Old grid points.
    xnew: np.ndarray
        Evaluation points.

    Notes
    -----
        Input data doesn't have to be defined at Chebyshev points but in this case the
        relevant node weights have to be provided.
    """
    xnew = np.array(xnew)
    if xnew.ndim == 0:
        xnew = np.array([xnew])

    m, n = x.size, xnew.size
    w = cheb_bary_weights(m)

    xnew_m = np.tile(xnew, (m, 1))
    x_m = np.tile(x.reshape(m, 1), (1, n))
    w_m = np.tile(w.reshape(m, 1), (1, n))
    f_m = np.tile(f.reshape(m, 1), (1, n))

    denom = xnew_m - x_m
    c_m = np.divide(w_m, denom, out=np.full_like(w_m, np.nan), where=(denom != 0))

    nom = np.sum(c_m * f_m, axis=0)
    denom = np.sum(c_m, axis=0)
    fnew = nom / denom

    # Treat nans (they occur where the old grid nodes coincide with the new ones).
    xc = xnew[np.isnan(fnew)]
    _, idx, _ = np.intersect1d(x, xc, return_indices=True)
    fnew[np.isnan(fnew)] = f[idx]

    return fnew
