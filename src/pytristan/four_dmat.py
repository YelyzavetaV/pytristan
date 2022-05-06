"""Fourier Differential Matrix Module

References:
    [1] L.N. Trefethen, "Spectral Methods in MATLAB". SIAM, Philadelphia, 2000.
"""

import numpy as np
from scipy.fft import fft, ifft
from ._base_dmat import BaseDiffMat


class FourDiffMat(BaseDiffMat):
    def __new__(cls, grid, order, axis):
        return super().__new__(cls, grid, order, axis)

    def _mat(self, grid, order, axis):
        """Computes 1D Fourier differential matrix.

        References
        ----------
        [1] L.N. Trefethen, "Spectral Methods in MATLAB". SIAM, Philadelphia, 2000.
        """
        nt = grid.npts[axis]
        # Modes should be arranged in the following order to be passed to ifft:
        # zero harmonic, positive harmonics, negative harmonics in the ascending
        # order.
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
