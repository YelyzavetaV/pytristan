"""Grid Module

Module provides a Grid class. It subclasses numpy.ndarray and intends to simplify
certain computational tasks for the user. Coordinate arrays are stored and manipulated
in column-major order. The ND numpy arrays produced by numpy.meshgrid can be retrie-
ved with mgrids method.

Relies on numpy and warnings modules.

* Grid - numpy.ndarray subclass extending its functionality to model N-dimensional
    structured grid.
"""

import numpy as np
from .def_mod import WP_real

__all__ = ['Grid', 'gausslob']


def gausslob(npoints, xlim=[]):
    if npoints < 1:
        raise ValueError('npoints must be >= 1.')
    elif npoints == 1:
        return np.array([1.0])

    x = np.cos(np.arange(npoints) * np.pi / (npoints - 1), dtype=WP_real)
    if xlim:
        x = (1.0 - x) / 2.0 * (xlim[1] - xlim[0]) + xlim[0]
    return x


class Grid(np.ndarray):
    def __new__(
        cls,
        arrs,
        name='',
        geom='cart',
    ):
        mgrids = np.meshgrid(*arrs, indexing='ij')

        # Linearize grid representation. Column-major order is adopted. Each coordinate
        # array is stored in the i-th row of the object where i is the serial number
        # of the corresponding axis.
        obj = np.array([mgrid.flatten(order='F') for mgrid in mgrids])

        obj = obj.view(cls)

        obj.name = name  # Unique name of the grid

        obj.ndims = obj.shape[0]  # Number of grid dimensions
        obj.npoints = tuple(len(arr) for arr in arrs)
        obj.geom = geom

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        # If the Grid instance is created through slicing, its name will be 'sub<pname>'
        # where <pname> is a name of a 'parent' Grid instance. If the Grid instance is
        # created through view-casting, name attribute will have the default value.
        try:
            self.name = 'sub' + obj.name
        except AttributeError:
            self.name = ''

        # Grid is designed in a way that the coordinate arrays are stored in a column-
        # major order. If the Grid instance is created through slicing or view-casting,
        # it becomes the responsibility of the user to ensure correct shaping and set
        # the attributes that are dependent on the ordering.
        self.ndims = None
        self.npoints = None
        # Geometry must be inherited if slicing.
        self.geom = getattr(obj, 'geom', 'cart')

    @classmethod
    def from_bounds(
        cls, xmin, xmax, npoints, name='', geom='cart', axes=[], mappers=[]
    ):
        """Create a grid from axes bounds.

        Parameters
        ----------
        xmin: array_like
            Sequence of lower bounds for each directions in the grid.
        xmax: array_like
            Sequence of upper bounds for each directions in the grid.
        npoints: array_like
            Sequence of number of points along each direction in the grid.
        name: str
            Name identifier of the grid. Default is ''.
        geom: str
            Geometry of the grid. Default is 'cart' for Cartesian.
        axes: array-like
            Axes, along which, mapping is to be applied.
        mappers: array-like
            Mappers to apply along the `axes` specified.
            To apply Gauss-Lobatto mapping, 'gausslob' should be passed as an element
            of `mappers`.
            Arbitrary mapping functions are supported as well. User must implement a
            Python function that implements a mapping and returns a np.ndarray. This
            function must be passed to from_bounds in an array-like followed by its
            arguments. Ex.: `mappers=[(my_mapper, npoints, xmin, eta)]` - it is assumed
            here that my_mapper function requires the number of grid points, the lower
            bound and some mapping constant `eta`.

        Returns
        -------
        Instance of Grid.

        Raises
        ------
        ValueError
            In the following cases:
            - when len(xmin), len(xmax) and len(npoints) don't match
            - when len(axes) != len(mappers)

        Examples
        --------
        Create a 2D grid from bounds and apply Gauss-Lobatto mapping along y-axis:

        >>> Grid.from_bounds((0, 10), (1, 15), (3, 9), axes=(1,), mappers=('gausslob',))
        [[ 0.          0.5         1.          0.          0.5         1.
           0.          0.5         1.          0.          0.5         1.
           0.          0.5         1.          0.          0.5         1.
           0.          0.5         1.          0.          0.5         1.
           0.          0.5         1.        ]
        [10.         10.         10.         10.19030117 10.19030117 10.19030117
           10.73223305 10.73223305 10.73223305 11.54329142 11.54329142 11.54329142
           12.5        12.5        12.5        13.45670858 13.45670858 13.45670858
           14.26776695 14.26776695 14.26776695 14.80969883 14.80969883 14.80969883
           15.         15.         15.        ]]
        """
        if len(xmin) != len(xmax) or len(xmin) != len(npoints):
            raise ValueError('Lengths of xmin, xmax and npoints do not match.')

        if axes or mappers:
            if len(axes) != len(mappers):
                raise ValueError('Lengths of axes and mappers do not match.')
            # Extract string keywords from mappers into a dictionary where keys are the
            # indices of those keywords in mappers.
            keywords = {i: m for i, m in enumerate(mappers) if isinstance(m, str)}
            # Replace 'gausslob' keyword with a tuple of gausslob function followed by
            # its arguments.
            mappers = [
                (gausslob, npoints[axes[i]], (xmin[axes[i]], xmax[axes[i]]))
                if 'gausslob' in keywords.get(i, '')
                else mapper
                for i, mapper in enumerate(mappers)
            ]
            # Collect all mapping data in one dict. Keys are the axes to map along, and
            # values are the mappers with their arguments.
        mapdict = dict(zip(axes, mappers))

        # Build 1D coordinate arrays. If no mappers are specified in the direction, it
        # will be equispaced. Otherwise, attempt will be made to call a mapper function
        # with the arguments provided, as they were ordered by the user. If 'gausslob'
        # keywords is provided instead of a mapper, arguments will be taken care of
        # implicitly.
        arrs = tuple(
            mapdict[ax][0](*mapdict[ax][1:])
            if ax in mapdict.keys()
            else np.linspace(i, j, n)
            for ax, (i, j, n) in enumerate(zip(xmin, xmax, npoints))
        )

        return cls(arrs, name, geom)

    @classmethod
    def from_arrs(cls, arrs, name='', geom='cart'):
        """Create a grid from axes bounds.

        Parameters
        ----------
        arrs: iterable filled with 1D array-like
            1D arrays must represent the axpoints of the grid.
        name: str
            Name identifier of the grid.

        Returns
        -------
        Constructor of Grid.
        """
        return cls(arrs, name, geom)

    def axpoints(self, axis):
        """Get coordinates along the axis.

        Parameters
        ----------
        axis: int
            Index of the axis to get coordinates along.

        Returns
        -------
        np.ndarray
            Coordinates along the axis.

        Raises
        ------
        IndexError from ValueError
            In the case if axis index is out of bounds.
        """
        try:
            self.npoints[axis]
        except IndexError as err:
            raise err from ValueError(f'grid does not have axis {axis} (out of bounds)')

        imax = int(np.prod(self.npoints[: axis + 1]))
        step = int(imax / self.npoints[axis])

        return np.array(self[axis][:imax:step])

    def mgrids(self):
        """Recover coordinate matrices of shape (n1, n2, n3,...) (the output of)
            numpy.meshgrid(X, Y, ..., indexing='ij').

        Returns
        -------
        list
            List of meshgrid matrices.
        """

        return [mat.reshape(self.npoints, order='F') for mat in self]
