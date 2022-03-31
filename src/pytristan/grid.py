"""Grid Module

This module provides API to create multi-dimensional grids (meshes). The grid is
represented by the Python class Grid that directly subclasses numpy.ndarray. All
coordinate arrays are stored in a linearized order.

- cheb - function that returns Chebyshev-Gauss-Lobatto points on an arbitrary interval.
- Grid - Python class that represents a computational grid. Subclasses numpy.ndarray.
- get_grid - get an existing grid or create a new one and `register' it to the grid
  manager.
- get_polar_grid - create and `register' a grid in a polar domain.
- drop_grid - `unregisters' grids from the grid manager.
- drop_last_grid - `unregisters' the last created grid from the manager.
- drop_all_grids - `unregisters' all grids from the manager.
"""

import inspect
import warnings
import operator
import numpy as np
from ._manager import ObjectManager

__all__ = [
    "cheb",
    "Grid",
    "_get_grid_manager",
    "get_grid",
    "get_polar_grid",
    "drop_grid",
    "drop_last_grid",
    "drop_all_grids",
]


def cheb(npts, xmin=None, xmax=None):
    """Returns (mapped) Chebyshev (Gauss-Lobatto) points.

    Standard Chebyshev points are defined in the interval [1, -1] as

        x_j = cos(pi * j / (N - 1)),

    where j = 0, ..., N - 1. They can be mapped onto an arbitrary interval [xmin, xmax]
    by

        x_j = (1 / 2) * (1.0 - x_j) * (xmax - xmin) + xmin.

    Parameters
    ----------
    npts: int
        Number of points.
    xmin: float, default=None
        Lower bound of the mapping interval.
    xmax: float, default=None
        Upper bound of the mapping interval.

    Returns
    -------
    x: numpy.ndarray
        (Mapped) Chebyshev points.

    Raises
    ------
    TypeError
        In the case if npts is not an integer.
    ValueError
        In the case if npts is a negative number.
    """
    try:
        npts = operator.index(npts)
    except TypeError as e:
        raise TypeError("npts must be an integer.") from e
    if npts < 0:
        raise ValueError(f"Number of grid points, {npts}, must be a positive integer.")

    x = np.cos(np.arange(npts) * np.pi / (npts - 1))
    if xmin is not None or xmax is not None:
        x = (1.0 - x) / 2.0 * (xmax - xmin) + xmin
    return x


class Grid(np.ndarray):
    def __new__(cls, arrs):
        mgrids = np.meshgrid(*arrs, indexing="ij")

        # Linearize grid representation. Column-major order is adopted. Each coordinate
        # array is stored in the i-th row of the object where i is the serial number
        # of the corresponding axis.
        obj = np.array([mgrid.flatten(order="F") for mgrid in mgrids])

        obj = obj.view(cls)

        obj._num = None  # Unique identifier of the grid

        obj.num_dim = obj.shape[0]  # Number of grid dimensions
        obj.npts = tuple(len(arr) for arr in arrs)

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self._num = None
        # Grid is designed in a way that the coordinate arrays are stored in a column-
        # major order. If the Grid instance is created through slicing or view-casting,
        # it becomes the responsibility of the user to ensure correct shaping and set
        # the attributes that are dependent on the ordering.
        self.num_dim = None
        self.npts = None

    # TODO: add support of custom mappers with *args (and **kwargs).
    # TODO: generate axes when not provided (arange).
    # TODO: support for axes and mappers that are just integer and callable.
    @classmethod
    def from_bounds(cls, *bounds, axes=[], mappers=[]):
        """Create a grid from axes bounds.

        Parameters
        ----------
        bound0, bound1,..., boundN: array-like
            Bounds for each dimension. Each bound must be an iterable containing three
            elements in the following order: lower bound, upper bound, number of points
            along the given direction. Number of points must be an integer.
        axes: array-like of int, default=[]
            Axes, along which mapping is to be applied.
        mappers: array-like of callable, default=[]
            Mapping functions to apply along the axes specified. Should be given in the
            same order as the corresponding axes in `axes`. A custom user-defined
            mapper must return a coordinate array.

        Returns
        -------
        Instance of Grid.

        Raises
        ------
        ValueError
            * If wrong format of bounds (see Parameters).
            * If number of indices in `axes` != number of mapping functions in
            `mappers`.
            * If repetitive indices in `axes`. This also corresponds to the case when
            there are negative indices referring to the same axes, as the positive
            ones. For instance, -1 and 2 for 3D grid.
        TypeError
            * If wrong formats of `axes` or/and `mappers` (see Parameters).
            * If elements of `axes` are not integer numbers.
        IndexError
            * If indices in `axes` are out of bounds for a grid of a given dimension.

        Examples
        --------
        >>> from pytristan import Grid
        Create a 2D grid Cartesian from bounds:
        >>> Grid.from_bounds((-1.0, 1.0, 3), (-1.0, 1.0, 3))
        [[-1.  0.  1. -1.  0.  1. -1.  0.  1.]
         [-1. -1. -1.  0.  0.  0.  1.  1.  1.]]
        >>> from pytristan import cheb
        Create a 1D grid based on Chebyshev-Gauss-Lobatto mapping:
        >>> Grid.from_bounds((0.0, 2.0, 8), axes=[0], mappers=[cheb])
        [[0.         0.09903113 0.3765102  0.77747907 1.22252093 1.6234898
          1.90096887 2.        ]]
        """
        for bound in bounds:
            if np.asarray(bound).shape != (3,):
                raise ValueError(
                    "Wrong bound format: each bound array must have a single dimension"
                    " and contain three elements in the following order: lower bound, "
                    "upper bound and number of grid points."
                )
        try:
            if len(axes) != len(mappers):
                raise ValueError(
                    "The number of axes must be equal to the number of mappers."
                )
        except TypeError as e:
            raise TypeError(
                "Axes and mappers must be one-dimensional array-like."
            ) from e

        if axes:
            axes = np.asarray(axes)
            if not np.issubdtype(axes.dtype, np.integer):
                raise TypeError("Axes' indices in `axes` must be integer numbers.")
            ndim = len(bounds)
            if ((axes > ndim - 1) | (axes < -ndim)).any():
                raise IndexError(
                    f"Axis' index out of bounds for the grid with {ndim} dimensions"
                    f" in axes. Allowed interval is from {-ndim} to {ndim - 1}."
                )
            # Convert negative indices to positive ones.
            axes[axes < 0] = axes[axes < 0] + ndim

            _, counts = np.unique(axes, return_counts=True)
            if not (counts == 1).all():
                raise ValueError(
                    "All indices in `axes` must be unique. It is advised you check "
                    "that there are no repetitive values and that negative indices, if"
                    " present, do not refer to the same axes as the positive ones."
                )

        mapdict = dict(zip(axes, mappers))
        arrs = tuple(
            mapdict[ax](*((bound[2],) + bound[:2]))
            if ax in axes
            else np.linspace(*bound)
            for ax, bound in enumerate(bounds)
        )

        return cls(arrs)

    @classmethod
    def from_arrs(cls, *arrs):
        """Create a grid from axes bounds.

        Parameters
        ----------
        arr0, arr1,..., arrN: array_like
            1D coordinate arrays.

        Returns
        -------
        Instance of Grid.

        Raises
        ------
        ValueError
            In the case if any of the arrs are not one-dimensional array-like.
        """
        for arr in arrs:
            if np.asarray(arr).ndim != 1:
                raise ValueError(
                    "Coordinate arrays must be one-dimensional array-like."
                )

        return cls(arrs)

    def __repr__(self):
        if self._num is None:
            msg = "no unique identifier"
        else:
            msg = f"unique identifier: {str(self._num)}"
        return f"Instance of {type(self).__name__} with {msg}."

    def axpoints(self, axis):
        """Get coordinates along the axis.

        Parameters
        ----------
        axis: int
            Index of the axis to get coordinates along.

        Returns
        -------
        numpy.ndarray
            Coordinates along the axis.

        Raises
        ------
        TypeError
            If axis is not integer.
        IndexError
            If axis out of bounds.
        """
        try:
            operator.index(axis)
        except TypeError as e:
            raise TypeError("Axis' index axis must be an integer.") from e
        if (axis > self.num_dim - 1) | (axis < -self.num_dim):
            raise IndexError(
                f"Axis' index {axis} out of bounds for the grid with {self.num_dim} "
                f"dimensions in axes. Allowed interval is from {-self.num_dim} to "
                f"{self.num_dim - 1}."
            )
        if axis < 0:
            axis += self.num_dim

        imax = int(np.prod(self.npts[: axis + 1]))
        step = int(imax / self.npts[axis])

        return np.array(self[axis][:imax:step])

    def mgrids(self):
        """Recover coordinate matrices of shape (n1, n2, n3,...).

        Equivalent to the output of numpy.meshgrid(x, y, ..., indexing='ij'), where
        x, y, ... are the coordinate arrays.

        Returns
        -------
        list containing 2D numpy.ndarray
            List of meshgrid matrices.
        """
        return [mat.reshape(self.npts, order="F") for mat in self]

    @property
    def num(self):
        return self._num

    # TODO: allow "registering" grids manually?
    @num.setter
    def num(self, number):
        """Setter of num property.

        get_grid is a sole allowed caller of the setter. It will assign an instance a
        unique num and register it in grid manager.
        """
        legal_caller = "get_grid"
        # Get names of a caller and its full filename.
        filename = inspect.stack()[1].filename
        funcname = inspect.stack()[1].function

        if not filename.endswith(legal_caller + ".py") and legal_caller not in funcname:
            raise RuntimeError(
                f"{funcname} in {filename} attempted to assign a name to an instance "
                f"of {type(self).__name__}. This action is only permitted to "
                f"pytristan.{legal_caller} - it properly registers an instance in "
                f"dedicated manager and allows its efficient re-usage."
            )

        self._num = number


def _get_grid_manager(_grid_manager_instance=ObjectManager()):
    """Getter for the GridManager's one and only instance."""
    return _grid_manager_instance


def get_grid(*args, from_bounds=False, axes=[], mappers=[], num=None, overwrite=False):
    """Get an existing grid or create a new one.

    Allows re-usage of the same instance of Grid during the run time using its unique
    identifier (num) (see Notes and Examples).

    Parameters
    ----------
    arg0, arg1,..., argN: array-like
        Either 1D coordinate arrays or axes' bounds. If bounds, each bound must be an
        iterable containing three elements in the following order: lower bound, upper
        bound, number of points along the given direction. Number of points must be an
        integer.
    from_bounds: bool, default=False
        Whether to create a grid from coordinate arrays or axes' bounds. Must be
        consistent with *args.
    axes: array-like of int, default=[]
        Axes, along which, mapping is to be applied. Only relevant when from_bounds is
        True.
    mappers: array-like of callable, default=[]
        Mapper functions to apply along the axes specified. Should be given in the same
        order as the corresponding axes in `axes`. A custom user-defined mapper must
        return a coordinate array.
    num: int, default=None
        Unique identifier of a grid. If no value is provided, the grid will be
        `registered' by num = max(nums) + 1, where nums are all existing identifiers
        known to the manager. If the manager is empty, num will be equal to 0. If the
        grid with a given num has already been created using get_grid, it will be
        returned.
    overwrite: bool, default=False
        Determines if the grid with identifier num (when provided) will be overwritten
        or not.

    Returns
    -------
    An instance of Grid.

    Raises
    ------
    TypeError
        If num is not integer.
    ValueError
        If grid not found by the manager and no grid data provided.

    Examples
    --------
    Re-usage of previously created instances of Grid.

    >>> from pytristan import get_grid
    Create a 2D cartesian grid from bounds. It will be automatically assigned a num=0.
    >>> grid = get_grid((-1.0, 1.0, 3), (-1.0, 1.0, 3), from_bounds=True)
    [[-1.  0.  1. -1.  0.  1. -1.  0.  1.]
     [-1. -1. -1.  0.  0.  0.  1.  1.  1.]]
    >>> print(grid.num)
    0
    Recover this grid using its unique identifier and verify it.
    >>> grid0 = get_grid(num=0)
    By default from_bounds is False, so if we don't specify that from_bounds is True,
    the following tuples will be interpreted as coordinate arrays with
    x=[-1.0, 1.0, 3.0].
    >>> get_grid((-1.0, 1.0, 3), (-1.0, 1.0, 3))
    [[-1.  1.  3. -1.  1.  3. -1.  1.  3.]
     [-1. -1. -1.  1.  1.  1.  3.  3.  3.]]
    """
    grid_manager = _get_grid_manager()

    if num is None:
        nums = grid_manager.nums()
        num = max(nums) + 1 if nums else 0
    else:
        try:
            operator.index(num)
        except TypeError as e:
            raise TypeError("Unique identifier num must be an integer") from e

    grid = getattr(grid_manager, str(num), None)

    if grid is None or overwrite:
        if not args:
            raise ValueError("Could not create a new grid - no grid data supplied.")

        if from_bounds:
            grid = Grid.from_bounds(
                *args,
                axes=axes,
                mappers=mappers,
            )
        else:
            grid = Grid.from_arrs(*args)

        grid.num = num
        setattr(grid_manager, str(num), grid)

    return grid


def get_polar_grid(nt, nr, axes=[], mappers=[], fornberg=False, num=None):
    """Creates a 2D polar grid.

    Parameters
    ----------
    nt: int
        Number of grid points in the azimuthal direction.
    nr: int
        Number of grid points in the radial direction.
    axes: array-like, default=[]
        Axes along which a mapping is to be applied.
    mappers: array-like, default=[]
        Mappers to apply along the axes specified. Should be given in the same order
        as the corresponding axes in `axes`.
        To apply Chebyshev mapping, cheb function should be passed as an element of
        mappers. Arbitrary mapping functions are supported as well. User must implement
        a Python function that implements a mapping and returns a np.ndarray.
    fornberg: bool, default=False
        Whether the Fornberg's grid is requested.
    num: int, default=None
        If provided, indicates the identifier of the grid to pass to the grid manager.
        If no value is provided, the grid will be `registered' by num = max(nums) + 1,
        where nums are all existing identifiers known to the manager. If the manager is
        empty, num will be equal to 0. If the grid with a given num has already been
        `registered', it will be returned.

    Notes
    -----
        `fornberg` set to True leads to that the radial coordinate r lies on the
        interval [-1, 1]. Such grid is redundant in the sense that a single point in
        Cartesian coordinates (x, y) will map to two distinct points in Fornberg's
        polar coordinates. However, it allows to use technique of constructing
        matrices of differential operators, as described in [1] and [2]. This technique
        spares one a necessity to treat the pole (r=0) in circular coordinate system as
        a boundary and helps avoid the clustering of the grid points near r=0.

    References
    ----------
    .. [1] B. Fornberg, "A Pseudospectral Approach for Polar and Spherical Geometries",
           SIAM J. Sci. Comp., 16(5):1071-1081, 1995.
    .. [2] L. Trefethen, "Spectral Methods in MATLAB", SIAM, Philadelphia, 2000.

    Examples
    --------
    >>> from pytristan import get_polar_grid, cheb
    Create a 2D polar grid with no streching and r in [0, 1]
    >>> get_polar_grid(4, 6)
    [[-3.14159265 -1.57079633  0.          1.57079633 -3.14159265 -1.57079633
       0.          1.57079633 -3.14159265 -1.57079633  0.          1.57079633
      -3.14159265 -1.57079633  0.          1.57079633 -3.14159265 -1.57079633
       0.          1.57079633 -3.14159265 -1.57079633  0.          1.57079633]
     [ 0.          0.          0.          0.          0.2         0.2
       0.2         0.2         0.4         0.4         0.4         0.4
       0.6         0.6         0.6         0.6         0.8         0.8
       0.8         0.8         1.          1.          1.          1.        ]]

    Create a 2D polar grid with streching in radial direction and r in [0, 1]
    >>> get_polar_grid(4, 6, axes=[1], mappers=[cheb])
    [[-3.14159265 -1.57079633  0.          1.57079633 -3.14159265 -1.57079633
       0.          1.57079633 -3.14159265 -1.57079633  0.          1.57079633
      -3.14159265 -1.57079633  0.          1.57079633 -3.14159265 -1.57079633
       0.          1.57079633 -3.14159265 -1.57079633  0.          1.57079633]
     [ 0.          0.          0.          0.          0.0954915   0.0954915
       0.0954915   0.0954915   0.3454915   0.3454915   0.3454915   0.3454915
       0.6545085   0.6545085   0.6545085   0.6545085   0.9045085   0.9045085
       0.9045085   0.9045085   1.          1.          1.          1.        ]]

    Create a 2D polar grid with grid streching in radial direction and r in [-1, 1].
    This grid is of Fornberg type and will have nr = 12 points in the radial direction
    >>> grid = get_polar_grid(4, 6, fornberg=True, axes=[1], mappers=[cheb])
    >>> grid.axpoints(1)
    [-1.         -0.95949297 -0.84125353 -0.65486073 -0.41541501 -0.14231484
      0.14231484  0.41541501  0.65486073  0.84125353  0.95949297  1.        ]
    """
    if fornberg:
        nr *= 2
        rmin = -1.0
    else:
        rmin = 0.0

    return get_grid(
        (-np.pi, np.pi - 2.0 * np.pi / nt, nt),
        (rmin, 1.0, nr),
        from_bounds=True,
        num=num,
        axes=axes,
        mappers=mappers,
    )


def drop_grid(num=None, nitem=0):
    """Allows to remove (drop) one or more grids from the grid manager.

    num or nitem != 0 should not be both passed as arguments in the same
    call to the function (see Examples).

    Parameters
    ----------
    num: int or array-like of int
        If provided, indicates the identifiers of the grids to be dropped from
        the list of grids stored in the grid manager. Default is None.
    nitem: int
        Number of grids to drop starting from the end of the list of grids
        stored in the grid manager. Default is 0.

    Raises
    ------
    TypeError
        * If nitem is not an integer.
        * If num is neither None, nor the integer or the array-like of integers.
    ValueError
        * If nitem is not a positive number.
        * If num is not None and nitem != 0: ambiguity.
        * If num has more than one dimension.
    RuntimeWarning
        * If num is None and nitem == 0: do nothing behaviour.
        * If no grids found with indices specified in num.

    Examples
    --------
    >>> grid_manager = _get_grid_manager()
    >>> print(grid_manager.nums())
    [0, 1, 2, 3, 4]
    >>> drop_grid(num=[0, 3]) # drops grids with identifiers 0 and 3.
    >>> print(grid_manager.nums())
    [1, 2, 4]
    >>> drop_grid(2) # drops grid with identifier 3.
    >>> print(grid_manager.nums())
    [1, 4]
    >>> drop_grid(num=3, nitem=2) # raises an error
    ValueError: num can only be used alongside nitem=0
    >>> drop_grid(nitem=2) # drops the last two grids from the grid manager.
    >>> print(grid_manager.nums())
    []
    """
    try:
        nitem = operator.index(nitem)
    except TypeError as e:
        raise TypeError("Number of drop items nitem must be an integer.") from e
    if nitem < 0:
        raise ValueError("Number of drop items nitem must be a positive integer.")

    grid_manager = _get_grid_manager()
    if num is None:
        if not nitem:
            warnings.warn(
                "No grids were dropped because num is None and nitem=0.", RuntimeWarning
            )
            return  # To ensure "do-nothing" behaviour
        nums = grid_manager.nums()
        drops = nums[-1 : -nitem - 1 : -1]
    else:
        if nitem:
            raise ValueError(
                "Providing num different from None and nitem different from 0 at the "
                "same time is ambiguous. To drop N last grid(s) from the manager AND "
                "to drop grid(s) with particular identifier(s), you have to perform "
                "two consecutive calls to drop_grid (see documentation)."
            )
        drops = np.asarray(num)
        if not np.issubdtype(drops.dtype, np.integer):
            raise TypeError("num must be an integer or an array-like of integers.")
        if drops.ndim != 1:
            if not drops.ndim:
                drops = drops[np.newaxis]
            else:
                raise ValueError("num cannot have more that one dimension.")

    for drop in drops:
        try:
            delattr(grid_manager, str(drop))
        except AttributeError:
            warnings.warn(
                f"Grid with identifier {drop} could not be dropped as it's not "
                f"registered in the manager.",
                RuntimeWarning,
            )


def drop_last_grid():
    """Shortcut for dropping the last grid contained in the grid manager."""
    drop_grid(nitem=1)


def drop_all_grids():
    """Shortcut for dropping all grids contained in the grid manager."""
    drop_grid(num=_get_grid_manager().nums())
