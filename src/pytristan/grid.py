"""Grid Module

Module provides a Grid class. It subclasses numpy.ndarray and intends to simplify
certain computational tasks for the user. Coordinate arrays are stored and manipulated
with in row-major order. The ND numpy arrays produced by numpy.meshgrid can be retrie-
ved with mgrids method.

Relies on numpy and warnings modules.

* Grid - numpy.ndarray subclass extending its functionality to model N-dimensional
    structured grid.
"""

import inspect
import warnings
import operator
import numpy as np
from ._manager import ObjectManager

__all__ = [
    "Grid",
    "cheb",
    "get_grid",
    "drop_grid",
    "drop_last_grid",
    "_get_grid_manager",
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
        raise TypeError("npts must be an integer") from e
    if npts < 0:
        raise ValueError(f"Number of grid points, {npts}, must be a positive integer")

    x = np.cos(np.arange(npts) * np.pi / (npts - 1))
    if xmin is not None or xmax is not None:
        x = (1.0 - x) / 2.0 * (xmax - xmin) + xmin
    return x


class Grid(np.ndarray):
    def __new__(
        cls,
        arrs,
        geom="cart",
        fornberg=False,
    ):
        mgrids = np.meshgrid(*arrs, indexing="ij")

        # Linearize grid representation. Column-major order is adopted. Each coordinate
        # array is stored in the i-th row of the object where i is the serial number
        # of the corresponding axis.
        obj = np.array([mgrid.flatten(order="F") for mgrid in mgrids])

        obj = obj.view(cls)

        obj._num = None  # Unique identifier of the grid

        obj.ndims = obj.shape[0]  # Number of grid dimensions
        obj.npts = tuple(len(arr) for arr in arrs)
        obj.geom = geom
        obj.fornberg = fornberg

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self._num = None
        # Grid is designed in a way that the coordinate arrays are stored in a column-
        # major order. If the Grid instance is created through slicing or view-casting,
        # it becomes the responsibility of the user to ensure correct shaping and set
        # the attributes that are dependent on the ordering.
        self.ndims = None
        self.npts = None
        # Geometry must be inherited if slicing.
        self.geom = getattr(obj, "geom", "cart")
        self.fornberg = getattr(obj, "fornberg", False)

    # TODO: add support of custom mappers with *args (and **kwargs).
    @classmethod
    def from_bounds(cls, *bounds, geom="cart", fornberg=False, axes=[], mappers=[]):
        """Create a grid from axes bounds.

        Parameters
        ----------
        xmin: array-like
            Sequence of lower bounds for each directions in the grid.
        xmax: array-like
            Sequence of upper bounds for each directions in the grid.
        npts: array-like
            Sequence of number of points along each direction in the grid.
        geom: str, default="cart"
            Geometry of the grid. Default is "cart" stands for Cartesian.
        fornberg: bool, default=False
            Whether the Fornberg method is to be applied when constructing differential
            operators based on this grid.
        axes: array-like, default=[]
            Axes, along which, mapping is to be applied.
        mappers: array-like, default=[]
            Mappers to apply along the axes specified. To apply Chebyshev mapping,
            pytristan.cheb function should be passed as an element of mappers.
            Arbitrary mapping functions are supported as well. User must implement a
            Python callable that implements a mapping and returns a np.ndarray.

        Returns
        -------
        Instance of Grid.

        Raises
        ------
        ValueError
            In the following cases:
            - when len(xmin), len(xmax) and len(npts) don't match
            - when len(axes) != len(mappers)

        Examples
        --------
        Create a 2D grid from bounds and apply Gauss-Lobatto mapping along y-axis:

        >>> Grid.from_bounds((0, 10), (1, 15), (3, 9), axes=(1,), mappers=(cheb,))
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
        for bound in bounds:
            if np.asarray(bound).ndim != 1 or np.asarray(bound).size != 3:
                raise ValueError(
                    "Each bound array must have a single dimension and contain three "
                    "elements in the following order: lower bound, upper bound and "
                    "number of grid points."
                )

        mapdict = dict(zip(axes, mappers))
        arrs = tuple(
            mapdict[ax](*((bound[2],) + bound[:2]))
            if ax in axes
            else np.linspace(*bound)
            for ax, bound in enumerate(bounds)
        )

        return cls(arrs, geom, fornberg)

    @classmethod
    def from_arrs(cls, *arrs, geom="cart", fornberg=False):
        """Create a grid from axes bounds.

        Parameters
        ----------
        arrs: iterable containing 1D array-like
            1D coordinate arrays.
        geom: str, default="cart"
            Geometry of the grid. Default is "cart" stands for Cartesian.
        fornberg: bool, default=False
            Whether the Fornberg method is to be applied when constructing differential
            operators based on this grid.

        Returns
        -------
        Instance of Grid.
        """
        arrs = list(map(np.asarray, arrs))
        for arr in arrs:
            if arr.ndim != 1:
                raise ValueError(
                    "Coordinate arrays must be one-dimensional array-like."
                )
            # Check if a coordinate only appears once in a coordinate array. Error or
            # warning?
            # TODO: more info?
            _, counts = np.unique(arr, return_counts=True)
            if not (counts == 1).all():
                warnings.warn(
                    "Detected repeated occurrence of coordinate in the same coordinate"
                    " array.",
                    RuntimeWarning,
                )

        return cls(arrs, geom, fornberg)

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
        """
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
    grid_manager = _get_grid_manager()
    # We need to ensure that nitem is positive to avoid unexpected content of drops
    # when slicing nums. This is the reason why I'm also checking that nitem is integer
    # - to avoid triggering a TypeError when checking that nitem is not negative.
    try:
        nitem = operator.index(nitem)
    except TypeError as e:
        raise TypeError("nitem must be an integer.") from e
    if nitem < 0:
        raise ValueError("nitem must be a positive integer.")

    if num is None:
        if nitem == 0:
            warnings.warn(
                "No grids were dropped because num is None and nitem=0.", RuntimeWarning
            )
            return  # To ensure "do-nothing" behaviour
        nums = grid_manager.nums()
        drops = nums[-1 : -nitem - 1 : -1]
    else:
        if nitem > 0:
            raise ValueError(
                "Providing num different from None and nitem different from 0 at the "
                "same time is ambiguous. To drop N last grid(s) from the manager AND "
                "to drop grid(s) with particular identifier(s), you have to perform "
                "two consecutive calls to drop_grid (see documentation)."
            )
        # This way we first check if num is just integer. operator.index is more broad
        # than isinstance(num, int). The latter will return False when some specialized
        # integer type is passed (for example np.int16(num)).
        #
        # If num is not an integer to begin with, it's gonna be too hideous to try to
        # check its type. I propose we create an array and allow trying to drop the
        # grids, and if they don't exist in manager, it doesn't matter, what's the type
        # of num. This is rather a "permissive" way to proceed. However, checking the
        # ndim of drops will trigger an error in some cases when the user passes num
        # of weird type - string for example.
        try:
            operator.index(num)
        except TypeError:
            drops = np.asarray(num)
        else:
            drops = np.asarray([num])
        if drops.ndim == 0:
            raise TypeError("num must be an interger of an array-like of integers.")

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


def get_grid(
    *args,
    from_bounds=False,
    num=None,
    geom="cart",
    fornberg=False,
    axes=[],
    mappers=[],
):
    """Getter and registrator of numerical grids.

    Allows re-usage of the same instance of Grid anywhere in the program at run time
    using unique identifier (num) of this instance (see Notes and Examples).

    Parameters
    ----------
    num: int or None
    arrs: iterable filled with 1D array-like or None
        If provided, 1D arrays must represent the axpoints of the grid. Default is None.
    xmin: array-like or None
        If provided, sequence of lower bounds for each directions in the grid. Default
        is None.
    xmax: array-like or None
        If provided, sequence of upper bounds for each directions in the grid. Default
        is None.
    npts: array-like or None
        If provided, sequence of number of points along each direction in the grid.
        Default is None.
    geom: str
        Geometry of the grid. Default is "cart" (Cartesian). Will be ignored, if an
        instance is retrieved from a manager by its id (this instance will have its own
        value of geom).
    fornberg: bool
        Whether the Fornberg grid is requested. Default is False. Will be ignored, if an
        instance is retrieved from a manager by its num (this instance will have its own
        value of fornberg).
    axes: array-like
        Axes, along which, mapping is to be applied. Default is an empty list.
    mappers: array-like
        Mappers to apply along the axes specified.
        To apply Chebyshev mapping, cheb function should be passed as an element of
        mappers.
        Arbitrary mapping functions are supported as well. User must implement a
        Python function that implements a mapping and returns a np.ndarray. Default is
        an empty list.

    Examples
    --------
    Re-usage of previously created instances of Grid.

    >>> from pytristan import get_grid, cheb
    >>> # Create some 2D grid.
    >>> grid = get_grid(
            xmin=[-1, -1],
            xmax=[1, 1],
            npts=[100, 50],
            axes=[0, 1],
            mappers=[cheb, cheb],
        )
    >>> # Since it's the first instance of Grid create at the run time, will have num
    >>> # equal to 0.
    >>> print(grid.num)
    0
    >>> grid0 = get_grid(num=0)
    >>> # Assert will pass.
    >>> assert grid.num == grid0.num and id(grid) == id(grid0)
    >>> # Create some other grid.
    >>> grid = get_grid(
            xmin=[0],
            xmax=[10],
            npts=[11],
        )
    >>> # This instance will have id equal to 1.
    >>> grid1 = get_grid(num=1)
    >>> assert grid.num == grid1.num and id(grid) == id(grid1)
    >>> # Other previously created instances can be retrieved at any time.
    >>> grid = get_grid(num=0)
    >>> assert grid.num == grid0.num and num(grid) == num(grid0)

    Usage of `fornberg` flag.

    Notes
    -----
    If an instance of a grid with an identifier (num) provided by the user has already
    been created using get_grid, this same instance will be returned. If num is not
    provided or no instance under num provided exists, a new instance will be created.
    An identifier of a newly created instance will be that provided by a user (in the
    case it was provided), 0 if manager's collection is empty, or max(ids) + 1, where
    ids is a list of all ids known to the manager.
    """

    grid_manager = _get_grid_manager()

    if num is None:
        nums = grid_manager.nums()
        num = max(nums) + 1 if nums else 0

    grid = getattr(grid_manager, str(num), None)

    if grid is None:
        if not args:
            raise ValueError("Could not create a new grid - no grid data supplied.")
        # TODO: it'd be good to use `switch` here when moved to Python 3.10.
        if "polar" in geom:
            if len(args) != 2:
                raise ValueError(
                    f"Expected two positional arguments, since polar coordinate system"
                    f" has two dimensions. Got {len(args)} instead."
                )
            for arg in args:
                if np.asarray(arg).ndim != 0 or np.asarray(arg).size != 1:
                    raise ValueError(
                        "Grid parameters along each axis, in the special case of polar"
                        " coordinate system, must be integers denoting the number of "
                        "grid points in a given direction."
                    )
            nt, nr = np.intp(args)
            if fornberg:
                nr *= 2
            tmin = -np.pi
            tmax = np.pi - 2.0 * np.pi / nt
            rmin = -1.0 if fornberg else 0.0
            rmax = 1.0

            grid = Grid.from_bounds(
                (tmin, tmax, nt),
                (rmin, rmax, nr),
                geom=geom,
                fornberg=fornberg,
                axes=axes,
                mappers=mappers,
            )
        else:
            if from_bounds:
                grid = Grid.from_bounds(
                    *args,
                    geom=geom,
                    fornberg=fornberg,
                    axes=axes,
                    mappers=mappers,
                )
            else:
                grid = Grid.from_arrs(*args, geom=geom, fornberg=fornberg)

        grid.num = num
        setattr(grid_manager, str(num), grid)

    return grid
