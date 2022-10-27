import warnings
from operator import index
import numpy as np


class ObjectManager:
    """Class to store and manage existing instances of objects."""

    def nums(self):
        """Get a list of ids of all stored instances."""

        nums = [
            [int(s) for s in key.split(sep="-") if s.isdigit()]
            for key in vars(self).keys()
        ]

        return [num[0] if len(num) == 1 else num for num in nums]


def _drop_items(manager, num=None, nitem=0):
    try:
        nitem = index(nitem)
    except TypeError as e:
        raise TypeError("Number of drop items nitem must be an integer.") from e
    if nitem < 0:
        raise ValueError("Number of drop items nitem must be a positive integer.")

    if num is None:
        if not nitem:
            warnings.warn(
                "No grids were dropped because num is None and nitem=0.", RuntimeWarning
            )
            return  # To ensure "do-nothing" behaviour
        nums = manager.nums()
        drops = nums[-1 : -nitem - 1 : -1]
    else:
        if nitem:
            raise ValueError(
                "Providing num different from None and nitem different from 0 at the "
                "same time is ambiguous. To drop N last grid(s) from the manager AND "
                "to drop item(s) with particular identifier(s), you have to perform "
                "two consecutive calls to drop_items (see documentation)."
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
            delattr(manager, str(drop))
        except AttributeError:
            warnings.warn(
                f"Item with identifier {drop} could not be dropped as it's not "
                f"registered in the manager.",
                RuntimeWarning,
            )
