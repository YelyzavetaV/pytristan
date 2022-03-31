[![Build and tests](https://github.com/YelyzavetaV/pytristan/actions/workflows/pytristan.yml/badge.svg)](https://github.com/YelyzavetaV/pytristan/actions/workflows/pytristan.yml)
[![codecov](https://codecov.io/gh/YelyzavetaV/pytristan/branch/main/graph/badge.svg?token=KWLTXFRDUE)](https://codecov.io/gh/YelyzavetaV/pytristan)

**pytristan** is a Python package, based on NumPy, that provides *tools for numerical solution of partial differential equations*, such as

* Multi-dimensional meshes
* *(coming soon)* Multi-dimensional differential operators (DO), constructed using different approximation techniques, such as
  - *Fourier spectral method*
  - *Chebyshev collocation method*
  - *Finite-differences method*

# Main features

## *Meshes and differential operators **are** NumPy arrays*

The dedicated mesh's (`Grid`) and DO's (`FourMat`, `ChebMat`, `FinDiffMat`) objects subclass `numpy.ndarray`, benefitting from all of its powerful tools and allowing for convenient and intuitive usage.

## *Simple API to treat cases of special geometry*

pytristan provides simplified interfaces to treat some specific cases, such as, for instance, *polar geometry*. Besides, there are pre-defined *mapping* functions in case the user aims to build a non-uniform mesh. It is also possible to define custom mappers and apply them as easily as the pre-defined ones.

For complete flexibility and control over the program, the user can opt for a more manual approach to construct the same objects using a generic interface.

## *Re-usage of once allocated objects*

If it's necessary to re-use the same mesh or DO multiple times in the same program, pytristan spares a user the need to reconstruct them or repeatedly pass their variables between functions. Once allocated, they can be extracted from memory by calling a dedicated getter function with a simple interface *anywhere in the same program*.

# Installation

You can install pytristan using pip:

```
pip install pytristan
```

Alternatively, the source code is available on [GitHub](https://github.com/YelyzavetaV/pytristan "pytristan on GitHub").

# Dependencies

* [NumPy - the fundamental package for scientific computing with Python](https://numpy.org "NumPy")

# License

[MIT](https://opensource.org/licenses/MIT "MIT license")
